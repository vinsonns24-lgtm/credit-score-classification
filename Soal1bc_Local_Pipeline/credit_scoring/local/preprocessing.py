import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SklearnPipeline


class CreditScorePreprocessor:
    """
    Step 2: Bersihkan data Credit Score yang kotor dan build sklearn ColumnTransformer.

    Data Credit Score punya banyak masalah:
    - Placeholder kotor: '_______', '#F%$D@*&8', '!@9#%8', '__10000__', 'NM'
    - Kolom numerik tersimpan sebagai string: '20364.57_', '3_', dll
    - Credit_History_Age dalam format teks: '9 Years and 8 Months' → 116 bulan
    - Nilai salah input: umur 4824, jumlah pinjaman -100, suku bunga 1663%
    - Missing values di banyak kolom
    - Satu nasabah muncul di beberapa bulan (25.000 baris dari 11.254 nasabah),
      sehingga data harus dibagi per nasabah, bukan per baris.

    """

    # Kolom pengelompokan: dipakai untuk membagi data per nasabah
    GROUP_COL = "Customer_ID"

    # Kolom identifier yang tidak prediktif → drop
    DROP_COLS = ["ID", "Customer_ID", "SSN", "Name", "Month", "Type_of_Loan"]

    # Nilai placeholder kotor → ganti dengan NaN
    DIRTY_VALUES = ["_______", "#F%$D@*&8", "!@9#%8", "__10000__", "NM", "_", ""]

    # Kolom yang tersimpan sebagai string tapi seharusnya float
    STR_TO_FLOAT_COLS = [
        "Age", "Annual_Income", "Num_of_Loan", "Num_of_Delayed_Payment",
        "Changed_Credit_Limit", "Outstanding_Debt", "Amount_invested_monthly",
    ]

    # Rentang nilai yang wajar. Nilai di luar rentang ini adalah salah input,
    # jadi diganti NaN (lalu diisi imputer), BUKAN dipotong ke batasnya.
    # Kalau dipotong, umur 4824 berubah jadi nasabah "berumur 90 tahun" yang palsu.
    # Batas dipilih dari data: nilai asli berkumpul di rentang ini, lalu ada celah
    # kosong sebelum nilai rusak. Contoh: suku bunga asli 1–34%, tidak ada nilai
    # 35–72%, dan nilai rusak mulai dari 73%.
    VALID_RANGES = {
        "Age":                    (14, 100),      # asli 14–56, rusak: -500 dan ratusan–ribuan
        "Annual_Income":          (0, 250_000),   # asli s.d. ~180 rb, rusak mulai ~267 rb
        "Num_Bank_Accounts":      (0, 15),
        "Num_Credit_Card":        (0, 15),
        "Interest_Rate":          (0, 50),
        "Num_of_Loan":            (0, 15),        # rusak: -100 dan puluhan–ribuan
        "Num_Credit_Inquiries":   (0, 25),
        "Num_of_Delayed_Payment": (0, 30),
        "Total_EMI_per_month":    (0, 5_000),     # 96% di bawah 652, rusak mulai ribuan
    }

    # Fitur numerik SETELAH cleaning (Credit_History_Age_Months menggantikan Credit_History_Age)
    NUMERIC_FEATURES = [
        "Age", "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts",
        "Num_Credit_Card", "Interest_Rate", "Num_of_Loan", "Delay_from_due_date",
        "Num_of_Delayed_Payment", "Changed_Credit_Limit", "Num_Credit_Inquiries",
        "Outstanding_Debt", "Credit_Utilization_Ratio", "Credit_History_Age_Months",
        "Total_EMI_per_month", "Amount_invested_monthly", "Monthly_Balance",
    ]

    # Fitur kategorikal
    CATEGORICAL_FEATURES = [
        "Occupation", "Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour"
    ]

    # Encoding target: label → angka
    TARGET_MAP = {"Good": 0, "Standard": 1, "Poor": 2}

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state

    # ─── Private helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _parse_credit_history_age(val) -> float:
        """Ubah '9 Years and 8 Months' → 116.0 (total bulan)."""
        if pd.isna(val) or not isinstance(val, str):
            return np.nan
        m = re.match(r"(\d+)\s+Years?\s+and\s+(\d+)\s+Months?", str(val), re.IGNORECASE)
        return float(int(m.group(1)) * 12 + int(m.group(2))) if m else np.nan

    @staticmethod
    def _str_to_float(val) -> float:
        """Strip karakter non-numerik dan konversi ke float. Contoh: '20364.57_' → 20364.57"""
        if pd.isna(val):
            return np.nan
        # Hanya sisakan digit, titik desimal, dan tanda minus
        s = re.sub(r"[^\d.\-]", "", str(val).strip())
        try:
            return float(s)
        except (ValueError, TypeError):
            return np.nan

    # ─── Public methods ───────────────────────────────────────────────────────

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Terapkan semua langkah cleaning pada raw DataFrame.
        Dipanggil saat training (clean_and_split) dan juga bisa untuk single-row inference.
        """
        df = df.copy()

        # 1. Ganti nilai placeholder kotor dengan NaN
        df.replace(self.DIRTY_VALUES, np.nan, inplace=True)

        # 2. Parse Credit_History_Age → total bulan
        #    (jika sudah ada Credit_History_Age_Months, skip langkah ini)
        if "Credit_History_Age" in df.columns and "Credit_History_Age_Months" not in df.columns:
            df["Credit_History_Age_Months"] = df["Credit_History_Age"].apply(
                self._parse_credit_history_age
            )
            df.drop(columns=["Credit_History_Age"], errors="ignore", inplace=True)

        # 3. Konversi string → float untuk kolom numerik yang kotor
        for col in self.STR_TO_FLOAT_COLS:
            if col in df.columns:
                df[col] = df[col].apply(self._str_to_float)

        # 4. Pastikan semua kolom numerik bertipe float
        for col in self.NUMERIC_FEATURES:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # 5. Nilai salah input (di luar rentang wajar) → NaN
        for col, (lo, hi) in self.VALID_RANGES.items():
            if col in df.columns:
                df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan

        # 6. Drop kolom identifier (tidak prediktif)
        df.drop(columns=[c for c in self.DROP_COLS if c in df.columns], inplace=True)

        return df

    def build_column_transformer(self) -> ColumnTransformer:
        """
        Build sklearn ColumnTransformer untuk DataFrame yang sudah di-clean.
        - Numerik : median imputation + standard scaling
        - Kategorikal: mode imputation + one-hot encoding
          (Occupation dan Payment_Behaviour tidak punya urutan, jadi one-hot lebih
          tepat daripada ordinal, terutama untuk Logistic Regression)
        """
        numeric_pipeline = SklearnPipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ])

        categorical_pipeline = SklearnPipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ])

        return ColumnTransformer(transformers=[
            ("num", numeric_pipeline,      self.NUMERIC_FEATURES),
            ("cat", categorical_pipeline,  self.CATEGORICAL_FEATURES),
        ], remainder="drop")

    def clean_and_split(
        self, data_path: Path
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Load ingested CSV → clean → encode target → split train/test per nasabah.
        Dipakai oleh CreditModelTrainer.

        Return: (X_train, X_test, y_train, y_test, groups_train)
        groups_train (Customer_ID data latih) dipakai untuk cross-validation per nasabah.
        """
        df = pd.read_csv(data_path)

        # Encode target
        y = df["Credit_Score"].map(self.TARGET_MAP)
        df.drop(columns=["Credit_Score"], inplace=True)

        # Simpan Customer_ID sebelum di-drop oleh clean()
        groups = df[self.GROUP_COL]

        # Clean
        df_clean = self.clean(df)

        # Pilih hanya kolom fitur yang diperlukan
        X = df_clean[self.NUMERIC_FEATURES + self.CATEGORICAL_FEATURES]

        print(f"  Fitur: {X.shape[1]} kolom | Sampel: {X.shape[0]:,} baris | Nasabah: {groups.nunique():,}")
        print(f"  Missing values total: {X.isna().sum().sum():,} sel")

        # Split per nasabah: semua baris milik satu nasabah masuk ke train saja atau test saja.
        # Kalau dibagi acak per baris, ~80% baris test nasabahnya juga ada di train,
        # sehingga skor test terlihat lebih tinggi dari kemampuan model yang sebenarnya.
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=self.test_size, random_state=self.random_state
        )
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))

        overlap = set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])
        assert not overlap, "❌ Ada nasabah yang muncul di train dan test!"

        return (
            X.iloc[train_idx], X.iloc[test_idx],
            y.iloc[train_idx], y.iloc[test_idx],
            groups.iloc[train_idx],
        )


if __name__ == "__main__":
    prep = CreditScorePreprocessor()
    X_train, X_test, y_train, y_test, _ = prep.clean_and_split(
        Path("ingested/credit_score.csv")
    )
    print(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}")
