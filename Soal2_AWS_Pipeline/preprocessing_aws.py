"""
preprocessing_aws.py — Script preprocessing untuk SageMaker ProcessingStep.

Aturan cleaning dan cara membagi data SAMA dengan preprocessing.py versi lokal:
- Placeholder kotor ('_______', '#F%$D@*&8', dll) → NaN
- Angka yang tersimpan sebagai teks ('20364.57_') → float
- '9 Years and 8 Months' → 116 bulan
- Nilai salah input (di luar rentang wajar) → NaN, BUKAN dipotong ke batas
- Data dibagi per nasabah (Customer_ID), bukan per baris

Step ini hanya membersihkan dan membagi data. Imputasi, scaling, dan encoding
dilakukan di dalam sklearn Pipeline di train_aws.py, supaya transformasi yang
sama ikut tersimpan di model dan dipakai lagi oleh endpoint (inference_aws.py).

Kolom customer_id ikut disimpan di train.csv, karena cross-validation di
train_aws.py juga dibagi per nasabah. Kolom ini bukan fitur model.
"""

import re
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# ─── Kolom dan konstanta (sama dengan preprocessing.py lokal) ────────────────

GROUP_COL        = "Customer_ID"
DROP_COLS        = ["ID", "Customer_ID", "SSN", "Name", "Month", "Type_of_Loan"]
DIRTY_VALUES     = ["_______", "#F%$D@*&8", "!@9#%8", "__10000__", "NM", "_", ""]
STR_TO_FLOAT_COLS= ["Age", "Annual_Income", "Num_of_Loan", "Num_of_Delayed_Payment",
                    "Changed_Credit_Limit", "Outstanding_Debt", "Amount_invested_monthly"]

# Rentang nilai yang wajar. Nilai di luar rentang ini adalah salah input,
# jadi diganti NaN (lalu diisi imputer), bukan dipotong ke batasnya.
# Kalau dipotong, umur 4824 berubah jadi nasabah "berumur 90 tahun" yang palsu.
VALID_RANGES = {
    "Age":                    (14, 100),
    "Annual_Income":          (0, 250_000),
    "Num_Bank_Accounts":      (0, 15),
    "Num_Credit_Card":        (0, 15),
    "Interest_Rate":          (0, 50),
    "Num_of_Loan":            (0, 15),
    "Num_Credit_Inquiries":   (0, 25),
    "Num_of_Delayed_Payment": (0, 30),
    "Total_EMI_per_month":    (0, 5_000),
}

NUMERIC_FEATURES = ["Age","Annual_Income","Monthly_Inhand_Salary","Num_Bank_Accounts",
                    "Num_Credit_Card","Interest_Rate","Num_of_Loan","Delay_from_due_date",
                    "Num_of_Delayed_Payment","Changed_Credit_Limit","Num_Credit_Inquiries",
                    "Outstanding_Debt","Credit_Utilization_Ratio","Credit_History_Age_Months",
                    "Total_EMI_per_month","Amount_invested_monthly","Monthly_Balance"]
CATEGORICAL_FEATURES = ["Occupation","Credit_Mix","Payment_of_Min_Amount","Payment_Behaviour"]
TARGET_MAP       = {"Good": 0, "Standard": 1, "Poor": 2}


def parse_credit_history_age(val) -> float:
    if pd.isna(val) or not isinstance(val, str):
        return np.nan
    m = re.match(r"(\d+)\s+Years?\s+and\s+(\d+)\s+Months?", str(val), re.IGNORECASE)
    return float(int(m.group(1)) * 12 + int(m.group(2))) if m else np.nan

def str_to_float(val) -> float:
    if pd.isna(val): return np.nan
    s = re.sub(r"[^\d.\-]", "", str(val).strip())
    try: return float(s)
    except (ValueError, TypeError): return np.nan

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Dipakai saat training (di sini) dan saat prediksi di endpoint (inference_aws.py)."""
    df = df.copy()
    df.replace(DIRTY_VALUES, np.nan, inplace=True)
    if "Credit_History_Age" in df.columns and "Credit_History_Age_Months" not in df.columns:
        df["Credit_History_Age_Months"] = df["Credit_History_Age"].apply(parse_credit_history_age)
        df.drop(columns=["Credit_History_Age"], errors="ignore", inplace=True)
    for col in STR_TO_FLOAT_COLS:
        if col in df.columns:
            df[col] = df[col].apply(str_to_float)
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col, (lo, hi) in VALID_RANGES.items():
        if col in df.columns:
            df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)
    return df


if __name__ == "__main__":
    # Deteksi environment
    if os.path.exists("/opt/ml/processing"):
        input_dir = "/opt/ml/processing/ingested"
        out_train = "/opt/ml/processing/train"
        out_test  = "/opt/ml/processing/test"
    else:
        base = "/home/ec2-user/SageMaker/credit_scoring"
        input_dir = f"{base}/ingested"
        out_train = f"{base}/train"
        out_test  = f"{base}/test"

    for d in [out_train, out_test]:
        os.makedirs(d, exist_ok=True)

    input_file = os.path.join(input_dir, "credit_score.csv")

    if os.path.exists(input_file):
        df = pd.read_csv(input_file)
        y  = df["Credit_Score"].map(TARGET_MAP)
        groups = df[GROUP_COL]
        df.drop(columns=["Credit_Score"], inplace=True)

        df_clean = clean(df)
        X = df_clean[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

        # Split per nasabah: semua baris milik satu nasabah masuk ke train saja atau test saja
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
        assert not set(groups.iloc[train_idx]) & set(groups.iloc[test_idx]), "Ada nasabah di train dan test!"

        # Simpan RAW (belum di-impute/scale/encode) — transformasi dilakukan di dalam
        # Pipeline saat training (train_aws.py), bukan di sini.
        train_df = X.iloc[train_idx].copy()
        train_df["customer_id"] = groups.iloc[train_idx].values
        train_df["target"] = y.iloc[train_idx].values
        test_df = X.iloc[test_idx].copy()
        test_df["target"] = y.iloc[test_idx].values

        train_df.to_csv(os.path.join(out_train, "train.csv"), index=False)
        test_df.to_csv(os.path.join(out_test,  "test.csv"),  index=False)

        print("✅ Preprocessing (cleaning + split per nasabah) complete — data masih RAW.")
        print(f"   Train: {len(train_df)} baris | Test: {len(test_df)} baris | "
              f"Nasabah: {groups.iloc[train_idx].nunique()} latih, {groups.iloc[test_idx].nunique()} uji")
    else:
        print(f"❌ File tidak ditemukan: {input_file}")
