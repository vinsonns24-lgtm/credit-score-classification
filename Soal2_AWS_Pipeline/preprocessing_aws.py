"""
preprocessing_aws.py — Script preprocessing untuk SageMaker ProcessingStep.

PERUBAHAN PENTING (perbaikan bug training-serving skew):
Step ini SEKARANG hanya melakukan CLEANING + SPLIT data mentah (raw), TIDAK
lagi fit/transform ColumnTransformer di sini. Kenapa? Karena sebelumnya
ColumnTransformer di-fit terpisah di step ini, sedangkan classifier dilatih
di step lain (train_aws.py) tanpa menyertakan transformer itu ke proses
inference — akibatnya endpoint menerima data mentah padahal model dilatih
di atas data yang sudah di-scale/encode (training-serving skew).

Solusi (pola Unified, sama seperti preprocessing.py + train.py versi lokal):
ColumnTransformer sekarang digabung jadi SATU sklearn Pipeline bersama
classifier di dalam train_aws.py, lalu disimpan sebagai satu best_model.pkl.
Jadi step ini cukup keluarkan train.csv/test.csv dalam bentuk RAW (belum
di-scale/encode) — transformasi terjadi otomatis saat training & inference
karena sudah "menempel" di dalam pipeline yang sama.
"""

import re
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ─── Kolom dan konstanta (sama dengan preprocessing.py lokal) ────────────────

DROP_COLS        = ["ID", "Customer_ID", "SSN", "Name", "Month", "Type_of_Loan"]
DIRTY_VALUES     = ["_______", "#F%$D@*&8", "!@9#%8", "__10000__", "NM", "_", ""]
STR_TO_FLOAT_COLS= ["Age", "Annual_Income", "Num_of_Loan", "Num_of_Delayed_Payment",
                    "Changed_Credit_Limit", "Outstanding_Debt", "Amount_invested_monthly"]
OUTLIER_CAPS     = {"Age":(18,90),"Num_Bank_Accounts":(0,20),"Num_Credit_Card":(0,20),
                    "Interest_Rate":(0,100),"Num_of_Loan":(0,20),
                    "Num_Credit_Inquiries":(0,50),"Delay_from_due_date":(0,180),
                    "Num_of_Delayed_Payment":(0,50)}
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
    except: return np.nan

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.replace(DIRTY_VALUES, np.nan, inplace=True)
    if "Credit_History_Age" in df.columns:
        df["Credit_History_Age_Months"] = df["Credit_History_Age"].apply(parse_credit_history_age)
        df.drop(columns=["Credit_History_Age"], errors="ignore", inplace=True)
    for col in STR_TO_FLOAT_COLS:
        if col in df.columns:
            df[col] = df[col].apply(str_to_float)
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col, (lo, hi) in OUTLIER_CAPS.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)
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
        df.drop(columns=["Credit_Score"], inplace=True)

        df_clean = clean(df)
        X = df_clean[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Simpan RAW (belum di-scale/encode) — transform dilakukan di dalam
        # Pipeline saat training (train_aws.py), bukan di sini.
        train_df = X_train.copy()
        train_df["target"] = y_train.values
        test_df  = X_test.copy()
        test_df["target"]  = y_test.values

        train_df.to_csv(os.path.join(out_train, "train.csv"), index=False)
        test_df.to_csv(os.path.join(out_test,  "test.csv"),  index=False)

        print("✅ Preprocessing (cleaning + split) complete — data masih RAW.")
        print(f"   Train: {X_train.shape} | Test: {X_test.shape}")
    else:
        print(f"❌ File tidak ditemukan: {input_file}")