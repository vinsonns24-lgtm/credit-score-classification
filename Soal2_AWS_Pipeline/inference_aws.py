"""
inference_aws.py — SageMaker Endpoint inference entry point untuk Credit Score.

POLA UNIFIED (perbaikan bug training-serving skew):
best_model.pkl sekarang berisi PIPELINE LENGKAP (ColumnTransformer + Classifier
digabung jadi satu, dilatih bersama di train_aws.py) — bukan classifier polos.
Artinya endpoint ini TIDAK perlu preprocessing manual lagi: cukup terima data
RAW (nilai asli seperti "Scientist", "Good", 60000.0, dst) dan panggil
model.predict_proba() langsung. Preprocessing (scaling + encoding) otomatis
terjadi di dalam pipeline itu sendiri.

4 fungsi SageMaker contract:
  model_fn   → load model dari disk (dipanggil sekali saat container start)
  input_fn   → parse request body (per request)
  predict_fn → jalankan prediksi (per request)
  output_fn  → serialize response (per request)
"""

import json
import os
import sys
import joblib
import numpy as np
import pandas as pd

# preprocessing_aws.py ikut dikemas bersama script ini (dependencies di deploy_endpoint.ipynb),
# supaya input endpoint dibersihkan dengan aturan yang sama persis seperti data latih.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing_aws import clean

JSON_CONTENT_TYPE = "application/json"
CSV_CONTENT_TYPE  = "text/csv"

LABEL_MAP = {0: "Good", 1: "Standard", 2: "Poor"}

# Urutan fitur RAW yang diharapkan model (harus sama persis dengan urutan
# NUMERIC_FEATURES + CATEGORICAL_FEATURES di preprocessing_aws.py / train_aws.py).
# Kolom kategorikal di sini adalah NILAI ASLI (string), bukan hasil encoding
# manual — karena encoding sekarang dilakukan otomatis di dalam pipeline.
FEATURE_NAMES = [
    "Age", "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts",
    "Num_Credit_Card", "Interest_Rate", "Num_of_Loan", "Delay_from_due_date",
    "Num_of_Delayed_Payment", "Changed_Credit_Limit", "Num_Credit_Inquiries",
    "Outstanding_Debt", "Credit_Utilization_Ratio", "Credit_History_Age_Months",
    "Total_EMI_per_month", "Amount_invested_monthly", "Monthly_Balance",
    "Occupation", "Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour",
]


def model_fn(model_dir: str):
    """Load model (Unified Pipeline: preprocessing + classifier) dari best_model.pkl."""
    model_path = os.path.join(model_dir, "best_model.pkl")
    model = joblib.load(model_path)
    print(f"✅ Model (Unified Pipeline) loaded from {model_path}")
    return model


def input_fn(request_body, request_content_type: str) -> pd.DataFrame:
    """
    Parse request body menjadi DataFrame RAW (belum di-scale/encode).
    Menerima JSON: {"instances": [[val1, val2, ..., val21], ...]}
    Nilai kategorikal (Occupation, Credit_Mix, dll) dikirim sebagai STRING asli,
    bukan angka hasil encoding manual.
    """
    if request_content_type == JSON_CONTENT_TYPE:
        payload   = json.loads(request_body)
        instances = payload["instances"]
        return pd.DataFrame(instances, columns=FEATURE_NAMES)

    if request_content_type == CSV_CONTENT_TYPE:
        if isinstance(request_body, (bytes, bytearray)):
            request_body = request_body.decode("utf-8")
        rows = [line.split(",") for line in request_body.strip().splitlines() if line.strip()]
        df = pd.DataFrame(rows, columns=FEATURE_NAMES)
        # Kolom numerik perlu di-cast eksplisit karena CSV parsing manual
        # di atas menghasilkan semua kolom bertipe string/object.
        numeric_cols = FEATURE_NAMES[:17]  # 17 kolom pertama = numerik
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        return df

    raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data: pd.DataFrame, model) -> dict:
    """
    Jalankan prediksi. Input dibersihkan dulu dengan clean() dari preprocessing_aws.py,
    lalu model (Unified Pipeline) melakukan imputasi, scaling, dan one-hot encoding
    sendiri di dalam predict_proba().
    """
    # Aturan cleaning yang sama dengan data latih, misalnya nilai di luar rentang wajar → NaN
    input_data = clean(input_data)[FEATURE_NAMES]
    probs     = model.predict_proba(input_data)
    class_ids = np.argmax(probs, axis=1)
    labels    = [LABEL_MAP[int(i)] for i in class_ids]
    return {
        "probabilities": probs.tolist(),
        "predictions":   class_ids.tolist(),
        "labels":        labels,
    }


def output_fn(prediction: dict, accept_content_type: str):
    """Serialize response."""
    if accept_content_type == JSON_CONTENT_TYPE:
        return json.dumps(prediction), JSON_CONTENT_TYPE
    raise ValueError(f"Unsupported accept type: {accept_content_type}")