import joblib
import pandas as pd
from pathlib import Path

from preprocessing import CreditScorePreprocessor

# Dekoder prediksi: angka → label yang bisa dibaca manusia
LABEL_MAP = {0: "Good", 1: "Standard", 2: "Poor"}

MODEL_PATH = Path(__file__).parent / "artifacts" / "best_model.pkl"

_preprocessor = CreditScorePreprocessor()

# ─── Lazy loading: model di-load sekali saja ─────────────────────────────────
_model = None

def _load_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model tidak ditemukan di {MODEL_PATH}.\n"
                "Jalankan pipeline.py terlebih dahulu untuk melatih model."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def predict(input_dict: dict) -> dict:
    """
    Prediksi credit score dari dictionary input nasabah.

    Catatan penting:
    - Input tetap dilewatkan ke CreditScorePreprocessor.clean(), sama seperti data latih,
      supaya aturan yang sama berlaku (misalnya nilai di luar rentang wajar → NaN)
    - 'Credit_History_Age_Months' harus sudah dihitung (years*12 + months) sebelum dikirim
    - Model yang dimuat adalah sklearn Pipeline lengkap:
      ColumnTransformer(imputer+scaler/encoder) → Classifier
      Jadi imputasi, scaling, dan encoding terjadi otomatis di dalam pipeline

    Args:
        input_dict: dict dengan key sesuai fitur model.

    Returns:
        dict dengan:
            'label'         : str  → "Good", "Standard", atau "Poor"
            'probabilities' : dict → {"Good": 0.82, "Standard": 0.14, "Poor": 0.04}
    """
    model = _load_model()

    # Konversi dict → DataFrame (model expects DataFrame), lalu bersihkan dengan aturan yang sama
    input_df = _preprocessor.clean(pd.DataFrame([input_dict]))
    input_df = input_df[_preprocessor.NUMERIC_FEATURES + _preprocessor.CATEGORICAL_FEATURES]

    # Prediksi kelas dan probabilitas
    pred_class = int(model.predict(input_df)[0])
    pred_proba = model.predict_proba(input_df)[0]

    return {
        "label": LABEL_MAP[pred_class],
        "probabilities": {
            "Good":     round(float(pred_proba[0]), 4),
            "Standard": round(float(pred_proba[1]), 4),
            "Poor":     round(float(pred_proba[2]), 4),
        },
    }


# ─── Test Case untuk Screenshot (mewakili setiap kelas) ──────────────────────
# Jalankan: python inferencing.py

# Ketiga contoh diambil dari DATA TEST (nasabah yang tidak pernah dilihat model saat training),
# tanpa nilai kosong atau salah input, dan riwayat kreditnya masuk akal untuk umurnya.
TEST_CASES = {
    # === Test Case 1: GOOD === (contoh nyata dari data test, model prediksi benar)
    # Profil: Accountant, income tinggi, suku bunga rendah (4%), credit history panjang (344 bln)
    "GOOD_CREDIT": {
        "Age": 47, "Annual_Income": 78039.48, "Monthly_Inhand_Salary": 6236.29,
        "Num_Bank_Accounts": 5, "Num_Credit_Card": 7, "Interest_Rate": 4,
        "Num_of_Loan": 4, "Delay_from_due_date": 21, "Num_of_Delayed_Payment": 17,
        "Changed_Credit_Limit": 2.59, "Num_Credit_Inquiries": 1,
        "Outstanding_Debt": 1188.91, "Credit_Utilization_Ratio": 28.62,
        "Credit_History_Age_Months": 344, "Total_EMI_per_month": 236.24,
        "Amount_invested_monthly": 97.54, "Monthly_Balance": 529.85,
        "Occupation": "Accountant", "Credit_Mix": "Good",
        "Payment_of_Min_Amount": "No",
        "Payment_Behaviour": "High_spent_Large_value_payments",
    },
    # === Test Case 2: STANDARD === (contoh nyata dari data test, model prediksi benar)
    # Profil: Entrepreneur, income menengah, outstanding debt rendah, suku bunga 15%
    "STANDARD_CREDIT": {
        "Age": 48, "Annual_Income": 27109.79, "Monthly_Inhand_Salary": 2224.15,
        "Num_Bank_Accounts": 3, "Num_Credit_Card": 5, "Interest_Rate": 15,
        "Num_of_Loan": 3, "Delay_from_due_date": 26, "Num_of_Delayed_Payment": 12,
        "Changed_Credit_Limit": 9.26, "Num_Credit_Inquiries": 1,
        "Outstanding_Debt": 320.07, "Credit_Utilization_Ratio": 28.58,
        "Credit_History_Age_Months": 184, "Total_EMI_per_month": 43.38,
        "Amount_invested_monthly": 71.97, "Monthly_Balance": 387.06,
        "Occupation": "Entrepreneur", "Credit_Mix": "Standard",
        "Payment_of_Min_Amount": "No",
        "Payment_Behaviour": "Low_spent_Medium_value_payments",
    },
    # === Test Case 3: POOR === (contoh nyata dari data test, model prediksi benar)
    # Profil: Developer, 9 pinjaman, 12 credit inquiries, outstanding debt tinggi, credit history pendek (94 bln)
    "POOR_CREDIT": {
        "Age": 32, "Annual_Income": 53751.21, "Monthly_Inhand_Salary": 4260.27,
        "Num_Bank_Accounts": 9, "Num_Credit_Card": 8, "Interest_Rate": 19,
        "Num_of_Loan": 9, "Delay_from_due_date": 16, "Num_of_Delayed_Payment": 21,
        "Changed_Credit_Limit": 8.98, "Num_Credit_Inquiries": 12,
        "Outstanding_Debt": 3351.48, "Credit_Utilization_Ratio": 28.91,
        "Credit_History_Age_Months": 94, "Total_EMI_per_month": 222.26,
        "Amount_invested_monthly": 145.59, "Monthly_Balance": 318.17,
        "Occupation": "Developer", "Credit_Mix": "Bad",
        "Payment_of_Min_Amount": "Yes",
        "Payment_Behaviour": "High_spent_Small_value_payments",
    },
}

if __name__ == "__main__":
    print("=" * 50)
    print("  Credit Score Inferencing — Smoke Test")
    print("=" * 50)
    for case_name, input_data in TEST_CASES.items():
        result = predict(input_data)
        print(f"\n[{case_name}]")
        print(f"  Prediksi  : {result['label']}")
        print(f"  Probabilitas:")
        for cls, prob in result["probabilities"].items():
            bar = "█" * int(prob * 30)
            print(f"    {cls:<10}: {prob:.2%}  {bar}")
