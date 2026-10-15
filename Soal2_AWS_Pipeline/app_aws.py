"""
app_aws.py — Streamlit app yang memanggil SageMaker Endpoint via boto3.

Aplikasi mengirim data mentah ke endpoint, termasuk kolom kategorikal sebagai teks
asli ("Scientist", "Good", dst). Pembersihan, imputasi, scaling, dan one-hot encoding
terjadi di sisi endpoint, dengan aturan yang sama seperti saat model dilatih.

Cara jalankan di EC2:
  export ENDPOINT_NAME="credit-score-endpoint"
  export AWS_REGION="us-east-1"
  streamlit run app_aws.py --server.port 8501
"""

import json
import os
import altair as alt
import boto3
import pandas as pd
import streamlit as st
from botocore.exceptions import ClientError, NoCredentialsError

from preprocessing_aws import VALID_RANGES as R

# Batas input form, sama dengan app.py versi lokal:
# - Kolom yang punya rentang wajar di preprocessing (R) memakai rentang itu, karena nilai di
#   luar rentang dianggap salah input oleh endpoint (diganti NaN).
# - Kolom lain dibatasi sekitar rentang nilai di data latih setelah dibersihkan (D).
D = {
    "Monthly_Inhand_Salary":    (0.0, 16_000.0),   # data: 304 – 15.205
    "Monthly_Balance":          (0.0, 1_600.0),    # data: 0 – 1.564
    "Outstanding_Debt":         (0.0, 5_000.0),    # data: 0 – 4.997
    "Credit_History_Years":     (0, 34),           # data: 2 – 404 bulan
    "Delay_from_due_date":      (-5, 70),          # data: -5 – 67 hari
    "Changed_Credit_Limit":     (-10.0, 40.0),     # data: -6,5 – 35,8
    "Credit_Utilization_Ratio": (20.0, 50.0),      # data: 20,1 – 50,0
    "Amount_invested_monthly":  (0.0, 2_000.0),    # data: 0 – 1.977
}

# ─── Config dari environment variable ────────────────────────────────────────
ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "credit-score-endpoint")
REGION        = os.environ.get("AWS_REGION",    "us-east-1")

# Daftar kategori (harus sama dengan nilai asli yang ada di dataset training).
# Ini HANYA dipakai buat isi pilihan dropdown Streamlit — nilainya dikirim
# apa adanya (string) ke endpoint, tidak di-encode manual di sini.
OCCUPATION_OPTIONS = [
    "Scientist","Teacher","Engineer","Entrepreneur","Developer",
    "Lawyer","Media_Manager","Doctor","Journalist","Manager",
    "Accountant","Musician","Mechanic","Writer","Architect",
]
CREDIT_MIX_OPTIONS        = ["Good", "Standard", "Bad"]
PAYMENT_MIN_OPTIONS       = ["Yes", "No"]
PAYMENT_BEHAVIOUR_OPTIONS = [
    "High_spent_Small_value_payments",
    "Low_spent_Large_value_payments",
    "High_spent_Medium_value_payments",
    "Low_spent_Small_value_payments",
    "High_spent_Large_value_payments",
    "Low_spent_Medium_value_payments",
]


@st.cache_resource
def get_runtime_client():
    return boto3.client("sagemaker-runtime", region_name=REGION)


def invoke_endpoint(features: list) -> dict:
    runtime = get_runtime_client()
    payload = {"instances": [features]}
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload),
    )
    return json.loads(response["Body"].read().decode("utf-8"))


# ─── Streamlit UI (sama dengan app.py lokal) ─────────────────────────────────
st.set_page_config(page_title="Credit Score — AWS", page_icon="☁️", layout="wide")
st.title("☁️ Credit Score Classification (AWS SageMaker)")
st.markdown(
    f"Model dijalankan via **SageMaker Endpoint**: `{ENDPOINT_NAME}` | "
    f"Region: `{REGION}`"
)
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📋 Data Pribadi & Pendapatan")
    age            = st.number_input("Umur",                        R["Age"][0], R["Age"][1], 34)
    annual_income  = st.number_input("Annual Income (USD)",         0.0, float(R["Annual_Income"][1]), 60000.0, step=1000.0)
    monthly_salary = st.number_input("Monthly Inhand Salary (USD)", *D["Monthly_Inhand_Salary"], 5000.0, step=100.0)
    monthly_balance= st.number_input("Monthly Balance (USD)",       *D["Monthly_Balance"], 400.0, step=50.0)
    occupation     = st.selectbox("Occupation", OCCUPATION_OPTIONS)

with col2:
    st.subheader("🏦 Data Kredit")
    num_bank_acc   = st.number_input("Jumlah Rekening Bank",    0, R["Num_Bank_Accounts"][1], 3)
    num_credit_card= st.number_input("Jumlah Kartu Kredit",     0, R["Num_Credit_Card"][1],   4)
    interest_rate  = st.number_input("Interest Rate (%)",       0, R["Interest_Rate"][1],     15)
    num_of_loan    = st.number_input("Jumlah Pinjaman",         0, R["Num_of_Loan"][1],       2)
    outstanding_debt=st.number_input("Outstanding Debt (USD)",  *D["Outstanding_Debt"], 1500.0, step=100.0)
    credit_mix     = st.selectbox("Credit Mix", CREDIT_MIX_OPTIONS)

    st.subheader("📅 Riwayat Kredit")
    ch_years       = st.number_input("Credit History (Tahun)",  *D["Credit_History_Years"], 10)
    ch_months_extra= st.number_input("Credit History (Bulan+)", 0, 11, 0)
    credit_hist_months = int(ch_years) * 12 + int(ch_months_extra)

with col3:
    st.subheader("💸 Perilaku Pembayaran")
    delay_due_date = st.number_input("Delay from Due Date (hari)", *D["Delay_from_due_date"], 10)
    num_delayed_pay= st.number_input("Jumlah Delayed Payment",    0, R["Num_of_Delayed_Payment"][1], 2)
    changed_credit = st.number_input("Changed Credit Limit (%)", *D["Changed_Credit_Limit"], 5.0, step=0.5)
    num_credit_inq = st.number_input("Jumlah Credit Inquiries",   0, R["Num_Credit_Inquiries"][1],   3)
    credit_util    = st.number_input("Credit Utilization (%)",    *D["Credit_Utilization_Ratio"], 30.0, step=1.0)
    total_emi      = st.number_input("Total EMI per Month (USD)", 0.0, float(R["Total_EMI_per_month"][1]), 250.0, step=10.0)
    amount_invested= st.number_input("Amount Invested Monthly",   *D["Amount_invested_monthly"], 300.0, step=10.0)
    payment_min    = st.radio("Payment of Min Amount", PAYMENT_MIN_OPTIONS)
    payment_beh    = st.selectbox("Payment Behaviour", PAYMENT_BEHAVIOUR_OPTIONS)

st.markdown("---")

if st.button("🔍 Prediksi via SageMaker", type="primary", use_container_width=True):
    # Nilai dikirim mentah. Pembersihan dan encoding terjadi di sisi endpoint.
    features = [
        int(age), float(annual_income), float(monthly_salary),
        int(num_bank_acc), int(num_credit_card), int(interest_rate),
        int(num_of_loan), int(delay_due_date), int(num_delayed_pay),
        float(changed_credit), int(num_credit_inq), float(outstanding_debt),
        float(credit_util), int(credit_hist_months), float(total_emi),
        float(amount_invested), float(monthly_balance),
        occupation, credit_mix, payment_min, payment_beh,   # teks asli
    ]

    try:
        with st.spinner("Memanggil SageMaker Endpoint..."):
            result = invoke_endpoint(features)
    except NoCredentialsError:
        st.error("❌ AWS credentials tidak ditemukan. Pastikan EC2 instance punya IAM role yang benar.")
        st.stop()
    except ClientError as e:
        st.error(f"❌ AWS error: {e.response['Error'].get('Message', str(e))}")
        st.stop()

    label = result["labels"][0]
    proba = result["probabilities"][0]  # list of 3 probs [Good, Standard, Poor]

    color_map = {"Good": "#28a745", "Standard": "#fd7e14", "Poor": "#dc3545"}
    icon_map  = {"Good": "✅",      "Standard": "⚠️",      "Poor": "❌"}

    st.markdown(f"""
    <div style="background-color:{color_map[label]}22; border-left:6px solid {color_map[label]};
                padding:20px; border-radius:8px; margin-top:10px;">
        <h2 style="color:{color_map[label]}; margin:0;">{icon_map[label]} Credit Score: <b>{label}</b></h2>
        <p style="margin:5px 0 0 0; color:gray;">Endpoint: {ENDPOINT_NAME}</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("📊 Probabilitas per Kelas")
    kelas = ["Good", "Standard", "Poor"]
    proba_df = pd.DataFrame({"Kelas": kelas, "Probabilitas": proba})
    # Urutan batang mengikuti tingkat risiko (Good → Standard → Poor), bukan urutan abjad
    grafik = (
        alt.Chart(proba_df)
        .mark_bar()
        .encode(
            x=alt.X("Kelas:N", sort=kelas, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Probabilitas:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Kelas:N", scale=alt.Scale(domain=kelas, range=[color_map[k] for k in kelas]), legend=None),
            tooltip=["Kelas", alt.Tooltip("Probabilitas:Q", format=".0%")],
        )
    )
    st.altair_chart(grafik, use_container_width=True)
