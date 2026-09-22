"""
app_aws.py — Streamlit app yang memanggil SageMaker Endpoint via boto3.

POLA UNIFIED (perbaikan bug training-serving skew):
Endpoint sekarang menerima DATA MENTAH (raw) — termasuk kolom kategorikal
sebagai STRING ASLI ("Scientist", "Good", dst), BUKAN hasil encoding manual
seperti sebelumnya. Preprocessing (scaling + encoding) terjadi otomatis di
dalam model (Unified Pipeline) di sisi endpoint. Mapping manual
(OCCUPATION_MAP, dkk) sudah dihapus karena sudah tidak diperlukan lagi.

Cara jalankan di EC2:
  export ENDPOINT_NAME="credit-score-endpoint"
  export AWS_REGION="us-east-1"
  streamlit run app_aws.py --server.port 8501
"""

import json
import os
import boto3
import pandas as pd
import streamlit as st
from botocore.exceptions import ClientError, NoCredentialsError

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
    age            = st.number_input("Umur",                        18, 90,       34)
    annual_income  = st.number_input("Annual Income (USD)",         0.0,500000.0, 60000.0, step=1000.0)
    monthly_salary = st.number_input("Monthly Inhand Salary (USD)", 0.0, 50000.0, 5000.0, step=100.0)
    monthly_balance= st.number_input("Monthly Balance (USD)",      -5000.0,50000.0,1000.0,step=100.0)
    occupation     = st.selectbox("Occupation", OCCUPATION_OPTIONS)

with col2:
    st.subheader("🏦 Data Kredit")
    num_bank_acc   = st.number_input("Jumlah Rekening Bank",    0, 20,  3)
    num_credit_card= st.number_input("Jumlah Kartu Kredit",     0, 20,  4)
    interest_rate  = st.number_input("Interest Rate (%)",       0, 100, 15)
    num_of_loan    = st.number_input("Jumlah Pinjaman",         0, 20,  2)
    outstanding_debt=st.number_input("Outstanding Debt (USD)",  0.0,50000.0,1500.0,step=100.0)
    credit_mix     = st.selectbox("Credit Mix", CREDIT_MIX_OPTIONS)

    st.subheader("📅 Riwayat Kredit")
    ch_years       = st.number_input("Credit History (Tahun)",  0, 50, 10)
    ch_months_extra= st.number_input("Credit History (Bulan+)", 0, 11, 0)
    credit_hist_months = int(ch_years) * 12 + int(ch_months_extra)

with col3:
    st.subheader("💸 Perilaku Pembayaran")
    delay_due_date = st.number_input("Delay from Due Date (hari)", 0, 180, 10)
    num_delayed_pay= st.number_input("Jumlah Delayed Payment",    0, 50,  2)
    changed_credit = st.number_input("Changed Credit Limit (%)", -20.0,50.0,5.0,step=0.5)
    num_credit_inq = st.number_input("Jumlah Credit Inquiries",   0, 50,  3)
    credit_util    = st.number_input("Credit Utilization (%)",    0.0,100.0,30.0,step=1.0)
    total_emi      = st.number_input("Total EMI per Month (USD)", 0.0,10000.0,250.0,step=10.0)
    amount_invested= st.number_input("Amount Invested Monthly",   0.0,10000.0,300.0,step=10.0)
    payment_min    = st.radio("Payment of Min Amount", PAYMENT_MIN_OPTIONS)
    payment_beh    = st.selectbox("Payment Behaviour", PAYMENT_BEHAVIOUR_OPTIONS)

st.markdown("---")

if st.button("🔍 Prediksi via SageMaker", type="primary", use_container_width=True):
    # Kirim nilai RAW apa adanya — TIDAK ada encoding manual lagi.
    # Preprocessing (scaling + encoding) terjadi otomatis di dalam model
    # (Unified Pipeline) di sisi endpoint.
    features = [
        int(age), float(annual_income), float(monthly_salary),
        int(num_bank_acc), int(num_credit_card), int(interest_rate),
        int(num_of_loan), int(delay_due_date), int(num_delayed_pay),
        float(changed_credit), int(num_credit_inq), float(outstanding_debt),
        float(credit_util), int(credit_hist_months), float(total_emi),
        float(amount_invested), float(monthly_balance),
        occupation,      # string asli, bukan hasil encoding manual
        credit_mix,      # string asli
        payment_min,     # string asli
        payment_beh,     # string asli
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
    proba_df = pd.DataFrame({
        "Kelas": ["Good", "Standard", "Poor"],
        "Probabilitas": proba,
    }).set_index("Kelas")
    st.bar_chart(proba_df)"""
app_aws.py — Streamlit app yang memanggil SageMaker Endpoint via boto3.

POLA UNIFIED (perbaikan bug training-serving skew):
Endpoint sekarang menerima DATA MENTAH (raw) — termasuk kolom kategorikal
sebagai STRING ASLI ("Scientist", "Good", dst), BUKAN hasil encoding manual
seperti sebelumnya. Preprocessing (scaling + encoding) terjadi otomatis di
dalam model (Unified Pipeline) di sisi endpoint. Mapping manual
(OCCUPATION_MAP, dkk) sudah dihapus karena sudah tidak diperlukan lagi.

Cara jalankan di EC2:
  export ENDPOINT_NAME="credit-score-endpoint"
  export AWS_REGION="us-east-1"
  streamlit run app_aws.py --server.port 8501
"""

import json
import os
import boto3
import pandas as pd
import streamlit as st
from botocore.exceptions import ClientError, NoCredentialsError

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
    age            = st.number_input("Umur",                        18, 90,       34)
    annual_income  = st.number_input("Annual Income (USD)",         0.0,500000.0, 60000.0, step=1000.0)
    monthly_salary = st.number_input("Monthly Inhand Salary (USD)", 0.0, 50000.0, 5000.0, step=100.0)
    monthly_balance= st.number_input("Monthly Balance (USD)",      -5000.0,50000.0,1000.0,step=100.0)
    occupation     = st.selectbox("Occupation", OCCUPATION_OPTIONS)

with col2:
    st.subheader("🏦 Data Kredit")
    num_bank_acc   = st.number_input("Jumlah Rekening Bank",    0, 20,  3)
    num_credit_card= st.number_input("Jumlah Kartu Kredit",     0, 20,  4)
    interest_rate  = st.number_input("Interest Rate (%)",       0, 100, 15)
    num_of_loan    = st.number_input("Jumlah Pinjaman",         0, 20,  2)
    outstanding_debt=st.number_input("Outstanding Debt (USD)",  0.0,50000.0,1500.0,step=100.0)
    credit_mix     = st.selectbox("Credit Mix", CREDIT_MIX_OPTIONS)

    st.subheader("📅 Riwayat Kredit")
    ch_years       = st.number_input("Credit History (Tahun)",  0, 50, 10)
    ch_months_extra= st.number_input("Credit History (Bulan+)", 0, 11, 0)
    credit_hist_months = int(ch_years) * 12 + int(ch_months_extra)

with col3:
    st.subheader("💸 Perilaku Pembayaran")
    delay_due_date = st.number_input("Delay from Due Date (hari)", 0, 180, 10)
    num_delayed_pay= st.number_input("Jumlah Delayed Payment",    0, 50,  2)
    changed_credit = st.number_input("Changed Credit Limit (%)", -20.0,50.0,5.0,step=0.5)
    num_credit_inq = st.number_input("Jumlah Credit Inquiries",   0, 50,  3)
    credit_util    = st.number_input("Credit Utilization (%)",    0.0,100.0,30.0,step=1.0)
    total_emi      = st.number_input("Total EMI per Month (USD)", 0.0,10000.0,250.0,step=10.0)
    amount_invested= st.number_input("Amount Invested Monthly",   0.0,10000.0,300.0,step=10.0)
    payment_min    = st.radio("Payment of Min Amount", PAYMENT_MIN_OPTIONS)
    payment_beh    = st.selectbox("Payment Behaviour", PAYMENT_BEHAVIOUR_OPTIONS)

st.markdown("---")

if st.button("🔍 Prediksi via SageMaker", type="primary", use_container_width=True):
    # Kirim nilai RAW apa adanya — TIDAK ada encoding manual lagi.
    # Preprocessing (scaling + encoding) terjadi otomatis di dalam model
    # (Unified Pipeline) di sisi endpoint.
    features = [
        int(age), float(annual_income), float(monthly_salary),
        int(num_bank_acc), int(num_credit_card), int(interest_rate),
        int(num_of_loan), int(delay_due_date), int(num_delayed_pay),
        float(changed_credit), int(num_credit_inq), float(outstanding_debt),
        float(credit_util), int(credit_hist_months), float(total_emi),
        float(amount_invested), float(monthly_balance),
        occupation,      # string asli, bukan hasil encoding manual
        credit_mix,      # string asli
        payment_min,     # string asli
        payment_beh,     # string asli
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
    proba_df = pd.DataFrame({
        "Kelas": ["Good", "Standard", "Poor"],
        "Probabilitas": proba,
    }).set_index("Kelas")
    st.bar_chart(proba_df)