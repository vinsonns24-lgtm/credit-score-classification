"""
app.py — Streamlit Web App untuk Credit Score Classification.

Cara jalankan:
    streamlit run app.py

Pola diambil dari app_churnPipeline.py (Unified approach):
- Load model SEKALI (via inferencing.py)
- Form input → DataFrame → model.predict() → tampilkan hasil
"""

import streamlit as st
import pandas as pd
from inferencing import predict
from preprocessing import CreditScorePreprocessor

# Batas input form mengikuti rentang wajar yang dipakai saat training.
# Nilai di luar rentang ini dianggap salah input oleh model (diganti NaN),
# jadi form tidak mengizinkannya supaya tidak diam-diam diabaikan.
R = CreditScorePreprocessor.VALID_RANGES

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Score Prediction",
    page_icon="💳",
    layout="wide",
)

# ─── Header ──────────────────────────────────────────────────────────────────
st.title("💳 Credit Score Classification")
st.markdown(
    "Masukkan data nasabah untuk memprediksi **Credit Score**: "
    "🟢 **Good** | 🟡 **Standard** | 🔴 **Poor**"
)
st.markdown("---")

# ─── Input Form (3 kolom) ────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📋 Data Pribadi & Pendapatan")
    age            = st.number_input("Umur (tahun)",             18, R["Age"][1], 34)
    annual_income  = st.number_input("Annual Income (USD)",      0.0, float(R["Annual_Income"][1]), 60000.0, step=1000.0)
    monthly_salary = st.number_input("Monthly Inhand Salary (USD)", 0.0, 50000.0, 5000.0, step=100.0)
    monthly_balance= st.number_input("Monthly Balance (USD)",   -5000.0, 50000.0, 1000.0, step=100.0)
    occupation     = st.selectbox("Occupation", [
        "Scientist", "Teacher", "Engineer", "Entrepreneur", "Developer",
        "Lawyer", "Media_Manager", "Doctor", "Journalist", "Manager",
        "Accountant", "Musician", "Mechanic", "Writer", "Architect",
    ])

with col2:
    st.subheader("🏦 Data Rekening & Kredit")
    num_bank_acc   = st.number_input("Jumlah Rekening Bank",     0, R["Num_Bank_Accounts"][1],  3)
    num_credit_card= st.number_input("Jumlah Kartu Kredit",      0, R["Num_Credit_Card"][1],    4)
    interest_rate  = st.number_input("Interest Rate (%)",        0, R["Interest_Rate"][1],     15)
    num_of_loan    = st.number_input("Jumlah Pinjaman Aktif",    0, R["Num_of_Loan"][1],        2)
    outstanding_debt=st.number_input("Outstanding Debt (USD)",   0.0, 50000.0, 1500.0, step=100.0)
    credit_mix     = st.selectbox("Credit Mix", ["Standard", "Good", "Bad"])

    st.subheader("📅 Riwayat Kredit")
    ch_years       = st.number_input("Credit History (Tahun)",   0, 50,  10)
    ch_months_extra= st.number_input("Credit History (Bulan tambahan)", 0, 11, 0)
    credit_hist_months = int(ch_years) * 12 + int(ch_months_extra)
    st.info(f"Total Credit History: **{credit_hist_months} bulan**")

with col3:
    st.subheader("💸 Perilaku Pembayaran")
    delay_due_date = st.number_input("Delay from Due Date (hari)", 0, 180, 10)
    num_delayed_pay= st.number_input("Jumlah Delayed Payment",    0, R["Num_of_Delayed_Payment"][1], 2)
    changed_credit = st.number_input("Changed Credit Limit (%)",  -20.0, 50.0, 5.0, step=0.5)
    num_credit_inq = st.number_input("Jumlah Credit Inquiries",   0, R["Num_Credit_Inquiries"][1],   3)
    credit_util    = st.number_input("Credit Utilization Ratio (%)", 0.0, 100.0, 30.0, step=1.0)
    total_emi      = st.number_input("Total EMI per Month (USD)", 0.0, float(R["Total_EMI_per_month"][1]), 250.0, step=10.0)
    amount_invested= st.number_input("Amount Invested Monthly (USD)", 0.0, 10000.0, 300.0, step=10.0)
    payment_min    = st.radio("Payment of Min Amount", ["Yes", "No"])
    payment_beh    = st.selectbox("Payment Behaviour", [
        "High_spent_Small_value_payments",
        "Low_spent_Large_value_payments",
        "High_spent_Medium_value_payments",
        "Low_spent_Small_value_payments",
        "High_spent_Large_value_payments",
        "Low_spent_Medium_value_payments",
    ])

st.markdown("---")

# ─── Prediction Button ───────────────────────────────────────────────────────
if st.button("🔍 Prediksi Credit Score", type="primary", width="stretch"):

    # Kumpulkan semua input ke dalam dict
    input_data = {
        "Age":                      int(age),
        "Annual_Income":            float(annual_income),
        "Monthly_Inhand_Salary":    float(monthly_salary),
        "Num_Bank_Accounts":        int(num_bank_acc),
        "Num_Credit_Card":          int(num_credit_card),
        "Interest_Rate":            int(interest_rate),
        "Num_of_Loan":              int(num_of_loan),
        "Delay_from_due_date":      int(delay_due_date),
        "Num_of_Delayed_Payment":   int(num_delayed_pay),
        "Changed_Credit_Limit":     float(changed_credit),
        "Num_Credit_Inquiries":     int(num_credit_inq),
        "Outstanding_Debt":         float(outstanding_debt),
        "Credit_Utilization_Ratio": float(credit_util),
        "Credit_History_Age_Months":int(credit_hist_months),
        "Total_EMI_per_month":      float(total_emi),
        "Amount_invested_monthly":  float(amount_invested),
        "Monthly_Balance":          float(monthly_balance),
        "Occupation":               occupation,
        "Credit_Mix":               credit_mix,
        "Payment_of_Min_Amount":    payment_min,
        "Payment_Behaviour":        payment_beh,
    }

    with st.spinner("Memproses prediksi..."):
        result = predict(input_data)

    label = result["label"]
    proba = result["probabilities"]

    # ─── Tampilkan hasil ─────────────────────────────────────────────────────
    color_map = {"Good": "#28a745", "Standard": "#fd7e14", "Poor": "#dc3545"}
    icon_map  = {"Good": "✅",      "Standard": "⚠️",      "Poor": "❌"}
    color = color_map[label]
    icon  = icon_map[label]

    st.markdown(f"""
    <div style="background-color:{color}22; border-left:6px solid {color};
                padding:20px; border-radius:8px; margin-top:10px;">
        <h2 style="color:{color}; margin:0;">{icon} Credit Score: <b>{label}</b></h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📊 Probabilitas per Kelas")
    proba_df = pd.DataFrame(
        {"Kelas": list(proba.keys()), "Probabilitas": list(proba.values())}
    ).set_index("Kelas")
    st.bar_chart(proba_df)

    # Tampilkan detail input
    with st.expander("📄 Lihat Detail Input"):
        # astype(str): kolom berisi campuran angka dan teks, jadi ditampilkan sebagai teks
        input_display = pd.DataFrame([input_data]).T.rename(columns={0: "Nilai"}).astype(str)
        st.dataframe(input_display, width="stretch")
