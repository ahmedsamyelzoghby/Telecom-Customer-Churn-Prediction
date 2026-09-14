import streamlit as st
import pandas as pd
from pathlib import Path

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Customer Churn", layout="wide") 

#==========================
# Data path
#==========================
BASE_DIR = Path(__file__).resolve()
while not (BASE_DIR / "dataset").exists():
    BASE_DIR = BASE_DIR.parent
data_path = BASE_DIR / "dataset" / "clean_Telco_Customer_Churn.csv"

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

df = load_data(data_path)

# =========================
# HOME PAGE
# =========================
st.title("📉🚶‍♂️ Customer Churn Prediction & Retention System")
st.markdown("##### Enterprise AI Solution for Real-Time Churn Risk Assessment & Customer Retention")
st.markdown("---")

# ========================
# KPIs
# ========================
st.subheader("📊 Key Performance Indicators (KPIs)")
col1, col2, col3, col4 = st.columns(4)

col1.metric("📄 Total Customers", df.shape[0])
col2.metric("📊 Baseline Churn Rate", (df['Churn'].value_counts(normalize=True).get('Yes', 0)*100).round(2).astype(str) + "%", delta_color="inverse")
col3.metric("📈 Monthly Revenue at Risk", f"${df[df['Churn'] == 'Yes']['MonthlyCharges'].sum():,.0f}")
col4.metric("📉 Model Recall Score", value="77.84%")
st.markdown("---")

st.markdown("""
    ### 📌 Overview    
    This end-to-end Machine Learning web application predicts customer churn for a telecommunications provider. 
    By analyzing customer demographics, subscribed services, and financial behavior, the system identifies 
    high-risk customers in real-time, enabling proactive retention strategies.
""")
st.markdown("---")

st.markdown("""
    #### 🧠 Model Details

    * **Algorithm:** XGBoost Classifier
    * **Resampling Technique:** SMOTE-ENN / Class Weighting
    * **Optimization Metric:** Balanced Recall & Precision
    * **Key Input Features:** Contract Type, Tenure, Monthly Charges, Internet Service, Tech Support, Payment Method
""")
st.markdown("---")

st.markdown("""
#### 🚀 System Capabilities

* **📊 Behavioral Analytics Dashboard:** Interactive visual EDA exploring key churn drivers, contract behaviors, and revenue impacts.
* **🤖 Real-time Churn Risk Assessor:** Instant prediction engine generating churn probability scores and tailored business recommendations
""")
st.markdown("---")

st.info("👈 Use the sidebar menu on the left to navigate between the Analytics Dashboard and the Prediction Engine.")  