import logging
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="wide"
)


# ============================================================
# Logging
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# Constants
# ============================================================

DECISION_THRESHOLD = 0.45

FEATURE_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges"
]

TARGET_COLUMN = "Churn"

NUMERIC_COLUMNS = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

INTERNET_DEPENDENT_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies"
]


# ============================================================
# Find Project Root
# ============================================================

def find_project_root():
    """
    Search upward from the current file until both the
    models and dataset directories are found.
    """

    current_dir = Path(__file__).resolve().parent

    for directory in [current_dir, *current_dir.parents]:

        if (
            (directory / "models").is_dir()
            and (directory / "dataset").is_dir()
        ):
            return directory

    return None


BASE_DIR = find_project_root()

if BASE_DIR is None:

    st.error(
        "Project structure could not be detected. "
        "Please make sure the app is inside the project directory."
    )

    st.stop()


MODEL_PATH = BASE_DIR / "models" / "customer_churn_model.pkl"
DATA_PATH = BASE_DIR / "dataset" / "clean_Telco_Customer_Churn.csv"


# ============================================================
# Validate Required Files
# ============================================================

if not MODEL_PATH.is_file():

    st.error(
        "The trained model could not be found. "
        "Please make sure 'customer_churn_model.pkl' exists "
        "inside the models folder."
    )

    st.stop()


if not DATA_PATH.is_file():

    st.error(
        "The dataset could not be found. "
        "Please make sure the cleaned Telco dataset exists "
        "inside the dataset folder."
    )

    st.stop()


# ============================================================
# Load Model
# ============================================================

@st.cache_resource
def load_model(path):

    # IMPORTANT:
    # Only load trusted local model files.
    # joblib uses pickle internally and should never be used
    # with untrusted/user-uploaded model files.

    return joblib.load(path)


# ============================================================
# Load Dataset
# ============================================================

@st.cache_data
def load_data(path):

    data = pd.read_csv(path)

    return data


# ============================================================
# Load Resources
# ============================================================

try:

    model = load_model(MODEL_PATH)
    df = load_data(DATA_PATH)

except Exception:

    logger.exception("Failed to load model or dataset.")

    st.error(
        "The model or dataset could not be loaded. "
        "Please verify that the project files are valid."
    )

    st.stop()


# ============================================================
# Dataset Validation
# ============================================================

required_dataset_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

missing_columns = [
    column
    for column in required_dataset_columns
    if column not in df.columns
]

if missing_columns:

    st.error(
        "The dataset is missing required columns."
    )

    st.code(", ".join(missing_columns))

    st.stop()


# ============================================================
# Numeric Data Validation
# ============================================================

for column in NUMERIC_COLUMNS:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


for column in ["tenure", "MonthlyCharges", "TotalCharges"]:

    if df[column].dropna().empty:

        st.error(
            f"The dataset contains no valid numeric values "
            f"for '{column}'."
        )

        st.stop()


# ============================================================
# Model Validation
# ============================================================

if not hasattr(model, "predict_proba"):

    st.error(
        "The loaded model does not support probability prediction."
    )

    st.stop()


# ============================================================
# Validate Model Feature Schema
# ============================================================

model_features = getattr(
    model,
    "feature_names_in_",
    None
)

if model_features is not None:

    model_features = list(model_features)

    missing_model_features = [
        column
        for column in FEATURE_COLUMNS
        if column not in model_features
    ]

    extra_model_features = [
        column
        for column in model_features
        if column not in FEATURE_COLUMNS
    ]

    if missing_model_features or extra_model_features:

        st.error(
            "The saved model expects a different feature schema "
            "than the Streamlit application."
        )

        st.stop()


# ============================================================
# Title
# ============================================================

st.markdown(
    """
    <h1 style="text-align:center;">
        📉🚶‍♂️ Customer Churn Predictor
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <p style="text-align:center; font-size:18px;">
        Predict customer churn risk and identify customers
        who may require retention actions.
    </p>
    """,
    unsafe_allow_html=True
)

st.write("")


# ============================================================
# Model Information
# ============================================================

info_col1, info_col2, info_col3 = st.columns(3)

with info_col1:

    st.metric(
        "Decision Threshold",
        f"{DECISION_THRESHOLD * 100:.0f}%"
    )

with info_col2:

    st.metric(
        "Input Features",
        len(FEATURE_COLUMNS)
    )

with info_col3:

    st.metric(
        "Model",
        "XGBoost"
    )


st.write("---")


# ============================================================
# Helper Functions
# ============================================================

def get_options(dataframe, column):

    """
    Return clean, sorted unique categorical values.
    """

    return sorted(
        dataframe[column]
        .dropna()
        .unique()
        .tolist()
    )


def get_churn_probability(model, input_data):

    """
    Safely extract the probability of the positive/churn class.
    """

    probabilities = model.predict_proba(input_data)

    if probabilities.ndim != 2 or probabilities.shape[1] < 2:

        raise ValueError(
            "The model did not return binary class probabilities."
        )

    classes = getattr(model, "classes_", None)

    if classes is not None:

        classes = list(classes)

        # Most binary churn models use 0 / 1.
        if 1 in classes:

            churn_index = classes.index(1)

            return float(
                probabilities[0][churn_index]
            )

        # Fallback for string labels.
        for label in ["Yes", "yes", "Churn", "churn"]:

            if label in classes:

                churn_index = classes.index(label)

                return float(
                    probabilities[0][churn_index]
                )

    # Safe fallback for a standard binary [0, 1] classifier.
    return float(
        probabilities[0][1]
    )


def get_risk_level(probability):

    """
    Business risk segmentation.

    Note:
    The actual churn decision is still controlled by the
    45% decision threshold.
    """

    if probability < 0.35:

        return "🟢 Low Risk"

    elif probability < 0.65:

        return "🟡 Medium Risk"

    else:

        return "🔴 High Risk"


def generate_recommendations(
    probability,
    contract,
    tenure,
    monthly_charges,
    internet_service,
    online_security,
    tech_support,
    payment_method
):

    """
    Generate profile-based retention suggestions.

    These are potential retention signals, not causal
    explanations of the model prediction.
    """

    recommendations = []

    # --------------------------------------------------------
    # Overall risk
    # --------------------------------------------------------

    if probability >= DECISION_THRESHOLD:

        recommendations.append(
            "Prioritize this customer for proactive retention outreach."
        )

        recommendations.append(
            "Consider a personalized retention offer based on "
            "the customer's service and contract profile."
        )

    else:

        recommendations.append(
            "No immediate retention intervention is required."
        )

        recommendations.append(
            "Continue monitoring the customer through normal "
            "engagement and retention processes."
        )

    # --------------------------------------------------------
    # Contract
    # --------------------------------------------------------

    if contract == "Month-to-month":

        recommendations.append(
            "Consider offering incentives for a longer-term contract."
        )

    # --------------------------------------------------------
    # Tenure
    # --------------------------------------------------------

    if tenure < 12:

        recommendations.append(
            "The customer has relatively short tenure; "
            "consider an early-lifecycle retention strategy."
        )

    # --------------------------------------------------------
    # Monthly Charges
    # --------------------------------------------------------

    monthly_charge_75th = df["MonthlyCharges"].quantile(0.75)

    if monthly_charges >= monthly_charge_75th:

        recommendations.append(
            "Review pricing and perceived value because the "
            "monthly charge is relatively high compared with "
            "the customer base."
        )

    # --------------------------------------------------------
    # Internet Service
    # --------------------------------------------------------

    if internet_service == "Fiber optic":

        recommendations.append(
            "Review fiber-service satisfaction and service quality."
        )

    # --------------------------------------------------------
    # Security / Support
    # --------------------------------------------------------

    if online_security == "No":

        recommendations.append(
            "Consider promoting Online Security as a value-added service."
        )

    if tech_support == "No":

        recommendations.append(
            "Consider offering or promoting technical support services."
        )

    # --------------------------------------------------------
    # Payment Method
    # --------------------------------------------------------

    if payment_method == "Electronic check":

        recommendations.append(
            "Consider encouraging a more convenient or automated "
            "payment method."
        )

    return recommendations


# ============================================================
# Input Section
# ============================================================

st.subheader("📋 Customer Profile & Service Details")

col1, col2, col3 = st.columns(3)


# ============================================================
# Column 1 - Demographic & Account
# ============================================================

with col1:

    st.markdown("##### 👤 Demographic & Account")

    gender = st.selectbox(
        "Gender",
        options=get_options(df, "gender")
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        options=[0, 1],
        format_func=lambda x: (
            "Yes" if x == 1 else "No"
        )
    )

    partner = st.selectbox(
        "Partner",
        options=get_options(df, "Partner")
    )

    dependents = st.selectbox(
        "Dependents",
        options=get_options(df, "Dependents")
    )

    tenure = st.number_input(
        "Tenure (months)",
        min_value=int(df["tenure"].min()),
        max_value=int(df["tenure"].max()),
        value=min(
            12,
            int(df["tenure"].max())
        ),
        step=1
    )


# ============================================================
# Column 2 - Services
# ============================================================

with col2:

    st.markdown("##### 🛠️ Subscribed Services")

    phone_service = st.selectbox(
        "Phone Service",
        options=get_options(df, "PhoneService")
    )

    # --------------------------------------------------------
    # Multiple Lines
    # --------------------------------------------------------

    if phone_service == "No":

        multiple_lines = "No phone service"

        st.selectbox(
            "Multiple Lines",
            options=["No phone service"],
            disabled=True
        )

    else:

        multiple_lines = st.selectbox(
            "Multiple Lines",
            options=[
                option
                for option in get_options(df, "MultipleLines")
                if option != "No phone service"
            ]
        )

    # --------------------------------------------------------
    # Internet Service
    # --------------------------------------------------------

    internet_service = st.selectbox(
        "Internet Service",
        options=get_options(df, "InternetService")
    )

    # --------------------------------------------------------
    # Internet-dependent Services
    # --------------------------------------------------------

    if internet_service == "No":

        online_security = "No internet service"
        online_backup = "No internet service"
        device_protection = "No internet service"

        st.selectbox(
            "Online Security",
            options=["No internet service"],
            disabled=True
        )

        st.selectbox(
            "Online Backup",
            options=["No internet service"],
            disabled=True
        )

        st.selectbox(
            "Device Protection",
            options=["No internet service"],
            disabled=True
        )

    else:

        online_security = st.selectbox(
            "Online Security",
            options=get_options(df, "OnlineSecurity")
        )

        online_backup = st.selectbox(
            "Online Backup",
            options=get_options(df, "OnlineBackup")
        )

        device_protection = st.selectbox(
            "Device Protection",
            options=get_options(df, "DeviceProtection")
        )


# ============================================================
# Column 3 - Financial & Contract
# ============================================================

with col3:

    st.markdown("##### 💳 Financials & Contract")

    # --------------------------------------------------------
    # Tech Support
    # --------------------------------------------------------

    if internet_service == "No":

        tech_support = "No internet service"

        st.selectbox(
            "Tech Support",
            options=["No internet service"],
            disabled=True
        )

    else:

        tech_support = st.selectbox(
            "Tech Support",
            options=get_options(df, "TechSupport")
        )

    # --------------------------------------------------------
    # Streaming TV
    # --------------------------------------------------------

    if internet_service == "No":

        streaming_tv = "No internet service"

        st.selectbox(
            "Streaming TV",
            options=["No internet service"],
            disabled=True
        )

    else:

        streaming_tv = st.selectbox(
            "Streaming TV",
            options=get_options(df, "StreamingTV")
        )

    # --------------------------------------------------------
    # Streaming Movies
    # --------------------------------------------------------

    if internet_service == "No":

        streaming_movies = "No internet service"

        st.selectbox(
            "Streaming Movies",
            options=["No internet service"],
            disabled=True
        )

    else:

        streaming_movies = st.selectbox(
            "Streaming Movies",
            options=get_options(df, "StreamingMovies")
        )

    # --------------------------------------------------------
    # Contract
    # --------------------------------------------------------

    contract = st.selectbox(
        "Contract",
        options=get_options(df, "Contract")
    )

    # --------------------------------------------------------
    # Paperless Billing
    # --------------------------------------------------------

    paperless_billing = st.selectbox(
        "Paperless Billing",
        options=get_options(df, "PaperlessBilling")
    )

    # --------------------------------------------------------
    # Payment Method
    # --------------------------------------------------------

    payment_method = st.selectbox(
        "Payment Method",
        options=get_options(df, "PaymentMethod")
    )

    # --------------------------------------------------------
    # Monthly Charges
    # --------------------------------------------------------

    monthly_charges = st.number_input(
        "Monthly Charges ($)",
        min_value=float(
            df["MonthlyCharges"].min()
        ),
        max_value=float(
            df["MonthlyCharges"].max()
        ),
        value=float(
            min(
                65.0,
                df["MonthlyCharges"].max()
            )
        ),
        step=1.0
    )

    # --------------------------------------------------------
    # Total Charges
    # --------------------------------------------------------

    total_charges_default = tenure * monthly_charges

    total_charges_default = min(
        max(
            total_charges_default,
            float(df["TotalCharges"].min())
        ),
        float(df["TotalCharges"].max())
    )

    total_charges = st.number_input(
        "Total Charges ($)",
        min_value=float(
            df["TotalCharges"].min()
        ),
        max_value=float(
            df["TotalCharges"].max()
        ),
        value=float(
            total_charges_default
        ),
        step=1.0,
        help=(
            "A reasonable default is estimated from "
            "tenure × monthly charges. You can adjust it "
            "to reflect the customer's actual historical charges."
        )
    )


# ============================================================
# Submit
# ============================================================

st.write("---")

submit_btn = st.button(
    "🚀 Run Churn Assessment",
    use_container_width=True
)

# ============================================================
# Prediction
# ============================================================

if submit_btn:

    # ========================================================
    # Build Input Data
    # ========================================================

    input_data = pd.DataFrame(
        [{
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }],
        columns=FEATURE_COLUMNS
    )

    # ========================================================
    # Final Input Validation
    # ========================================================

    if list(input_data.columns) != FEATURE_COLUMNS:

        st.error(
            "The prediction input does not match the trained "
            "model feature schema."
        )

        st.stop()

    try:

        # ====================================================
        # Prediction Probability
        # ====================================================

        churn_probability = get_churn_probability(
            model,
            input_data
        )

        churn_percent = churn_probability * 100

        # ====================================================
        # Decision
        # ====================================================

        is_churn = (
            churn_probability >= DECISION_THRESHOLD
        )

        # ====================================================
        # Risk Level
        # ====================================================

        risk_level = get_risk_level(
            churn_probability
        )

        # ====================================================
        # Business Recommendations
        # ====================================================

        recommendations = generate_recommendations(
            probability=churn_probability,
            contract=contract,
            tenure=tenure,
            monthly_charges=monthly_charges,
            internet_service=internet_service,
            online_security=online_security,
            tech_support=tech_support,
            payment_method=payment_method
        )

        # ====================================================
        # Results
        # ====================================================

        st.write("---")

        st.subheader("🎯 Churn Risk Assessment")

        res_col1, res_col2 = st.columns(
            [1, 1.2]
        )

        # ====================================================
        # Gauge
        # ====================================================

        with res_col1:

            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=churn_percent,
                    number={
                        "suffix": "%",
                        "font": {
                            "size": 36
                        }
                    },
                    title={
                        "text": "Churn Risk Score"
                    },
                    gauge={
                        "axis": {
                            "range": [0, 100]
                        },
                        "bar": {
                            "color": "#1E293B"
                        },
                        "steps": [
                            {
                                "range": [0, 35],
                                "color": "#2ecc71"
                            },
                            {
                                "range": [35, 65],
                                "color": "#f39c12"
                            },
                            {
                                "range": [65, 100],
                                "color": "#e74c3c"
                            }
                        ],
                        "threshold": {
                            "line": {
                                "color": "#000000",
                                "width": 4
                            },
                            "thickness": 0.75,
                            "value": DECISION_THRESHOLD * 100
                        }
                    }
                )
            )

            fig_gauge.update_layout(
                height=350,
                margin=dict(
                    l=20,
                    r=20,
                    t=60,
                    b=20
                )
            )

            st.plotly_chart(
                fig_gauge,
                use_container_width=True
            )

        # ====================================================
        # Business Result
        # ====================================================

        with res_col2:

            metric_col1, metric_col2 = st.columns(2)

            with metric_col1:

                st.metric(
                    "Churn Probability",
                    f"{churn_percent:.1f}%"
                )

            with metric_col2:

                st.metric(
                    "Decision Threshold",
                    f"{DECISION_THRESHOLD * 100:.0f}%"
                )

            st.markdown(
                f"### {risk_level}"
            )

            # ------------------------------------------------
            # Prediction Decision
            # ------------------------------------------------

            if is_churn:

                st.error(
                    "⚠️ Prediction: Customer is classified "
                    "as likely to churn."
                )

            else:

                st.success(
                    "✅ Prediction: Customer is classified "
                    "as unlikely to churn."
                )

            st.caption(
                "The churn decision is based on the selected "
                "45% decision threshold."
            )

        # ====================================================
        # Business Recommendations
        # ====================================================

        st.write("---")

        st.subheader(
            "💼 Recommended Retention Actions"
        )

        for recommendation in recommendations:

            st.markdown(
                f"- {recommendation}"
            )

        # ====================================================
        # Customer Profile Summary
        # ====================================================

        st.write("---")

        st.subheader(
            "👤 Customer Profile Summary"
        )

        summary_col1, summary_col2, summary_col3, summary_col4 = (
            st.columns(4)
        )

        with summary_col1:

            st.metric(
                "Tenure",
                f"{tenure} months"
            )

        with summary_col2:

            st.metric(
                "Monthly Charges",
                f"${monthly_charges:,.2f}"
            )

        with summary_col3:

            st.metric(
                "Total Charges",
                f"${total_charges:,.2f}"
            )

        with summary_col4:

            st.metric(
                "Contract",
                contract
            )

    except Exception:

        logger.exception(
            "Prediction failed."
        )

        st.error(
            "Unable to generate the churn prediction. "
            "Please verify the customer inputs and the trained model."
        )