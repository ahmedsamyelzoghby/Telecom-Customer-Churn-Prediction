import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px
import seaborn as sns

# =========================
# Page Config
# =========================
st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")
st.title("📉🚶‍♂️ Customer Churn Dashboard")
st.markdown("Full Interactive Analysis 🔥")

# =========================
# Data path
# =========================
BASE_DIR = Path(__file__).resolve()
while not (BASE_DIR / "dataset").exists():
    if BASE_DIR == BASE_DIR.parent:
        break
    BASE_DIR = BASE_DIR.parent

data_path = BASE_DIR / "dataset" / "clean_Telco_Customer_Churn.csv"

# =========================
# Load Data
# =========================
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

df = load_data(data_path)

# =========================
# Sidebar Filters
# =========================
st.sidebar.header("🔎 Filter Options")

contract_type = st.sidebar.multiselect(
    "1. Contract Type",
    options=df["Contract"].unique(),
    default=df["Contract"].unique()
)

internet_filter = st.sidebar.multiselect(
    "2. Internet Service",
    options=df["InternetService"].unique(),
    default=df["InternetService"].unique()
)

payment_filter = st.sidebar.multiselect(
    "3. Payment Method",
    options=df["PaymentMethod"].unique(),
    default=df["PaymentMethod"].unique()
)

min_tenure, max_tenure = df["tenure"].min(), df["tenure"].max()
tenure_range = st.sidebar.slider(
    "4. Tenure Range (Months)",
    min_value=int(min_tenure),
    max_value=int(max_tenure),
    value=(int(min_tenure), int(max_tenure))
)

filtered_df = df[
    (df["Contract"].isin(contract_type)) &
    (df["InternetService"].isin(internet_filter)) &
    (df["PaymentMethod"].isin(payment_filter)) &
    (df["tenure"].between(tenure_range[0], tenure_range[1]))
]

st.sidebar.caption(f"📊 Showing **{len(filtered_df):,}** of **{len(df):,}** customers")

# =========================
# Tabs
# =========================
tab1, tab2, tab3, tab4 = st.tabs([
    "📌 Overview",
    "📊 Univariate Analysis",
    "📈 Bivariate Analysis",
    "📉 Multivariate Analysis"
])

# =========================
# TAB 1: Overview
# =========================
with tab1:
    st.subheader("Overview of Customer Churn")
    st.markdown("This section provides a high-level overview of the customer churn dataset, including key metrics and visualizations.")

    total_customers = len(filtered_df)
    churned_customers = filtered_df[filtered_df["Churn"] == "Yes"].shape[0] if "Churn" in filtered_df.columns else 0
    churn_rate = (churned_customers / total_customers) * 100 if total_customers > 0 else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Churned Customers", f"{churned_customers:,}")
    col3.metric("Churn Rate", f"{churn_rate:.2f}%")

    st.dataframe(filtered_df.head(10), use_container_width=True)

# =========================
# TAB 2: Univariate
# =========================
with tab2:
    st.subheader("📊 Univariate Feature Analysis")
    st.caption("Explore distributions and patterns for individual numerical or categorical variables.")
    
    selected_feature = st.selectbox(
        "Select Feature to Analyze:", 
        options=filtered_df.columns,
        index=0
    )
    
    st.markdown("---")

    if pd.api.types.is_numeric_dtype(filtered_df[selected_feature]):
        fig = px.histogram(
            filtered_df, 
            x=selected_feature, 
            marginal="box",                   
            color_discrete_sequence=['#2E86AB'], 
            title=f"<b>Distribution of {selected_feature}</b>",
            opacity=0.85
        )
        
        fig.update_layout(
            template="plotly_white",           
            title_font=dict(size=18, family="Arial", color="#1E293B"),
            xaxis_title=selected_feature,
            yaxis_title="Count",
            hovermode="x unified",            
            margin=dict(l=40, r=40, t=60, b=40)
        )
        fig.update_traces(marker_line_width=1, marker_line_color="white")

    else:
        counts_df = filtered_df[selected_feature].value_counts().reset_index()
        counts_df.columns = [selected_feature, "count"]
        
        fig = px.bar(
            counts_df, 
            x=selected_feature, 
            y="count",
            text="count",                     
            color="count",                    
            color_continuous_scale="Tealgrn", 
            title=f"<b>Frequency of {selected_feature} Categories</b>"
        )
        
        fig.update_layout(
            template="plotly_white",
            title_font=dict(size=18, family="Arial", color="#1E293B"),
            xaxis_title=selected_feature,
            yaxis_title="Number of Customers",
            coloraxis_showscale=False,         
            margin=dict(l=40, r=40, t=60, b=40)
        )
        fig.update_traces(
            texttemplate='%{text}', 
            textposition='outside',
            marker_line_color='rgb(8,48,107)',
            marker_line_width=1.5
        )

    st.plotly_chart(fig, use_container_width=True)

# =========================
# TAB 3: Bivariate
# =========================
with tab3:
    st.subheader("📈 Bivariate Feature Analysis")
    st.caption("Explore relationships between two variables and their connection to Customer Churn.")

    col1, col2 = st.columns(2)
    with col1:
        x_feature = st.selectbox(
            "Select X-axis Feature:", 
            options=filtered_df.columns,
            index=0
        )
    with col2:
        default_y_idx = 1 if len(filtered_df.columns) > 1 else 0
        y_feature = st.selectbox(
            "Select Y-axis Feature:", 
            options=filtered_df.columns,
            index=default_y_idx
        )
    
    st.markdown("---")

    if x_feature == y_feature:
        st.warning("⚠️ Please select two **different** features to perform Bivariate Analysis.")
    else:
        churn_color_map = {
            "Yes": "#E63946",
            "No": "#2A9D8F",
            1: "#E63946",
            0: "#2A9D8F"
        }

        is_x_continuous = pd.api.types.is_numeric_dtype(filtered_df[x_feature]) and filtered_df[x_feature].nunique() > 2
        is_y_continuous = pd.api.types.is_numeric_dtype(filtered_df[y_feature]) and filtered_df[y_feature].nunique() > 2

        if is_x_continuous and is_y_continuous:
            fig = px.scatter(
                filtered_df, 
                x=x_feature, 
                y=y_feature, 
                color="Churn" if "Churn" in filtered_df.columns else None,
                color_discrete_map=churn_color_map,
                title=f"<b>Scatter Plot: {x_feature} vs {y_feature}</b>",
                opacity=0.75
            )

        elif is_x_continuous != is_y_continuous:
            fig = px.box(
                filtered_df, 
                x=x_feature, 
                y=y_feature, 
                color="Churn" if "Churn" in [x_feature, y_feature] else (x_feature if not is_x_continuous else y_feature), 
                color_discrete_map=churn_color_map,
                title=f"<b>Box Plot: {y_feature} by {x_feature}</b>",
                notched=False
            )
            fig.update_traces(marker=dict(size=3, opacity=0.6))

        else:
            grouped_df = filtered_df.groupby([x_feature, y_feature]).size().reset_index(name="Count")
            grouped_df[x_feature] = grouped_df[x_feature].astype(str)
            grouped_df[y_feature] = grouped_df[y_feature].astype(str)

            fig = px.bar(
                grouped_df, 
                x=x_feature, 
                y="Count", 
                color=y_feature,
                barmode="group",
                text="Count",
                title=f"<b>Grouped Bar Chart: {x_feature} by {y_feature}</b>"
            )
            fig.update_traces(textposition='outside')

        fig.update_layout(
            template="plotly_white",
            title_font=dict(size=18, family="Arial", color="#1E293B"),
            xaxis_title=x_feature,
            yaxis_title=y_feature if (is_x_continuous or is_y_continuous) else "Count",
            hovermode="closest",
            margin=dict(l=40, r=40, t=60, b=40)
        )

        st.plotly_chart(fig, use_container_width=True)

# =========================
# TAB 4: Multivariate
# =========================
with tab4:
    st.subheader("🌐 Multivariate Feature Analysis")
    st.caption("Analyze complex interactions across three or more features simultaneously.")

    analysis_type = st.radio(
        "Select Multivariate Plot Type:",
        options=["Correlation Heatmap", "Multi-Feature Scatter & 3D Plot", "Categorical Feature Interaction"],
        horizontal=True
    )

    st.markdown("---")

    churn_color_map = {
        "Yes": "#E63946",
        "No": "#2A9D8F",
        1: "#E63946",
        0: "#2A9D8F"
    }

    # 1. Correlation Heatmap
    if analysis_type == "Correlation Heatmap":
        numeric_cols = [col for col in filtered_df.columns if pd.api.types.is_numeric_dtype(filtered_df[col])]
        
        if len(numeric_cols) < 2:
            st.warning("⚠️ Need at least two numerical columns to calculate correlation.")
        else:
            corr_matrix = filtered_df[numeric_cols].corr().round(2)
            
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                color_continuous_scale="RdBu_r",
                aspect="auto",
                title="<b>Feature Correlation Matrix</b>"
            )
            fig.update_layout(
                template="plotly_white",
                title_font=dict(size=18, family="Arial", color="#1E293B"),
                margin=dict(l=40, r=40, t=60, b=40)
            )
            st.plotly_chart(fig, use_container_width=True)

    # 2. Multi-Feature Continuous Analysis (Scatter 2D & 3D)
    elif analysis_type == "Multi-Feature Scatter & 3D Plot":
        numeric_cols = [col for col in filtered_df.columns if pd.api.types.is_numeric_dtype(filtered_df[col]) and filtered_df[col].nunique() > 2]
        
        if len(numeric_cols) < 2:
            st.warning("⚠️ At least 2 continuous numerical features are required.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                x_num = st.selectbox("Select X-axis Feature:", options=numeric_cols, index=0)
            with col2:
                default_y = 1 if len(numeric_cols) > 1 else 0
                y_num = st.selectbox("Select Y-axis Feature:", options=numeric_cols, index=default_y)

            if x_num == y_num:
                st.warning("⚠️ Please select two **different** features.")
            else:
                fig_scatter = px.scatter(
                    filtered_df,
                    x=x_num,
                    y=y_num,
                    color="Churn" if "Churn" in filtered_df.columns else None,
                    color_discrete_map=churn_color_map,
                    opacity=0.5, 
                    title=f"<b>Scatter Plot: {x_num} vs {y_num} by Churn Status</b>"
                )
                
                fig_scatter.update_traces(marker=dict(size=7))
                fig_scatter.update_layout(
                    template="plotly_white",
                    xaxis_title=x_num,
                    yaxis_title=y_num,
                    title_font=dict(size=18, family="Arial", color="#1E293B"),
                    margin=dict(l=40, r=40, t=60, b=40)
                )
                
                st.plotly_chart(fig_scatter, use_container_width=True)

                with st.expander("👉 Optional: View 3D Scatter"):
                    if len(numeric_cols) >= 3:
                        z_num = st.selectbox("Select Z-axis Feature:", options=[c for c in numeric_cols if c not in [x_num, y_num]])
                        fig_3d = px.scatter_3d(
                            filtered_df,
                            x=x_num,
                            y=y_num,
                            z=z_num,
                            color="Churn" if "Churn" in filtered_df.columns else None,
                            color_discrete_map=churn_color_map,
                            opacity=0.6,
                            title=f"<b>3D Scatter: {x_num} vs {y_num} vs {z_num}</b>"
                        )
                        st.plotly_chart(fig_3d, use_container_width=True)
                    else:
                        st.info("Need 3 numerical features for 3D plot.")

    # 3. Categorical Feature Interaction (Faceted Bar Chart)
    else:
        cat_cols = [col for col in filtered_df.columns if not (pd.api.types.is_numeric_dtype(filtered_df[col]) and filtered_df[col].nunique() > 2)]
        
        if len(cat_cols) < 2:
            st.warning("⚠️ At least 2 categorical/binary features are required.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                col_a = st.selectbox("Select Main Categorical Feature:", options=cat_cols, index=0)
            with col2:
                default_b = 1 if len(cat_cols) > 1 else 0
                col_b = st.selectbox("Select Secondary Grouping Feature:", options=cat_cols, index=default_b)

            if col_a == col_b:
                st.warning("⚠️ Please select two **different** categorical features.")
            else:
                grouped_df = filtered_df.groupby([col_a, col_b, "Churn"]).size().reset_index(name="Count")
                grouped_df[col_a] = grouped_df[col_a].astype(str)
                grouped_df[col_b] = grouped_df[col_b].astype(str)

                fig_bar = px.bar(
                    grouped_df,
                    x=col_a,
                    y="Count",
                    color="Churn",
                    facet_col=col_b, 
                    barmode="group",
                    text="Count",
                    color_discrete_map=churn_color_map,
                    title=f"<b>Customer Breakdown: {col_a} vs {col_b} by Churn Status</b>"
                )
                
                fig_bar.update_traces(textposition='outside')
                fig_bar.update_layout(
                    template="plotly_white",
                    title_font=dict(size=18, family="Arial", color="#1E293B"),
                    margin=dict(l=40, r=40, t=60, b=40)
                )
                
                st.plotly_chart(fig_bar, use_container_width=True)