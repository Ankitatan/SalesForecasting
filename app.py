import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

# -------------------------------
# App Configuration
# -------------------------------
st.set_page_config(
    page_title="Rossmann Sales Forecasting",
    layout="wide"
)

st.title("Rossmann Store Sales Forecasting")
st.write("Predict daily sales using a trained Machine Learning model")

# -------------------------------
# Load Model
# -------------------------------
MODEL_PATH = "models/best_sales_forecast_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("Input Parameters")

customers = st.sidebar.number_input("Customers", min_value=0, value=500)
promo = st.sidebar.selectbox("Promo", [0, 1])
day_of_week = st.sidebar.selectbox(
    "Day of Week",
    options=[0,1,2,3,4,5,6],
    format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x]
)
is_weekend = 1 if day_of_week >= 5 else 0
month = st.sidebar.selectbox("Month", list(range(1,13)))

store_type = st.sidebar.selectbox("Store Type", ["a", "b", "c", "d"])
assortment = st.sidebar.selectbox("Assortment", ["a", "b", "c"])

competition_distance = st.sidebar.number_input(
    "Competition Distance (meters)", min_value=1.0, value=1000.0
)

# -------------------------------
# Feature Engineering (MATCH TASK 2)
# -------------------------------
sales_per_customer = customers / max(customers, 1)
log_competition_distance = np.log1p(competition_distance)
promo_store_type = f"{promo}_{store_type}"

# Build input dataframe
input_df = pd.DataFrame([{
    "Customers": customers,
    "Sales_per_Customer": sales_per_customer,
    "Promo": promo,
    "DayOfWeek": day_of_week,
    "IsWeekend": is_weekend,
    "Month": month,
    "StoreType": store_type,
    "Assortment": assortment,
    "Promo_StoreType": promo_store_type,
    "Log_CompetitionDistance": log_competition_distance
}])

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Sales"):
    prediction = model.predict(input_df)[0]

    st.success(f"Predicted Sales: ₹ {prediction:,.2f}")

    # Visualization
    fig, ax = plt.subplots()
    ax.bar(["Predicted Sales"], [prediction])
    ax.set_ylabel("Sales Amount")
    st.pyplot(fig)

    # Downloadable CSV
    result_df = input_df.copy()
    result_df["Predicted_Sales"] = prediction

    csv = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Prediction CSV",
        data=csv,
        file_name="sales_prediction.csv",
        mime="text/csv"
    )

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Built using sklearn Pipelines | Deployed with Streamlit")


## Dataset Usage Justification (Why Only `train.csv` and `store.csv` Were Used)


# Although four datasets were provided (`train.csv`, `store.csv`, `test.csv`, and `sample_submission.csv`), only **`train.csv`** and **`store.csv`** were used # throughout Task 1–3. This design choice is intentional and aligned with real-world machine learning systems.


### `train.csv` — Used for Modeling
# - Contains the target variable **`Sales`**, which is mandatory for:
#   - Exploratory Data Analysis (EDA)
#   - Feature engineering
#  - Model training and validation
# - Represents historical, labeled data used to learn sales patterns.


### `store.csv` — Used for Feature Enrichment
# - Provides static store-level attributes such as:
#  - StoreType
#  - Assortment
#  - CompetitionDistance
#  - Promo participation
#- These features significantly influence sales behavior and are merged with the training data to improve predictive performance.


### `test.csv` — Not Used by Design
# - Does **not** contain the target variable (`Sales`), making it unsuitable for:
#   - Model evaluation
#   - Loss computation
# - In this project, the **Streamlit application replaces `test.csv`** by generating real-time, unseen inputs, which mirrors real business forecasting scenarios.


### `sample_submission.csv` — Not Used
# - Serves only as a submission format template for Kaggle-style competitions.
# - Provides no analytical or modeling value in a production or deployment-focused project.


### Business & MLOps Alignment
# This approach reflects how forecasting systems are built in industry:
# - Models are trained on historical labeled data
# - Reference data (store metadata) is joined during training and inference
# - Future sales are predicted using live inputs, not pre-defined test files


# **Therefore, excluding `test.csv` and `sample_submission.csv` is both technically correct and business-realistic.**


# ---
