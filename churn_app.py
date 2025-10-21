from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import pickle
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")

# Load model and scaler
with open('xgboost_churn_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('xgboost_scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Features that model expects (must match training exactly)
feature_names = [
    "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary",
    "Geography_France", "Geography_Germany", "Geography_Spain",
    "Gender_Female", "Gender_Male",
    "HasCrCard", "IsActiveMember"
]

# Columns requiring scaling
scale_vars = ["CreditScore", "EstimatedSalary", "Tenure", "Balance", "Age", "NumOfProducts"]

# Default values
default_values = {
    "CreditScore": 600,
    "Age": 30,
    "Tenure": 2,
    "Balance": 8000,
    "NumOfProducts": 2,
    "EstimatedSalary": 60000,
    "Geography_France": True,
    "Geography_Germany": False,
    "Geography_Spain": False,
    "Gender_Female": True,
    "Gender_Male": False,
    "HasCrCard": 1,
    "IsActiveMember": 1
}

# Sidebar Inputs
st.sidebar.image("Pic 1.png", use_container_width=True)
st.sidebar.header("User Inputs")

user_inputs = {}
for feature in feature_names:
    if feature in scale_vars:
        user_inputs[feature] = st.sidebar.number_input(feature, value=default_values[feature])
    elif feature.startswith("Geography_") or feature.startswith("Gender_"):
        user_inputs[feature] = st.sidebar.checkbox(feature, value=default_values[feature])
    elif feature in ["HasCrCard", "IsActiveMember"]:
        user_inputs[feature] = st.sidebar.selectbox(feature, [0, 1], index=default_values[feature])
    else:
        user_inputs[feature] = st.sidebar.number_input(feature, value=default_values[feature])

# ✅ Convert inputs to DataFrame
input_data = pd.DataFrame([user_inputs])

# ✅ Align columns exactly as seen during training
expected_columns = list(scaler.feature_names_in_)  # Columns seen by scaler at fit time

# Add missing columns with 0 values
for col in expected_columns:
    if col not in input_data.columns:
        input_data[col] = 0

# Drop any extra columns not seen at training time
input_data = input_data[expected_columns]

# ✅ Scale using the same scaler
input_data_scaled = pd.DataFrame(
    scaler.transform(input_data),
    columns=expected_columns
)



# App layout
st.image("Pic 2.png", use_container_width=True)
st.title("Customer Churn Prediction")

left_col, right_col = st.columns(2)

# Left column: Feature Importance
with left_col:
    st.header("Feature Importance")
    feature_importance_df = pd.read_excel("feature_importance.xlsx", usecols=["Feature", "Feature Importance Score"])
    fig = px.bar(
        feature_importance_df.sort_values(by="Feature Importance Score", ascending=True),
        x="Feature Importance Score",
        y="Feature",
        orientation="h",
        title="Feature Importance",
        width=400,
        height=500
    )
    st.plotly_chart(fig)

# Right column: Prediction
with right_col:
    st.header("Prediction")
    if st.button("Predict"):
        probabilities = model.predict_proba(input_data_scaled)[0]
        prediction = model.predict(input_data_scaled)[0]
        label = "Churned" if prediction == 1 else "Retained"

        st.subheader(f"Predicted Value: {label}")
        st.write(f"Churn Probability: {probabilities[1]:.2%}")
        st.write(f"Retention Probability: {probabilities[0]:.2%}")
        st.markdown(f"### Output: **{label}**")
