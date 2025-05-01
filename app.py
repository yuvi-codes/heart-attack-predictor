import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('heart_attack_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🫀 Heart Attack Risk Predictor")
st.subheader("Enter your health info to calculate risk percentage")

# User inputs
age = st.slider("Age", 18, 100, 45)
sex = st.selectbox("Sex", ["Male", "Female"])
diabetes = st.selectbox("Do you have Diabetes?", ["Yes", "No"])
smoking = st.selectbox("Do you smoke?", ["Yes", "No"])
alcohol = st.selectbox("Do you drink alcohol regularly?", ["Yes", "No"])
stroke = st.selectbox("Have you had a stroke?", ["Yes", "No"])
physical = st.selectbox("Are you physically active?", ["Yes", "No"])
bmi = st.slider("Body Mass Index (BMI)", 10.0, 50.0, 25.0)

# Convert inputs
def map_input(val, yes_val=1, no_val=0):
    return yes_val if val == "Yes" or val == "Male" else no_val

user_data = np.array([[
    age,
    map_input(sex, 1, 0),
    bmi,
    map_input(diabetes),
    map_input(smoking),
    map_input(alcohol),
    map_input(stroke),
    map_input(physical)
]])

# Scale input
user_data_scaled = scaler.transform(user_data)

# Predict
if st.button("Check Heart Attack Risk"):
    prediction_proba = model.predict_proba(user_data_scaled)[0][1]
    percentage = round(prediction_proba * 100, 2)
    st.success(f"🩺 Your risk of heart attack is: **{percentage}%**")
