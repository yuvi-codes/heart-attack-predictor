import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

st.set_page_config(page_title="Heart Attack Risk Predictor", layout="centered")

st.title("❤️ Heart Attack Risk Predictor")
st.write("Enter your health details to estimate the percentage chance of a heart attack.")

# Load and preprocess the dataset
@st.cache_data
def load_and_train_model():
    df = pd.read_csv("heart.csv")

    # Encode categorical features
    df['Diabetes'] = df['Diabetes'].map({True: 1, False: 0})
    df['Smoking'] = df['Smoking'].map({True: 1, False: 0})
    df['AlcoholDrinking'] = df['AlcoholDrinking'].map({True: 1, False: 0})
    df['Stroke'] = df['Stroke'].map({True: 1, False: 0})
    df['Sex'] = df['Sex'].map({'Male': 1, 'Female': 0})
    df['PhysicalActivity'] = df['PhysicalActivity'].map({True: 1, False: 0})
    df['DiffWalking'] = df['DiffWalking'].map({True: 1, False: 0})
    df['Asthma'] = df['Asthma'].map({True: 1, False: 0})
    df['KidneyDisease'] = df['KidneyDisease'].map({True: 1, False: 0})
    df['SkinCancer'] = df['SkinCancer'].map({True: 1, False: 0})

    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    return model, scaler

model, scaler = load_and_train_model()

# Collect user input
st.header("📝 Your Health Info")

age = st.slider("Age", 18, 100, 30)
bmi = st.slider("BMI", 10.0, 50.0, 22.0)
physical_health = st.slider("Physical Health (0–30 days)", 0, 30, 0)
mental_health = st.slider("Mental Health (0–30 days)", 0, 30, 0)
sleep_time = st.slider("Average Sleep Time (hrs)", 1, 24, 7)

sex = st.radio("Sex", ["Male", "Female"])
diabetes = st.radio("Diabetes", ["Yes", "No"])
smoking = st.radio("Smoking", ["Yes", "No"])
alcohol = st.radio("Alcohol Drinking", ["Yes", "No"])
stroke = st.radio("Ever had a Stroke", ["Yes", "No"])
physical_activity = st.radio("Physically Active", ["Yes", "No"])
diff_walking = st.radio("Difficulty Walking", ["Yes", "No"])
asthma = st.radio("Asthma", ["Yes", "No"])
kidney = st.radio("Kidney Disease", ["Yes", "No"])
skin_cancer = st.radio("Skin Cancer", ["Yes", "No"])

# Convert inputs to numerical form
input_data = np.array([[
    1 if diabetes == "Yes" else 0,
    1 if smoking == "Yes" else 0,
    1 if alcohol == "Yes" else 0,
    1 if stroke == "Yes" else 0,
    1 if sex == "Male" else 0,
    age,
    bmi,
    physical_health,
    mental_health,
    sleep_time,
    1 if physical_activity == "Yes" else 0,
    1 if diff_walking == "Yes" else 0,
    1 if asthma == "Yes" else 0,
    1 if kidney == "Yes" else 0,
    1 if skin_cancer == "Yes" else 0
]])

input_scaled = scaler.transform(input_data)

if st.button("🩺 Check Heart Attack Risk"):
    prediction = model.predict_proba(input_scaled)[0][1]
    percentage = round(prediction * 100, 2)

    st.subheader("💡 Result")
    st.success(f"Estimated Heart Attack Risk: **{percentage}%**")

