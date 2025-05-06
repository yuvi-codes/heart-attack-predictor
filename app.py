import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

@st.cache_data
def load_and_train_model():
    df = pd.read_csv("heart.csv")
    
    selected_columns = ['age', 'sex', 'cp', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'target']
    df = df[selected_columns]

    X = df.drop("target", axis=1)
    y = df["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)

    svm = SVC(probability=True, random_state=42)
    svm.fit(X_train, y_train)

    return rf, svm, scaler

# Load model and scaler
rf_model, svm_model, scaler = load_and_train_model()

# UI
st.title("Heart Attack Risk Predictor 💓")
st.write("This version uses only the top 8 features most relevant for heart attack prediction.")

# Inputs
age = st.slider("Age (15–80)", 15, 80, 45)

sex = st.selectbox("Sex", ["Male", "Female"])

cp = st.selectbox("Chest Pain Type 🔗 [What is Chest Pain Type?](https://www.google.com/search?q=chest+pain+types)", [0, 1, 2, 3])

thalach = st.slider("Max Heart Rate Achieved 🔗 [What is Thalach?](https://www.google.com/search?q=thalach+heart+rate)", 60, 202, 150)

exang = st.selectbox("Exercise Induced Angina 🔗 [What is Exercise Angina?](https://www.google.com/search?q=exercise+induced+angina)", ["Yes", "No"])

oldpeak = st.slider("ST Depression 🔗 [What is ST depression?](https://www.google.com/search?q=what+is+ST+depression)", 0.0, 6.0, 1.0, step=0.1)

slope = st.selectbox("Slope of ST Segment 🔗 [What is ST Slope?](https://www.google.com/search?q=slope+of+ST+segment)", [0, 1, 2])

ca = st.selectbox("Number of Major Vessels (ca) 🔗 [What is CA in heart test?](https://www.google.com/search?q=major+vessels+fluoroscopy)", [0, 1, 2, 3, 4])

# Prepare input
user_input = pd.DataFrame([[
    age,
    1 if sex == "Male" else 0,
    cp,
    thalach,
    1 if exang == "Yes" else 0,
    oldpeak,
    slope,
    ca
]], columns=['age', 'sex', 'cp', 'thalach', 'exang', 'oldpeak', 'slope', 'ca'])

scaled_input = scaler.transform(user_input)

# Predict
rf_pred = rf_model.predict_proba(scaled_input)[0][1]
svm_pred = svm_model.predict_proba(scaled_input)[0][1]
final_pred = (rf_pred + svm_pred) / 2

# Output
st.subheader("🩺 Estimated Heart Attack Risk:")
st.metric(label="Risk Percentage", value=f"{final_pred * 100:.2f}%")
