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

rf_model, svm_model, scaler = load_and_train_model()

st.title("Heart Attack Risk Predictor 💓")
st.write("Enter your details below to check your heart attack risk percentage.")

# Input Form
age = st.slider("Age (15–80)", 15, 80, 45)

sex = st.selectbox("Sex", ["Male", "Female"])

cp = st.selectbox("Chest Pain Type (cp) 🔗 [What is Chest Pain Type?](https://www.google.com/search?q=chest+pain+types+cp) ", [0, 1, 2, 3])

trestbps = st.slider("Resting Blood Pressure (trestbps) 🔗 [What is Resting BP?](https://www.google.com/search?q=resting+blood+pressure)", 80, 200, 120)

chol = st.slider("Serum Cholesterol (chol) 🔗 [What is Serum Cholesterol?](https://www.google.com/search?q=serum+cholesterol)", 100, 600, 200)

fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs) 🔗 [What is Fasting Blood Sugar?](https://www.google.com/search?q=fasting+blood+sugar)", ["Yes", "No"])

restecg = st.selectbox("Resting ECG (restecg) 🔗 [What is Resting ECG?](https://www.google.com/search?q=resting+ECG)", [0, 1, 2])

thalach = st.slider("Max Heart Rate Achieved (thalach) 🔗 [What is Thalach?](https://www.google.com/search?q=thalach+heart+rate)", 60, 202, 150)

exang = st.selectbox("Exercise Induced Angina (exang) 🔗 [What is Exercise Angina?](https://www.google.com/search?q=exercise+induced+angina)", ["Yes", "No"])

oldpeak = st.slider("ST Depression (oldpeak) 🔗 [What is ST depression?](https://www.google.com/search?q=what+is+ST+depression)", 0.0, 6.0, 1.0, step=0.1)

slope = st.selectbox("Slope of Peak Exercise ST Segment (slope) 🔗 [What is ST Slope?](https://www.google.com/search?q=slope+of+ST+segment)", [0, 1, 2])

ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy (ca) 🔗 [What is CA in heart test?](https://www.google.com/search?q=number+of+major+vessels+fluoroscopy)", [0, 1, 2, 3, 4])

thal = st.selectbox("Thalassemia (thal) 🔗 [What is Thalassemia?](https://www.google.com/search?q=thalassemia+heart+test)", [0, 1, 2, 3])

# User Input
user_input = pd.DataFrame([[
    age,
    1 if sex == "Male" else 0,
    cp,
    trestbps,
    chol,
    1 if fbs == "Yes" else 0,
    restecg,
    thalach,
    1 if exang == "Yes" else 0,
    oldpeak,
    slope,
    ca,
    thal
]], columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
             'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])

scaled_input = scaler.transform(user_input)

# Predict using both models and average
rf_pred = rf_model.predict_proba(scaled_input)[0][1]
svm_pred = svm_model.predict_proba(scaled_input)[0][1]
final_pred = (rf_pred + svm_pred) / 2

# Display result
st.subheader("🩺 Estimated Heart Attack Risk:")
st.metric(label="Risk Percentage", value=f"{final_pred * 100:.2f}%")
