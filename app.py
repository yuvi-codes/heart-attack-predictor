import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

@st.cache_data
def load_and_train_model():
    df = pd.read_csv("heart.csv")

    # Use only 8 most important features
    selected_columns = ['age', 'sex', 'cp', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'target']
    df = df[selected_columns]

    X = df.drop("target", axis=1)
    y = df["target"]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Define base models
    rf = RandomForestClassifier(random_state=42)
    svm = SVC(probability=True, random_state=42)
    logreg = LogisticRegression(max_iter=1000)

    # Voting Classifier
    voting_model = VotingClassifier(
        estimators=[('rf', rf), ('svm', svm), ('logreg', logreg)],
        voting='soft'
    )

    # Train ensemble
    voting_model.fit(X_train, y_train)

    return voting_model, scaler

# Load model and scaler
model, scaler = load_and_train_model()

# Streamlit UI
st.title("Heart Attack Risk Predictor 💓 ")
st.write("This version uses a Voting Classifier combining RandomForest, SVM, and Logistic Regression.")

# Input fields
age = st.slider("Age (15–80)", 15, 80, 45)

sex = st.selectbox("Sex", ["Male", "Female"])

cp = st.selectbox("Chest Pain Type 🔗 [What is Chest Pain Type?](https://www.google.com/search?q=chest+pain+types)", [0, 1, 2, 3])

thalach = st.slider("Max Heart Rate Achieved 🔗 [What is Thalach?](https://www.google.com/search?q=thalach+heart+rate)", 60, 202, 150)

exang = st.selectbox("Exercise Induced Angina 🔗 [What is Exercise Angina?](https://www.google.com/search?q=exercise+induced+angina)", ["Yes", "No"])

oldpeak = st.slider("ST Depression 🔗 [What is ST depression?](https://www.google.com/search?q=what+is+ST+depression)", 0.0, 6.0, 1.0, step=0.1)

slope = st.selectbox("Slope of ST Segment 🔗 [What is ST Slope?](https://www.google.com/search?q=slope+of+ST+segment)", [0, 1, 2])

ca = st.selectbox("Number of Major Vessels (ca) 🔗 [What is CA in heart test?](https://www.google.com/search?q=major+vessels+fluoroscopy)", [0, 1, 2, 3, 4])

# Convert input to DataFrame
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

# Scale input
scaled_input = scaler.transform(user_input)

# Predict probability
risk_prob = model.predict_proba(scaled_input)[0][1]

# Output
st.subheader("🩺 Estimated Heart Attack Risk:")
st.metric(label="Risk Percentage", value=f"{risk_prob * 100:.2f}%")
