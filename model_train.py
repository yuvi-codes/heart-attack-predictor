import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Load data
df = pd.read_csv('heart.csv')

# Map categorical values
df['Diabetes'] = df['Diabetes'].map({True: 1, False: 0})
df['Smoking'] = df['Smoking'].map({True: 1, False: 0})
df['AlcoholDrinking'] = df['AlcoholDrinking'].map({True: 1, False: 0})
df['Stroke'] = df['Stroke'].map({True: 1, False: 0})
df['Sex'] = df['Sex'].map({'Male': 1, 'Female': 0})
df['PhysicalActivity'] = df['PhysicalActivity'].map({True: 1, False: 0})

X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save model and scaler
joblib.dump(model, 'heart_attack_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model training complete and saved.")
