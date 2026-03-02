# ==========================================================
# HEART DISEASE AI - CLEAN UI VERSION
# ==========================================================

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# ----------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------

st.set_page_config(page_title="Heart Disease Predictor", page_icon="🫀", layout="wide")

st.markdown("""
    <h1 style='text-align: center; color: #C0392B;'>
    🫀 Heart Disease Risk Prediction System
    </h1>
""", unsafe_allow_html=True)

st.markdown("---")

# ----------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------

df = pd.read_csv("heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

# ----------------------------------------------------------
# TRAIN MODEL
# ----------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = Pipeline([
    ("smote", SMOTE(random_state=42)),
    ("scaler", StandardScaler()),
    ("classifier", XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        eval_metric="logloss"
    ))
])

model.fit(X_train, y_train)

# ----------------------------------------------------------
# SHOW MODEL PERFORMANCE
# ----------------------------------------------------------

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

col1, col2 = st.columns(2)
col1.metric("Model Accuracy", f"{accuracy:.2f}")
col2.metric("Model AUC Score", f"{auc:.2f}")

st.markdown("---")

# ----------------------------------------------------------
# INPUT SECTION
# ----------------------------------------------------------

st.subheader("Enter Patient Information")

col1, col2 = st.columns(2)

# Left Column
age = col1.slider("Age", 20, 100, 50)
sex_option = col1.selectbox("Sex", ["Female", "Male"])
cp = col1.selectbox("Chest Pain Type (cp)", [0,1,2,3])
fbs = col1.selectbox("Fasting Blood Sugar >120 mg/dl (fbs)", [0,1])
restecg = col1.selectbox("Resting ECG (restecg)", [0,1,2])
exang = col1.selectbox("Exercise Induced Angina (exang)", [0,1])
slope = col1.selectbox("Slope", [0,1,2])
ca = col1.selectbox("Number of Major Vessels (ca)", [0,1,2,3])
thal = col1.selectbox("Thalassemia (thal)", [0,1,2,3])

# Right Column
trestbps = col2.number_input("Resting Blood Pressure", 80, 200, 120)
chol = col2.number_input("Cholesterol", 100, 600, 200)
thalach = col2.number_input("Maximum Heart Rate", 60, 220, 150)
oldpeak = col2.number_input("Oldpeak", 0.0, 10.0, 1.0)

# Convert sex to numeric
sex = 1 if sex_option == "Male" else 0

# ----------------------------------------------------------
# PREDICT
# ----------------------------------------------------------

if st.button("Predict Risk"):

    input_data = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }])

    probability = model.predict_proba(input_data)[0][1]

    st.markdown("## Prediction Result")

    st.metric("Risk Probability", f"{probability:.2f}")

    if probability > 0.6:
        st.error("⚠ High Risk")
    elif probability > 0.3:
        st.warning("⚡ Moderate Risk")
    else:
        st.success("✅ Low Risk")

st.markdown("---")
st.markdown("Developed using XGBoost | Clean Clinical UI")