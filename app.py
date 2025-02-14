import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# ✅ Set Page Configuration
st.set_page_config(page_title="Heart Safe AI", layout="centered")

# Load dataset
heart = pd.read_csv("heart.csv")

# Features & Labels
features = ['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exng', 'oldpeak', 'slp', 'caa', 'thall']
X = heart[features]
Y = heart['output']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train KNN Model
knn = KNeighborsClassifier(n_neighbors=7, weights="distance")  # Increased k for better predictions
knn.fit(X_scaled, Y)

# 🎯 **Custom Styling**
st.markdown("""
    <style>
    .stApp { background-color: #F0F8FF; color: black; }
    .stButton>button { background-color: #D32F2F !important; color: white !important; border-radius: 10px; padding: 10px; }
    .stTextInput>label, .stSelectbox>label, .stSlider>label, .stNumberInput>label { color: #D32F2F; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# 🎯 **Title & Description**
st.title("💓 Heart Safe AI - Heart Attack Prediction")
st.markdown("### 🏥 Enter your health details below to assess your heart attack risk.")

# 🎯 **Form Inputs**
st.markdown("### 🔍 Patient Information")
input_data = {}

input_data['age'] = st.number_input("📅 Age", min_value=1, max_value=120, step=1)
input_data['sex'] = st.selectbox("⚧ Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
input_data['cp'] = st.slider("💔 Chest Pain Type (0: None, 3: Severe)", min_value=0, max_value=3, step=1)
input_data['trtbps'] = st.number_input("🩸 Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, step=1)
input_data['chol'] = st.number_input("🍔 Cholesterol (mg/dL)", min_value=100, max_value=600, step=1)
input_data['fbs'] = st.selectbox("🧪 Fasting Blood Sugar > 120 mg/dL?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
input_data['restecg'] = st.selectbox("📊 Resting ECG (0: Normal, 2: Abnormal)", options=[0, 1, 2])
input_data['thalachh'] = st.number_input("❤️ Max Heart Rate Achieved", min_value=60, max_value=220, step=1)
input_data['exng'] = st.selectbox("🏃‍♂️ Exercise-Induced Angina?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
input_data['oldpeak'] = st.number_input("📉 ST Depression (Oldpeak)", min_value=0.0, max_value=6.0, step=0.1)
input_data['slp'] = st.selectbox("📈 Slope of ST Segment (0: Down, 2: Up)", options=[0, 1, 2])
input_data['caa'] = st.slider("🩸 Number of Major Blocked Vessels (0-4)", min_value=0, max_value=4, step=1)
input_data['thall'] = st.selectbox("🧬 Thalassemia (0-3)", options=[0, 1, 2, 3])

# 🎯 **Prediction Button**
if st.button("🔎 Predict Heart Attack Risk"):
    try:
        user_input = np.array([[input_data[key] for key in features]])
        user_input_scaled = scaler.transform(user_input)
        prediction = knn.predict(user_input_scaled)[0]

        # 🔥 **Fixed Severe Risk Logic**
        severe_risk = (
            input_data['cp'] == 3 or
            input_data['trtbps'] >= 180 or
            input_data['chol'] >= 500 or
            input_data['exng'] == 1 or
            input_data['oldpeak'] >= 5.0 or
            input_data['caa'] >= 3
        )

        if severe_risk or prediction == 1:
            st.error("🚨 **High Risk of Heart Attack!** 🚨")
            st.markdown("### **🛑 Severe Risk Detected!**")
            st.markdown("### **⚠️ Risk Factors:**")
            if input_data['cp'] == 3:
                st.markdown("- **Severe Chest Pain**")
            if input_data['trtbps'] >= 180:
                st.markdown("- **Extremely High Blood Pressure**")
            if input_data['chol'] >= 500:
                st.markdown("- **Dangerously High Cholesterol**")
