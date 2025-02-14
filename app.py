import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
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
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train KNN Model
knn = KNeighborsClassifier(n_neighbors=5, weights="distance")  # Weight by distance for better severe case detection
knn.fit(X_scaled, Y)

# 🎯 **Custom Styling**
st.markdown("""
    <style>
    .stApp { background-color: #DFFFD6; color: black; }
    .stButton>button { background-color: #4CAF50 !important; color: white !important; border-radius: 10px; padding: 10px; }
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

        if prediction == 1:
            st.error("🚨 **High Risk of Heart Attack!** 🚨")
            st.markdown("### **🛑 Warning!** Based on your inputs, there is a **severe risk** of heart complications.")
            st.markdown("### **⚠️ Risk Factors:**")
            st.markdown("- **Severe chest pain (cp)**")
            st.markdown("- **High blood pressure (trtbps) and cholesterol (chol)**")
            st.markdown("- **Exercise-induced angina (exng: Yes)**")
            st.markdown("- **High ST depression (oldpeak)**")
            st.markdown("- **Blocked arteries (caa)**")
            st.markdown("---")
            st.markdown("### 🏥 **Recommendations:**")
            st.markdown("- **Consult a doctor immediately!** 🏥")
            st.markdown("- Lifestyle changes: **Healthy diet, regular exercise, no smoking.**")
            st.markdown("- Regular **BP and cholesterol check-ups.**")
        else:
            st.success("✅ **Low Risk of Heart Attack!**")
            st.markdown("### 💪 Keep maintaining a healthy lifestyle!")

    except ValueError:
        st.error("❌ Please enter valid numbers in all fields!")

# 🎯 **Find Maximum Risk Case from Dataset**
if st.button("📊 Show Example of Severe Risk Case"):
    severe_case = heart[heart['output'] == 1].sort_values(by=['oldpeak', 'caa', 'chol'], ascending=False).iloc[0]
    st.markdown("### 🛑 **Example of a Severe Risk Patient from the Dataset:**")
    st.write(severe_case.to_dict())
