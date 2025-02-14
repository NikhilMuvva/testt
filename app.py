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
st.title("Heart Safe AI - Heart Attack Prediction")
st.markdown("### Enter the details below to predict the risk of heart attack.")

# 🎯 **Form Inputs**
input_data = {}
for label, key in {
    "Age": "age",
    "Sex (0: Female, 1: Male)": "sex",
    "Chest Pain Type (0-3)": "cp",
    "Resting BP (mm Hg)": "trtbps",
    "Cholesterol (mg/dL)": "chol",
    "Fasting Blood Sugar (0: No, 1: Yes)": "fbs",
    "Resting ECG (0-2)": "restecg",
    "Max Heart Rate (bpm)": "thalachh",
    "Exercise Angina (0: No, 1: Yes)": "exng",
    "Old Peak (ST Depression)": "oldpeak",
    "Slope (0-2)": "slp",
    "Number of Vessels (0-4)": "caa",
    "Thal (0-2)": "thall"
}.items():
    input_data[key] = st.number_input(label, min_value=0.0, step=1.0 if key != 'oldpeak' else 0.1)

# 🎯 **Prediction Button**
if st.button("Predict"):
    try:
        user_input = np.array([[input_data[key] for key in features]])
        user_input_scaled = scaler.transform(user_input)
        prediction = knn.predict(user_input_scaled)[0]

        if prediction == 1:
            st.error("⚠️ **High Risk of Heart Attack!**")
            st.markdown("### **Risk Factors to Consider:**")
            st.markdown("- **Chest pain type (cp):** Higher values indicate more severe angina.")
            st.markdown("- **Resting BP (trtbps) & Cholesterol:** Elevated levels increase risk.")
            st.markdown("- **Exercise Angina (exng):** If positive, signals stress-related issues.")
            st.markdown("- **ST Depression (oldpeak) & Slope:** Downsloping and high ST depression are risk indicators.")
            st.markdown("- **Number of Blocked Vessels (caa):** More blockages = higher risk.")
            st.markdown("---")
            st.markdown("### 🚑 **Recommendations:**")
            st.markdown("- Consult a cardiologist **immediately.**")
            st.markdown("- Lifestyle changes: **Diet, exercise, no smoking/alcohol.**")
            st.markdown("- Regular **blood pressure & cholesterol checks.**")
        else:
            st.success("✅ **Low Risk of Heart Attack!**")
            st.markdown("### Keep maintaining a healthy lifestyle! 🏃‍♂️🥗")

    except ValueError:
        st.error("❌ Please enter valid numbers in all fields!")

# 🎯 **Find Maximum Risk Case from Dataset**
if st.button("Show Example of Severe Risk Case"):
    severe_case = heart[heart['output'] == 1].sort_values(by=['oldpeak', 'caa', 'chol'], ascending=False).iloc[0]
    st.write("💀 **Example of a Severe Risk Patient:**")
    st.write(severe_case.to_dict())
