import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

# ✅ Set Page Configuration (Must be the first Streamlit command)
st.set_page_config(page_title="Heart Safe AI", layout="centered")

# Load dataset and train model
heart = pd.read_csv("heart.csv")
features = ['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exng', 'oldpeak', 'slp', 'caa', 'thall']
X = heart[features]
Y = heart['output']
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_scaled, Y)

# ✅ Custom CSS for Light Green Background & Black Text
st.markdown("""
    <style>
    .stApp {
        background-color: #DFFFD6;
        color: black;
    }
    h1, h2, h3, h4, h5, h6, p, label, span, div {
        color: black !important;
    }
    .stButton>button {
        background-color: #4CAF50 !important;
        color: white !important;
        font-size: 16px !important;
        border-radius: 10px;
        padding: 10px;
    }
    .input-box {
        border: 2px solid #4CAF50;
        border-radius: 8px;
        padding: 8px;
        font-size: 14px;
        background: white;
    }
    .image-container {
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# 🎯 **Title**
st.title("Heart Safe AI - Heart Attack Risk Prediction")
st.markdown("### Enter the details below to predict the risk level of heart attack.")

# 🎯 **Centered Image (Splash Image)**
st.markdown('<div class="image-container">', unsafe_allow_html=True)
st.image("splash_image.png", width=150)  # Adjust size as needed
st.markdown('</div>', unsafe_allow_html=True)

# 🎯 **Form for Input Fields**
input_data = {}
labels = {
    "Age (Years)": "age",
    "Sex (0: Female, 1: Male)": "sex",
    "Chest Pain Type (0-3)": "cp",
    "Resting BP (mm Hg)": "trtbps",
    "Cholesterol (mg/dL)": "chol",
    "Fasting Blood Sugar (0-1)": "fbs",
    "Resting ECG (0-2)": "restecg",
    "Max Heart Rate (bpm)": "thalachh",
    "Exercise Angina (0-1)": "exng",
    "Old Peak (ST Depression)": "oldpeak",
    "Slope (0-2)": "slp",
    "Number of Vessels (0-4)": "caa",
    "Thal (0-2)": "thall",
}

for label, key in labels.items():
    input_data[key] = st.number_input(label, min_value=0.0, step=1.0 if key not in ['oldpeak'] else 0.1)

# 🎯 **Prediction Button**
if st.button("Predict"):
    try:
        user_input = np.array([[input_data[key] for key in features]])
        user_input_scaled = scaler.transform(user_input)
        prediction = knn.predict(user_input_scaled)[0]
        
        risk_levels = {
            0: "Minimal Risk - Keep maintaining a healthy lifestyle!",
            1: "Moderate Risk - Take precautions and monitor your health.",
            2: "High Risk - Consider lifestyle changes and regular check-ups.",
            3: "Severe Risk - Consult a doctor immediately!"
        }

        st.warning(risk_levels.get(prediction, "Unknown Risk Level"))

    except ValueError:
        st.error("Please enter valid numbers in all fields!")
