import streamlit as st
import pandas as pd
import numpy as np

from src.pipeline.predict_pipeline import PredictPipeline, CustomData
from src.exception import CustomException

st.title('Heart Risk Detection App')

st.write(
    "This app predicts the risk of heart disease based on input health parameters. "
    "Fill in the details below, and the model will assess the likelihood of heart risk."
)

# User input fields
age = st.number_input("Age", min_value=1, max_value=120, value=50)
gender = st.selectbox("Gender", ["Male", "Female"])
origin = st.selectbox("Origin", ["Cleveland", "Hungary", "VA Long Beach", "Switzerland"])
cp = st.selectbox("Chest Pain", ["asymptomatic", "non-anginal", "atypical angina", "typical angina"])
trestbps = st.number_input("Resting Blood Pressure", min_value=0, max_value=200, value=120)
chol = st.number_input("Cholesterol Level", min_value=0, max_value=610, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["False", "True"])
fbs = fbs == "True"
restecg = st.selectbox("Resting ECG Results", ["normal", "lv hypertrophy", "st-t abnormality"])
thalch = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=250, value=150)
exang = st.selectbox("Exercise Induced Angina", ["False", "True"])
exang = exang == "True"
oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=-2.6, max_value=7.0, value=1.0)

if st.button("Evaluate Risk"):
    try:
        data = CustomData(age, gender, origin, cp, trestbps, chol, fbs, restecg, thalch, exang, oldpeak)
        pred_df = data.get_data_as_data_frame()

        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(pred_df)

        if prediction[0] > 0.5:
            st.write("⚠️ **Risk Detected:** The patient may have a heart condition. Please consult a doctor for further evaluation.")
        else:
            st.write("✅ **Low Risk:** The patient is unlikely to have a heart condition. However, regular check-ups are recommended.")
    
    except Exception as e:
        st.error(f"Error occurred: {str(e)}")