import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
import joblib

# ------------------navbar-----------------------
from component.nav import navbar
navbar()

# ------------------Model Prediction Function-----------------------
def predict_risk_level(input_data):
    # Load the trained SVM model and scaler from the saved files
    scaler = joblib.load('scaler.pkl')
    knn_model = joblib.load('knn.pkl')
    
    # Scale the input data using the same scaler that was used during training
    input_data_scaled = scaler.transform(input_data)
    
    # Predict risk level using the trained SVM model
    prediction = knn_model.predict(input_data_scaled)
    
    return prediction[0]

# ------------------Streamlit UI-----------------------
def model():
    # Title
    st.markdown(
        '<h1 align="center">Maternal Health Risk Prediction</h1>',
        unsafe_allow_html=True
    )
    
    st.markdown(
        "Silakan masukkan data untuk memprediksi tingkat risiko kesehatan maternal berdasarkan fitur berikut:"
    )
    
    # Input fields for features
    age = st.number_input('Age', min_value=10, max_value=80, value=23)
    systolic_bp = st.number_input('Systolic BP', min_value=70, max_value=160, value=130)
    diastolic_bp = st.number_input('Diastolic BP', min_value=49, max_value=100, value=70)
    blood_sugar = st.number_input('Blood Sugar', min_value=6.0, max_value=20.0, value=7.0, step=0.1)
    body_temp = st.number_input('Body Temperature (°F)', min_value=98, max_value=110, value=98)
    heart_rate = st.number_input('Heart Rate', min_value=7, max_value=190, value=78)
    
    # Creating a button to trigger prediction
    if st.button('Predict'):
        # Collect input data into a list
        input_data = [[age, systolic_bp, diastolic_bp, blood_sugar, body_temp, heart_rate]]
        
        # Make prediction
        prediction = predict_risk_level(input_data)
        print(prediction)
        
        # Display the prediction result
        if prediction == "high risk":
            st.error("Risk Level: High")
        elif prediction == "low risk":
            st.success("Risk Level: Low")
        elif prediction == "mid risk":
            st.warning("Risk Level: Mid")

# Call the model function to run the application
model()
