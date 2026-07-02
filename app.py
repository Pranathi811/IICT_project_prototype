import streamlit as st
import numpy as np
from src.predict import predict_yield

st.set_page_config(page_title="Chemical Yield Predictor", layout="centered")

st.title("🔬 Chemical Yield Prediction System")

st.write("Enter process parameters to predict yield percentage")

# ---- Input fields ----
temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=1000.0)
pressure = st.number_input("Pressure (atm)", min_value=0.0, max_value=500.0)
reaction_time = st.number_input("Reaction Time (hours)", min_value=0.0, max_value=100.0)
concentration = st.number_input("Concentration (mol/L)", min_value=0.0, max_value=10.0)
catalyst = st.number_input("Catalyst Amount (g)", min_value=0.0, max_value=50.0)

# ---- Predict Button ----
if st.button("Predict Yield"):
    inputs = np.array([[temperature, pressure, reaction_time, concentration, catalyst]])
    
    predicted_yield = predict_yield(inputs)

    st.success(f"✅ Predicted Yield: {predicted_yield:.2f}%")