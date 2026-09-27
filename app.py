import streamlit as st
import numpy as np
import joblib

# Load artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load('water_model.pkl')
    scaler = joblib.load('scaler.pkl')
    imputer = joblib.load('imputer.pkl')
    return model, scaler, imputer

model, scaler, imputer = load_artifacts()

st.title("💧 Water Quality Risk Classification Prototype")
st.markdown("Capstone Project - Environment Domain: Predict water safety and risk categories based on physicochemical properties.")

st.sidebar.header("Input Water Parameters")

def user_input_features():
    ph = st.sidebar.slider("pH Level", 0.0, 14.0, 7.2)
    turbidity = st.sidebar.slider("Turbidity (NTU)", 0.0, 20.0, 3.1)
    temperature = st.sidebar.slider("Temperature (°C)", 0.0, 50.0, 23.5)
    dissolved_oxygen = st.sidebar.slider("Dissolved Oxygen (mg/L)", 0.0, 15.0, 7.0)
    conductivity = st.sidebar.slider("Conductivity (μS/cm)", 0.0, 1000.0, 350.0)
    
    data = np.array([[ph, turbidity, temperature, dissolved_oxygen, conductivity]])
    return data

input_df = user_input_features()

st.subheader("User Specified Parameters")
st.write(input_df)

if st.button("Predict Water Risk"):
    # Apply matching preprocessing pipeline used during training
    input_imputed = imputer.transform(input_df)
    input_scaled = scaler.transform(input_imputed)
    
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)
    
    st.subheader("Prediction Results")
    if prediction[0] == 0:
        st.success("✅ **Safe / Low Risk:** The water sample meets acceptable quality thresholds.")
    else:
        st.error("⚠️ **High Risk / Polluted:** The water sample exhibits anomalous attributes indicating potential hazards.")
        
    st.write(f"Confidence Score: {np.max(prediction_proba) * 100:.2f}%")