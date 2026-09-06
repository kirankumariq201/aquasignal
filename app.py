import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="AquaSignal",
    page_icon="💧",
    layout="centered"
)

st.title("💧 AquaSignal")
st.subheader("Global Water Safety Predictor")
st.markdown("Enter water parameters below to check if water is safe to drink.")

st.sidebar.header("About AquaSignal")
st.sidebar.info(
    "AquaSignal uses Machine Learning trained on "
    "3276 global water samples to predict drinking "
    "water safety and explain why water is safe or unsafe."
)

# Input sliders
st.header("Water Parameters")

col1, col2 = st.columns(2)

with col1:
    ph = st.slider("pH Level", 0.0, 14.0, 7.0)
    hardness = st.slider("Hardness (mg/L)", 0.0, 400.0, 150.0)
    solids = st.slider("Solids (ppm)", 0.0, 60000.0, 20000.0)
    chloramines = st.slider("Chloramines (ppm)", 0.0, 15.0, 7.0)
    sulfate = st.slider("Sulfate (mg/L)", 0.0, 500.0, 250.0)

with col2:
    conductivity = st.slider("Conductivity", 0.0, 800.0, 400.0)
    organic_carbon = st.slider("Organic Carbon", 0.0, 30.0, 15.0)
    trihalomethanes = st.slider("Trihalomethanes", 0.0, 130.0, 65.0)
    turbidity = st.slider("Turbidity (NTU)", 0.0, 7.0, 3.5)

# Predict button
if st.button("🔍 Analyze Water Safety"):
    
    # Load and train model on the fly
    df = pd.read_csv("water_potability.csv")
    df['ph'] = df['ph'].fillna(df['ph'].median())
    df['Sulfate'] = df['Sulfate'].fillna(df['Sulfate'].median())
    df['Trihalomethanes'] = df['Trihalomethanes'].fillna(
        df['Trihalomethanes'].median()
    )
    
    X = df.drop('Potability', axis=1)
    y = df['Potability']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Make prediction
    input_data = np.array([[ph, hardness, solids, chloramines,
                            sulfate, conductivity, organic_carbon,
                            trihalomethanes, turbidity]])
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    
    st.header("Results")
    
    if prediction == 1:
        st.success("✅ SAFE — This water appears safe to drink")
        st.metric("Safety Confidence", 
                  f"{probability[1]*100:.1f}%")
    else:
        st.error("❌ UNSAFE — This water is NOT safe to drink")
        st.metric("Risk Confidence", 
                  f"{probability[0]*100:.1f}%")
    
    # Show which parameters are concerning
    st.header("Parameter Analysis")
    
    params = {
        'pH': (ph, 6.5, 8.5),
        'Sulfate': (sulfate, 0, 250),
        'Turbidity': (turbidity, 0, 4),
        'Chloramines': (chloramines, 0, 4),
        'Organic Carbon': (organic_carbon, 0, 10)
    }
    
    for param, (value, safe_min, safe_max) in params.items():
        if value < safe_min or value > safe_max:
            st.warning(f"⚠️ {param}: {value} is outside safe range "
                      f"({safe_min} - {safe_max})")
        else:
            st.success(f"✅ {param}: {value} is within safe range")

st.markdown("---")
st.markdown("Built with ❤️ by AquaSignal | "
            "Trained on 3276 global water samples")