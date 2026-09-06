import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import shap

st.set_page_config(
    page_title="AquaSignal",
    page_icon="💧",
    layout="centered"
)

st.title("💧 AquaSignal")
st.subheader("Global Water Safety Predictor")
st.markdown("Slide or type exact values to check water safety.")

st.sidebar.header("About AquaSignal")
st.sidebar.info(
    "AquaSignal uses Machine Learning trained on "
    "3276 global water samples to predict drinking "
    "water safety and explain why water is safe or unsafe."
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Model:** Random Forest")
st.sidebar.markdown("**Accuracy:** 65.09%")
st.sidebar.markdown("**Key Drivers:** Sulfate, pH")

@st.cache_resource
def load_and_train():
    url = "https://raw.githubusercontent.com/kirankumariq201/aquasignal/main/water_potability.csv"
    df = pd.read_csv(url)
    df['ph'] = df['ph'].fillna(df['ph'].median())
    df['Sulfate'] = df['Sulfate'].fillna(df['Sulfate'].median())
    df['Trihalomethanes'] = df['Trihalomethanes'].fillna(
        df['Trihalomethanes'].median()
    )
    X = df.drop('Potability', axis=1)
    y = df['Potability']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, X.columns.tolist()

with st.spinner("Loading AquaSignal model..."):
    model, feature_names = load_and_train()

def param_input(label, min_val, max_val, default, step=0.01, key=""):
    st.markdown(f"**{label}**")
    slider_val = st.slider(
        label, min_val, max_val, default,
        step=step, key=f"slider_{key}",
        label_visibility="collapsed"
    )
    number_val = st.number_input(
        f"Exact value for {label}",
        min_value=min_val,
        max_value=max_val,
        value=slider_val,
        step=0.01,
        format="%.2f",
        key=f"number_{key}",
        label_visibility="collapsed"
    )
    # Number input overrides slider if different
    if number_val != slider_val:
        return number_val
    return slider_val

st.header("Water Parameters")
st.markdown("*Use slider for quick input — type exact value below each slider for precision*")

col1, col2 = st.columns(2)

with col1:
    ph = param_input("pH Level", 0.0, 14.0, 7.0, key="ph")
    hardness = param_input("Hardness (mg/L)", 0.0, 400.0, 150.0, key="hardness")
    solids = param_input("Solids (ppm)", 0.0, 60000.0, 20000.0, step=1.0, key="solids")
    chloramines = param_input("Chloramines (ppm)", 0.0, 15.0, 7.0, key="chloramines")
    sulfate = param_input("Sulfate (mg/L)", 0.0, 500.0, 250.0, key="sulfate")

with col2:
    conductivity = param_input("Conductivity", 0.0, 800.0, 400.0, key="conductivity")
    organic_carbon = param_input("Organic Carbon", 0.0, 30.0, 15.0, key="organic_carbon")
    trihalomethanes = param_input("Trihalomethanes", 0.0, 130.0, 65.0, key="trihalomethanes")
    turbidity = param_input("Turbidity (NTU)", 0.0, 7.0, 3.5, key="turbidity")

if st.button("🔍 Analyze Water Safety"):

    input_data = np.array([[ph, hardness, solids, chloramines,
                            sulfate, conductivity, organic_carbon,
                            trihalomethanes, turbidity]])

    input_df = pd.DataFrame(input_data, columns=feature_names)

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    score = int(probability[1] * 100)

    st.header("Results")

    col_a, col_b = st.columns(2)
    with col_a:
        if prediction == 1:
            st.success("✅ SAFE — Safe to drink")
        else:
            st.error("❌ UNSAFE — Not safe to drink")
    with col_b:
        st.metric("Water Quality Score", f"{score}/100")

    st.progress(score)
    if score >= 60:
        st.caption("🟢 Good quality water")
    elif score >= 40:
        st.caption("🟡 Borderline — treat before drinking")
    else:
        st.caption("🔴 Poor quality — do not drink")

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
            st.warning(
                f"⚠️ {param}: {value:.2f} is outside safe range "
                f"({safe_min} - {safe_max})"
            )
        else:
            st.success(
                f"✅ {param}: {value:.2f} is within safe range"
            )

    st.header("Why This Prediction?")
    st.markdown("*Which parameters influenced this result the most:*")

    with st.spinner("Generating explanation..."):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)

        fig, ax = plt.subplots(figsize=(8, 4))
        shap_vals = shap_values[1][0] if isinstance(
            shap_values, list) else shap_values[0]

        colors = ['#ff4444' if v < 0 else '#44bb44'
                  for v in shap_vals]

        ax.barh(feature_names, shap_vals, color=colors)
        ax.axvline(x=0, color='white', linewidth=0.8)
        ax.set_xlabel("Impact on Safety Score")
        ax.set_title("Green = Safer | Red = Riskier")
        ax.set_facecolor('#0e1117')
        fig.patch.set_facecolor('#0e1117')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.title.set_color('white')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

st.markdown("---")
st.markdown(
    "Built with ❤️ by KIRAN KUMAR| SAVE WATER💧,SAVE LIFE🌍"
    "Trained on 3276 global water samples"
)