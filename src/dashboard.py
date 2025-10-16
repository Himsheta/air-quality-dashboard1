import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# ============================
# Load model and data
# ============================
model = joblib.load('src/aqi_model.pkl')
df = pd.read_csv('data/processed/clean_data.csv')

# ============================
# Dashboard Layout
# ============================
st.set_page_config(page_title="Air Quality Dashboard", layout="centered")
st.title("🌍 Air Quality Prediction Dashboard")

# ============================
# AQI Trend
# ============================
st.subheader("📊 AQI Trend")
st.line_chart(df['AQI'])

# ============================
# Input Fields
# ============================
st.subheader("📈 Predict AQI")

pm25 = st.number_input("PM2.5", value=50.0)
pm10 = st.number_input("PM10", value=70.0)
no = st.number_input("NO", value=30.0)
co = st.number_input("CO", value=0.8)
so2 = st.number_input("SO2", value=20.0)

# ============================
# Predict Button
# ============================
if st.button("🚀 Predict AQI"):
    input_data = [[pm25, pm10, no, co, so2]]
    prediction = model.predict(input_data)
    st.success(f"Predicted AQI: {prediction[0]:.2f}")

# ============================
# Predicted vs Actual AQI (Sample)
# ============================
st.subheader("📊 Predicted vs Actual AQI (Sample)")

# Select a random sample of rows
features = ['PM2.5', 'PM10', 'NO', 'CO', 'SO2']
sample_df = df.sample(20, random_state=42)  # you can change sample size

# Extract features and actual AQI
X_sample = sample_df[features]
y_true = sample_df['AQI']

# Make predictions
y_pred = model.predict(X_sample)

# Create a comparison table
comparison_df = sample_df.copy()
comparison_df['Predicted_AQI'] = np.round(y_pred, 2)

# Show only useful columns
st.dataframe(comparison_df[['Date', 'City', 'AQI', 'Predicted_AQI']].head(10))

# ============================
# Real vs Predicted Scatter Plot
# ============================
st.subheader("📈 Real vs Predicted AQI Scatter Plot")

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_true, y_pred, alpha=0.6)
ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
ax.set_xlabel('Actual AQI')
ax.set_ylabel('Predicted AQI')
ax.set_title('Real vs Predicted AQI (Sample)')
st.pyplot(fig)
