import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from datetime import timedelta

# Streamlit config
st.set_page_config(page_title="Apple Stock Predictor", layout="centered")

st.title("🍏 Apple Stock Price Forecast (Next 30 Days)")
st.markdown("This app uses a trained **ARIMA(2,2,1)** model to forecast Apple stock closing prices.")

# Load model
@st.cache_resource
def load_model():
    with open('arima_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

# Load historical data for last date
df = pd.read_csv('AAPL (4).csv')
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df = df.sort_values('Date')
last_date = df['Date'].max()

# Forecast
forecast = model.forecast(steps=30)

# Create forecast DataFrame
forecast_dates = [last_date + timedelta(days=i+1) for i in range(30)]
forecast_df = pd.DataFrame({'Date': forecast_dates, 'Predicted_Close': forecast})

# Plot
st.subheader("📈 Forecasted Prices")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(forecast_df['Date'], forecast_df['Predicted_Close'], label='Forecasted Price', color='orange')
ax.set_title('Next 30 Days Stock Price Forecast')
ax.set_xlabel('Date')
ax.set_ylabel('Price ($)')
ax.legend()
st.pyplot(fig)

# Show table
st.subheader("🔢 Forecast Data")
st.dataframe(forecast_df)

st.success("Forecast generated using ARIMA(2,2,1) model.")
