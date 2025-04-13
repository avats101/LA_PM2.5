import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE_DIR, 'data', 'xgb_1103.pkl'), "rb") as f:
    model_1103 = pickle.load(f)
with open(os.path.join(BASE_DIR, 'data', 'xgb_1201.pkl'), "rb") as f:
    model_1201 = pickle.load(f)
with open(os.path.join(BASE_DIR, 'data', 'scaler_1103.pkl'), "rb") as f:
    scaler_1103 = pickle.load(f)
with open(os.path.join(BASE_DIR, 'data', 'scaler_1201.pkl'), "rb") as f:
    scaler_1201 = pickle.load(f)
df1103 = pd.read_csv(os.path.join(BASE_DIR, 'data', 'df1103_clean.csv'))
df1201 = pd.read_csv(os.path.join(BASE_DIR, 'data', 'df1201_clean.csv'))

# Feature columns
sensor_columns = ['NO2_Measurement', 'CO_Measurement', 'SO2_Measurement','RH_Measurement', 'Temp_Measurement', 'ozone_Measurement']
feature_cols = sensor_columns + [
    'hour', 'day', 'month',
    'month_sin', 'month_cos', 'hour_sin', 'hour_cos',
    'is_weekend', 'week_of_year', 'day_of_year'
]

# Add seasonal features
def add_seasonal_features(df):
    df['week_of_year'] = df['ds'].dt.isocalendar().week
    df['day_of_year'] = df['ds'].dt.dayofyear
    df['hour'] = df['ds'].dt.hour
    df['day'] = df['ds'].dt.day
    df['month'] = df['ds'].dt.month

    df['season'] = df['month'].apply(lambda m: 
        'Winter' if m in [12, 1, 2] else
        'Spring' if m in [3, 4, 5] else
        'Summer' if m in [6, 7, 8] else 'Fall')

    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['is_weekend'] = df['ds'].dt.weekday >= 5

    return df

# Estimate features
def estimate_features(df, target_time, n_years=5):
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['hour'] = df['datetime'].dt.hour

    past_years = [target_time.year - i for i in range(1, n_years + 1)]

    subset = df[
        (df['year'].isin(past_years)) &
        (df['month'] == target_time.month) &
        (df['day'] == target_time.day) &
        (df['hour'] == target_time.hour)
    ]

    return subset[sensor_columns].mean().to_dict()

# Predict
def predict_pm25(model, scaler, data_point):
    df = pd.DataFrame([data_point])
    df['ds'] = pd.to_datetime(df['ds'])
    X = df[sensor_columns]
    X_scaled = scaler.transform(X)
    df[sensor_columns] = X_scaled
    df = add_seasonal_features(df)
    X = df[feature_cols]
    y_pred_log = model.predict(X)
    return np.expm1(y_pred_log[0])

# --- Streamlit UI ---
st.title("PM2.5 Prediction App")

# Select model
model_choice = st.selectbox("Choose a model", ["1103", "1201"])
model = model_1103 if model_choice == "1103" else model_1201
scaler = scaler_1103 if model_choice == "1103" else scaler_1201
df_data = df1103 if model_choice == "1103" else df1201

# DateTime selection
from datetime import datetime
st.markdown("### 📅 Select a Date and Time")
col1, col2 = st.columns(2)
with col1:
    date = st.date_input("Choose a date")
with col2:
    time = st.time_input("Choose a time")
selected_datetime= datetime.combine(date, time)
st.write(f"🕒 You selected: `{selected_datetime}`")
selected_datetime = pd.to_datetime(selected_datetime).round("H")

input_type = st.radio("Sensor data input method", ["Estimate from history", "Manual input"])

# Collect sensor values
if input_type == "Manual input":
    user_inputs = {}
    for col in sensor_columns:
        user_inputs[col] = st.number_input(f"{col}", min_value=0.0, step=0.1)
else:
    st.write("Estimating sensor values from historical patterns...")
    user_inputs = estimate_features(df_data, pd.Timestamp(selected_datetime), n_years=5)
    for k, v in user_inputs.items():
        st.write(f"{k}: {v:.2f}")

def classify_pm25(pm25):
    if pm25 <= 12.0:
        return "Good", "🟢", "#A8E05F"
    elif pm25 <= 35.4:
        return "Moderate", "🟡", "#FDD64B"
    elif pm25 <= 55.4:
        return "Unhealthy for Sensitive Groups", "🟠", "#FF9B57"
    elif pm25 <= 150:
        return "Unhealthy", "🔴", "#FE6A69"
    elif pm25 <= 250:
        return "Very Unhealthy", "🟣", "#A97ABC"
    else:
        return "Hazardous", "⚫️", "#A87383"


# Predict button
if st.button("Predict PM2.5"):
    data_point = {'ds': selected_datetime}
    data_point.update(user_inputs)
    predicted_value = predict_pm25(model,  scaler, data_point)
    category, emoji, color = classify_pm25(predicted_value)

    st.markdown(f"""
    <div style="background-color:{color}; padding:20px; border-radius:10px; text-align:center">
        <h2 style="color:white;">{emoji} Air Quality: {category}</h2>
        <p style="font-size:18px; color:white;">PM2.5: <b>{predicted_value:.2f} µg/m³</b></p>
    </div>
    """, unsafe_allow_html=True)

