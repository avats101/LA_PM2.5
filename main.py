import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import plotly.express as px


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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

def classify_pm25(pm25):
    if pm25 <= 12:
        return "Good", "🟢", "#00E400"
    elif pm25 <= 35.4:
        return "Moderate", "🟡", "#FFFF00"
    elif pm25 <= 55.4:
        return "Unhealthy for Sensitive Groups", "🟠", "#FF7E00"
    elif pm25 <= 150.4:
        return "Unhealthy", "🔴", "#FF0000"
    elif pm25 <= 250.4:
        return "Very Unhealthy", "🟣", "#8F3F97"
    else:
        return "Hazardous", "🟤", "#7E0023" 

# --- Streamlit UI ---
with st.sidebar:
        st.markdown("## About PM-2.5")
        st.markdown("""
    **PM-2.5** refers to fine inhalable particles with diameters that are 2.5 micrometers and smaller.  
    These tiny pollutants can enter the lungs and bloodstream, leading to serious health effects.

    ### 🔍 Sources of PM-2.5:
    - Combustion (power plants, vehicles)
    - Wildfires, burning waste
    - Industrial chemical reactions

    ### 🧠 Health Effects:
    **Short-term:** Coughing, asthma attacks, throat and nose irritation  
    **Long-term:** Lung damage, cancer, stroke, premature death""")
        st.image("images/pm25_aqi_chart.png", caption="PM-2.5 AQI Chart")
        st.markdown("""
    📊 **Data Source for this project:**  
    Hourly sensor values (since 2015) from the [EPA AirData](https://aqs.epa.gov/aqsweb/airdata/download_files.html#Raw)
    """)
st.title("PM-2.5 Prediction In Los Angeles")
st.markdown("---")

# Select model
sensor_locations = {
    "1103": {"lat": 34.06659, "lon": -118.22688, "name": "Downtown LA"},
    "1201": {"lat": 34.19925, "lon": -118.53276, "name": "San Fernando Valley"}
}
# Initialize session state for selected sensor
if "selected_sensor" not in st.session_state:
    st.session_state.selected_sensor = None

# Create a Folium map centered between the two sensors
center_lat = (sensor_locations["1103"]["lat"] + sensor_locations["1201"]["lat"]) / 2
center_lon = (sensor_locations["1103"]["lon"] + sensor_locations["1201"]["lon"]) / 2
m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

# Add markers for each sensor
for sensor_id, coords in sensor_locations.items():
    folium.Marker(
        location=[coords["lat"], coords["lon"]],
        tooltip=f"Sensor at {coords['name']}"
    ).add_to(m)

# Display the map and capture click events
st.markdown("### 🗺️ Pick a location")
map_data = st_folium(m, width=700, height=500)

# Determine which sensor was clicked based on the clicked coordinates
if map_data and map_data.get("last_clicked"):
    clicked_lat = map_data["last_clicked"]["lat"]
    clicked_lon = map_data["last_clicked"]["lng"]
    # Find the closest sensor to the clicked location
    min_distance = float("inf")
    selected_sensor = None
    for sensor_id, coords in sensor_locations.items():
        distance = ((coords["lat"] - clicked_lat) ** 2 + (coords["lon"] - clicked_lon) ** 2) ** 0.5
        if distance < min_distance:
            min_distance = distance
            selected_sensor = sensor_id
    st.session_state.selected_sensor = selected_sensor

# Display the selected sensor
if st.session_state.selected_sensor:
    # Load the corresponding model and scaler
    with open(os.path.join(BASE_DIR, 'data', f'xgb_{st.session_state.selected_sensor}.pkl'), "rb") as f:
        model= pickle.load(f)
    with open(os.path.join(BASE_DIR, 'data', f'scaler_{st.session_state.selected_sensor}.pkl'), "rb") as f:
        scaler = pickle.load(f)
    df_data = pd.read_csv(os.path.join(BASE_DIR, 'data', f'df{st.session_state.selected_sensor}_clean.csv'))
    st.markdown("---")
    
    # DateTime selection
    from datetime import datetime
    st.markdown("### 📅 Pick a Date and Time")
    col1, col2 = st.columns(2)
    with col1:
        date = st.date_input("Choose a date")
    with col2:
        time = st.time_input("Choose a time")
    selected_datetime= datetime.combine(date, time)
    if st.session_state.selected_sensor == "1103":
        st.success(f"Using data from the sensor at {sensor_locations['1103']['name']} and predicting for the time {selected_datetime}")
    else:
        st.success(f"Using data from the sensor at {sensor_locations['1201']['name']} and predicting for the time {selected_datetime}")
    selected_datetime = pd.to_datetime(selected_datetime).round("H")
    st.markdown("---")
    
    #Sensor Selection
    st.markdown("### 📡 Sensor Data Input Method")
    input_type = st.radio(" ",["Estimate from History", "Manual Input"])
    sensor_units = {
        "NO2_Measurement": "µg/m³",
        "CO_Measurement": "µg/m³",
        "SO2_Measurement": "µg/m³",
        "RH_Measurement": "%",
        "Temp_Measurement": "°F",
        "ozone_Measurement": "µg/m³"
    }
    st.markdown("---")
    user_inputs = {}

    if input_type == "Manual Input":
        st.markdown("### 🛠️ Enter Sensor Readings Manually")
        for col in sensor_columns:
            label = f"{col.replace('_', ' ')} ({sensor_units[col]})"
            user_inputs[col] = st.number_input(label, min_value=0.0, step=0.1)
    else:
        st.markdown("### 🔎 Estimating Sensor Values from Historical Patterns")
        user_inputs = estimate_features(df_data, pd.Timestamp(selected_datetime), n_years=5)

        estimated_table = {
            "Sensor": [],
            "Estimated Value": [],
            "Unit": []
        }

        for k, v in user_inputs.items():
            estimated_table["Sensor"].append(k.replace("_", " "))
            estimated_table["Estimated Value"].append(f"{v:.2f}")
            estimated_table["Unit"].append(sensor_units.get(k, ""))

        st.table(estimated_table)
    st.markdown("---")
    
    # Prediction  
    if "clicked_predict" not in st.session_state:
        st.session_state.clicked_predict = None
        
    left, middle, right = st.columns(3)
    
    if right.button("🚀 Predict", use_container_width=True):
        st.session_state.clicked_predict = True
    
    if st.session_state.clicked_predict == True:
        data_point = {'ds': selected_datetime}
        data_point.update(user_inputs)
        predicted_value = predict_pm25(model,  scaler, data_point)
        category, emoji, color = classify_pm25(predicted_value)       
        st.markdown(f"""
    <div style="width: 100%; display: flex; justify-content: left;">
    <table style="width:66%; font-size:16px; color:white; text-align:center; border-radius: 10px; border-collapse:separate; border-spacing: 0; border: 2px solid #45b6fe; ">
        <tr style="background-color: #45b6fe;">
            <th style="padding: 8px 12px;">Air Quality</th>
            <th style="padding: 8px 12px;">PM₂.₅ (µg/m³)</th>
        </tr>
        <tr style="background-color:{color}">
            <td style="padding: 8px 12px;">{emoji} <b>{category}</b></td>
            <td style="padding: 8px 12px;"><b>{predicted_value:.2f}</b></td>
        </tr>
    </table>
    </div>
""", unsafe_allow_html=True)
   
        st.markdown("---")     
        st.markdown("## 📊 PM-2.5 Forecasts")

        # --- Generate 30-day Daily Forecast (Same hour)
        daily_preds = []
        for i in range(30):
            future_date = selected_datetime + pd.Timedelta(days=i)
            input_features = estimate_features(df_data, future_date)
            data_point = {'ds': future_date}
            data_point.update(input_features)
            pm25_value = predict_pm25(model, scaler, data_point)
            daily_preds.append({'datetime': future_date, 'PM2.5': pm25_value})

        df_daily = pd.DataFrame(daily_preds)

        # --- Generate 24-hour Hourly Forecast (Same day)
        hourly_preds = []
        for i in range(24):
            future_time = selected_datetime.replace(minute=0, second=0, microsecond=0) + pd.Timedelta(hours=i)
            input_features = estimate_features(df_data, future_time)
            data_point = {'ds': future_time}
            data_point.update(input_features)
            pm25_value = predict_pm25(model, scaler, data_point)
            hourly_preds.append({'datetime': future_time, 'PM2.5': pm25_value})

        df_hourly = pd.DataFrame(hourly_preds)

        # --- Show in Streamlit
        st.markdown("### 📆 30-Day PM2.5 Prediction")
        fig_daily = px.line(df_daily, x='datetime', y='PM2.5', markers=True,
                            labels={'datetime': 'Date', 'PM2.5': 'PM2.5 (µg/m³)'})
        st.plotly_chart(fig_daily, use_container_width=True)

        st.markdown("### 🕒 24-Hour PM2.5 Prediction")
        fig_hourly = px.line(df_hourly, x='datetime', y='PM2.5', markers=True,
                            labels={'datetime': 'Hour', 'PM2.5': 'PM2.5 (µg/m³)'})
        st.plotly_chart(fig_hourly, use_container_width=True)
        st.markdown("---")
