import pandas as pd
import joblib

# input data
input_data = {
    'Precipitation (mm)': [12.5],
    'Ambient Temp (°C)': [28],
    'Humidity (%)': [70],
    'Soil Temp (°C)': [22],
    'Soil Moisture (%)': [45]
}

# Convert to DataFrame
input_df = pd.DataFrame(input_data)

# Load  trained model
model = joblib.load('watering_model.pkl')
scaler = joblib.load('scaler.pkl')
scaled_input_data = scaler.transform(input_df)

#  prediction
predicted_watering_duration = model.predict(scaled_input_data)
print(f'Predicted Watering Duration (minutes): {predicted_watering_duration[0]}')
