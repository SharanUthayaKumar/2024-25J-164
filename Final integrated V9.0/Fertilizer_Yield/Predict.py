import joblib
import numpy as np
import pandas as pd

# Load the trained models
fertilizer_model = joblib.load('fertilizer_model.pkl')
yield_model = joblib.load('yield_model.pkl')
harvest_model = joblib.load('harvest_model.pkl')
scaler = joblib.load('scaler.pkl')

# New data
new_data = pd.DataFrame({
    'Soil Temperature (°C)': [25.5],
    'Air Temperature (°C)': [31.0],
    'Humidity (%)': [50],
    'Growth Stage': [2]  # Fruiting
})

# Preprocess
new_data_scaled = scaler.transform(new_data)

# predictions
new_fertilizer_pred = fertilizer_model.predict(new_data_scaled)
new_yield_pred = yield_model.predict(new_data_scaled)
new_harvest_pred = harvest_model.predict(new_data_scaled)

# Output the predictions
print(f"Predicted Fertilizer Plan (kg): {new_fertilizer_pred[0]}")
print(f"Predicted Yield (kg): {new_yield_pred[0]}")
print(f"Predicted Harvest Readiness: {'Yes' if new_harvest_pred[0] == 1 else 'No'}")
