from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

fertilizer_model = joblib.load('fertilizer_model.pkl')
yield_model = joblib.load('yield_model.pkl')
harvest_model = joblib.load('harvest_model.pkl')
scaler = joblib.load('scaler.pkl')

# Route
@app.route('/predict', methods=['POST'])
def predict():

    data = request.get_json()


    if not all(key in data for key in ['Soil Temperature (°C)', 'Air Temperature (°C)', 'Humidity (%)', 'Growth Stage']):
        return jsonify({"error": "Missing required input fields"}), 400


    soil_temp = data['Soil Temperature (°C)']
    air_temp = data['Air Temperature (°C)']
    humidity = data['Humidity (%)']
    growth_stage = data['Growth Stage']


    new_data = pd.DataFrame({
        'Soil Temperature (°C)': [soil_temp],
        'Air Temperature (°C)': [air_temp],
        'Humidity (%)': [humidity],
        'Growth Stage': [growth_stage]
    })


    new_data_scaled = scaler.transform(new_data)


    new_fertilizer_pred = fertilizer_model.predict(new_data_scaled)
    new_yield_pred = yield_model.predict(new_data_scaled)
    new_harvest_pred = harvest_model.predict(new_data_scaled)


    output = {
        "Predicted Fertilizer Plan (kg/ha)": new_fertilizer_pred[0],
        "Predicted Yield (kg/ha)": new_yield_pred[0],
        "Predicted Harvest Readiness": 'Yes' if new_harvest_pred[0] == 1 else 'No'
    }


    return jsonify(output)


if __name__ == '__main__':
    app.run(debug=True)
