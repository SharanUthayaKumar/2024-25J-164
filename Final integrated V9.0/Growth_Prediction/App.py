from flask import Flask, render_template, request
import joblib
import numpy as np
import requests

app = Flask(__name__)

# Load trained models
height_model = joblib.load('height_model.pkl')
leaf_area_model = joblib.load('leaf_area_model.pkl')

# Firebase endpoint
FIREBASE_URL = 'https://myiot-80b9f-default-rtdb.firebaseio.com/soil_data.json'

def get_soil_moisture_from_firebase():
    try:
        response = requests.get(FIREBASE_URL)
        if response.status_code == 200:
            data = response.json()
            return float(data.get('moisture', 0))
    except Exception as e:
        print("Error fetching moisture:", e)
    return None

# Prediction function
def predict_growth(soil_moisture, precipitation, fertilizer_amount):
    input_data = np.array([[soil_moisture, precipitation, fertilizer_amount]])
    predicted_height = height_model.predict(input_data)[0]
    predicted_leaf_area = leaf_area_model.predict(input_data)[0]
    return predicted_height, predicted_leaf_area

@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_height = None
    predicted_leaf_area = None
    error = None

    # Get auto-fetched soil moisture value from Firebase
    auto_soil_moisture = get_soil_moisture_from_firebase()

    if request.method == 'POST':
        try:
            # Use auto-fetched soil moisture if input is empty
            soil_moisture = float(request.form['soil_moisture']) if request.form['soil_moisture'] else auto_soil_moisture
            precipitation = float(request.form['precipitation'])
            fertilizer_amount = float(request.form['fertilizer_amount'])

            # Predict
            predicted_height, predicted_leaf_area = predict_growth(soil_moisture, precipitation, fertilizer_amount)
        except ValueError:
            error = "Please enter valid numbers for all fields."

    return render_template('index.html',
                           predicted_height=predicted_height,
                           predicted_leaf_area=predicted_leaf_area,
                           auto_soil_moisture=auto_soil_moisture,
                           error=error)

if __name__ == '__main__':
    app.run(debug=True)
