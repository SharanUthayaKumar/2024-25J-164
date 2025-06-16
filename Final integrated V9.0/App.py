from flask import Flask, render_template, jsonify, request
import serial
import threading
import time
import joblib
import pandas as pd
import numpy as np
import requests
from serial.tools import list_ports

app = Flask(__name__)

# Load all ML models
try:
    # Growth prediction models
    height_model = joblib.load('models/height_model.pkl')
    leaf_area_model = joblib.load('models/leaf_area_model.pkl')
    
    # Decision support models
    fertilizer_model = joblib.load('models/fertilizer_model.pkl')
    yield_model = joblib.load('models/yield_model.pkl')
    harvest_model = joblib.load('models/harvest_model.pkl')
    scaler1 = joblib.load('models/scaler1.pkl')
    # Watering model
    watering_model = joblib.load('models/watering_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    
    print("All ML models loaded successfully")
except Exception as e:
    print(f"Error loading ML models: {e}")
    height_model = leaf_area_model = fertilizer_model = yield_model = harvest_model = watering_model = scaler = None

# Firebase configuration
FIREBASE_URL = 'https://myiot-80b9f-default-rtdb.firebaseio.com/soil_data.json'

# Data storage
data_buffer = ["Initializing..."]
buffer_lock = threading.Lock()
latest_data = {
    'soil_temp': None,
    'env_temp': None,
    'humidity': None,
    'soil_moisture': None
}
serial_connected = False
ser = None

def get_soil_moisture_from_firebase():
    try:
        response = requests.get(FIREBASE_URL)
        if response.status_code == 200:
            data = response.json()

            # Extract values with defaults
            moisture = float(data.get('moisture', 0))
            N = data.get('N', 0)
            P = data.get('P', 0)
            K = data.get('K', 0)

            return {
                "moisture": moisture,
                "N": N,
                "P": P,
                "K": K
            }
    except Exception as e:
        print(f"Error fetching soil data from Firebase: {e}")
    return {
        "moisture": 0,
        "N": 0,
        "P": 0,
        "K": 0
    }


def get_soil_data_from_firebase():
    try:
        response = requests.get(FIREBASE_URL)
        if response.status_code == 200:
            data = response.json()
            if data:
                latest_data.update({
                    'soil_moisture': float(data.get('moisture', 0)),
                    'N': data.get('N', 0),
                    'P': data.get('P', 0),
                    'K': data.get('K', 0)
                })
            return data
    except Exception as e:
        print(f"Error fetching soil data from Firebase: {e}")
    return {}
 
        
# Serial connection management
def connect_serial():
    global ser, serial_connected
    ports = list_ports.comports()
    print("Available ports:")
    for port in ports:
        print(f" - {port.device}")

    for attempt in range(3):  # Retry connection
        try:
            ser = serial.Serial(
                port='COM7',
                baudrate=115200,
                timeout=1,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
            serial_connected = True
            print("Serial connection established")
            return ser
        except serial.SerialException as e:
            print(f"Serial connection failed (attempt {attempt + 1}/3): {e}")
            time.sleep(2)

    serial_connected = False
    print("❌ Failed to establish serial connection")
    return None

def read_serial():
    global latest_data, serial_connected
    while True:
        if ser and ser.is_open:
            try:
                if ser.in_waiting > 0:
                    raw_data = ser.readline()
                    try:
                        line = raw_data.decode('utf-8').strip()
                        print(f"📩 Received: {line}")

                        with buffer_lock:
                            data_buffer.append(line)
                            if len(data_buffer) > 100:
                                data_buffer.pop(0)

                        # Parse sensor data
                        if line and ',' in line:
                            try:
                                parts = [float(x) for x in line.split(',')]
                                if len(parts) == 3:
                                    latest_data.update({
                                        'soil_temp': parts[0],
                                        'env_temp': parts[1],
                                        'humidity': parts[2]
                                    })
                                    print(f"Parsed data: {latest_data}")
                            except (ValueError, IndexError) as e:
                                print(f"Parsing error: {e} | Data: {line}")
                    except UnicodeDecodeError:
                        print(f"Decode error: {raw_data}")
            except Exception as e:
                print(f"Serial error: {e}")
                serial_connected = False
                time.sleep(1)
                connect_serial()
        else:
            time.sleep(1)
            connect_serial()

# Start serial thread if connection established
if connect_serial():
    serial_thread = threading.Thread(target=read_serial, daemon=True)
    serial_thread.start()

# Background task to update soil moisture from Firebase
def update_sensor_data():
    while True:
        get_soil_moisture_from_firebase()
        get_soil_data_from_firebase()
        time.sleep(60)  # Update every minute

sensor_update_thread = threading.Thread(target=update_sensor_data, daemon=True)
sensor_update_thread.start()

# Prediction functions
def predict_growth(soil_moisture, precipitation, fertilizer_amount):
    input_data = np.array([[soil_moisture, precipitation, fertilizer_amount]])
    predicted_height = height_model.predict(input_data)[0]
    predicted_leaf_area = leaf_area_model.predict(input_data)[0]
    return predicted_height, predicted_leaf_area

def predict_watering(precipitation, ambient_temp, humidity, soil_temp, soil_moisture):
    input_data = {
        'Precipitation (mm)': [precipitation],
        'Ambient Temp (°C)': [ambient_temp],
        'Humidity (%)': [humidity],
        'Soil Temp (°C)': [soil_temp],
        'Soil Moisture (%)': [soil_moisture]
    }
    input_df = pd.DataFrame(input_data)
    scaled_input = scaler.transform(input_df)
    return watering_model.predict(scaled_input)[0]

def predict_decision_support(soil_temp, air_temp, humidity, growth_stage):
    features = scaler.transform([[soil_temp, air_temp, humidity, growth_stage]])
    return {
        'fertilizer': round(float(fertilizer_model.predict(features)[0]), 2),
        'yield': round(float(yield_model.predict(features)[0]), 2),
        'harvest': "Yes" if harvest_model.predict(features)[0] == 1 else "No"
    }

# Routes
 

 
@app.route('/')
def index():
    soil_data = get_soil_moisture_from_firebase()
    print(soil_data)
    return render_template('index.html', soil_data=soil_data)
    



# API endpoints
@app.route('/api/data')
def get_data():
    with buffer_lock:
        last_line = data_buffer[-1] if data_buffer else "No data"
        return jsonify({
            'line': last_line,
            'connected': serial_connected
        })

@app.route('/api/sensor_data')
def get_sensor_data():
    return jsonify({
        'sensors': latest_data,
        'connected': serial_connected
    })

@app.route('/api/predict/growth', methods=['POST'])
def api_predict_growth():
    try:
        data = request.get_json()
        soil_moisture = float(data.get('soil_moisture', latest_data['soil_moisture'] or 0))
        precipitation = float(data['precipitation'])
        fertilizer_amount = float(data['fertilizer_amount'])

        height, leaf_area = predict_growth(soil_moisture, precipitation, fertilizer_amount)
        return jsonify({
            'predicted_height': height,
            'predicted_leaf_area': leaf_area
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/predict/watering', methods=['POST'])
def api_predict_watering():
    try:
        data = request.get_json()
        precipitation = float(data['precipitation'])
        ambient_temp = float(data.get('ambient_temp', latest_data['env_temp'] or 0))
        humidity = float(data.get('humidity', latest_data['humidity'] or 0))
        soil_temp = float(data.get('soil_temp', latest_data['soil_temp'] or 0))
        soil_moisture = float(data.get('soil_moisture', latest_data['soil_moisture'] or 0))

        duration = predict_watering(precipitation, ambient_temp, humidity, soil_temp, soil_moisture)
        return jsonify({'predicted_watering_duration': duration})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/predict/decisions', methods=['POST'])
def api_predict_decisions():
    if not all([fertilizer_model, yield_model, harvest_model, scaler]):
        return jsonify({'error': 'ML models not loaded'}), 500

    try:
        data = request.get_json()
        soil_temp = float(data.get('soil_temp', latest_data['soil_temp'] or 0))
        air_temp = float(data.get('air_temp', latest_data['env_temp'] or 0))
        humidity = float(data.get('humidity', latest_data['humidity'] or 0))
        growth_stage = int(data['growth_stage'])

        return jsonify(predict_decision_support(soil_temp, air_temp, humidity, growth_stage))
    except Exception as e:
        return jsonify({'error': str(e)}), 400


import requests
from urllib.parse import quote_plus
from flask import Flask, render_template, request


def get_latest_precipitation():
    latitude = 8.0
    longitude = 79.8
    timezone = "Asia/Colombo"

    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&hourly=precipitation&timezone={quote_plus(timezone)}"
        response = requests.get(url, timeout=5)  # Added timeout

        if response.status_code == 200:
            data = response.json()
            precipitation_list = data.get('hourly', {}).get('precipitation', [])
            if precipitation_list:
                return float(precipitation_list[-1])
        return 0.0  # Default value if no data

    except (requests.RequestException, ValueError, KeyError) as e:
        print(f"Error getting precipitation data: {e}")
        return 0.0  # Default value on error


@app.route('/watering_recommendation', methods=['GET', 'POST'])
def watering_recommendation():
    predicted_duration = None
    error = None

    # Get current sensor data
    current_data = {
        'soil_temp': latest_data.get('soil_temp'),
        'env_temp': latest_data.get('env_temp'),
        'humidity': latest_data.get('humidity'),
        'soil_moisture': latest_data.get('soil_moisture')
    }

    # Get latest precipitation data
    latest_precipitation = get_latest_precipitation()

    if request.method == 'POST':
        try:
            precipitation = float(request.form.get('precipitation', latest_precipitation))
            ambient_temp = float(request.form.get('ambient_temp', current_data.get('env_temp', 0)))
            humidity = float(request.form.get('humidity', current_data.get('humidity', 0)))
            soil_temp = float(request.form.get('soil_temp', current_data.get('soil_temp', 0)))
            soil_moisture = float(request.form.get('soil_moisture', current_data.get('soil_moisture', 0)))

            predicted_duration = predict_watering(
                precipitation, ambient_temp, humidity, soil_temp, soil_moisture
            )
        except Exception as e:
            error = f"Invalid input: {str(e)}"

    return render_template(
        'watering.html',
        predicted_duration=predicted_duration,
        error=error,
        latest_data=current_data,
        weather_forecast=latest_precipitation  # This is the critical line that was missing
    )


@app.route('/growth_prediction', methods=['GET', 'POST'])
def growth_prediction():
    predicted_height = None
    predicted_leaf_area = None
    error = None

    # Get current sensor data
    current_data = {
        'soil_temp': latest_data.get('soil_temp'),
        'env_temp': latest_data.get('env_temp'),
        'humidity': latest_data.get('humidity'),
        'soil_moisture': latest_data.get('soil_moisture')
    }

    # Get latest data
    auto_soil_moisture = get_soil_moisture_from_firebase() or current_data.get('soil_moisture', 0)
    latest_precipitation = get_latest_precipitation()

    if request.method == 'POST':
        try:
            soil_moisture = float(request.form.get('soil_moisture', auto_soil_moisture))
            precipitation = float(request.form.get('precipitation', latest_precipitation))
            fertilizer_amount = float(request.form['fertilizer_amount'])

            predicted_height, predicted_leaf_area = predict_growth(
                soil_moisture, precipitation, fertilizer_amount
            )
        except ValueError as e:
            error = f"Invalid input: {str(e)}"

    return render_template('growth_prediction.html',
                           predicted_height=predicted_height,
                           predicted_leaf_area=predicted_leaf_area,
                           error=error,
                           latest_data=current_data,
                           auto_soil_moisture=auto_soil_moisture,
                           weather_forecast=latest_precipitation)



@app.route('/decision_support', methods=['GET', 'POST'])
def decision_support():
    predictions = None
    error = None

    # Get current sensor data
    current_data = {
        'soil_temp': latest_data.get('soil_temp', 0),
        'env_temp': latest_data.get('env_temp', 0),
        'humidity': latest_data.get('humidity', 0),
        'soil_moisture': latest_data.get('soil_moisture', 0)
    }

    if request.method == 'POST':
        if not all([fertilizer_model, yield_model, harvest_model, scaler]):
            error = 'ML models not loaded'
        else:
            try:
                # Get input values
                soil_temp = float(request.form.get('soil_temp', current_data['soil_temp']))
                air_temp = float(request.form.get('air_temp', current_data['env_temp']))
                humidity = float(request.form.get('humidity', current_data['humidity']))
                growth_stage = float(request.form.get('growth_stage', 0))

                # Prepare input with correct feature names
                input_df = pd.DataFrame([{
                    'Soil Temperature (°C)': soil_temp,
                    'Air Temperature (°C)': air_temp,
                    'Humidity (%)': humidity,
                    'Growth Stage': growth_stage
                }])

                # Scale and predict
                scaled_input = scaler1.transform(input_df)

                predictions = {
                    'fertilizer': round(float(fertilizer_model.predict(scaled_input)[0]), 2),
                    'yield': round(float(yield_model.predict(scaled_input)[0]), 2),
                    'harvest': "Yes" if harvest_model.predict(scaled_input)[0] == 1 else "No"
                }

            except Exception as e:
                error = f"Invalid input: {str(e)}"

    return render_template('decision_support.html',
                           predictions=predictions,
                           error=error,
                           latest_data=current_data)



                         
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5000)