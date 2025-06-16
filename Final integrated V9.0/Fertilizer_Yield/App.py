from flask import Flask, render_template, jsonify, request
import serial
import threading
import time
import joblib
import pandas as pd
from serial.tools import list_ports

app = Flask(__name__)

# Load ML models
try:
    fertilizer_model = joblib.load('fertilizer_model.pkl')
    yield_model = joblib.load('yield_model.pkl')
    harvest_model = joblib.load('harvest_model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("✅ ML models loaded successfully")
except Exception as e:
    print(f"❌ Error loading ML models: {e}")
    fertilizer_model = yield_model = harvest_model = scaler = None

# Data storage
data_buffer = ["Initializing..."]
buffer_lock = threading.Lock()
latest_data = {'soil_temp': None, 'env_temp': None, 'humidity': None}
serial_connected = False


# Serial connection setup
def connect_serial():
    global ser, serial_connected
    ports = list_ports.comports()
    print("Available ports:")
    for port in ports:
        print(f" - {port.device}")

    for _ in range(3):  # Retry connection
        try:
            ser = serial.Serial(
                port='COM6',
                baudrate=115200,
                timeout=1,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
            serial_connected = True
            print("✅ Serial connection established")
            return ser
        except serial.SerialException as e:
            print(f"⚠️ Serial connection failed (attempt {_ + 1}/3): {e}")
            time.sleep(2)

    serial_connected = False
    print("❌ Failed to establish serial connection")
    return None


ser = connect_serial()


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
                                    latest_data = {
                                        'soil_temp': parts[0],
                                        'env_temp': parts[1],
                                        'humidity': parts[2]
                                    }
                                    print(f"📊 Parsed data: {latest_data}")
                            except (ValueError, IndexError) as e:
                                print(f"⚠️ Parsing error: {e} | Data: {line}")
                    except UnicodeDecodeError:
                        print(f"⚠️ Decode error: {raw_data}")
            except Exception as e:
                print(f"❌ Serial error: {e}")
                serial_connected = False
                time.sleep(1)
                connect_serial()
        else:
            time.sleep(1)
            connect_serial()


# Start serial thread
if ser:
    serial_thread = threading.Thread(target=read_serial, daemon=True)
    serial_thread.start()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/data')
def get_data():
    with buffer_lock:
        last_line = data_buffer[-1] if data_buffer else "No data"
        return jsonify({
            'line': last_line,
            'connected': serial_connected
        })


@app.route('/sensor_data')
def get_sensor_data():
    return jsonify({
        'sensors': latest_data,
        'connected': serial_connected
    })


@app.route('/predict', methods=['POST'])
def predict():
    if not all([fertilizer_model, yield_model, harvest_model, scaler]):
        return jsonify({'error': 'ML models not loaded'})

    try:
        data = {
            'soil_temp': float(request.form['soil_temp']),
            'air_temp': float(request.form['air_temp']),
            'humidity': float(request.form['humidity']),
            'growth_stage': int(request.form['growth_stage'])
        }

        features = scaler.transform([[data['soil_temp'], data['air_temp'],
                                      data['humidity'], data['growth_stage']]])

        return jsonify({
            'fertilizer': round(float(fertilizer_model.predict(features)[0]), 2),
            'yield': round(float(yield_model.predict(features)[0]), 2),
            'harvest': "Yes" if harvest_model.predict(features)[0] == 1 else "No"
        })
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5000)