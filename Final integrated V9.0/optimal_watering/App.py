from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import requests
import threading
import serial
import time
from serial.tools import list_ports

app = Flask(__name__)

# Load model and scaler
model = joblib.load('watering_model.pkl')
scaler = joblib.load('scaler.pkl')

# Firebase endpoint
FIREBASE_URL = 'https://myiot-80b9f-default-rtdb.firebaseio.com/soil_data.json'

# Data storage
data_buffer = ["Initializing..."]
buffer_lock = threading.Lock()
latest_data = {'soil_temp': None, 'env_temp': None, 'humidity': None}
serial_connected = False
ser = None

def get_soil_moisture_from_firebase():
    try:
        response = requests.get(FIREBASE_URL)
        if response.status_code == 200:
            data = response.json()
            return float(data.get('moisture', 0))
    except Exception as e:
        print("Error fetching soil moisture:", e)
    return None

def connect_serial():
    global ser, serial_connected
    ports = list_ports.comports()
    print("Available ports:")
    for port in ports:
        print(f" - {port.device}")

    for _ in range(3):  # Retry connection
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
            print("✅ Serial connection established")
            return ser
        except serial.SerialException as e:
            print(f"⚠️ Serial connection failed (attempt {_ + 1}/3): {e}")
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
if connect_serial():
    serial_thread = threading.Thread(target=read_serial, daemon=True)
    serial_thread.start()

@app.route('/', methods=['GET', 'POST'])
def home():
    predicted_watering_duration = None
    auto_soil_moisture = get_soil_moisture_from_firebase()
    error = None

    if request.method == 'POST':
        try:
            # Get form data or use sensor data
            precipitation = float(request.form['precipitation'])
            ambient_temp = float(request.form.get('ambient_temp', latest_data['env_temp'] or 0))
            humidity = float(request.form.get('humidity', latest_data['humidity'] or 0))
            soil_temp = float(request.form.get('soil_temp', latest_data['soil_temp'] or 0))
            soil_moisture = float(request.form.get('soil_moisture', auto_soil_moisture or 0))

            # Prepare DataFrame
            input_data = {
                'Precipitation (mm)': [precipitation],
                'Ambient Temp (°C)': [ambient_temp],
                'Humidity (%)': [humidity],
                'Soil Temp (°C)': [soil_temp],
                'Soil Moisture (%)': [soil_moisture]
            }
            input_df = pd.DataFrame(input_data)

            # Scale and predict
            scaled_input = scaler.transform(input_df)
            predicted_watering_duration = model.predict(scaled_input)[0]

        except Exception as e:
            error = f"Invalid input: {str(e)}"

    return render_template('index.html', 
                         predicted_duration=predicted_watering_duration,
                         auto_soil_moisture=auto_soil_moisture,
                         error=error,
                         sensor_data=latest_data)

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

if __name__ == '__main__':
    app.run(debug=True)