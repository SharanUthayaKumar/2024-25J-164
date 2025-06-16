import serial

# Adjust 'COM3' to your ESP32's port (e.g., /dev/ttyUSB0 for Linux)
ser = serial.Serial('COM7', 115200, timeout=1)

try:
    while True:
        line = ser.readline().decode('utf-8').strip()
        if line:
            try:
                soil_temp, env_temp, humidity = map(float, line.split(","))
                print(f"Soil Temp: {soil_temp} °C, Env Temp: {env_temp} °C, Humidity: {humidity} %")
            except ValueError:
                print("Invalid data received:", line)
except KeyboardInterrupt:
    ser.close()
    print("Serial connection closed.")
