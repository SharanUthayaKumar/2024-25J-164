import requests
import datetime
from urllib.parse import quote_plus

#  coordinates
latitude = 8.0
longitude = 79.8

# timezone
timezone = "Asia/Colombo"

# API URL
url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&hourly=precipitation&timezone={quote_plus(timezone)}"

# request to the Open-Meteo API
response = requests.get(url)

# Check if the response
if response.status_code == 200:
    data = response.json()
    
    # forecast for the next few hours
    print("Rainfall Forecast for the next few hours in Puttalam, Sri Lanka:\n")
    for i in range(len(data['hourly']['precipitation'])):
        timestamp = data['hourly']['time'][i]
        precipitation = data['hourly']['precipitation'][i]

        date_time = datetime.datetime.strptime(timestamp, "%Y-%m-%dT%H:%M")
        formatted_time = date_time.strftime('%Y-%m-%d %H:%M:%S')

        print(f"{formatted_time} - Precipitation: {precipitation}mm")
else:
    print(f"Error fetching weather data. Status code: {response.status_code}")
    print(f"Response Text: {response.text}")
