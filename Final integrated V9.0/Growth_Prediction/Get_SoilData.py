import requests

# Firebase Realtime Database endpoint (make sure to add the full path to your data)
firebase_url = 'https://myiot-80b9f-default-rtdb.firebaseio.com/soil_data.json'

try:
    response = requests.get(firebase_url)
    if response.status_code == 200:
        data = response.json()
        if data:
            # Assuming 'moisture' is one of the keys in the latest object
            print("Retrieved Soil Data:", data)
            soil_moisture = data.get('moisture', 'No moisture key found')
            print(f"Soil Moisture Value: {soil_moisture}")
        else:
            print("No data found at the endpoint.")
    else:
        print(f"Failed to retrieve data. HTTP {response.status_code}")
except Exception as e:
    print("An error occurred:", str(e))
