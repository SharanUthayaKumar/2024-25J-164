from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)


model = joblib.load('watering_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict_watering_duration():
    input_data = request.get_json()
    input_df = pd.DataFrame(input_data)
    scaled_input_data = scaler.transform(input_df)
    predicted_watering_duration = model.predict(scaled_input_data)
    return jsonify({'predicted_watering_duration': predicted_watering_duration[0]})


if __name__ == '__main__':
    app.run(debug=True)
