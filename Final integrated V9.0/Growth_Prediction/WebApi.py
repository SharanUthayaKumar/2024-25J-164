from flask import Flask, request, jsonify
import joblib
import numpy as np
app = Flask(__name__)


height_model = joblib.load('height_model.pkl')
leaf_area_model = joblib.load('leaf_area_model.pkl')

def predict_growth(soil_moisture, precipitation, fertilizer_amount):
    input_data = np.array([[soil_moisture, precipitation, fertilizer_amount]])
    predicted_height = height_model.predict(input_data)[0]
    predicted_leaf_area = leaf_area_model.predict(input_data)[0]
    return predicted_height, predicted_leaf_area


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not all(key in data for key in ['soil_moisture', 'precipitation', 'fertilizer_amount']):
        return jsonify({"error": "Missing required input fields"}), 400
    soil_moisture = data['soil_moisture']
    precipitation = data['precipitation']
    fertilizer_amount = data['fertilizer_amount']
    predicted_height, predicted_leaf_area = predict_growth(soil_moisture, precipitation, fertilizer_amount)

    output = {
        "Predicted Plant Height (cm)": round(predicted_height, 2),
        "Predicted Leaf Area (cm²)": round(predicted_leaf_area, 2)
    }
    return jsonify(output)


if __name__ == '__main__':
    app.run(debug=True)
