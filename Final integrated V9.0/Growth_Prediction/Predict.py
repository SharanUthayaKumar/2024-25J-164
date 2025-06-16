import joblib
import numpy as np

# rained models
height_model = joblib.load('height_model.pkl')
leaf_area_model = joblib.load('leaf_area_model.pkl')

# prediction
def predict_growth(soil_moisture, precipitation, fertilizer_amount):

    input_data = np.array([[soil_moisture, precipitation, fertilizer_amount]])

    predicted_height = height_model.predict(input_data)[0]
    predicted_leaf_area = leaf_area_model.predict(input_data)[0]
    
    return predicted_height, predicted_leaf_area

# input
soil_moisture = 45
precipitation = 3
fertilizer_amount = 2.2

#  prediction
predicted_height, predicted_leaf_area = predict_growth(soil_moisture, precipitation, fertilizer_amount)
print(f"Predicted Plant Height: {predicted_height:.2f} cm")
print(f"Predicted Leaf Area: {predicted_leaf_area:.2f} cm²")
