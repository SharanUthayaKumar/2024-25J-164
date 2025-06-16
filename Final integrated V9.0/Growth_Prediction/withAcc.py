import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import joblib

# Load dataset
df = pd.read_csv('plant_growth_data.csv')

# Prepare Features and Target Variables
X = df[['Soil Moisture (%)', 'Precipitation (mm)', 'Fertilizer Amount (kg)']]

# Targets
y_height = df['Plant Height (cm)']
y_leaf_area = df['Leaf Area (cm²)']

# Split the Data
X_train, X_test, y_height_train, y_height_test = train_test_split(X, y_height, test_size=0.3, random_state=42)
_, _, y_leaf_area_train, y_leaf_area_test = train_test_split(X, y_leaf_area, test_size=0.3, random_state=42)

# Train Models
height_model = RandomForestRegressor(n_estimators=100, random_state=42)
height_model.fit(X_train, y_height_train)

leaf_area_model = RandomForestRegressor(n_estimators=100, random_state=42)
leaf_area_model.fit(X_train, y_leaf_area_train)

# Evaluate Models
y_height_pred = height_model.predict(X_test)
height_mape = mean_absolute_percentage_error(y_height_test, y_height_pred) * 100   
height_accuracy = 100 - height_mape   

y_leaf_area_pred = leaf_area_model.predict(X_test)
leaf_area_mape = mean_absolute_percentage_error(y_leaf_area_test, y_leaf_area_pred) * 100
leaf_area_accuracy = 100 - leaf_area_mape

# Compute Overall Accuracy
overall_accuracy = (height_accuracy + leaf_area_accuracy) / 2   


print(f"Overall Model Accuracy: {overall_accuracy:.2f}%")

# Save Models
joblib.dump(height_model, 'height_model.pkl')
joblib.dump(leaf_area_model, 'leaf_area_model.pkl')
