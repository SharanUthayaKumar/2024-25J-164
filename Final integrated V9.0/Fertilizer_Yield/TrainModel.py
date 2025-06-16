import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
import joblib

# Load  dataset
df = pd.read_csv('agricultural_data.csv')

# Preprocessing
df['Growth Stage'] = df['Growth Stage'].map({
    'Vegetative': 0,
    'Flowering': 1,
    'Fruiting': 2,
    'Ripening': 3,
    'Harvesting': 4
})

df['Harvest Readiness (Yes/No)'] = df['Harvest Readiness (Yes/No)'].map({
    'Yes': 1,
    'No': 0
})

# Split data
X = df[['Soil Temperature (°C)', 'Air Temperature (°C)', 'Humidity (%)', 'Growth Stage']]
y_fertilizer = df['Fertilizer Applied (kg/ha)']
y_yield = df['Predicted Yield (kg/ha)']
y_harvest = df['Harvest Readiness (Yes/No)']

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset
X_train, X_test, y_fertilizer_train, y_fertilizer_test, y_yield_train, y_yield_test, y_harvest_train, y_harvest_test = train_test_split(
    X_scaled, y_fertilizer, y_yield, y_harvest, test_size=0.2, random_state=42
)

# Train models
fertilizer_model = RandomForestRegressor(n_estimators=100, random_state=42)
yield_model = RandomForestRegressor(n_estimators=100, random_state=42)
harvest_model = RandomForestClassifier(n_estimators=100, random_state=42)

fertilizer_model.fit(X_train, y_fertilizer_train)
yield_model.fit(X_train, y_yield_train)
harvest_model.fit(X_train, y_harvest_train)

# Evaluate
fertilizer_pred = fertilizer_model.predict(X_test)
yield_pred = yield_model.predict(X_test)
harvest_pred = harvest_model.predict(X_test)

fertilizer_mse = mean_squared_error(y_fertilizer_test, fertilizer_pred)
yield_mse = mean_squared_error(y_yield_test, yield_pred)
harvest_accuracy = accuracy_score(y_harvest_test, harvest_pred)


print(f'Harvest Readiness Model Accuracy: {harvest_accuracy}')

# Save models
joblib.dump(fertilizer_model, 'fertilizer_model.pkl')
joblib.dump(yield_model, 'yield_model.pkl')
joblib.dump(harvest_model, 'harvest_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Models and scaler saved successfully!")
