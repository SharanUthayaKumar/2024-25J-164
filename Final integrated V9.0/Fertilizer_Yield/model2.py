import pandas as pd
import numpy as np
from sklearn.svm import SVR, SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
import joblib

# Load dataset
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

# Features and targets
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

# Train SVM models
fertilizer_model = SVR(kernel='rbf')
yield_model = SVR(kernel='rbf')
harvest_model = SVC(kernel='rbf')

fertilizer_model.fit(X_train, y_fertilizer_train)
yield_model.fit(X_train, y_yield_train)
harvest_model.fit(X_train, y_harvest_train)

# Predict
fertilizer_preds = fertilizer_model.predict(X_test)
yield_preds = yield_model.predict(X_test)
harvest_preds = harvest_model.predict(X_test)

# Evaluate
fertilizer_rmse = mean_squared_error(y_fertilizer_test, fertilizer_preds, squared=False)
yield_rmse = mean_squared_error(y_yield_test, yield_preds, squared=False)
harvest_acc = accuracy_score(y_harvest_test, harvest_preds)

print(f'Fertilizer Prediction RMSE: {fertilizer_rmse:.2f}')
print(f'Yield Prediction RMSE: {yield_rmse:.2f}')
print(f'Harvest Readiness Accuracy: {harvest_acc:.2f}')

# Save models and scaler
joblib.dump(fertilizer_model, 'svm_fertilizer_model.pkl')
joblib.dump(yield_model, 'svm_yield_model.pkl')
joblib.dump(harvest_model, 'svm_harvest_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
