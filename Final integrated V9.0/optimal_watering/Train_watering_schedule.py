import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
import logging
from sklearn.model_selection import train_test_split
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_preprocess_data(file_path):
    try:
        dataset = pd.read_csv(file_path)
        logging.info("Dataset loaded")
    except Exception as e:
        logging.error(f"Error loading{e}")
        raise

    # Check for missing values
    if dataset.isnull().sum().any():
        logging.warning("Missing values found in the dataset. Filling missing values with the mean.")
        dataset.fillna(dataset.mean(), inplace=True)

    # Split dataset
    X = dataset[['Precipitation (mm)', 'Ambient Temp (°C)', 'Humidity (%)', 'Soil Temp (°C)', 'Soil Moisture (%)']]
    y = dataset['Watering Duration (minutes)']
    return X, y

def scale_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logging.info("Data scaled")
    return X_scaled, scaler

def train_model(X_scaled, y):

    try:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)
        logging.info("Model trained successfully.")
    except Exception as e:
        logging.error(f"Error training{e}")
        raise
    return model

def save_model(model, scaler, model_file='watering_model.pkl', scaler_file='scaler.pkl'):
    try:
        joblib.dump(model, model_file)
        joblib.dump(scaler, scaler_file)
        logging.info(f"Model  saved successfully as {model_file} and {scaler_file}.")
    except Exception as e:
        logging.error(f"Error saving {e}")
        raise

def main():
    dataset_file = 'optimal_watering_schedule.csv'
    # preprocess  data
    X, y = load_and_preprocess_data(dataset_file)
    # Scale  data
    X_scaled, scaler = scale_data(X)
    model = train_model(X_scaled, y)
    save_model(model, scaler)
    
    logging.info("Training complete.")

if __name__ == "__main__":
    main()
