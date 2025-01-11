import os
import sqlite3
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Define paths (relative to repo root)
DATABASE_PATH = "./nifty50_data_v1.db"
PREDICTION_FOLDER = "./predictions"
MODELS_FOLDER = "./models"
PREDICTION_DATE = "2024-01-10"  # Change as needed

def load_data(table_name):
    conn = sqlite3.connect(DATABASE_PATH)
    query = f"SELECT * FROM {table_name}"
    data = pd.read_sql(query, conn)
    print(data.head())
    data["date_str"] = pd.to_datetime(data["Datetime"]).strftime("%Y-%m-%d")
    # data["date_str"] = date_str
    print(data.head())
    conn.close()
    return data
print(load_data("TCS_NS"))

def load_model_and_scaler(table_name):
    model_path = f"{MODELS_FOLDER}/{table_name}_model.h5"
    scaler_path = f"{MODELS_FOLDER}/{table_name}_scaler.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"Model or scaler missing for {table_name}. Skipping...")
        return None, None
    
    model = load_model(model_path)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

def make_predictions(table_name):
    data = load_data(table_name)
    if data.empty:
        print(f"No data for prediction date in table {table_name}")
        return pd.DataFrame()

    actual_data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    if actual_data.empty:
        print(f"No valid data for prediction in table {table_name}")
        return pd.DataFrame()

    model, scaler = load_model_and_scaler(table_name)
    if model is None or scaler is None:
        return pd.DataFrame()

    scaled_data = scaler.transform(actual_data)
    X_test = [scaled_data[i-12:i] for i in range(12, len(scaled_data))]
    X_test = np.array(X_test)
    
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    prediction_df = pd.DataFrame(predictions, columns=['Open', 'High', 'Low', 'Close', 'Volume'])
    prediction_df['Datetime'] = data['Datetime'].values[12:]
    prediction_df.set_index('Datetime', inplace=True)

    actual_data['Datetime'] = data['Datetime']
    actual_data.set_index('Datetime', inplace=True)

    result = pd.concat([actual_data, prediction_df], axis=1, keys=['Actual', 'Predicted'])
    return result

def save_predictions_to_csv(predictions, table_name):
    if not os.path.exists(PREDICTION_FOLDER):
        os.makedirs(PREDICTION_FOLDER)
    csv_path = os.path.join(PREDICTION_FOLDER, f"{table_name}_predictions.csv")
    predictions.to_csv(csv_path)
    print(f"Predictions saved to {csv_path}")

# Main script logic
def main():
    # List all tables
    conn = sqlite3.connect(DATABASE_PATH)
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
    conn.close()

    # Make predictions for each table and save to CSV files
    for table in tables['name']:
        if table != 'sqlite_sequence':
            try:
                predictions = make_predictions(table)
                if not predictions.empty:
                    save_predictions_to_csv(predictions, table)
                else:
                    print(f"No predictions generated for table {table}")
            except Exception as e:
                print(f"Error processing table {table}: {e}")

if __name__ == "__main__":
    main()
