import os
import sqlite3
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

DATABASE_PATH = "nifty50_data_v1.db"
PREDICTION_DATABASE_PATH = "prediction.db"
MODELS_FOLDER = "models"
PREDICTION_DATE = "2024-01-10"

def load_data(table_name):
    conn = sqlite3.connect(DATABASE_PATH)
    query = f"SELECT * FROM {table_name} WHERE Datetime LIKE '{PREDICTION_DATE}%'"
    data = pd.read_sql(query, conn)
    conn.close()
    return data

def load_model_and_scaler(table_name):
    model_path = f"{MODELS_FOLDER}/{table_name}_model.h5"
    scaler_path = f"{MODELS_FOLDER}/{table_name}_scaler.pkl"
    model = load_model(model_path)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

def make_predictions(table_name):
    data = load_data(table_name)
    if data.empty:
        return pd.DataFrame()  # No data for prediction date

    actual_data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    if actual_data.empty:
        return pd.DataFrame()  # No valid data for prediction

    model, scaler = load_model_and_scaler(table_name)
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

def save_predictions_to_db(predictions, table_name):
    conn = sqlite3.connect(PREDICTION_DATABASE_PATH)
    predictions.to_sql(table_name, conn, if_exists='append')
    conn.close()

# Create the prediction database if it does not exist
if not os.path.exists(PREDICTION_DATABASE_PATH):
    conn = sqlite3.connect(PREDICTION_DATABASE_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS dummy (id INTEGER PRIMARY KEY)")
    conn.close()

# List all tables
conn = sqlite3.connect(DATABASE_PATH)
tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
conn.close()

# Make predictions for each table and save to prediction.db
for table in tables['name']:
    if table != 'sqlite_sequence':
        predictions = make_predictions(table)
        if not predictions.empty:
            save_predictions_to_db(predictions, table)
