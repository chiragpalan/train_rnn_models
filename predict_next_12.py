import os
import sqlite3
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime, timedelta

# Paths
DATABASE_PATH = "nifty50_data_v1.db"
PREDICTIONS_FOLDER = "predictions"
PREDICTIONS_DB = os.path.join(PREDICTIONS_FOLDER, "predictions_next_12.db")
MODELS_FOLDER = "models"

# Ensure the predictions folder exists
os.makedirs(PREDICTIONS_FOLDER, exist_ok=True)

# Connect to the database
conn = sqlite3.connect(DATABASE_PATH)
cursor = conn.cursor()

# Predict for each table in the database
for table_name in cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall():
    table_name = table_name[0]

    if table_name == "sqlite_sequence":
        continue

    # Load data
    query = f"SELECT * FROM {table_name} ORDER BY Datetime DESC LIMIT 12"
    data = pd.read_sql(query, conn)

    if data.empty:
        print(f"No data found in table {table_name}. Skipping...")
        continue

    # Preprocess data
    features = ["Open", "High", "Low", "Close", "Volume", "Adj_Close"]
    if not all(feature in data.columns for feature in features):
        print(f"Missing required features in table {table_name}. Skipping...")
        continue

    data = data[features + ["Datetime"]]

    # Load the model
    model_path = os.path.join(MODELS_FOLDER, f"{table_name}_rnn_model.h5")
    if not os.path.exists(model_path):
        print(f"Model for table {table_name} not found. Skipping...")
        continue

    model = tf.keras.models.load_model(model_path)

    # Prepare data for prediction
    X_input = data[features].values[::-1]  # Reverse order for proper time sequence
    X_input = X_input.reshape(1, X_input.shape[0], X_input.shape[1])  # Reshape for RNN input

    # Predict next 12 timestamps
    predictions = model.predict(X_input)
    predictions = predictions.reshape(-1, len(features))

    # Generate Datetime for next 12 predictions
    last_datetime = datetime.strptime(data["Datetime"].iloc[0], "%Y-%m-%d %H:%M:%S")
    prediction_timestamps = [
        last_datetime + timedelta(minutes=5 * i) for i in range(1, 13)
    ]

    # Save predictions
    predictions_df = pd.DataFrame(
        predictions, columns=[f"Predicted_{col}" for col in features]
    )
    predictions_df["Datetime"] = prediction_timestamps

    with sqlite3.connect(PREDICTIONS_DB) as predictions_conn:
        predictions_df.to_sql(table_name, predictions_conn, if_exists="replace", index=False)

print(f"Predictions database saved at {PREDICTIONS_DB}")
conn.close()
