import os
import sqlite3
import pandas as pd
import tensorflow as tf
from keras.losses import MeanSquaredError
tf.keras.utils.get_custom_objects().update({"mse": MeanSquaredError()})

DATABASE_PATH = "nifty50_data_v1.db"
PREDICTIONS_FOLDER = "predictions"
PREDICTIONS_DB = os.path.join(PREDICTIONS_FOLDER, "predictions.db")
MODELS_FOLDER = "models"

# Ensure the folder for PREDICTIONS_DB exists
os.makedirs(PREDICTIONS_FOLDER, exist_ok=True)

# Connect to the database
conn = sqlite3.connect(DATABASE_PATH)
cursor = conn.cursor()

# Load trained models and make predictions for each table
for table_name in cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall():
    table_name = table_name[0]

    if table_name == "sqlite_sequence":
        continue

    # Load data
    query = f"SELECT * FROM {table_name}"
    data = pd.read_sql(query, conn)

    # Preprocess data
    features = ["Open", "High", "Low", "Close", "Volume"]
    if not all(feature in data.columns for feature in features):
        print(f"Missing required features in table {table_name}. Skipping...")
        continue

    # Use Datetime as index only
    data.set_index("Datetime", inplace=True)

    # Load model
    model_path = os.path.join(MODELS_FOLDER, f"{table_name}_model.h5")
    print(os.path.exists(model_path))
    print(model_path)
    if not os.path.exists(model_path):
        print(f"Model for table {table_name} not found. Skipping...")
        continue

    model = tf.keras.models.load_model(model_path, custom_objects={"mse": MeanSquaredError()})

    # Make predictions
    X_test = data[-12:][features].values.astype('float32').reshape(1, 12, len(features))  # Last 12 steps for prediction
    predictions = model.predict(X_test).reshape(12, len(features))

    # Ensure predictions_df has the correct length
    predictions_df = pd.DataFrame(predictions, columns=[f"Predicted_{col}" for col in features])
    predictions_df.index = data.index[-12:]

    # Save predictions to a new database
    predictions_df["Datetime"] = data.index[-12:].values
    predictions_df["Actual"] = data[features].iloc[-12:].values.tolist()

    with sqlite3.connect(PREDICTIONS_DB) as predictions_conn:
        predictions_df.to_sql(table_name, predictions_conn, if_exists="replace", index=False)

print(f"Predictions database saved at {PREDICTIONS_DB}")

conn.close()
