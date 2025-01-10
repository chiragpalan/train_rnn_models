import os
import sqlite3
import pandas as pd
import tensorflow as tf
from keras.losses import MeanSquaredError
from sklearn.preprocessing import MinMaxScaler

tf.keras.utils.get_custom_objects().update({"mse": MeanSquaredError()})

DATABASE_PATH = "nifty50_data_v1.db"
PREDICTIONS_FOLDER = "predictions"
MODELS_FOLDER = "models"

# Ensure the folder for predictions exists
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

    # Scale data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[features])

    # Load model
    model_path = os.path.join(MODELS_FOLDER, f"{table_name}_model.h5")
    if not os.path.exists(model_path):
        print(f"Model for table {table_name} not found. Skipping...")
        continue

    model = tf.keras.models.load_model(model_path, custom_objects={"mse": MeanSquaredError()})

    # Make predictions
    X_test = scaled_data[-12:].astype('float32').reshape(1, 12, len(features))  # Last 12 steps for prediction
    predictions = model.predict(X_test)

    # Reverse MinMax scaling
    predictions = scaler.inverse_transform(predictions.reshape(-1, len(features)))

    # Prepare predictions DataFrame
    predictions_df = pd.DataFrame(predictions, columns=[f"Predicted_{col}" for col in features])
    predictions_df.index = data.index[-12:]

    # Load existing predictions if they exist
    csv_path = os.path.join(PREDICTIONS_FOLDER, f"{table_name}_predictions.csv")
    if os.path.exists(csv_path):
        existing_predictions = pd.read_csv(csv_path)
        predictions_df = pd.concat([existing_predictions, predictions_df])

    # Save predictions to CSV
    predictions_df.to_csv(csv_path, index_label="Datetime")

print("Predictions saved to CSV files")

conn.close()
