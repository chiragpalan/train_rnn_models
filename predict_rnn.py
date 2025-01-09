import os
import sqlite3
import pandas as pd
import tensorflow as tf

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
    features = ["Open", "High", "Low", "Close", "Volume", "Adj_Close"]
    if not all(feature in data.columns for feature in features):
        print(f"Missing required features in table {table_name}. Skipping...")
        continue

    data = data[features + ["Datetime"]]

    # Load model
    # model_path = os.path.join(MODELS_FOLDER, f"{table_name}_rnn_model.h5")
    
    script_dir = r"/home/runner/work/train_rnn_models" #os.path.dirname(os.path.abspath(__file__))
    print(script_dir)
    models_folder = os.path.join(script_dir, MODELS_FOLDER)    
    model_path = os.path.join(models_folder, f"{table_name}_model.h5")
    print(os.path.exists(model_path))
    print(model_path)
    if not os.path.exists(model_path):
        print(f"Model for table {table_name} not found. Skipping...")
        continue

    model = tf.keras.models.load_model(model_path)

    # Make predictions
    X_test = data[-12:].values[:, :-1].reshape(1, 12, len(features))  # Last 12 steps for prediction
    predictions = model.predict(X_test)

    # Save predictions to a new database
    predictions_df = pd.DataFrame(predictions, columns=[f"Predicted_{col}" for col in features[:-1]])
    predictions_df["Datetime"] = data["Datetime"].iloc[-12:].values
    predictions_df["Actual"] = data.iloc[-12:][features[:-1]].values.tolist()

    with sqlite3.connect(PREDICTIONS_DB) as predictions_conn:
        predictions_df.to_sql(table_name, predictions_conn, if_exists="replace", index=False)

print(f"Predictions database saved at {PREDICTIONS_DB}")

conn.close()
