"""
STEP 6: Train the Model
Loads train_data.csv and trains Isolation Forest on it ONLY.
"""
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

TRAIN_INPUT = "train_data.csv"
MODEL_FILE = "isolation_forest_model.pkl"

df = pd.read_csv(TRAIN_INPUT)

# Dynamically pick available numeric feature columns from your dataset
possible_features = [
    # Full names from extraction
    "failed_login_count",
    "successful_login_count",
    "network_connection_count",
    "process_creation_count",
    "total_events",
    "is_off_hours",
    "day_of_week",
    "hour_of_day",
    # Short names (current extract_features.py)
    "failed_login",
    "successful_login",
    "network_connection",
    "process_creation",
    "command_execution",
    "off_hours_ratio",
    "novel_event_count",
    "orphan_process_count",
    "priv_esc_attempt_count",
    "suspicious_cmd_count",
    "unique_processes_count",
    "session_duration_min",
]

# Use whichever feature columns actually exist in train_data.csv
FEATURE_COLUMNS = [col for col in possible_features if col in df.columns]

print(f"Features detected for training: {FEATURE_COLUMNS}")

CONTAMINATION = 'auto'

X_train = df[FEATURE_COLUMNS].fillna(0)

print(f"Training Isolation Forest on {len(X_train)} rows (contamination={CONTAMINATION})...")

model = IsolationForest(n_estimators=100, contamination=CONTAMINATION, random_state=42)
model.fit(X_train)

joblib.dump(model, MODEL_FILE)
print(f"\n[+] Model trained successfully and saved to {MODEL_FILE}")

# Quick sanity check
preds = model.predict(X_train)
n_anomalies = (preds == -1).sum()
print(f"Sanity check - on training data itself: {n_anomalies}/{len(X_train)} flagged as anomalies")
print("(This is expected to roughly match the contamination rate - it's not a real test,")
print(" the real test comes in step 8 using data the model has NEVER seen.)")
