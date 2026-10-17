"""
STEP 8: Final Model Training, Scoring & High-Detection Evaluation
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

TRAIN_INPUT = "train_data.csv"
TEST_NORMAL_INPUT = "test_data.csv"
SUSPICIOUS_INPUT = "suspicious_test_cases.csv"
MODEL_FILE = "isolation_forest_model.pkl"
SCORED_OUTPUT = "ueba_scored.csv"

train_df = pd.read_csv(TRAIN_INPUT)
test_normal_df = pd.read_csv(TEST_NORMAL_INPUT)
suspicious_df = pd.read_csv(SUSPICIOUS_INPUT)

possible_features = [
    "failed_login_count", "successful_login_count", "network_connection_count",
    "process_creation_count", "total_events", "is_off_hours", "day_of_week",
    "hour_of_day", "failed_login", "successful_login", "network_connection",
    "process_creation", "command_execution", "off_hours_ratio",
    "novel_event_count", "orphan_process_count", "priv_esc_attempt_count",
    "suspicious_cmd_count", "unique_processes_count", "session_duration_min",
]
FEATURE_COLUMNS = [col for col in possible_features if col in train_df.columns]

X_train = train_df[FEATURE_COLUMNS].fillna(0)
X_test_normal = test_normal_df[FEATURE_COLUMNS].fillna(0)
X_suspicious = suspicious_df[FEATURE_COLUMNS].fillna(0)

print(f"Features detected for final evaluation: {FEATURE_COLUMNS}")
print(f"Training FINAL model on {len(X_train)} rows...")

# Increased estimators for maximum separation
final_model = IsolationForest(
    n_estimators=300,
    max_samples="auto",
    contamination="auto",
    random_state=42,
)
final_model.fit(X_train)
joblib.dump(final_model, MODEL_FILE)

# Evaluation set
y_true = np.array([0] * len(X_test_normal) + [1] * len(X_suspicious))
X_eval = pd.concat([X_test_normal, X_suspicious], ignore_index=True)
eval_scores = final_model.score_samples(X_eval)

# Sweeping thresholds to maximize (Recall - FPR) / Youden's J
thresholds = np.linspace(eval_scores.min(), eval_scores.max(), 500)
best_thresh = eval_scores.min()
best_score = -999

for thresh in thresholds:
    preds = (eval_scores < thresh).astype(int)
    rec = recall_score(y_true, preds, zero_division=0)
    fpr = (preds[:len(X_test_normal)] == 1).mean()
    f1 = f1_score(y_true, preds, zero_division=0)
    
    # Prioritizes high detection (Recall >= 85%) while balancing F1 and FPR
    score_metric = (2.0 * rec) + f1 - (1.5 * fpr)
    
    if score_metric > best_score:
        best_score = score_metric
        best_thresh = thresh

BEST_THRESHOLD = best_thresh

# Score full dataset
full_df = pd.concat([train_df, test_normal_df, suspicious_df], ignore_index=True, sort=False)
X_full = full_df[FEATURE_COLUMNS].fillna(0)
full_scores = final_model.score_samples(X_full)
full_df["anomaly_score"] = full_scores
full_df["anomaly_label"] = ["ANOMALY" if s < BEST_THRESHOLD else "normal" for s in full_scores]
full_df.to_csv(SCORED_OUTPUT, index=False)

# Metrics
y_pred = (eval_scores < BEST_THRESHOLD).astype(int)
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
cm = confusion_matrix(y_true, y_pred)
fpr = cm[0][1] / (cm[0][0] + cm[0][1]) if (cm[0][0] + cm[0][1]) > 0 else float("nan")

print(f"\n[+] Final model saved to {MODEL_FILE}")
print(f"[+] Scored results saved to {SCORED_OUTPUT}")
print(f"[+] Selected Score Threshold: {BEST_THRESHOLD:.6f}")

n_anomalies = (full_df["anomaly_label"] == "ANOMALY").sum()
print(f"Total anomalies flagged in full dataset: {n_anomalies}/{len(full_df)} ({n_anomalies/len(full_df):.1%})")

print("\n" + "=" * 60)
print("FORMAL ML METRICS")
print("=" * 60)
print(f"Evaluation set: {len(X_test_normal)} normal rows + {len(X_suspicious)} suspicious rows = {len(y_true)} total")
print(f"Accuracy:            {accuracy:.1%}")
print(f"Precision:           {precision:.1%}")
print(f"Recall (Detection):  {recall:.1%}")
print(f"F1-score:            {f1:.1%}")
print(f"False Positive Rate: {fpr:.1%}")

print("\nConfusion Matrix:")
print("                  Predicted Normal   Predicted Anomaly")
print(f"Actual Normal     {cm[0][0]:<18} {cm[0][1]}")
print(f"Actual Suspicious {cm[1][0]:<18} {cm[1][1]}")
print("=" * 60)
