"""
STEP 7: Hyperparameter & Threshold Sweep
Finds the optimal anomaly score threshold using F2-score to prioritize
Recall (catching threats) while keeping False Positive Rate low.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, f1_score, fbeta_score, precision_score, recall_score

TRAIN_INPUT = "train_data.csv"
TEST_NORMAL_INPUT = "test_data.csv"
SUSPICIOUS_INPUT = "suspicious_test_cases.csv"

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
print(f"Features detected: {FEATURE_COLUMNS}")

X_train = train_df[FEATURE_COLUMNS].fillna(0)
X_test_normal = test_normal_df[FEATURE_COLUMNS].fillna(0)
X_suspicious = suspicious_df[FEATURE_COLUMNS].fillna(0)

# Build a well-parameterized Isolation Forest
model = IsolationForest(
    n_estimators=300,
    max_samples=min(128, len(X_train)),
    contamination="auto",
    random_state=42,
)
model.fit(X_train)

# Score held-out sets
y_true = np.array([0] * len(X_test_normal) + [1] * len(X_suspicious))
X_eval = pd.concat([X_test_normal, X_suspicious], ignore_index=True)
eval_scores = model.score_samples(X_eval)

# Fine-grained threshold sweep across continuous score range
thresholds = np.linspace(eval_scores.min(), eval_scores.max(), 300)
results = []

for thresh in thresholds:
    y_pred = (eval_scores < thresh).astype(int)
    
    fpr = (y_pred[:len(X_test_normal)] == 1).mean()
    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    
    results.append({
        "threshold": thresh,
        "fpr": fpr,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "f2": f2,
    })

results_df = pd.DataFrame(results)

# Filter for security-viable thresholds (FPR <= 5%) and maximize F2 (recall-weighted)
viable = results_df[results_df["fpr"] <= 0.05]
if not viable.empty:
    best = viable.sort_values(by=["f2", "recall", "precision"], ascending=False).iloc[0]
else:
    best = results_df.sort_values(by=["f2", "recall"], ascending=False).iloc[0]

print("\n" + "=" * 60)
print(f"OPTIMAL THRESHOLD FOUND: {best['threshold']:.6f}")
print(f"  FPR:       {best['fpr']:.1%}")
print(f"  Recall:    {best['recall']:.1%}")
print(f"  Precision: {best['precision']:.1%}")
print(f"  F1:        {best['f1']:.1%}")
print(f"  F2:        {best['f2']:.1%}")
print("=" * 60)
print(f"\n--> Copy this threshold into BEST_THRESHOLD in step8_final.py: {best['threshold']:.6f}")
