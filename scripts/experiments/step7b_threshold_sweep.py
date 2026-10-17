"""
STEP 7b: Fine-Grained Threshold Sweep
Instead of testing 6 preset contamination values, this trains ONE model
(contamination='auto' just to get it to fit; contamination doesn't affect
score_samples) and then sweeps many threshold cutoffs on the continuous
anomaly score to find the operating point with the best F1.

This finds a better precision/recall balance than the 6 discrete presets
in step7_tune.py can, because it's searching a much finer grid.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score

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
print(f"Features detected for training: {FEATURE_COLUMNS}")

X_train = train_df[FEATURE_COLUMNS].fillna(0)
X_test_normal = test_normal_df[FEATURE_COLUMNS].fillna(0)
X_suspicious = suspicious_df[FEATURE_COLUMNS].fillna(0)

print(f"Training rows: {len(X_train)}")
print(f"Held-out NORMAL test rows: {len(X_test_normal)}")
print(f"Known-SUSPICIOUS test rows: {len(X_suspicious)}")

# Train once. contamination='auto' here only affects the .predict() default
# cutoff, which we ignore - we use score_samples() and sweep our own cutoffs.
model = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
model.fit(X_train)

y_true = [0] * len(X_test_normal) + [1] * len(X_suspicious)
X_eval = pd.concat([X_test_normal, X_suspicious], ignore_index=True)

# Lower score = more anomalous in sklearn's IsolationForest
scores = model.score_samples(X_eval)

# Sweep 200 thresholds across the observed score range
lo, hi = scores.min(), scores.max()
thresholds = np.linspace(lo, hi, 200)

results = []
for t in thresholds:
    y_pred = [1 if s < t else 0 for s in scores]
    if sum(y_pred) == 0:
        continue  # skip thresholds that flag nothing
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    n_flagged = sum(y_pred)
    results.append((t, precision, recall, f1, n_flagged))

# Best by F1
best = max(results, key=lambda r: r[3])

print(f"\n{'threshold':<12}{'precision':<12}{'recall':<10}{'f1':<10}{'n_flagged'}")
print("-" * 56)
# Show a handful of points around the best one for context
results_sorted = sorted(results, key=lambda r: -r[3])[:10]
for t, p, r, f1, n in results_sorted:
    print(f"{t:<12.4f}{p:<12.1%}{r:<10.1%}{f1:<10.1%}{n}")

print(f"\nBest threshold by F1: {best[0]:.4f}")
print(f"  precision: {best[1]:.1%}")
print(f"  recall:    {best[2]:.1%}")
print(f"  f1:        {best[3]:.1%}")
print(f"  rows flagged anomaly out of {len(y_true)}: {best[4]}")

print("""
NOTE: this threshold is tuned directly against your 90-row eval set
(80 normal + 10 suspicious). With only 10 suspicious cases, treat this as
an upper bound on what's achievable with current data, not a guaranteed
generalizing threshold. To use this threshold in production instead of
contamination, save it and compare new anomaly scores against it:
    scores = model.score_samples(X_new)
    is_anomaly = scores < best_threshold
""")
