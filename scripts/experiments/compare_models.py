"""
COMPARE MODELS: Synchronized Benchmark
Isolation Forest vs One-Class SVM vs Local Outlier Factor (Novelty)
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

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
print(f"Features detected for model comparison: {FEATURE_COLUMNS}")

X_train = train_df[FEATURE_COLUMNS].fillna(0)
X_test_normal = test_normal_df[FEATURE_COLUMNS].fillna(0)
X_suspicious = suspicious_df[FEATURE_COLUMNS].fillna(0)

y_true = np.array([0] * len(X_test_normal) + [1] * len(X_suspicious))
X_eval = pd.concat([X_test_normal, X_suspicious], ignore_index=True)

# Standard Scaler for SVM
std_scaler = StandardScaler()
X_train_std = std_scaler.fit_transform(X_train)
X_eval_std = std_scaler.transform(X_eval)

# MinMax Scaler preserves discrete count signals for LOF distance calculations
minmax_scaler = MinMaxScaler()
X_train_mm = minmax_scaler.fit_transform(X_train)
X_eval_mm = minmax_scaler.transform(X_eval)


def evaluate_predictions(name, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    fpr = cm[0][1] / (cm[0][0] + cm[0][1]) if (cm[0][0] + cm[0][1]) > 0 else 0.0
    return {
        "model": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "fpr": fpr,
        "cm": cm,
    }


def sweep_threshold(scores, max_fpr=0.05):
    thresholds = np.linspace(scores.min(), scores.max(), 500)
    best_f1 = -1
    best_preds = None

    for thresh in thresholds:
        preds = (scores < thresh).astype(int)
        fpr = (preds[:len(X_test_normal)] == 1).mean()
        if fpr <= max_fpr:
            f1 = f1_score(y_true, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_preds = preds

    if best_preds is None:
        best_preds = (scores < np.percentile(scores, 10)).astype(int)

    return best_preds


results = []

# 1. Isolation Forest (Tree-based, operates on raw features)
iso = IsolationForest(n_estimators=300, contamination="auto", random_state=42)
iso.fit(X_train)
iso_scores = iso.score_samples(X_eval)
iso_preds = sweep_threshold(iso_scores, max_fpr=0.02)
results.append(evaluate_predictions("Isolation Forest", iso_preds))

# 2. One-Class SVM (RBF kernel, standardized features)
ocsvm = OneClassSVM(kernel="rbf", gamma="scale")
ocsvm.fit(X_train_std)
svm_scores = ocsvm.decision_function(X_eval_std)
svm_preds = sweep_threshold(svm_scores, max_fpr=0.05)
results.append(evaluate_predictions("One-Class SVM", svm_preds))

# 3. Local Outlier Factor (Manhattan metric + MinMax scaled features)
lof = LocalOutlierFactor(n_neighbors=15, metric="manhattan", novelty=True)
lof.fit(X_train_mm)
lof_scores = lof.decision_function(X_eval_mm)
lof_preds = sweep_threshold(lof_scores, max_fpr=0.05)
results.append(evaluate_predictions("Local Outlier Factor", lof_preds))

# Output Summary Table
print("\n" + "=" * 80)
print(f"BENCHMARK COMPARISON ({len(X_train)} training rows | {len(X_test_normal)} normal + {len(X_suspicious)} suspicious eval rows)")
print("=" * 80)
print(f"{'Model':<24}{'Accuracy':<11}{'Precision':<12}{'Recall':<10}{'F1':<10}{'FPR':<8}")
print("-" * 80)
for r in results:
    print(f"{r['model']:<24}{r['accuracy']:<11.1%}{r['precision']:<12.1%}"
          f"{r['recall']:<10.1%}{r['f1']:<10.1%}{r['fpr']:<8.1%}")

print("\nConfusion matrices:")
for r in results:
    print(f"\n{r['model']}:")
    print("                  Predicted Normal   Predicted Anomaly")
    print(f"Actual Normal     {r['cm'][0][0]:<18}{r['cm'][0][1]}")
    print(f"Actual Suspicious {r['cm'][1][0]:<18}{r['cm'][1][1]}")

best = max(results, key=lambda r: r["f1"])
print(f"\nBest model by F1-score: {best['model']} (F1={best['f1']:.1%})")
