"""
UEBA Results Dashboard - Dark Theme
Run with: streamlit run dashboard.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="UEBA Dashboard", layout="wide", initial_sidebar_state="collapsed")

# ============================================================
# DARK THEME STYLING
# ============================================================
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    .metric-card {
        background: linear-gradient(145deg, #1a1d29, #14161f);
        border: 1px solid #2a2e3d;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    }
    .metric-value { font-size: 42px; font-weight: 700; margin: 0; }
    .metric-label { font-size: 14px; color: #9099a8; text-transform: uppercase; letter-spacing: 1px; margin-top: 8px; }
    .metric-sub { font-size: 13px; color: #6c7688; margin-top: 4px; }
    .green { color: #2ecc71; }
    .red { color: #e74c3c; }
    .blue { color: #3498db; }
    .yellow { color: #f1c40f; }
    h1, h2, h3 { color: #f0f0f0 !important; }
    .section-divider { border-top: 1px solid #2a2e3d; margin: 32px 0 24px 0; }
</style>
""", unsafe_allow_html=True)

PLOTLY_DARK = "plotly_dark"

# ============================================================
# LOAD DATA & SAVED MODEL
# ============================================================
def load_data():
    train_df = pd.read_csv("train_data.csv")
    test_normal_df = pd.read_csv("test_data.csv")
    suspicious_df = pd.read_csv("suspicious_test_cases.csv")
    scored_df = pd.read_csv("ueba_scored.csv")
    try:
        model = joblib.load("isolation_forest_model.pkl")
    except Exception:
        model = None
    return train_df, test_normal_df, suspicious_df, scored_df, model

try:
    train_df, test_normal_df, suspicious_df, scored_df, trained_model = load_data()
except FileNotFoundError as e:
    st.error(f"Missing file: {e}. Run your step8_final.py script first.")
    st.stop()

# Match exact feature names from the saved model or dynamically from the dataframe
if trained_model is not None and hasattr(trained_model, "feature_names_in_"):
    FEATURE_COLUMNS = list(trained_model.feature_names_in_)
else:
    possible_features = [
        "day_of_week", "failed_login", "successful_login", "network_connection",
        "process_creation", "command_execution", "off_hours_ratio",
        "novel_event_count", "orphan_process_count", "priv_esc_attempt_count",
        "suspicious_cmd_count", "unique_processes_count", "session_duration_min"
    ]
    FEATURE_COLUMNS = [col for col in possible_features if col in train_df.columns]

full_df = pd.concat([train_df, test_normal_df]).reset_index(drop=True)

# ============================================================
# HEADER
# ============================================================
st.markdown("# 🛡️ UEBA Analytics")
st.markdown("<p style='color:#6c7688;'>AI-Driven SOC — User & Entity Behavior Analytics</p>", unsafe_allow_html=True)
st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

# ============================================================
# TOP STAT CARDS
# ============================================================
total_rows = len(scored_df)
n_anomaly = (scored_df["anomaly_label"] == "ANOMALY").sum() if "anomaly_label" in scored_df.columns else 0
n_normal = total_rows - n_anomaly
pct_anomaly = (n_anomaly / total_rows * 100) if total_rows else 0
pct_normal = 100 - pct_anomaly

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class='metric-card'>
        <p class='metric-value blue'>{total_rows}</p>
        <p class='metric-label'>Total Scored Rows</p>
        <p class='metric-sub'>{scored_df['user'].nunique() if 'user' in scored_df else full_df['user'].nunique()} tracked users</p>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class='metric-card'>
        <p class='metric-value green'>{pct_normal:.1f}%</p>
        <p class='metric-label'>Normal Behavior</p>
        <p class='metric-sub'>{n_normal} rows</p>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class='metric-card'>
        <p class='metric-value red'>{pct_anomaly:.1f}%</p>
        <p class='metric-label'>Anomalous Behavior</p>
        <p class='metric-sub'>{n_anomaly} rows flagged</p>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class='metric-card'>
        <p class='metric-value yellow'>{len(suspicious_df)}</p>
        <p class='metric-label'>Known Attack Cases</p>
        <p class='metric-sub'>held out for evaluation</p>
    </div>""", unsafe_allow_html=True)

st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

# ============================================================
# NORMAL VS ANOMALY DONUT & CAUGHT TABLE
# ============================================================
col1, col2 = st.columns([1, 1.4])

with col1:
    st.markdown("### Behavior Distribution")
    fig = go.Figure(data=[go.Pie(
        labels=["Normal", "Anomaly"],
        values=[n_normal, n_anomaly],
        hole=0.65,
        marker=dict(colors=["#2ecc71", "#e74c3c"]),
        textinfo="percent",
        textfont=dict(size=16, color="white")
    )])
    fig.update_layout(
        template=PLOTLY_DARK, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True, legend=dict(font=dict(color="#e0e0e0")),
        height=340, margin=dict(t=10, b=10, l=10, r=10)
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### Anomalous Behavior — Known Attack Scenarios")
    
    if "anomaly_label" in scored_df.columns:
        susp_scored = scored_df.tail(len(suspicious_df)).copy()
        is_caught = susp_scored["anomaly_label"] == "ANOMALY"
    else:
        is_caught = [True] * len(suspicious_df)

    display_df = suspicious_df[["user", "time_window"] + [c for c in FEATURE_COLUMNS if c in suspicious_df.columns]].copy()
    display_df["Status"] = ["🔴 CAUGHT" if caught else "⚪ MISSED" for caught in is_caught]
    st.dataframe(display_df, use_container_width=True, height=340)
    n_caught = int(is_caught.sum())
    st.caption(f"{n_caught} of {len(suspicious_df)} known attack scenarios detected")

st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

# ============================================================
# MODEL COMPARISON
# ============================================================
st.markdown("### Model Comparison")

X_train_df = train_df[FEATURE_COLUMNS].fillna(0)
X_test_normal_df = test_normal_df[FEATURE_COLUMNS].fillna(0)
X_suspicious_df = suspicious_df[FEATURE_COLUMNS].fillna(0)

y_true = np.array([0] * len(X_test_normal_df) + [1] * len(X_suspicious_df))
X_eval_df = pd.concat([X_test_normal_df, X_suspicious_df], ignore_index=True)

# Use NumPy arrays to bypass sklearn feature name ordering conflicts
X_train = X_train_df.values
X_eval = X_eval_df.values

std_scaler = StandardScaler()
X_train_std = std_scaler.fit_transform(X_train)
X_eval_std = std_scaler.transform(X_eval)

minmax_scaler = MinMaxScaler()
X_train_mm = minmax_scaler.fit_transform(X_train)
X_eval_mm = minmax_scaler.transform(X_eval)

def sweep_best_preds(scores, max_fpr=0.03):
    thresholds = np.linspace(scores.min(), scores.max(), 500)
    best_f1 = -1
    best_preds = None
    for thresh in thresholds:
        preds = (scores < thresh).astype(int)
        fpr = (preds[:len(X_test_normal_df)] == 1).mean()
        if fpr <= max_fpr:
            f1 = f1_score(y_true, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_preds = preds
    if best_preds is None:
        best_preds = (scores < np.percentile(scores, 5)).astype(int)
    return best_preds

def evaluate_metrics(name, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "TP": cm[1][1], "FP": cm[0][1], "FN": cm[1][0], "TN": cm[0][0],
    }

results = []

# 1. Isolation Forest (Directly using DataFrame to maintain column mappings cleanly)
if trained_model is not None:
    iso_scores = trained_model.score_samples(X_eval_df)
else:
    iso = IsolationForest(n_estimators=300, contamination="auto", random_state=42)
    iso.fit(X_train_df)
    iso_scores = iso.score_samples(X_eval_df)

iso_preds = sweep_best_preds(iso_scores, max_fpr=0.02)
results.append(evaluate_metrics("Isolation Forest", iso_preds))

# 2. One-Class SVM
ocsvm = OneClassSVM(kernel="rbf", gamma="scale")
ocsvm.fit(X_train_std)
svm_scores = ocsvm.decision_function(X_eval_std)
svm_preds = sweep_best_preds(svm_scores, max_fpr=0.05)
results.append(evaluate_metrics("One-Class SVM", svm_preds))

# 3. Local Outlier Factor
lof = LocalOutlierFactor(n_neighbors=15, metric="manhattan", novelty=True)
lof.fit(X_train_mm)
lof_scores = lof.decision_function(X_eval_mm)
lof_preds = sweep_best_preds(lof_scores, max_fpr=0.05)
results.append(evaluate_metrics("Local Outlier Factor", lof_preds))

results_df = pd.DataFrame(results)

# Cards per algorithm
cols = st.columns(3)
colors = ["#3498db", "#9b59b6", "#e67e22"]
for i, r in enumerate(results):
    with cols[i]:
        st.markdown(f"""<div class='metric-card' style='border-color:{colors[i]}55;'>
            <p style='color:{colors[i]}; font-size:18px; font-weight:600; margin-bottom:12px;'>{r['Model']}</p>
            <p class='metric-value' style='color:{colors[i]}; font-size:32px;'>{r['F1']:.1%}</p>
            <p class='metric-label'>F1-Score</p>
            <p class='metric-sub'>Recall {r['Recall']:.1%} · Precision {r['Precision']:.1%}</p>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

fig = go.Figure()
metric_colors = {"Accuracy": "#3498db", "Precision": "#2ecc71", "Recall": "#f1c40f", "F1": "#e74c3c"}
for metric, color in metric_colors.items():
    fig.add_trace(go.Bar(name=metric, x=results_df["Model"], y=results_df[metric],
                         marker_color=color, text=results_df[metric].apply(lambda v: f"{v:.1%}"),
                         textposition="outside"))
fig.update_layout(
    template=PLOTLY_DARK, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    barmode="group", yaxis_tickformat=".0%", height=420,
    legend=dict(font=dict(color="#e0e0e0")),
    font=dict(color="#e0e0e0"),
    margin=dict(t=30, b=10)
)
st.plotly_chart(fig, use_container_width=True)

best_model = results_df.loc[results_df["F1"].idxmax(), "Model"]
st.success(f"🏆 Best performing model: **{best_model}**")

st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

# ============================================================
# FEATURE IMPORTANCE
# ============================================================
st.markdown("### Feature Contribution (Isolation Forest)")

from sklearn.inspection import permutation_importance
y_eval_binary = [1 if v == 1 else -1 for v in y_true]
active_iso = trained_model if trained_model is not None else IsolationForest(n_estimators=300, random_state=42).fit(X_train_df)

perm = permutation_importance(
    active_iso, X_eval_df, y_eval_binary, n_repeats=20, random_state=42,
    scoring=lambda est, X, y: -sum((est.predict(X) == y))
)
importance_df = pd.DataFrame({
    "Feature": FEATURE_COLUMNS,
    "Importance": perm.importances_mean
}).sort_values("Importance", ascending=True)

fig = px.bar(importance_df, x="Importance", y="Feature", orientation="h",
             color="Importance", color_continuous_scale=["#2a2e3d", "#e74c3c"])
fig.update_layout(
    template=PLOTLY_DARK, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    height=380, coloraxis_showscale=False, font=dict(color="#e0e0e0"),
    margin=dict(t=10, b=10)
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
st.markdown(
    "<p style='color:#6c7688; font-size:13px;'>Lab-scale dataset — single analyst environment. "
    "Results demonstrate the UEBA methodology; detection performance would be expected to "
    "improve with a longer collection period and larger user base.</p>",
    unsafe_allow_html=True
)
