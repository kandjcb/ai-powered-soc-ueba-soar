"""
UEBA Exploratory Data Analysis (EDA)
Run this BEFORE training any model - understand the data first.
"""
import pandas as pd

INPUT_CSV = "ueba_features.csv"

# Comprehensive list of system noise accounts & Windows security groups to exclude
SYSTEM_ACCOUNTS_TO_EXCLUDE = [
    "NT AUTHORITY\\SYSTEM",
    "NT AUTHORITY\\LOCAL SERVICE",
    "NT AUTHORITY\\NETWORK SERVICE",
    "Window Manager\\DWM-1",
    "SYSTEM",
    "LOCAL SERVICE",
    "NETWORK SERVICE",
    "DWM-1",
    "UMFD-0",
    "UMFD-1",
    "gdm",
    "unset",
    "unknown",
    "?",
    "(invalid user)",
    # Windows Security Groups (not actual user accounts)
    "Administrators",
    "Backup Operators",
    "Users",
    "Guests",
    "Remote Management Users",
    "Access Control Assistance Operators",
    "Certificate Service DCOM Access",
    "Cryptographic Operators",
    "Device Owners",
    "Distributed COM Users",
    "Event Log Readers",
    "Hyper-V Administrators",
    "IIS_IUSRS",
    "Network Configuration Operators",
    "Performance Log Users",
    "Performance Monitor Users",
    "Power Users",
    "Print Operators",
    "RDS Endpoint Servers",
    "RDS Management Servers",
    "RDS Remote Access Servers",
    "Remote Desktop Users",
    "Replicator",
    "Storage Replica Administrators",
    "System Managed Accounts Group"
]

df = pd.read_csv(INPUT_CSV, parse_dates=["time_window"])

possible_features = [
    # Full names from feature extractor
    "failed_login_count",
    "successful_login_count",
    "network_connection_count",
    "process_creation_count",
    "total_events",
    "is_off_hours",
    "day_of_week",
    "hour_of_day",
    # Short names fallback
    "failed_login",
    "network_connection",
    "process_creation",
    "command_execution",
    "off_hours_ratio",
    "novel_event_count",
    "orphan_process_count",
    "suspicious_cmd_count",
    "unique_processes_count",
    "failed_success_ratio"
]

# Dynamically pick available numeric feature columns from your CSV
FEATURE_COLUMNS = [col for col in possible_features if col in df.columns]

print("=" * 60)
print("1. BASIC SHAPE")
print("=" * 60)
print(f"Total rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")
print(f"Date range: {df['time_window'].min()} to {df['time_window'].max()}")

print("\n" + "=" * 60)
print("2. USERS IN DATASET (before filtering)")
print("=" * 60)
print(df["user"].value_counts())

# Filter system accounts
df_filtered = df[~df["user"].isin(SYSTEM_ACCOUNTS_TO_EXCLUDE)].copy()

print("\n" + "=" * 60)
print("3. USERS AFTER FILTERING SYSTEM ACCOUNTS")
print("=" * 60)
print(f"Rows before: {len(df)}, rows after: {len(df_filtered)}")
print(df_filtered["user"].value_counts())

print("\n" + "=" * 60)
print("4. FEATURE STATISTICS (real users only)")
print("=" * 60)
print(df_filtered[FEATURE_COLUMNS].describe().round(2).T[["min", "mean", "50%", "max"]])

print("\n" + "=" * 60)
print("5. MISSING / NULL VALUES CHECK")
print("=" * 60)
print(df_filtered[FEATURE_COLUMNS].isnull().sum())

print("\n" + "=" * 60)
print("6. FEATURE CORRELATIONS (real users only)")
print("=" * 60)
print(df_filtered[FEATURE_COLUMNS].corr().round(2))

print("\n" + "=" * 60)
print("7. PER-USER FEATURE AVERAGES")
print("=" * 60)
print(df_filtered.groupby("user")[FEATURE_COLUMNS].mean().round(2))

print("\n" + "=" * 60)
print("8. OUTLIER CHECK - rows with extreme values (top 1% by count features)")
print("=" * 60)
for col in ["failed_login_count", "network_connection_count", "process_creation_count", "total_events"]:
    if col in df_filtered.columns:
        threshold = df_filtered[col].quantile(0.99)
        outliers = df_filtered[df_filtered[col] > threshold]
        if not outliers.empty:
            print(f"\nTop outliers for '{col}' (> {threshold:.1f}):")
            print(outliers[["user", "time_window", col]].to_string(index=False))

print("\n" + "=" * 60)
print("9. ROWS PER USER PER DAY (checking activity density)")
print("=" * 60)
df_filtered["day"] = df_filtered["time_window"].dt.date
print(df_filtered.groupby(["user", "day"]).size().reset_index(name="rows_that_day"))
