"""
UEBA Data Cleaning Script
Run this AFTER extract_features.py and AFTER reviewing eda.py output.
Produces two clean outputs:
  - ueba_features_clean.csv      -> real users only, for TRAINING
  - suspicious_test_cases.csv    -> invalid-user login attempts, for EVALUATION only (never train on these)
"""
import pandas as pd

INPUT_CSV = "ueba_features.csv"
TRAIN_OUTPUT = "ueba_features_clean.csv"
TEST_OUTPUT = "suspicious_test_cases.csv"

# Pure noise - not real behavioral entities at all
NOISE_ACCOUNTS = [
    "NT AUTHORITY\\SYSTEM",
    "NT AUTHORITY\\LOCAL SERVICE",
    "NT AUTHORITY\\NETWORK SERVICE",
    "Window Manager\\DWM-1",
    "gdm",
    "unset",
    "unknown",
]

df = pd.read_csv(INPUT_CSV, parse_dates=["time_window"])

before = len(df)
print(f"Starting rows: {before}")

# Step 1: remove pure noise accounts
df = df[~df["user"].isin(NOISE_ACCOUNTS)]
print(f"After removing noise accounts: {len(df)}")

# Step 2: separate "invalid user" attempts into their own evaluation set
# these represent failed logins with non-existent usernames - inherently
# suspicious events, not a real user with a behavioral baseline to learn.
KNOWN_ATTACK_ACCOUNTS = ["testvictim", "WIN-T9PH3OLG25E\\testvictim"]

invalid_user_mask = (
    df["user"].str.startswith("invalid user", na=False)
    | df["user"].isin(KNOWN_ATTACK_ACCOUNTS)
)
suspicious_df = df[invalid_user_mask].copy()
suspicious_df["label"] = "suspicious"  # manual label for evaluation later

train_df = df[~invalid_user_mask].copy()
print(f"Real user rows (for training): {len(train_df)}")
print(f"Invalid-user rows (for evaluation only): {len(suspicious_df)}")

# Step 3: drop redundant feature (failed_success_ratio duplicates failed_login
# in this dataset, since network_connection is near-zero for these Linux-side events)
if "failed_login" in train_df.columns and "failed_success_ratio" in train_df.columns:
    corr = train_df["failed_login"].corr(train_df["failed_success_ratio"])
    print(f"\nCorrelation between failed_login and failed_success_ratio: {corr:.3f}")
    if corr > 0.95 or pd.isna(corr):
        print("-> Dropping failed_success_ratio (redundant / no additional signal)")
        train_df = train_df.drop(columns=["failed_success_ratio"])
        suspicious_df = suspicious_df.drop(columns=["failed_success_ratio"], errors="ignore")

# Step 4: basic sanity checks (part of proper cleaning, not just filtering)
print(f"\nDuplicate rows in training set: {train_df.duplicated().sum()}")
train_df = train_df.drop_duplicates()

print(f"Negative value check (should all be 0):")
numeric_cols = train_df.select_dtypes(include="number").columns
print((train_df[numeric_cols] < 0).sum())

train_df.to_csv(TRAIN_OUTPUT, index=False)
suspicious_df.to_csv(TEST_OUTPUT, index=False)

print(f"\nSaved clean training data to {TRAIN_OUTPUT} ({len(train_df)} rows)")
print(f"Saved suspicious test cases to {TEST_OUTPUT} ({len(suspicious_df)} rows)")
