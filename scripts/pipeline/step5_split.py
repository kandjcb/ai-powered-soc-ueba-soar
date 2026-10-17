import pandas as pd

INPUT_CSV = "ueba_features_clean.csv"
df = pd.read_csv(INPUT_CSV, parse_dates=["time_window"])
df = df.sort_values(["user", "time_window"])

train_parts = []
test_parts = []

for user, group in df.groupby("user"):
    n = len(group)
    if n < 2:
        train_parts.append(group)
        continue
    split_idx = int(n * 0.75)
    train_parts.append(group.iloc[:split_idx])
    test_parts.append(group.iloc[split_idx:])

train_df = pd.concat(train_parts).reset_index(drop=True)
test_df = pd.concat(test_parts).reset_index(drop=True)

train_df.to_csv("train_data.csv", index=False)
test_df.to_csv("test_data.csv", index=False)

print(f"Training on {len(train_df)} clean human rows | Testing on {len(test_df)} held-out rows")
