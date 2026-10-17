import pandas as pd

# Load the existing CSV
df = pd.read_csv("ueba_features.csv", parse_dates=["time_window"])

# --- Time-based features ---
df["hour_of_day"] = df["time_window"].dt.hour
df["day_of_week"] = df["time_window"].dt.dayofweek  # 0=Monday, 6=Sunday
df["off_hours_flag"] = ((df["hour_of_day"] < 7) | (df["hour_of_day"] > 20)).astype(int)

# --- Ratio features ---
df["failed_success_ratio"] = df["failed_login"] / (df["network_connection"] + 1)

# Save back into the SAME file, overwriting it with the new columns added
df.to_csv("ueba_features.csv", index=False)

print("Done. New columns added directly to ueba_features.csv:")
print(df[["user", "time_window", "hour_of_day", "day_of_week", "off_hours_flag", "failed_success_ratio"]].head(10))
