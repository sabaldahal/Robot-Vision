import pandas as pd

# Load the CSV
df = pd.read_csv("pose_errors.csv")

# Count how many are above 30 degrees
above_30 = df[df["Rotational_Error_deg"] > 30]
below_or_equal_30 = df[df["Rotational_Error_deg"] <= 30]

count_above_30 = len(above_30)
avg_rot_error_rest = below_or_equal_30["Rotational_Error_deg"].mean()

print(f"Number of samples with rotational error > 30°: {count_above_30}")
print(f"Average rotational error for the rest (≤ 30°): {avg_rot_error_rest:.2f}°")

# Optional: save separated rows to new CSVs
above_30.to_csv("pose_errors_above30.csv", index=False)
below_or_equal_30.to_csv("pose_errors_below_or_equal_30.csv", index=False)
