import pandas as pd

# Load the CSV
df = pd.read_csv("pose_errors.csv")

all = df["Translational_Error_m"]

avg_rot_error_rest = df["Translational_Error_m"].mean()

print(f"Average rotational error for the rest (≤ 30°): {avg_rot_error_rest:.2f}°")