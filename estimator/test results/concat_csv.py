import pandas as pd


df1 = pd.read_csv('estimator/test results/debug_confidence_1/v1_2026-01-09/file_singleclass_1_2026-01-09_pose_errors.csv')

df2 = pd.read_csv('estimator/test results/debug_confidence_1/v1_2026-01-09/file_multiclass_1_2026-01-09_pose_errors.csv')

combined_df = pd.concat([df1, df2], ignore_index=True)

combined_df.index = combined_df.index + 1
combined_df.to_csv('estimator/test results/debug_confidence_1/v1_2026-01-09/combined_file_3_pose_errors.csv', index_label='ID')