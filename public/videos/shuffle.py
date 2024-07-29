import pandas as pd
import random

# Load the CSV file
file_path = 'mapping.csv'
df = pd.read_csv(file_path)

# Separate the first two rows
first_two_rows = df.iloc[:2, :]
remaining_rows = df.iloc[2:, :]

# Shuffle the remaining rows
remaining_rows = remaining_rows.sample(frac=1).reset_index(drop=True)

# Concatenate the first two rows with the shuffled remaining rows
result_df = pd.concat([first_two_rows, remaining_rows], ignore_index=True)

# Save the result to a new CSV file
result_df.to_csv(file_path, index=False)

print(f"Rows have been shuffled and saved to '{file_path}'")