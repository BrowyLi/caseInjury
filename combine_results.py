import os
import pandas as pd

# Path to the folder containing your CSV files
folder_path = 'result'

# The name for the combined CSV file
output_file = 'combined_result.csv'

# List all files in the directory and read them into pandas dataframes
all_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')])
print(all_files)
dfs = [pd.read_csv(f) for f in all_files]

# Concatenate all dataframes into one dataframe
combined_df = pd.concat(dfs, ignore_index=True)

# Write the combined dataframe to a new CSV file
combined_df.to_csv(output_file, index=False)

print(f"All CSV files in {folder_path} have been combined into {output_file}")