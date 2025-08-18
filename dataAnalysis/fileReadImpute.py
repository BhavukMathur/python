import pandas as pd

# Read the CSV file
input_file = 'dataAnalysis/input.csv'
df = pd.read_csv(input_file)

# Drop rows with any missing (NaN) values
df_cleaned = df.dropna()

# Print the cleaned data
print(df_cleaned)

# Optionally, write the cleaned data to a new CSV file
df_cleaned.to_csv('cleaned_output.csv', index=False)
