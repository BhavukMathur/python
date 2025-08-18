import pandas as pd

# Read the CSV file
input_file = 'dataAnalysis/input.csv'
df = pd.read_csv(input_file)

# Drop rows with missing values (to ensure clean calculations)
df = df.dropna()

# Convert 'age' column to numeric (in case of parsing issues)
df['age'] = pd.to_numeric(df['age'])

# Calculate IQR
Q1 = df['age'].quantile(0.25)
Q3 = df['age'].quantile(0.75)
IQR = Q3 - Q1

# Define bounds for non-outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter out rows where age is outside the bounds
df_no_outliers = df[(df['age'] >= lower_bound) & (df['age'] <= upper_bound)]

# Print the filtered DataFrame
print(df_no_outliers)

# Optionally, write the result to a CSV
df_no_outliers.to_csv('no_outliers.csv', index=False)
