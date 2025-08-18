import pandas as pd

# Read the CSV file
input_file = 'dataAnalysis/input.csv'
df = pd.read_csv(input_file) #For excel, pd.read_excel, json: read_json

# Print the contents (optional)
print("Contents of input.csv:")
print(df)

# Write the DataFrame to a new CSV file
output_file = 'dataAnalysis/output.csv'
df.to_csv(output_file, index=False)  # index=False avoids writing row numbers

print(f"\nData written to {output_file}")
