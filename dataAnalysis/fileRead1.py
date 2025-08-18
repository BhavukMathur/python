import pandas as pd

# Read the CSV file
input_file = 'dataAnalysis/input.csv'
df = pd.read_csv(input_file)

# Store columns in separate variables
names = df['name'].tolist()
ages = df['age'].tolist()
cities = df['city'].tolist()

# Print the values
print("Names:", names)
print("Ages:", ages)
print("Cities:", cities)
