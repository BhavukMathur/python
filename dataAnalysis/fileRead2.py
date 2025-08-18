import pandas as pd

# Read the CSV file
input_file = 'dataAnalysis/input.csv'
df = pd.read_csv(input_file)

# Deduplicate rows — remove both fully identical and partially duplicated rows (based on name, age, city)
df = df.drop_duplicates()                          # Remove fully duplicated rows
df = df.drop_duplicates(subset=['name', 'city'])   # Remove rows with same name and city

# Filter rows — by segment and date period
filtered_df = df[
    (df['segment'] == 'A') &
    (df['date'] >= '2024-01-01') &
    (df['date'] <= '2024-12-31')
]

# Filter columns — keep only relevant columns for analysis
filtered_df = filtered_df[['name', 'city', 'segment', 'value']]

# Aggregate data — group by city and segment, and sum the value
aggregated_df = filtered_df.groupby(['city', 'segment']).agg({'value': 'sum'}).reset_index()

# Output final result
print("Aggregated Data:")
print(aggregated_df)
