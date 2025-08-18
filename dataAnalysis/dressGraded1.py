
import pandas as pd

# Load the dataset
df = pd.read_csv('dataAnalysis/attributeDataSet.csv')

# Normalize the Size column
size_mapping = {
    'M': 'Medium',
    'L': 'Large',
    'XL': 'Extra large',
    'Free': 'Free',
    'S': 'Small',
    's': 'Small',
    'small': 'Small'
}

# Apply mapping
df['Size'] = df['Size'].map(lambda x: size_mapping.get(str(x).strip(), x))

# Calculate size distribution
size_distribution = df['Size'].value_counts(normalize=True) * 100

# Get required percentages
lowest_percentage = size_distribution.min()
highest_percentage = size_distribution.max()
small_percentage = size_distribution.get('Small', 0)

print(f"Lowest percentage: {lowest_percentage:.2f}%")
print(f"Highest percentage: {highest_percentage:.2f}%")
print(f"Percentage of Small size: {small_percentage:.2f}%")
