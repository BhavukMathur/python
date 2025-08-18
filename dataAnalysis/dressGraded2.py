#Import the required Libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

## Data Cleaning

### Data Reading & Data Types

#Read the data in pandas
inp0 = pd.read_csv('dataAnalysis/attributeDataSet.csv')
inp1 = pd.read_csv('dataAnalysis/dressSales.csv')

# Print the information about the attributes of inp0 and inp1.
print("Attribute Dataset Info:")
print(inp0.info())
print("\nDress Sales Dataset Info:")
print(inp1.info())

### Fixing the Rows and Columns

# Column fixing, correcting size abbreviation. count the percentage of each size category in "Size" column.
size_mapping = {
    'M': 'Medium',
    'L': 'Large',
    'XL': 'Extra large',
    'free': 'Free',
    'S': 'Small',
    's': 'Small',
    'small': 'Small'
}

inp0['Size'] = inp0['Size'].map(size_mapping).fillna(inp0['Size'])

# Print the value counts of each category in "Size" column.
size_counts = inp0['Size'].value_counts()
size_percentages = (size_counts / len(inp0)) * 100
print("Size category percentages:")
print(size_percentages)

### Impute/Remove Missing values

# Print the null count of each variables of inp0 and inp1.
print("Null counts in Attribute Dataset:")
print(inp0.isnull().sum())
print("\nNull counts in Dress Sales Dataset:")
print(inp1.isnull().sum())

# Print the data types information of inp1 i.e. "Dress Sales" data.
print("\nDress Sales data types:")
print(inp1.dtypes)

# Try to convert the object type into float type of data. YOU GET ERROR MESSAGE.

# Do the required changes in the "Dress Sales" data set to get null values on string values.
for col in inp1.columns:
    if inp1[col].dtype == 'object':
        inp1[col] = pd.to_numeric(inp1[col], errors='coerce')

# Convert the object type columns in "Dress Sales" into float type of data type.
# This is already done above with pd.to_numeric

# Print the null percetange of each column of inp1.
null_percentages = (inp1.isnull().sum() / len(inp1)) * 100
print("\nNull percentages in Dress Sales:")
print(null_percentages)

# Drop the columns in "Dress Sales" which have more than 40% of missing values.
columns_to_drop = inp1.columns[null_percentages > 40].tolist()
inp1 = inp1.drop(columns=columns_to_drop)
print(f"\nDropped columns with >40% missing values: {columns_to_drop}")

# Create the four seasons columns in inp1, according to the above criteria.
def get_season(date_col):
    month = int(date_col.split('-')[1])
    if month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Autumn'
    elif month in [12, 1, 2]:
        return 'Winter'
    else:
        return 'Spring'

# Create season columns
for col in inp1.columns:
    if col != 'Dress_ID' and '-' in col:
        season = get_season(col)
        if season not in inp1.columns:
            inp1[season] = 0
        inp1[season] += inp1[col].fillna(0)

# calculate the sum of sales in each seasons in inp1 i.e. "Dress Sales".
season_sales = inp1[['Summer', 'Autumn', 'Winter', 'Spring']].sum()
print("Seasonal sales:")
print(season_sales)

# Now let's merge inp1 with inp0 with left join manner, so that the information of inp0 should remain intact.

# Merge inp0 with inp1 into inp0. this is also called left merge.
inp0 = pd.merge(left=inp0, right=inp1, how='left', left_on='Dress_ID', right_on='Dress_ID')

# Now Drop the Date columns from inp0 as it is already combined into four seasons.
date_columns = [col for col in inp0.columns if '-' in col and col != 'Dress_ID']
inp0 = inp0.drop(columns=date_columns)

# Print the null count of each columns in inp0 dataframe i.e. combined data frame of inp0 and inp1 without date columns.
print("Null counts in merged dataset:")
print(inp0.isnull().sum())

# Deal with the missing values of Type-1 columns: Price, Season, NeckLine, SleeveLength, Winter and Autumn.
type1_columns = ['Price', 'Season', 'NeckLine', 'SleeveLength', 'Winter', 'Autumn']
for col in type1_columns:
    if col in inp0.columns:
        inp0[col] = inp0[col].fillna(inp0[col].mode()[0] if inp0[col].dtype == 'object' else inp0[col].median())

# Deal with the missing values for Type-2 columns: Material, FabricType, Decoration and Pattern Type.
type2_columns = ['Material', 'FabricType', 'Decoration', 'Pattern Type']
for col in type2_columns:
    if col in inp0.columns:
        inp0[col] = inp0[col].fillna('Others')

### Standardise value

#correcting the spellings.
spelling_corrections = {
    'fabricType': 'FabricType',
    'patern type': 'Pattern Type',
    'patern': 'Pattern'
}

#correcting the Spellings.
for col in inp0.columns:
    if col in spelling_corrections:
        inp0 = inp0