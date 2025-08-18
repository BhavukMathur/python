import pandas as pd

# Step 1: Long format data
df_long = pd.DataFrame({
    'ID': [1, 1, 2, 2],
    'Date': ['2024-01-01', '2024-01-01', '2024-01-01', '2024-01-01'],
    'Variable': ['Height', 'Weight', 'Height', 'Weight'],
    'Value': [180, 75, 165, 60]
})

print("Step 1: Original Long Format:")
print(df_long)

# Step 2: Convert Long → Wide using pivot
df_wide = df_long.pivot(index=['ID', 'Date'], columns='Variable', values='Value').reset_index()

print("\nStep 2: Converted to Wide Format:")
print(df_wide)

# Step 3: Convert Wide → Long using melt
df_long_back = pd.melt(df_wide, id_vars=['ID', 'Date'], 
                       value_vars=['Height', 'Weight'],
                       var_name='Variable', value_name='Value')

print("\nStep 3: Converted Back to Long Format:")
print(df_long_back)
