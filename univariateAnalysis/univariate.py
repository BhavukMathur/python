import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('univariateAnalysis/input.csv')

# Display first few rows
print("Data Preview:\n", df.head())

# Basic statistics for numerical columns
print("\nDescriptive Statistics:\n", df.describe())

# Frequency distribution for categorical variables
print("\nGender Distribution:\n", df['gender'].value_counts())
print("\nCity Distribution:\n", df['city'].value_counts())

# Plotting - Histogram for Age
plt.figure(figsize=(6, 4))
sns.histplot(df['age'], bins=5, kde=True, color='skyblue')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plotting - Boxplot for Income
plt.figure(figsize=(6, 4))
sns.boxplot(x=df['income'], color='lightgreen')
plt.title('Boxplot of Income')
plt.xlabel('Income')
plt.tight_layout()
plt.show()

# Bar chart for Gender
plt.figure(figsize=(4, 4))
sns.countplot(x='gender', data=df, palette='pastel')
plt.title('Gender Count')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.tight_layout()
plt.show()
