import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("univariateAnalysis/input.csv")

# Preview data
print("Data:\n", df.head())

# ===============================
# 1. Numerical vs Numerical (age vs income)
# ===============================
print("\nCorrelation between age and income:", df['age'].corr(df['income']))

plt.figure(figsize=(6, 5))
sns.scatterplot(data=df, x='age', y='income', hue='gender', palette='Set1')
plt.title('Age vs Income (Colored by Gender)')
plt.xlabel('Age')
plt.ylabel('Income')
plt.grid(True)
plt.tight_layout()
plt.show()

# ===============================
# 2. Categorical vs Numerical (gender vs income)
# ===============================
plt.figure(figsize=(6, 5))
sns.boxplot(data=df, x='gender', y='income', palette='pastel')
plt.title('Income Distribution by Gender')
plt.xlabel('Gender')
plt.ylabel('Income')
plt.tight_layout()
plt.show()

# ===============================
# 3. Categorical vs Categorical (gender vs city)
# ===============================
cross_tab = pd.crosstab(df['city'], df['gender'])
print("\nGender count by City:\n", cross_tab)

# Stacked bar chart
cross_tab.plot(kind='bar', stacked=True, colormap='Accent', figsize=(8, 5))
plt.title('Gender Distribution by City')
plt.xlabel('City')
plt.ylabel('Count')
plt.tight_layout()
plt.show()
