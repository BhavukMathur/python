import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("univariateAnalysis/input.csv")

# Display data preview
print("Data:\n", df.head())

# Segmented descriptive statistics
print("\nSegmented Income Stats by Gender:\n", df.groupby("gender")["income"].describe())

# Histogram of Income segmented by Gender
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="income", hue="gender", bins=5, kde=True, palette="pastel", multiple="stack")
plt.title("Income Distribution by Gender")
plt.xlabel("Income")
plt.ylabel("Count")
plt.grid(True)
plt.tight_layout()
plt.show()

# Boxplot of Income segmented by Gender
plt.figure(figsize=(6, 5))
sns.boxplot(data=df, x="gender", y="income", palette="Set2")
plt.title("Income Distribution by Gender (Boxplot)")
plt.xlabel("Gender")
plt.ylabel("Income")
plt.tight_layout()
plt.show()
