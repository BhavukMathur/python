# Data Loading (do not edit)
import pandas as pd
df = pd.read_csv('https://d3ejq4mxgimsmf.cloudfront.net/athletes_dataset-bbcc67f2702d42babc76a9d519c9d131.csv')

# Code here
# Group by 'Sport', calculate median age, and filter athletes younger than or equal to the median
young_athletes_by_sport = (
    df[df.groupby("Sport")["Age"].transform("median") >= df["Age"]]
    .groupby("Sport")["Name"]
    .apply(list)
)


# Input/Output Processing (do not edit)
print(young_athletes_by_sport[input()])