import pandas as pd
filename = 'https://d3ejq4mxgimsmf.cloudfront.net/customer_risk_dataset-bd8995d80cf648b1b797318bd0b802a3.csv'
df = pd.read_csv(filename)


def classify_risk(df):
    def get_risk(row):
        # Check for High Risk
        if (row['Occupation'] == 'Unemployed' and row['Education_Level'] == 'High School') or \
           (row['Account_Type'] == 'Current' and row['Marital_Status'] == 'Divorced'):
            return 'High'
        
        # Check for Low Risk
        if (row['Occupation'] in ['Engineer', 'Doctor', 'Scientist', 'Lawyer'] and row['Education_Level'] == 'Postgraduate') or \
           (row['Account_Type'] == 'Fixed Deposit' and row['Marital_Status'] == 'Married'):
            return 'Low'
        
        # Else Medium Risk
        return 'Medium'

    df['Risk_Category'] = df.apply(get_risk, axis=1)
    return df

# Processing input and output (do not edit)
print(classify_risk(df).loc[int(input()), 'Risk_Category'])