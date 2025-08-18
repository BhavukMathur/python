# Define function to remove illegal votes and return valid data set
import pandas as pd

def remove_illegal_votes(df):
    # Step 1: Remove votes where both candidates are selected
    df = df[~((df['candidate_A'] == 1) & (df['candidate_B'] == 1))]

    # Step 2: Group by voter_id and filter out those who voted more than once
    def is_illegal(group):
        has_voted = ((group['candidate_A'] + group['candidate_B']) > 0).any()
        return len(group) > 1 and has_voted

    illegal_voters = df.groupby('voter_id').filter(is_illegal)['voter_id'].unique()

    # Step 3: Remove illegal voters' rows
    df = df[~df['voter_id'].isin(illegal_voters)]

    # Step 4: Drop duplicates for valid (0,0) voters keeping only one row
    df = df.drop_duplicates(subset='voter_id', keep='first')

    return df
    
      
    return deduped_df

# Input and output processing (do not edit)
from ast import literal_eval
import pandas as pd
filename = 'https://d3ejq4mxgimsmf.cloudfront.net/votes-5d1c9ce4560d4bcbace2742f26ca9c24.csv'
df = pd.read_csv(filename)
x, y = literal_eval(input())
print(int(remove_illegal_votes(df).iloc[x, y]))