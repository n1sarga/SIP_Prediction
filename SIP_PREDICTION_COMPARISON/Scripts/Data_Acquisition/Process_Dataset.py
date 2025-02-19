'''
This code will do the following tasks for processing the dataset collected from the IntAct Database:

1. Trim the garbage portion from the protein identifiers
2. Replace the ticks from the Interaction columns
3. Identify Self-Interacting Proteins (SIPs)
'''
import pandas as pd

df = pd.read_csv('C:/Users/nextn/Downloads/GitHub/SIP_PREDICTION_COMPARISON/Datasets/Yeast/Yeast_All_Interactions.csv') # Dataframe df has three columns: Identifier A, Identifier B, and Interaction [Validated]

# Task 1
df['Identifier A'] = df['Identifier A'].str.replace('UniProt', '')
df['Identifier B'] = df['Identifier B'].str.replace('UniProt', '')

# Task 2
df['Interaction'] = df['Interaction'].apply(lambda x: '1' if x != '1' else x)

# Task 3
sips = df[df['Identifier A'] == df['Identifier B']] # Dataframe sips has 577 rows and 3 columns [Validated]

# Distinct Proteins
dist = sips[['Identifier A']].rename(columns={'Identifier A': 'Proteins Identifier'})
dist['Protein Sequences'] = None

# Dataframe sips will be saved as 'Maize_SIPs.csv'
sips.to_csv('C:/Users/nextn/Downloads/GitHub/SIP_PREDICTION_COMPARISON/Datasets/Yeast/Yeast_SIPs.csv', index=False)
dist.to_csv('C:/Users/nextn/Downloads/GitHub/SIP_PREDICTION_COMPARISON/Datasets/Yeast/Unique_Proteins.csv', index=False)