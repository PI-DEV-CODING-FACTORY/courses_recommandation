import pandas as pd

# Load the data
df = pd.read_csv('Segmentation_des_utilisateurs.csv')

# Print column names
print("\nColumn names:")
print(df.columns.tolist())

# Print unique values in the target column (cluster labels)
print("\nUnique cluster labels:")
print(sorted(df['Cluster'].unique()))

# Print basic info about the dataset
print("\nDataset info:")
df.info() 