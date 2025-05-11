import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
print("Loading data...")
df = pd.read_csv('Segmentation_des_utilisateurs.csv')

# 1. Analyze numerical variables distribution before preprocessing
numerical_cols = ['Questions_Postees', 'Reponses_Fournies', 'Upvotes_Recus', 'Score_Engagement']

# Create distribution plots before standardization
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(2, 2, i)
    sns.histplot(data=df, x=col, bins=30)
    plt.title(f'Distribution of {col} (Before Standardization)')
plt.tight_layout()
plt.savefig('distributions_before_standardization.png')
plt.close()

# 2. Standardization
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Create distribution plots after standardization
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(2, 2, i)
    sns.histplot(data=df_scaled, x=col, bins=30)
    plt.title(f'Distribution of {col} (After Standardization)')
plt.tight_layout()
plt.savefig('distributions_after_standardization.png')
plt.close()

# 3. Analyze Tags_Expertise
# Convert string tags to list
df['Tags_List'] = df['Tags_Expertise'].str.split(',')

# Use MultiLabelBinarizer for one-hot encoding
mlb = MultiLabelBinarizer()
tags_encoded = pd.DataFrame(mlb.fit_transform(df['Tags_List']),
                          columns=mlb.classes_,
                          index=df.index)

# Visualize tag distribution
plt.figure(figsize=(12, 6))
tag_counts = tags_encoded.sum().sort_values(ascending=False)
sns.barplot(x=tag_counts.values, y=tag_counts.index)
plt.title('Distribution of Expertise Tags')
plt.xlabel('Count')
plt.tight_layout()
plt.savefig('tags_distribution.png')
plt.close()

# 4. Correlation Analysis
# Combine numerical features
correlation_df = df[numerical_cols]
correlation_matrix = correlation_df.corr()

# Create correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Numerical Features')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

# Print summary statistics
print("\nSummary Statistics of Original Data:")
print(df[numerical_cols].describe())

print("\nSummary Statistics after Standardization:")
print(df_scaled[numerical_cols].describe())

# Save preprocessed data
preprocessed_data = pd.concat([df_scaled[numerical_cols], tags_encoded], axis=1)
preprocessed_data['Cluster'] = df['Cluster']
preprocessed_data.to_csv('preprocessed_data.csv', index=False)

print("\nPreprocessing completed. Visualizations saved as PNG files.") 