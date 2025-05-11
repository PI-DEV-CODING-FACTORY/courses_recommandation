import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA

def load_data():
    print("Loading dataset...")
    return pd.read_csv('Segmentation_des_utilisateurs.csv')

def analyze_basic_stats(df):
    print("\n=== Basic Dataset Statistics ===")
    print(f"\nDataset Shape: {df.shape}")
    print("\nColumns in dataset:")
    for col in df.columns:
        print(f"- {col}")
    
    print("\nNumerical Columns Summary:")
    print(df.describe())
    
    print("\nMissing Values:")
    print(df.isnull().sum())

def analyze_expertise_distribution(df):
    print("\n=== Expertise Tags Analysis ===")
    # Split the Tags_Expertise string and count frequencies
    tags = df['Tags_Expertise'].str.split(', ', expand=True).stack()
    tag_counts = tags.value_counts()
    
    print("\nExpertise Tags Distribution:")
    print(tag_counts)
    
    # Visualize tag distribution
    plt.figure(figsize=(12, 6))
    tag_counts.plot(kind='bar')
    plt.title('Distribution of Expertise Tags')
    plt.xlabel('Tags')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('expertise_distribution.png')
    plt.close()

def analyze_cluster_distribution(df):
    print("\n=== Cluster Distribution Analysis ===")
    cluster_counts = df['Cluster'].value_counts()
    print("\nCluster Distribution:")
    print(cluster_counts)
    
    # Visualize cluster distribution
    plt.figure(figsize=(8, 6))
    cluster_counts.plot(kind='pie', autopct='%1.1f%%')
    plt.title('Distribution of Clusters')
    plt.axis('equal')
    plt.savefig('cluster_distribution.png')
    plt.close()

def feature_importance_analysis(df):
    print("\n=== Feature Importance Analysis ===")
    
    # Prepare features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != 'Cluster' and col != 'Étudiant_ID']
    
    X = df[numeric_cols]
    y = df['Cluster']
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Feature importance using SelectKBest
    selector = SelectKBest(score_func=f_classif, k='all')
    selector.fit(X_scaled, y)
    
    # Create feature importance DataFrame
    feature_scores = pd.DataFrame({
        'Feature': numeric_cols,
        'Score': selector.scores_
    })
    feature_scores = feature_scores.sort_values('Score', ascending=False)
    
    print("\nFeature Importance Scores:")
    print(feature_scores)
    
    # Visualize feature importance
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Score', y='Feature', data=feature_scores)
    plt.title('Feature Importance Scores')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    return feature_scores

def pca_analysis(df):
    print("\n=== PCA Analysis ===")
    
    # Prepare features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != 'Cluster' and col != 'Étudiant_ID']
    
    X = df[numeric_cols]
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # Calculate explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    print("\nExplained Variance Ratio by Components:")
    for i, var in enumerate(explained_variance_ratio):
        print(f"PC{i+1}: {var:.4f} ({cumulative_variance_ratio[i]:.4f} cumulative)")
    
    # Visualize explained variance
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(explained_variance_ratio) + 1), cumulative_variance_ratio, 'bo-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('PCA Explained Variance Ratio')
    plt.grid(True)
    plt.savefig('pca_explained_variance.png')
    plt.close()

def analyze_cluster_characteristics(df):
    print("\n=== Cluster Characteristics Analysis ===")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != 'Cluster' and col != 'Étudiant_ID']
    
    # Calculate mean values for each cluster
    cluster_means = df.groupby('Cluster')[numeric_cols].mean()
    print("\nCluster Means:")
    print(cluster_means)
    
    # Visualize cluster characteristics
    plt.figure(figsize=(15, 8))
    cluster_means.T.plot(kind='bar')
    plt.title('Cluster Characteristics - Mean Values')
    plt.xlabel('Features')
    plt.ylabel('Mean Value')
    plt.legend(title='Cluster')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('cluster_characteristics.png')
    plt.close()

def main():
    # Load data
    df = load_data()
    
    # Perform analyses
    analyze_basic_stats(df)
    analyze_expertise_distribution(df)
    analyze_cluster_distribution(df)
    feature_scores = feature_importance_analysis(df)
    pca_analysis(df)
    analyze_cluster_characteristics(df)
    
    # Print key findings
    print("\n=== Key Findings ===")
    print("\nTop 5 Most Important Features:")
    print(feature_scores.head())

if __name__ == "__main__":
    main() 