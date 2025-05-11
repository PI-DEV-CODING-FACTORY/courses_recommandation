import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Set random seed for reproducibility
np.random.seed(42)

# Load the data
def load_data(file_path):
    df = pd.read_csv(file_path)
    print("Dataset shape:", df.shape)
    return df

# Preprocess the data
def preprocess_data(df):
    # Create a copy of the dataframe
    df_processed = df.copy()
    
    # Convert Tags_Expertise to numeric features using one-hot encoding
    tags = df_processed['Tags_Expertise'].str.get_dummies(', ')
    
    # Drop the original Tags_Expertise column and combine with numeric features
    df_processed = df_processed.drop('Tags_Expertise', axis=1)
    df_processed = pd.concat([df_processed, tags], axis=1)
    
    # Store the true labels (if available)
    true_labels = None
    if 'Cluster' in df_processed.columns:
        true_labels = df_processed['Cluster']
        df_processed = df_processed.drop('Cluster', axis=1)
    
    # Remove Étudiant_ID as it's not relevant for clustering
    df_processed = df_processed.drop('Étudiant_ID', axis=1)
    
    # Scale the features
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_processed), columns=df_processed.columns)
    
    return df_scaled, true_labels, tags.columns

# Perform clustering
def perform_clustering(data, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(data)
    return kmeans, clusters

# Calculate evaluation metrics
def calculate_metrics(true_labels, predicted_labels):
    # Ensure labels are aligned (clustering may assign different cluster numbers)
    # We'll use Hungarian algorithm to find the best mapping
    from scipy.optimize import linear_sum_assignment
    
    # Create confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    
    # Find optimal label mapping
    row_ind, col_ind = linear_sum_assignment(-cm)
    
    # Remap predicted labels
    mapped_labels = np.zeros_like(predicted_labels)
    for i, j in zip(row_ind, col_ind):
        mapped_labels[predicted_labels == j] = i
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, mapped_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, mapped_labels, average='weighted')
    
    # Calculate specificity for each class
    specificities = []
    for class_label in np.unique(true_labels):
        true_neg = np.sum((true_labels != class_label) & (mapped_labels != class_label))
        false_pos = np.sum((true_labels != class_label) & (mapped_labels == class_label))
        specificity = true_neg / (true_neg + false_pos) if (true_neg + false_pos) > 0 else 0
        specificities.append(specificity)
    
    mean_specificity = np.mean(specificities)
    
    # Create metrics dictionary
    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall (Sensibilité)': recall,
        'Specificity (Spécificité)': mean_specificity,
        'F-measure': f1
    }
    
    return metrics

# Visualize results
def visualize_clusters(data, clusters, true_labels=None):
    # Reduce dimensionality for visualization
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot predicted clusters
    plt.subplot(1, 2, 1)
    plt.scatter(data_2d[:, 0], data_2d[:, 1], c=clusters, cmap='viridis')
    plt.title('Predicted Clusters')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    
    # Plot true labels if available
    if true_labels is not None:
        plt.subplot(1, 2, 2)
        plt.scatter(data_2d[:, 0], data_2d[:, 1], c=true_labels, cmap='viridis')
        plt.title('True Labels')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
    
    plt.tight_layout()
    plt.savefig('cluster_visualization.png')
    plt.close()

# Analyze cluster characteristics
def analyze_clusters(original_data, clusters, tag_columns):
    df_analysis = original_data.copy()
    df_analysis['Predicted_Cluster'] = clusters
    
    # Select only numeric columns for mean calculation
    numeric_columns = ['Questions_Postees', 'Reponses_Fournies', 'Upvotes_Recus', 'Score_Engagement']
    
    # Calculate mean values for numeric features
    cluster_means = df_analysis.groupby('Predicted_Cluster')[numeric_columns].mean()
    
    # Calculate most common expertise tags per cluster
    expertise_by_cluster = {}
    for cluster in df_analysis['Predicted_Cluster'].unique():
        cluster_tags = df_analysis[df_analysis['Predicted_Cluster'] == cluster]['Tags_Expertise'].value_counts()
        expertise_by_cluster[f'Cluster {cluster}'] = cluster_tags.head(3)

    print("\nCluster Characteristics:")
    print("\nNumeric Features (Means):")
    print(cluster_means)
    
    print("\nMost Common Expertise Tags by Cluster:")
    for cluster, tags in expertise_by_cluster.items():
        print(f"\n{cluster}:")
        print(tags)
    
    # Create heatmap for numeric features
    plt.figure(figsize=(10, 6))
    sns.heatmap(cluster_means, annot=True, cmap='YlOrRd', fmt='.2f')
    plt.title('Cluster Characteristics Heatmap (Numeric Features)')
    plt.tight_layout()
    plt.savefig('cluster_characteristics.png')
    plt.close()
    
    # Create bar plot for cluster sizes
    plt.figure(figsize=(8, 5))
    df_analysis['Predicted_Cluster'].value_counts().plot(kind='bar')
    plt.title('Cluster Sizes')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Users')
    plt.tight_layout()
    plt.savefig('cluster_sizes.png')
    plt.close()

def main():
    # Load data
    print("Loading data...")
    df = load_data('Segmentation_des_utilisateurs.csv')
    
    # Preprocess data
    print("\nPreprocessing data...")
    df_scaled, true_labels, tag_columns = preprocess_data(df)
    
    # Perform clustering
    print("\nPerforming clustering...")
    kmeans, clusters = perform_clustering(df_scaled)
    
    # Calculate and display evaluation metrics
    if true_labels is not None:
        print("\nEvaluation Metrics:")
        metrics = calculate_metrics(true_labels, clusters)
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")
    
    # Visualize results
    print("\nVisualizing results...")
    visualize_clusters(df_scaled, clusters, true_labels)
    
    # Analyze clusters
    print("\nAnalyzing cluster characteristics...")
    analyze_clusters(df, clusters, tag_columns)
    
    print("\nAnalysis complete! Check the following files for visualizations:")
    print("1. cluster_visualization.png - PCA visualization of clusters")
    print("2. cluster_characteristics.png - Heatmap of numeric features")
    print("3. cluster_sizes.png - Distribution of users across clusters")

if __name__ == "__main__":
    main() 