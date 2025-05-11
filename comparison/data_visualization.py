import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    print("Loading dataset...")
    return pd.read_csv('Segmentation_des_utilisateurs.csv')

def create_distribution_plots(df):
    # Set the style for all plots
    sns.set_theme(style="whitegrid")
    
    # 1. Distribution of numerical variables
    numerical_cols = ['Questions_Postees', 'Reponses_Fournies', 'Upvotes_Recus', 'Score_Engagement']
    
    # Create subplots for numerical variables
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Distribution of Numerical Variables', fontsize=16, y=1.02)
    
    for idx, col in enumerate(numerical_cols):
        row = idx // 2
        col_idx = idx % 2
        
        # Histogram with KDE
        sns.histplot(data=df, x=col, kde=True, ax=axes[row, col_idx])
        axes[row, col_idx].set_title(f'Distribution of {col}')
        axes[row, col_idx].set_xlabel(col)
        axes[row, col_idx].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('numerical_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Box plots for numerical variables by cluster
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Distribution by Cluster', fontsize=16, y=1.02)
    
    for idx, col in enumerate(numerical_cols):
        row = idx // 2
        col_idx = idx % 2
        
        sns.boxplot(data=df, x='Cluster', y=col, ax=axes[row, col_idx])
        axes[row, col_idx].set_title(f'{col} by Cluster')
    
    plt.tight_layout()
    plt.savefig('cluster_boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Violin plots for numerical variables by cluster
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Detailed Distribution by Cluster', fontsize=16, y=1.02)
    
    for idx, col in enumerate(numerical_cols):
        row = idx // 2
        col_idx = idx % 2
        
        sns.violinplot(data=df, x='Cluster', y=col, ax=axes[row, col_idx])
        axes[row, col_idx].set_title(f'{col} by Cluster')
    
    plt.tight_layout()
    plt.savefig('cluster_violinplots.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Correlation heatmap
    plt.figure(figsize=(10, 8))
    correlation_matrix = df[numerical_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap of Numerical Variables')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Expertise tags distribution
    plt.figure(figsize=(15, 6))
    tags = df['Tags_Expertise'].str.split(', ', expand=True).stack()
    tag_counts = tags.value_counts()
    
    sns.barplot(x=tag_counts.index, y=tag_counts.values)
    plt.title('Distribution of Expertise Tags')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Tags')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('expertise_tags_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 6. Scatter plot matrix
    sns.pairplot(df[numerical_cols + ['Cluster']], hue='Cluster', diag_kind='kde')
    plt.tight_layout()
    plt.savefig('scatter_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Load data
    df = load_data()
    
    # Create visualizations
    print("Creating distribution plots...")
    create_distribution_plots(df)
    print("\nVisualization files have been created:")
    print("1. numerical_distributions.png - Histograms with KDE for numerical variables")
    print("2. cluster_boxplots.png - Box plots showing distribution by cluster")
    print("3. cluster_violinplots.png - Violin plots showing detailed distribution by cluster")
    print("4. correlation_heatmap.png - Correlation heatmap of numerical variables")
    print("5. expertise_tags_distribution.png - Bar plot of expertise tags distribution")
    print("6. scatter_matrix.png - Scatter plot matrix showing relationships between variables")

if __name__ == "__main__":
    main() 