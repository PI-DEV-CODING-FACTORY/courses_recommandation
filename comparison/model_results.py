import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class ClusteringAnalyzer:
    def __init__(self):
        self.models = {
            'K-Means': KMeans(n_clusters=3, random_state=42),
            'DBSCAN': DBSCAN(eps=0.5, min_samples=5),
            'Hierarchical': AgglomerativeClustering(n_clusters=3),
            'Gaussian Mixture': GaussianMixture(n_components=3, random_state=42)
        }
        
    def load_and_preprocess_data(self):
        print("Chargement et prétraitement des données...")
        df = pd.read_csv('Segmentation_des_utilisateurs.csv')
        self.features = ['Questions_Postees', 'Reponses_Fournies', 'Upvotes_Recus', 'Score_Engagement']
        X = df[self.features]
        
        # Standardisation
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(X)
        self.df = df
        return self.X_scaled

    def analyze_clusters(self, model_name, labels, X):
        print(f"\n{'='*20} Résultats pour {model_name} {'='*20}")
        
        # Nombre de clusters
        n_clusters = len(np.unique(labels[labels >= 0]))
        print(f"\nNombre de clusters trouvés: {n_clusters}")
        
        # Taille des clusters
        unique, counts = np.unique(labels, return_counts=True)
        print("\nDistribution des clusters:")
        for cluster, count in zip(unique, counts):
            if cluster >= 0:  # Ignore noise points (DBSCAN)
                print(f"Cluster {cluster}: {count} utilisateurs ({count/len(labels)*100:.1f}%)")
            else:
                print(f"Points de bruit: {count} utilisateurs ({count/len(labels)*100:.1f}%)")
        
        # Caractéristiques des clusters
        X_with_labels = np.column_stack([X, labels])
        df_clusters = pd.DataFrame(X_with_labels, columns=self.features + ['Cluster'])
        
        print("\nCaractéristiques moyennes par cluster:")
        for cluster in range(n_clusters):
            if cluster in df_clusters['Cluster'].values:
                cluster_data = df_clusters[df_clusters['Cluster'] == cluster]
                print(f"\nCluster {cluster}:")
                for feature in self.features:
                    mean_val = cluster_data[feature].mean()
                    std_val = cluster_data[feature].std()
                    print(f"{feature}:")
                    print(f"  - Moyenne: {mean_val:.2f}")
                    print(f"  - Écart-type: {std_val:.2f}")
        
        # Visualisation de la distribution des caractéristiques par cluster
        self.plot_cluster_distributions(df_clusters, model_name)

    def plot_cluster_distributions(self, df_clusters, model_name):
        plt.figure(figsize=(15, 10))
        for i, feature in enumerate(self.features, 1):
            plt.subplot(2, 2, i)
            sns.boxplot(x='Cluster', y=feature, data=df_clusters)
            plt.title(f'{feature} par Cluster')
        plt.tight_layout()
        plt.savefig(f'cluster_distributions_{model_name.lower().replace(" ", "_")}.png')
        plt.close()

    def run_analysis(self):
        X = self.load_and_preprocess_data()
        
        for name, model in self.models.items():
            print(f"\nAnalyse de {name}...")
            model.fit(X)
            
            # Obtention des labels
            if hasattr(model, 'labels_'):
                labels = model.labels_
            else:
                labels = model.predict(X)
            
            # Analyse des clusters
            self.analyze_clusters(name, labels, X)

def main():
    analyzer = ClusteringAnalyzer()
    analyzer.run_analysis()

if __name__ == "__main__":
    main() 