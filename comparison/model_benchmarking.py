import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tabulate import tabulate
import warnings
warnings.filterwarnings('ignore')

# Configuration du style Seaborn
sns.set_theme(style="whitegrid")

class ModelBenchmarker:
    def __init__(self):
        self.models = {
            'K-Means': {
                'model': KMeans(n_clusters=3, random_state=42),
                'params': {'n_clusters': 3},
                'type': 'Partitionnement'
            },
            'DBSCAN': {
                'model': DBSCAN(eps=0.5, min_samples=5),
                'params': {'eps': 0.5, 'min_samples': 5},
                'type': 'Densité'
            },
            'Hierarchical': {
                'model': AgglomerativeClustering(n_clusters=3),
                'params': {'n_clusters': 3},
                'type': 'Hiérarchique'
            },
            'Gaussian Mixture': {
                'model': GaussianMixture(n_components=3, random_state=42),
                'params': {'n_components': 3},
                'type': 'Probabiliste'
            }
        }
        
    def load_and_preprocess_data(self):
        print("Chargement et prétraitement des données...")
        df = pd.read_csv('Segmentation_des_utilisateurs.csv')
        features = ['Questions_Postees', 'Reponses_Fournies', 'Upvotes_Recus', 'Score_Engagement']
        X = df[features]
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, df

    def benchmark_models(self, X):
        results = []
        
        for name, model_info in self.models.items():
            print(f"\nBenchmarking de {name}...")
            
            # Mesures de performance
            metrics = self._evaluate_model(X, model_info['model'], name)
            
            # Caractéristiques du modèle
            model_chars = {
                'Nom': name,
                'Type': model_info['type'],
                'Paramètres': str(model_info['params']),
                **metrics
            }
            
            results.append(model_chars)
        
        return pd.DataFrame(results)

    def _evaluate_model(self, X, model, name):
        # Temps d'entraînement
        start_time = time.time()
        model.fit(X)
        train_time = time.time() - start_time
        
        # Prédictions
        if hasattr(model, 'labels_'):
            labels = model.labels_
        else:
            labels = model.predict(X)
        
        # Métriques d'évaluation
        try:
            silhouette = silhouette_score(X, labels)
            calinski = calinski_harabasz_score(X, labels)
            davies = davies_bouldin_score(X, labels)
            n_clusters = len(np.unique(labels[labels >= 0]))  # Compte les clusters (ignore le bruit pour DBSCAN)
        except:
            silhouette = np.nan
            calinski = np.nan
            davies = np.nan
            n_clusters = np.nan
        
        return {
            'Temps (s)': round(train_time, 3),
            'Silhouette': round(silhouette, 3) if not np.isnan(silhouette) else "N/A",
            'Calinski-Harabasz': round(calinski, 3) if not np.isnan(calinski) else "N/A",
            'Davies-Bouldin': round(davies, 3) if not np.isnan(davies) else "N/A",
            'Nb Clusters': n_clusters
        }

    def generate_comparison_plots(self, results):
        # 1. Comparaison des scores de performance
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Comparaison des Performances des Modèles', fontsize=16)
        
        # Scores Silhouette
        self._plot_metric(axes[0, 0], results, 'Silhouette', 'Scores Silhouette')
        
        # Scores Calinski-Harabasz
        self._plot_metric(axes[0, 1], results, 'Calinski-Harabasz', 'Scores Calinski-Harabasz')
        
        # Scores Davies-Bouldin
        self._plot_metric(axes[1, 0], results, 'Davies-Bouldin', 'Scores Davies-Bouldin')
        
        # Temps d'exécution
        self._plot_metric(axes[1, 1], results, 'Temps (s)', 'Temps d\'exécution (secondes)')
        
        plt.tight_layout()
        plt.savefig('model_benchmarking.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_metric(self, ax, results, metric, title):
        if metric in results.columns:
            data = results[metric].astype(float)
            sns.barplot(x=results['Nom'], y=data, ax=ax)
            ax.set_title(title)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    def generate_report(self, results):
        print("\n=== Rapport de Benchmarking des Modèles de Clustering ===\n")
        
        # Tableau comparatif
        print("Tableau Comparatif des Performances:")
        print(tabulate(results, headers='keys', tablefmt='grid', showindex=False))
        
        # Analyse des résultats
        best_silhouette = results.loc[results['Silhouette'].astype(float).idxmax()]
        best_calinski = results.loc[results['Calinski-Harabasz'].astype(float).idxmax()]
        fastest = results.loc[results['Temps (s)'].astype(float).idxmin()]
        
        print("\nAnalyse des Résultats:")
        print(f"- Meilleur score Silhouette: {best_silhouette['Nom']} ({best_silhouette['Silhouette']})")
        print(f"- Meilleur score Calinski-Harabasz: {best_calinski['Nom']} ({best_calinski['Calinski-Harabasz']})")
        print(f"- Modèle le plus rapide: {fastest['Nom']} ({fastest['Temps (s)']} secondes)")
        
        print("\nForces et Faiblesses des Modèles:")
        for _, row in results.iterrows():
            print(f"\n{row['Nom']} ({row['Type']}):")
            print("Forces:")
            if row['Silhouette'] == best_silhouette['Silhouette']:
                print("- Meilleure cohésion des clusters (Score Silhouette)")
            if row['Temps (s)'] == fastest['Temps (s)']:
                print("- Performance computationnelle optimale")
            if row['Type'] == 'Densité':
                print("- Détection automatique du nombre de clusters")
                print("- Robuste aux outliers")
            elif row['Type'] == 'Hiérarchique':
                print("- Structure hiérarchique des clusters")
                print("- Pas d'hypothèse sur la forme des clusters")
            elif row['Type'] == 'Probabiliste':
                print("- Approche probabiliste flexible")
                print("- Adapté aux clusters de tailles différentes")
            
            print("Faiblesses:")
            if float(row['Temps (s)']) > float(fastest['Temps (s)']) * 2:
                print("- Temps de calcul élevé")
            if row['Type'] == 'Partitionnement':
                print("- Nécessite de spécifier le nombre de clusters")
                print("- Sensible aux outliers")
            elif row['Type'] == 'Densité':
                print("- Sensible aux paramètres de densité")
                print("- Peut avoir des difficultés avec des densités variables")
            elif row['Type'] == 'Probabiliste':
                print("- Sensible à l'initialisation")
                print("- Suppose une distribution gaussienne")

def main():
    benchmarker = ModelBenchmarker()
    
    # Chargement et prétraitement des données
    X, df = benchmarker.load_and_preprocess_data()
    
    # Benchmarking des modèles
    results = benchmarker.benchmark_models(X)
    
    # Génération des visualisations
    benchmarker.generate_comparison_plots(results)
    
    # Génération du rapport
    benchmarker.generate_report(results)

if __name__ == "__main__":
    main() 