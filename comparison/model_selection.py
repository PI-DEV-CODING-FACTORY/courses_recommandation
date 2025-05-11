import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns
import time

class ModelEvaluator:
    def __init__(self):
        self.models = {
            'K-Means': {
                'model': KMeans(n_clusters=3, random_state=42),
                'complexity': 'O(k * n * i)',
                'avantages': [
                    'Simple et facile à comprendre',
                    'Rapide sur de grands ensembles de données',
                    'Garantit la convergence',
                    'Adapté aux clusters de forme sphérique'
                ],
                'inconvenients': [
                    'Nécessite de spécifier le nombre de clusters',
                    'Sensible aux outliers',
                    'Suppose des clusters de forme sphérique',
                    'Sensible à l'initialisation'
                ]
            },
            'DBSCAN': {
                'model': DBSCAN(eps=0.5, min_samples=5),
                'complexity': 'O(n * log(n))',
                'avantages': [
                    'Ne nécessite pas de spécifier le nombre de clusters',
                    'Peut détecter des clusters de forme arbitraire',
                    'Robuste aux outliers',
                    'Pas d\'hypothèse sur la forme des clusters'
                ],
                'inconvenients': [
                    'Sensible aux paramètres eps et min_samples',
                    'Performances réduites si densités variables',
                    'Difficile avec données de haute dimension',
                    'Peut être plus lent que K-Means'
                ]
            },
            'Hierarchical': {
                'model': AgglomerativeClustering(n_clusters=3),
                'complexity': 'O(n²)',
                'avantages': [
                    'Hiérarchie de clusters visualisable',
                    'Pas d\'hypothèse sur la forme des clusters',
                    'Flexible sur le critère de liaison',
                    'Déterministe'
                ],
                'inconvenients': [
                    'Coût calculatoire élevé',
                    'Utilisation mémoire importante',
                    'Peut être sensible au bruit',
                    'Moins adapté aux grands datasets'
                ]
            },
            'Gaussian Mixture': {
                'model': GaussianMixture(n_components=3, random_state=42),
                'complexity': 'O(k * n * i)',
                'avantages': [
                    'Clusters souples (probabilistes)',
                    'Adapté aux clusters de tailles différentes',
                    'Peut capturer des corrélations',
                    'Base statistique solide'
                ],
                'inconvenients': [
                    'Sensible à l\'initialisation',
                    'Peut converger vers optimum local',
                    'Suppose distribution gaussienne',
                    'Complexité calculatoire modérée'
                ]
            }
        }

    def load_and_preprocess_data(self):
        print("Chargement et prétraitement des données...")
        df = pd.read_csv('Segmentation_des_utilisateurs.csv')
        features = ['Questions_Postees', 'Reponses_Fournies', 'Upvotes_Recus', 'Score_Engagement']
        X = df[features]
        
        # Standardisation
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled

    def evaluate_models(self, X):
        results = {}
        
        for name, model_info in self.models.items():
            print(f"\nÉvaluation de {name}:")
            start_time = time.time()
            
            # Entraînement du modèle
            model = model_info['model']
            model.fit(X)
            
            # Temps d'exécution
            execution_time = time.time() - start_time
            
            # Prédictions
            if hasattr(model, 'labels_'):
                labels = model.labels_
            else:
                labels = model.predict(X)
            
            # Métriques d'évaluation
            try:
                silhouette = silhouette_score(X, labels)
                calinski = calinski_harabasz_score(X, labels)
            except:
                silhouette = "N/A"
                calinski = "N/A"
            
            results[name] = {
                'silhouette': silhouette,
                'calinski': calinski,
                'time': execution_time,
                'complexity': model_info['complexity'],
                'avantages': model_info['avantages'],
                'inconvenients': model_info['inconvenients']
            }
            
            print(f"Temps d'exécution: {execution_time:.2f} secondes")
            print(f"Score Silhouette: {silhouette}")
            print(f"Score Calinski-Harabasz: {calinski}")
        
        return results

    def plot_comparison(self, results):
        # Création d'un graphique comparatif
        plt.figure(figsize=(12, 6))
        
        models = list(results.keys())
        silhouette_scores = [results[m]['silhouette'] for m in models]
        execution_times = [results[m]['time'] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Score Silhouette
        ax1.bar(x, silhouette_scores, width, label='Score Silhouette')
        ax1.set_ylabel('Score Silhouette')
        ax1.set_title('Comparaison des Scores Silhouette')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45)
        
        # Temps d'exécution
        ax2.bar(x, execution_times, width, label='Temps d\'exécution')
        ax2.set_ylabel('Temps (secondes)')
        ax2.set_title('Comparaison des Temps d\'exécution')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, rotation=45)
        
        plt.tight_layout()
        plt.savefig('model_comparison.png')
        plt.close()

def main():
    evaluator = ModelEvaluator()
    
    # Chargement et prétraitement des données
    X = evaluator.load_and_preprocess_data()
    
    # Évaluation des modèles
    print("\n=== Évaluation des Modèles de Clustering ===")
    results = evaluator.evaluate_models(X)
    
    # Visualisation des comparaisons
    evaluator.plot_comparison(results)
    
    # Affichage du résumé
    print("\n=== Résumé des Modèles ===")
    for name, info in results.items():
        print(f"\n{name}:")
        print("Avantages:")
        for adv in info['avantages']:
            print(f"- {adv}")
        print("\nInconvénients:")
        for inc in info['inconvenients']:
            print(f"- {inc}")
        print(f"\nComplexité: {info['complexity']}")
        print(f"Performance (Silhouette): {info['silhouette']}")
        print(f"Temps d'exécution: {info['time']:.2f} secondes")

if __name__ == "__main__":
    main() 