import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import time

class ModelEvaluator:
    def __init__(self):
        self.models = {
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'SVM': SVC(kernel='rbf', random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'XGBoost': XGBClassifier(random_state=42)
        }
        self.metrics = {}
        
    def load_and_preprocess_data(self):
        print("Chargement et prétraitement des données...")
        # Charger les données
        df = pd.read_csv('Segmentation_des_utilisateurs.csv')
        
        # Sélectionner les features pertinentes (à adapter selon votre dataset)
        features = ['Recency', 'Frequency', 'Monetary', 'Age', 'Family_Size', 'Points_in_Wallet']
        X = df[features]
        y = df['Cluster'] - 1  # Ajuster les étiquettes pour commencer à 0
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test

    def evaluate_models(self, X_train, X_test, y_train, y_test):
        results = []
        
        for name, model in self.models.items():
            print(f"\nÉvaluation de {name}...")
            start_time = time.time()
            
            # Entraînement
            model.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            # Prédictions
            y_pred = model.predict(X_test)
            
            # Calcul des métriques
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            # Cross-validation pour la robustesse
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            
            results.append({
                'Modèle': name,
                'Accuracy': accuracy,
                'F1-Score': f1,
                'RMSE': rmse,
                'CV Mean': cv_scores.mean(),
                'CV Std': cv_scores.std(),
                'Temps (s)': train_time
            })
            
            self.metrics[name] = {
                'accuracy': accuracy,
                'f1': f1,
                'rmse': rmse,
                'train_time': train_time
            }
        
        return pd.DataFrame(results)

    def plot_metrics_comparison(self):
        metrics_df = pd.DataFrame(self.metrics).T
        
        # Configuration du style
        plt.style.use('seaborn')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Comparaison des Performances des Modèles', fontsize=16)
        
        # Accuracy
        sns.barplot(x=metrics_df.index, y='accuracy', data=metrics_df, ax=axes[0, 0])
        axes[0, 0].set_title('Accuracy')
        axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=45)
        
        # F1-Score
        sns.barplot(x=metrics_df.index, y='f1', data=metrics_df, ax=axes[0, 1])
        axes[0, 1].set_title('F1-Score')
        axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45)
        
        # RMSE
        sns.barplot(x=metrics_df.index, y='rmse', data=metrics_df, ax=axes[1, 0])
        axes[1, 0].set_title('RMSE')
        axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45)
        
        # Temps d'exécution
        sns.barplot(x=metrics_df.index, y='train_time', data=metrics_df, ax=axes[1, 1])
        axes[1, 1].set_title('Temps d\'exécution (secondes)')
        axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig('model_metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    evaluator = ModelEvaluator()
    
    # Chargement et prétraitement des données
    X_train, X_test, y_train, y_test = evaluator.load_and_preprocess_data()
    
    # Évaluation des modèles
    print("\n=== Évaluation des Modèles de Classification ===")
    results = evaluator.evaluate_models(X_train, X_test, y_train, y_test)
    
    # Affichage des résultats
    print("\nRésultats détaillés:")
    print(results.to_string(index=False))
    
    # Génération des visualisations
    evaluator.plot_metrics_comparison()
    
    # Identification du meilleur modèle
    best_accuracy = results.loc[results['Accuracy'].idxmax()]
    best_f1 = results.loc[results['F1-Score'].idxmax()]
    best_rmse = results.loc[results['RMSE'].idxmin()]
    fastest = results.loc[results['Temps (s)'].idxmin()]
    
    print("\nMeilleurs modèles par métrique:")
    print(f"Meilleure Accuracy: {best_accuracy['Modèle']} ({best_accuracy['Accuracy']:.3f})")
    print(f"Meilleur F1-Score: {best_f1['Modèle']} ({best_f1['F1-Score']:.3f})")
    print(f"Meilleur RMSE: {best_rmse['Modèle']} ({best_rmse['RMSE']:.3f})")
    print(f"Plus rapide: {fastest['Modèle']} ({fastest['Temps (s)']:.3f}s)")

if __name__ == "__main__":
    main() 