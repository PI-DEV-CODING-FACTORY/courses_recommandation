import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    print("Chargement des données...")
    return pd.read_csv('Segmentation_des_utilisateurs.csv')

def analyze_missing_values(df):
    print("\n1. Analyse des valeurs manquantes:")
    missing_values = df.isnull().sum()
    print(missing_values)
    
    if missing_values.sum() > 0:
        # Visualiser les valeurs manquantes
        plt.figure(figsize=(10, 6))
        missing_values.plot(kind='bar')
        plt.title('Distribution des valeurs manquantes')
        plt.xlabel('Colonnes')
        plt.ylabel('Nombre de valeurs manquantes')
        plt.tight_layout()
        plt.savefig('missing_values.png')
        plt.close()

def handle_outliers(df, columns):
    print("\n2. Traitement des valeurs aberrantes:")
    df_clean = df.copy()
    
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_count = df[(df[column] < lower_bound) | (df[column] > upper_bound)].shape[0]
        print(f"{column}: {outliers_count} valeurs aberrantes détectées")
        
        # Remplacer les valeurs aberrantes par les bornes
        df_clean.loc[df_clean[column] < lower_bound, column] = lower_bound
        df_clean.loc[df_clean[column] > upper_bound, column] = upper_bound
    
    return df_clean

def scale_features(df, columns):
    print("\n3. Normalisation des caractéristiques:")
    
    # StandardScaler (standardisation)
    standard_scaler = StandardScaler()
    df_standardized = pd.DataFrame(
        standard_scaler.fit_transform(df[columns]),
        columns=[f"{col}_standardized" for col in columns]
    )
    
    # MinMaxScaler (normalisation)
    minmax_scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(
        minmax_scaler.fit_transform(df[columns]),
        columns=[f"{col}_normalized" for col in columns]
    )
    
    # RobustScaler (mise à l'échelle robuste)
    robust_scaler = RobustScaler()
    df_robust = pd.DataFrame(
        robust_scaler.fit_transform(df[columns]),
        columns=[f"{col}_robust" for col in columns]
    )
    
    print("Méthodes de mise à l'échelle appliquées:")
    print("- StandardScaler (moyenne=0, écart-type=1)")
    print("- MinMaxScaler (échelle [0,1])")
    print("- RobustScaler (robuste aux valeurs aberrantes)")
    
    return df_standardized, df_normalized, df_robust

def encode_categorical_features(df):
    print("\n4. Encodage des variables catégorielles:")
    
    # One-Hot Encoding pour Tags_Expertise
    print("One-Hot Encoding pour Tags_Expertise:")
    tags_dummies = df['Tags_Expertise'].str.get_dummies(sep=', ')
    print(f"Nombre de colonnes créées: {tags_dummies.shape[1]}")
    
    return tags_dummies

def main():
    # Charger les données
    df = load_data()
    
    # Colonnes numériques à traiter
    numerical_cols = ['Questions_Postees', 'Reponses_Fournies', 'Upvotes_Recus', 'Score_Engagement']
    
    print("\n=== Rapport de Prétraitement des Données ===")
    
    # 1. Analyse des valeurs manquantes
    analyze_missing_values(df)
    
    # 2. Traitement des valeurs aberrantes
    df_clean = handle_outliers(df, numerical_cols)
    
    # 3. Normalisation des caractéristiques
    df_standardized, df_normalized, df_robust = scale_features(df_clean, numerical_cols)
    
    # 4. Encodage des variables catégorielles
    tags_encoded = encode_categorical_features(df)
    
    print("\n5. Résumé des transformations:")
    print(f"- Dimensions originales du dataset: {df.shape}")
    print(f"- Dimensions après nettoyage: {df_clean.shape}")
    print(f"- Nombre de caractéristiques après encodage des tags: {tags_encoded.shape[1]}")
    
    # Sauvegarder les données prétraitées
    df_clean.to_csv('data_cleaned.csv', index=False)
    print("\nDonnées prétraitées sauvegardées dans 'data_cleaned.csv'")

if __name__ == "__main__":
    main() 