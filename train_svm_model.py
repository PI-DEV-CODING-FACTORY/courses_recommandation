import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import json
import os

# Load the dataset
df = pd.read_csv("data/df_reference.csv")

# Columns to encode
categorical_cols = ['Compétence_1', 'Compétence_2', 'Compétence_3', 
                   'Formation_Suivie', 'Centre_Intérêt', 'Formation_Recommandée']

# Create and fit encoders
encoders = {}
competence_mappings = {}
for col in categorical_cols:
    encoders[col] = LabelEncoder()
    df[col] = encoders[col].fit_transform(df[col])
    if col in ['Compétence_1', 'Compétence_2', 'Compétence_3']:
        # Store both the original values and their encoded versions
        competence_mappings[col] = {
            'values': encoders[col].classes_.tolist(),
            'encoded': list(range(len(encoders[col].classes_)))
        }

# Save encoders
joblib.dump(encoders, 'data/encoders_svm.pkl')

# Save competence mappings
with open('data/competence_mappings.json', 'w') as f:
    json.dump(competence_mappings, f, indent=4)

# Features for training
X = df[['Compétence_1', 'Compétence_2', 'Compétence_3', 
        'Formation_Suivie', 'Durée_Formation', 'Note_Formation', 'Centre_Intérêt']]
y = df['Formation_Recommandée']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train SVM model
svm_model = SVC(kernel='rbf', probability=True, random_state=42)
svm_model.fit(X_train, y_train)

# Save model
joblib.dump(svm_model, 'data/svm_model.pkl')

# Save model metadata
metadata = {
    'model_type': 'SVM',
    'kernel': 'rbf',
    'C': 1.0,
    'gamma': 'scale',
    'accuracy': svm_model.score(X_test, y_test),
    'features': X.columns.tolist(),
    'target_name': 'Formation_Recommandée'
}

with open('data/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=4)

print("Model training complete!")
print(f"Test accuracy: {metadata['accuracy']}")

# Print valid competences for reference
print("\nValid competences for each position:")
for col, mapping in competence_mappings.items():
    print(f"\n{col}:")
    print(sorted(mapping['values']))