import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Load preprocessed data
print("Loading preprocessed data...")
data = pd.read_csv('preprocessed_data.csv')
X = data.drop('Cluster', axis=1)
y = data['Cluster']

# Adjust labels for XGBoost (0-based indexing)
le = LabelEncoder()
y = le.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'SVM': SVC(kernel='rbf', random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42)
}

# Dictionary to store results
results = []
confusion_matrices = {}

print("\nTraining and evaluating models...")
for name, model in models.items():
    print(f"\nEvaluating {name}...")
    start_time = time.time()
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Store results
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'F1-Score': f1,
        'CV Mean': cv_scores.mean(),
        'CV Std': cv_scores.std(),
        'Training Time': time.time() - start_time
    })
    
    confusion_matrices[name] = conf_matrix
    
    # Print detailed classification report
    print(f"\nClassification Report for {name}:")
    print(classification_report(y_test, y_pred))

# Convert results to DataFrame
results_df = pd.DataFrame(results)
print("\nModel Performance Summary:")
print(results_df.to_string(index=False))

# Save results to CSV
results_df.to_csv('model_performance_results.csv', index=False)

# Create visualizations
# 1. Performance Comparison
plt.figure(figsize=(12, 6))
metrics = ['Accuracy', 'F1-Score', 'CV Mean']
bar_width = 0.25
x = np.arange(len(models))

for i, metric in enumerate(metrics):
    plt.bar(x + i*bar_width, results_df[metric], width=bar_width, label=metric)

plt.xlabel('Models')
plt.ylabel('Score')
plt.title('Model Performance Comparison')
plt.xticks(x + bar_width, results_df['Model'], rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('model_performance_comparison.png')
plt.close()

# 2. Confusion Matrices
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
for (name, conf_matrix), ax in zip(confusion_matrices.items(), axes.ravel()):
    sns.heatmap(conf_matrix, annot=True, fmt='d', ax=ax)
    ax.set_title(f'Confusion Matrix - {name}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrices.png')
plt.close()

# 3. Training Time Comparison
plt.figure(figsize=(10, 6))
plt.bar(results_df['Model'], results_df['Training Time'])
plt.title('Training Time Comparison')
plt.xlabel('Model')
plt.ylabel('Time (seconds)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('training_time_comparison.png')
plt.close()

print("\nEvaluation completed. Results and visualizations have been saved.") 