import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import pickle
import os

# --- Configuration ---
DATASET_PATH = "data/df_reference.csv"  # Path to your dataset file
MODEL_SAVE_PATH = "data/your_knn_model.joblib"  # Matches MODEL_PATH in app/services.py
ENCODERS_PATH = "data/label_encoders.pkl"  # Matches ENCODERS_PATH in app/services.py
SCALER_PATH = "data/standard_scaler.pkl"  # Matches SCALER_PATH in app/services.py
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_NEIGHBORS = 5

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

def load_data(path: str) -> pd.DataFrame:
    """Loads dataset from the specified path."""
    print(f"Loading dataset from {path}...")
    try:
        df = pd.read_csv(path)
        print("Dataset loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {path}. Please place your dataset there.")
        return None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def preprocess_data(df: pd.DataFrame):
    """
    Preprocesses the data with label encoding for categorical features and scaling for numerical features.
    """
    print("Preprocessing data...")
    if df is None or df.empty:
        print("Cannot preprocess empty or None dataframe.")
        return None, None, None, None

    # Define categorical and numerical columns
    categorical_cols = ['Compétence_1', 'Compétence_2', 'Compétence_3',
                        'Formation_Suivie', 'Centre_Intérêt', 'Formation_Recommandée']
    numerical_cols = ['Durée_Formation', 'Note_Formation']
    
    # Check if all required columns exist
    for col in categorical_cols + numerical_cols:
        if col not in df.columns:
            print(f"Error: '{col}' column not found in the dataset.")
            return None, None, None, None
    
    # Create a copy to avoid modifying the original dataframe
    df_processed = df.copy()
    
    # Initialize dictionaries to store encoders
    label_encoders = {}
    
    # Label encode categorical columns
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        label_encoders[col] = le
        print(f"Encoded {col} with {len(le.classes_)} unique values")
    
    # Extract features (X) and target (y)
    X = df_processed.drop('Formation_Recommandée', axis=1)
    y = df_processed['Formation_Recommandée']
    
    # Scale numerical features
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    print("Data preprocessing complete.")
    return X, y, label_encoders, scaler

def train_knn_model(X_train, y_train):
    """Trains the KNN model."""
    print(f"Training KNN model with {N_NEIGHBORS} neighbors...")
    # Initialize KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors=N_NEIGHBORS, weights='distance')
    
    model.fit(X_train, y_train)
    print("Model training complete.")
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluates the trained model."""
    print("Evaluating model...")
    predictions = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy:.4f}")
    
    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

def save_artifacts(model, label_encoders, scaler):
    """Saves the trained model, label encoders, and scaler to the specified paths."""
    try:
        # Save the model
        print(f"Saving model to {MODEL_SAVE_PATH}...")
        joblib.dump(model, MODEL_SAVE_PATH)
        print(f"Model saved successfully to {MODEL_SAVE_PATH}")
        
        # Save the label encoders
        print(f"Saving label encoders to {ENCODERS_PATH}...")
        with open(ENCODERS_PATH, 'wb') as f:
            pickle.dump(label_encoders, f)
        print(f"Label encoders saved successfully to {ENCODERS_PATH}")
        
        # Save the scaler
        print(f"Saving scaler to {SCALER_PATH}...")
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Scaler saved successfully to {SCALER_PATH}")
        
    except Exception as e:
        print(f"Error saving artifacts: {e}")

def main():
    """Main function to run the training pipeline."""
    print("--- Starting Model Training Pipeline ---")
    
    # 1. Load Data
    df = load_data(DATASET_PATH)
    if df is None:
        print("Halting pipeline due to data loading issues.")
        return

    # 2. Preprocess Data
    X, y, label_encoders, scaler = preprocess_data(df.copy())
    if X is None or y is None:
        print("Halting pipeline due to preprocessing issues.")
        return

    # 3. Split Data
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    print(f"Training set size: {X_train.shape[0]}, Testing set size: {X_test.shape[0]}")

    # 4. Train Model
    model = train_knn_model(X_train, y_train)

    # 5. Evaluate Model
    evaluate_model(model, X_test, y_test)

    # 6. Save Model and other artifacts
    save_artifacts(model, label_encoders, scaler)
    
    # 7. Save the reference data for inference
    print(f"Saving reference data to {DATASET_PATH}...")
    df.to_csv(DATASET_PATH, index=False)
    
    print("--- Model Training Pipeline Finished ---")

if __name__ == "__main__":
    main()
    print(f"\nArtifacts have been saved to the following locations:")
    print(f"  - Model: {MODEL_SAVE_PATH}")
    print(f"  - Label Encoders: {ENCODERS_PATH}")
    print(f"  - Scaler: {SCALER_PATH}")
    print(f"  - Reference Data: {DATASET_PATH}")
    print("\nThese paths match the ones in app/services.py for inference.")
