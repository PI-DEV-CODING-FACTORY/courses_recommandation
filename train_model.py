import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier # Or KNeighborsRegressor if it's a regression task
from sklearn.preprocessing import StandardScaler # Optional: if you need feature scaling
from sklearn.metrics import accuracy_score # Or other relevant metrics
import joblib
import os

# --- Configuration ---
DATASET_PATH = "data/your_dataset.csv"  # IMPORTANT: Update this to your dataset file
MODEL_SAVE_PATH = "data/your_knn_model.pkl" # This should match MODEL_PATH in app/services.py
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_NEIGHBORS = 5 # Example: Number of neighbors for KNN

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
    Preprocesses the data.
    This is where you'd handle missing values, feature engineering, scaling, etc.
    Modify this function based on your dataset's needs.
    """
    print("Preprocessing data...")
    # --- TODO: Add your preprocessing steps here ---
    # Example: Separating features (X) and target (y)
    # Assuming the last column is the target variable
    # X = df.iloc[:, :-1]
    # y = df.iloc[:, -1]

    # Example: Feature Scaling (optional, but often good for KNN)
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)
    # X = pd.DataFrame(X_scaled, columns=X.columns)
    
    # For this placeholder, we'll assume X and y are defined.
    # Replace this with your actual feature and target separation.
    if df is None or df.empty:
        print("Cannot preprocess empty or None dataframe.")
        return None, None

    # --- Replace with your actual feature and target column names/logic ---
    print("Please update the preprocess_data function in train_model.py with your actual feature (X) and target (y) separation logic.")
    # As a placeholder, let's assume 'target' is the name of your target column
    # And all other columns are features.
    if 'target' not in df.columns:
        print("Error: 'target' column not found in the dataset. Please define your target variable (y) and features (X) in preprocess_data().")
        return None, None
        
    X = df.drop('target', axis=1)
    y = df['target']
    # --- End of placeholder ---
    
    print("Data preprocessing complete.")
    return X, y

def train_knn_model(X_train, y_train):
    """Trains the KNN model."""
    print(f"Training KNN model with {N_NEIGHBORS} neighbors...")
    # Initialize KNeighborsClassifier or KNeighborsRegressor
    model = KNeighborsClassifier(n_neighbors=N_NEIGHBORS)
    # model = KNeighborsRegressor(n_neighbors=N_NEIGHBORS) # If it's a regression task

    # --- TODO: Add any specific model training parameters here ---
    
    model.fit(X_train, y_train)
    print("Model training complete.")
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluates the trained model."""
    print("Evaluating model...")
    predictions = model.predict(X_test)
    
    # Example for classification:
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy:.4f}")
    
    # --- TODO: Add other relevant evaluation metrics ---
    # For regression, you might use:
    # from sklearn.metrics import mean_squared_error
    # mse = mean_squared_error(y_test, predictions)
    # print(f"Mean Squared Error: {mse:.4f}")

def save_model(model, path: str):
    """Saves the trained model to the specified path."""
    print(f"Saving model to {path}...")
    try:
        joblib.dump(model, path)
        print(f"Model saved successfully to {path}")
    except Exception as e:
        print(f"Error saving model: {e}")

def main():
    """Main function to run the training pipeline."""
    print("--- Starting Model Training Pipeline ---")
    
    # 1. Load Data
    df = load_data(DATASET_PATH)
    if df is None:
        print("Halting pipeline due to data loading issues.")
        return

    # 2. Preprocess Data
    X, y = preprocess_data(df.copy()) # Use .copy() to avoid modifying the original DataFrame
    if X is None or y is None:
        print("Halting pipeline due to preprocessing issues.")
        return

    # 3. Split Data
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    print("Data splitting complete.")

    # 4. Train Model
    model = train_knn_model(X_train, y_train)

    # 5. Evaluate Model (optional, but good practice)
    evaluate_model(model, X_test, y_test)

    # 6. Save Model
    save_model(model, MODEL_SAVE_PATH)
    
    print("--- Model Training Pipeline Finished ---")

if __name__ == "__main__":
    main()
    print(f"\nTo use the trained model, ensure '{MODEL_SAVE_PATH}' is correctly referenced in 'app/services.py'.")
    print(f"Make sure you have placed your dataset at '{DATASET_PATH}'.")
