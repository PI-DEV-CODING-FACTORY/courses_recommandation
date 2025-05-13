import joblib
import pickle
import numpy as np
import pandas as pd
import traceback
import os
import json
from typing import Dict, Any, List
from sklearn.neighbors import NearestNeighbors
from .models import CompetencesInput, RecommendationItem

# --- Artifact Paths ---
MODEL_PATH = "data/your_knn_model.joblib"  # Matches MODEL_SAVE_PATH in train_model.py
ENCODERS_PATH = "data/label_encoders.pkl"  # Matches ENCODERS_PATH in train_model.py
SCALER_PATH = "data/standard_scaler.pkl"  # Matches SCALER_PATH in train_model.py
REF_DATA_PATH = "data/df_reference.csv"  # Matches DATASET_PATH in train_model.py

# SVM model paths
SVM_MODEL_PATH = "data/svm_model.pkl"
SVM_ENCODERS_PATH = "data/encoders_svm.pkl"

# --- Global Variables to Hold Loaded Artifacts ---
knn_model = None
svm_model = None
label_encoders: Dict[str, Any] = {}
svm_encoders: Dict[str, Any] = {}
standard_scaler = None
df_reference = None
competence_mappings = None

# Columns that were label encoded
CATEGORICAL_COLS = ['Compétence_1', 'Compétence_2', 'Compétence_3',
                    'Formation_Suivie', 'Centre_Intérêt', 'Formation_Recommandée']
# Columns that were scaled
NUMERICAL_COLS = ['Durée_Formation', 'Note_Formation']

def load_artifacts():
    global knn_model, label_encoders, standard_scaler, df_reference

    print("Loading artifacts...")
    try:
        # First check if files exist and print absolute paths
        print(f"Checking if model exists at {os.path.abspath(MODEL_PATH)}")
        if not os.path.exists(MODEL_PATH):
            print(f"ERROR: Model file not found at {MODEL_PATH}")
            return False
            
        print(f"Checking if encoders exist at {os.path.abspath(ENCODERS_PATH)}")
        if not os.path.exists(ENCODERS_PATH):
            print(f"ERROR: Encoders file not found at {ENCODERS_PATH}")
            return False
            
        print(f"Checking if scaler exists at {os.path.abspath(SCALER_PATH)}")
        if not os.path.exists(SCALER_PATH):
            print(f"ERROR: Scaler file not found at {SCALER_PATH}")
            return False
            
        print(f"Checking if reference data exists at {os.path.abspath(REF_DATA_PATH)}")
        if not os.path.exists(REF_DATA_PATH):
            print(f"ERROR: Reference data file not found at {REF_DATA_PATH}")
            # Create a copy from the original CSV file if available
            original_csv = "data/Recommandation_de_formations_expanded (1).csv"
            if os.path.exists(original_csv):
                print(f"Found original CSV, creating reference copy...")
                df = pd.read_csv(original_csv)
                df.to_csv(REF_DATA_PATH, index=False)
            else:
                return False
        
        # Now attempt to load each file
        print(f"Loading model from {MODEL_PATH}")
        knn_model = joblib.load(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")

        print(f"Loading encoders from {ENCODERS_PATH}")
        with open(ENCODERS_PATH, "rb") as f:
            label_encoders = pickle.load(f)
        print(f"Label encoders loaded successfully from {ENCODERS_PATH}")

        print(f"Loading scaler from {SCALER_PATH}")
        with open(SCALER_PATH, "rb") as f:
            standard_scaler = pickle.load(f)
        print(f"Standard scaler loaded successfully from {SCALER_PATH}")
        
        # Load reference data
        print(f"Loading reference data from {REF_DATA_PATH}")
        df_reference = pd.read_csv(REF_DATA_PATH)
        print(f"Reference data loaded successfully from {REF_DATA_PATH}")

        print("All artifacts loaded.")
        return True
    except FileNotFoundError as e:
        print(f"Error: Artifact file not found. {e}")
        print(f"Current working directory: {os.getcwd()}")
        return False
    except Exception as e:
        print(f"Error loading artifacts: {e}")
        print(traceback.format_exc())  # This prints the full traceback
        return False

def load_svm_artifacts():
    """Load SVM model and its encoders"""
    global svm_model, svm_encoders
    
    try:
        print("Loading SVM artifacts...")
        
        if not os.path.exists(SVM_MODEL_PATH):
            print(f"ERROR: SVM model file not found at {SVM_MODEL_PATH}")
            return False
            
        if not os.path.exists(SVM_ENCODERS_PATH):
            print(f"ERROR: SVM encoders file not found at {SVM_ENCODERS_PATH}")
            return False

        # Load SVM model
        svm_model = joblib.load(SVM_MODEL_PATH)
        print(f"SVM model loaded successfully from {SVM_MODEL_PATH}")

        # Load SVM encoders
        svm_encoders = joblib.load(SVM_ENCODERS_PATH)
        print(f"SVM encoders loaded successfully from {SVM_ENCODERS_PATH}")
        
        # Load competence mappings if available
        global competence_mappings
        mappings_path = "data/competence_mappings.json"
        if os.path.exists(mappings_path):
            with open(mappings_path, 'r') as f:
                competence_mappings = json.load(f)
            print(f"Competence mappings loaded successfully from {mappings_path}")

        return True
    except Exception as e:
        print(f"Error loading SVM artifacts: {e}")
        print(traceback.format_exc())
        return False

def get_recommendations(data: CompetencesInput) -> List[RecommendationItem]:
    """
    Get top N recommendations based on a list of competences using KNN.
    """
    global df_reference

    if df_reference is None:
        print("Reference data not loaded. Cannot predict.")
        if not load_artifacts():  # Attempt to load them
            raise ValueError("Error: Reference data not available.")

    try:
        # Extract input competences and top_n
        input_competences = data.competences
        top_n = min(data.top_n, 20)  # Limit to 20 max recommendations

        print(f"Processing request for competences: {input_competences}, top_n: {top_n}")

        # Validate that the competences exist in our dataset
        all_competences = set()
        for comp_col in ['Compétence_1', 'Compétence_2', 'Compétence_3']:
            all_competences.update(df_reference[comp_col].unique())

        for comp in input_competences:
            if comp not in all_competences:
                raise ValueError(f"Invalid competence value: {comp}")

        # Calculate how many unique formations we have in the dataset
        unique_formation_count = len(df_reference['Formation_Recommandée'].unique())
        
        # Request more neighbors initially to account for duplicates
        initial_neighbors = min(top_n * 3, unique_formation_count)
        
        # Initialize a nearest neighbors model for finding similar formations
        nearest_neighbors = NearestNeighbors(n_neighbors=initial_neighbors, metric='cosine')
        
        # Build a binary feature vector for the input competences
        competence_vector = np.zeros(len(all_competences))
        all_competences_list = list(all_competences)
        
        for competence in input_competences:
            idx = all_competences_list.index(competence)
            competence_vector[idx] = 1
            
        # Build feature vectors for all records in the reference data
        reference_vectors = np.zeros((len(df_reference), len(all_competences)))
        
        for i, row in df_reference.iterrows():
            for comp_col in ['Compétence_1', 'Compétence_2', 'Compétence_3']:
                competence = row[comp_col]
                idx = all_competences_list.index(competence)
                reference_vectors[i, idx] = 1
                    
        # Fit the nearest neighbors model
        nearest_neighbors.fit(reference_vectors)
        
        # Find the nearest neighbors
        distances, indices = nearest_neighbors.kneighbors(competence_vector.reshape(1, -1))
        
        # Convert distances to similarity scores (1 - distance)
        base_similarity_scores = 1 - distances[0]
        
        # Create lists to store enhanced scoring factors
        enhanced_scores = []
        
        # Apply enhanced scoring that considers multiple factors
        for i, idx in enumerate(indices[0]):
            record = df_reference.iloc[idx]
            base_score = base_similarity_scores[i]
            
            # 1. Base similarity score (60% weight)
            final_score = 0.6 * base_score
            
            # 2. Rating boost (0-20% weight based on rating from 4.0-5.0)
            rating_factor = min(1.0, max(0, (record['Note_Formation'] - 4.0)))
            final_score += 0.2 * rating_factor
            
            # 3. Duration boost - favor medium length courses (10% weight)
            duration_factor = 1.0 - min(1.0, abs(record['Durée_Formation'] - 20) / 10)
            final_score += 0.1 * duration_factor
            
            # 4. Competence match bonus (10% weight)
            match_count = sum(1 for comp in input_competences if comp in [
                record['Compétence_1'],
                record['Compétence_2'],
                record['Compétence_3']
            ])
            match_factor = match_count / len(input_competences)
            final_score += 0.1 * match_factor
            
            # Ensure score stays in 0-1 range
            final_score = min(1.0, max(0, final_score))
            enhanced_scores.append(final_score)
        
        # Convert to numpy array
        enhanced_scores = np.array(enhanced_scores)
        
        # Prepare recommendations
        recommendations = []
        seen_formations = set()  # Track formations we've already recommended
        
        # Sort by enhanced score (descending)
        sorted_indices = np.argsort(-enhanced_scores)
        
        # Process in order of enhanced score
        for i in sorted_indices:
            idx = indices[0][i]
            score = enhanced_scores[i]
            
            record = df_reference.iloc[idx]
            formation_name = record['Formation_Recommandée']
            
            # Skip this formation if we've already recommended it
            if formation_name in seen_formations:
                continue
                
            # Add to tracking set
            seen_formations.add(formation_name)
            
            # Get competences for this formation
            competences = [
                record['Compétence_1'],
                record['Compétence_2'],
                record['Compétence_3']
            ]
            
            # Create a recommendation item
            recommendation = RecommendationItem(
                formation=formation_name,
                similarity_score=float(score),  # Convert numpy float to Python float
                competences=competences,
                centre_interet=record['Centre_Intérêt'],
                duree=int(record['Durée_Formation']),
                note=float(record['Note_Formation'])
            )
            
            recommendations.append(recommendation)
            
            # If we have enough unique recommendations, stop
            if len(recommendations) >= top_n:
                break
                
        return recommendations
            
    except Exception as e:
        print(f"Error during recommendation: {e}")
        print(traceback.format_exc())
        raise ValueError(f"Error processing recommendation: {str(e)}")

def get_svm_recommendation(data: CompetencesInput):
    """Get course recommendation using SVM model"""
    global svm_model, svm_encoders, df_reference, competence_mappings

    if svm_model is None or svm_encoders is None:
        if not load_svm_artifacts():
            raise ValueError("SVM model or encoders not available")

    try:
        # Get reference data if not loaded
        if df_reference is None:
            df_reference = pd.read_csv("data/df_reference.csv")

        # Validate input length
        if not 1 <= len(data.competences) <= 3:
            raise ValueError("Between 1 and 3 competences are required")

        # Collect all valid competences across positions
        valid_competences = set()
        if competence_mappings:
            for col in ['Compétence_1', 'Compétence_2', 'Compétence_3']:
                valid_competences.update(competence_mappings[col]['values'])
        else:
            # Fallback to using reference data
            for col in ['Compétence_1', 'Compétence_2', 'Compétence_3']:
                valid_competences.update(df_reference[col].unique())
        
        # Validate input competences
        for comp in data.competences:
            if comp not in valid_competences:
                raise ValueError(f"Invalid competence value: {comp}. Valid competences are: {sorted(valid_competences)}")
        
        # Find best positions for provided competences
        input_competences = data.competences.copy()
        final_competences = [None, None, None]
        
        # Limit to first 3 competences if more are provided
        if len(valid_input_competences) > 3:
            valid_input_competences = valid_input_competences[:3]
        
        print(f"Using competences: {valid_input_competences}")
        
        # Find optimal positions for each competence based on reference data
        final_competences = [None, None, None]  # Initialize with None for all 3 positions
        
        for comp in valid_input_competences:
            # Count occurrences in each position
            counts = [0, 0, 0]
            for i, col in enumerate(['Compétence_1', 'Compétence_2', 'Compétence_3']):
                matching_rows = df_reference[df_reference[col] == comp]
                counts[i] = len(matching_rows)
            
            # Find the position where this competence appears most often
            best_pos = counts.index(max(counts))
            
            # If that position is already taken, try next best position
            while final_competences[best_pos] is not None:
                counts[best_pos] = -1  # Mark this position as invalid
                if max(counts) <= 0:  # No more valid positions
                    best_pos = None
                    break
                best_pos = counts.index(max(counts))
            
            if best_pos is None:
                # If we couldn't find a position, put it in the first available slot
                for i in range(3):
                    if final_competences[i] is None:
                        best_pos = i
                        break
            
            final_competences[best_pos] = comp

        # Fill any remaining positions with the last valid competence
        last_comp = final_competences[max(i for i, x in enumerate(final_competences) if x is not None)]
        final_competences = [c if c is not None else last_comp for c in final_competences]

        # Create a query point with the input competences
        query_point = []
        
        # For each competence position, encode the corresponding competence
        for i, comp_col in enumerate(['Compétence_1', 'Compétence_2', 'Compétence_3']):
            try:
                # Get the competence for this position
                comp = final_competences[i]
                # Encode it
                encoded_value = label_encoders[comp_col].transform([comp])[0]
                query_point.append(encoded_value)
            except Exception as e:
                print(f"Error encoding competence {comp} for position {i+1}: {e}")
                # Use a default value if encoding fails
                query_point.append(0)
        
        # Add placeholder values for other required features
        query_point.extend([
            0,  # Formation_Suivie (using a default value)
            0,  # Durée_Formation (will be scaled by the scaler)
            0,  # Note_Formation (will be scaled by the scaler)
            0,  # Centre_Intérêt (default value)
        ])
        
        # Scale the numerical features in the query point
        numerical_indices = [3, 4]  # Assuming positions 3 and 4 are Durée_Formation and Note_Formation
        query_array = np.array([query_point])
        query_array[:, numerical_indices] = standard_scaler.transform(query_array[:, numerical_indices])
        
        # Get the feature matrix from the reference data
        X_ref = df_reference.drop('Formation_Recommandée', axis=1)
        for col in X_ref.columns:
            if col in label_encoders:
                X_ref[col] = label_encoders[col].transform(X_ref[col])
        
        # Scale the numerical features in the reference data
        X_ref[NUMERICAL_COLS] = standard_scaler.transform(X_ref[NUMERICAL_COLS])
        
        # Use the KNN model to predict
        # First approach: Use the model directly if it's compatible
        try:
            # Get the distances and indices of the nearest neighbors
            distances, indices = knn_model.kneighbors(query_array)
            
            # Convert distances to similarity scores (1 / (1 + distance))
            similarities = 1 / (1 + distances[0])
            
            # Get the indices of the nearest neighbors
            neighbor_indices = indices[0]
        except Exception as e:
            print(f"Could not use KNN model directly: {e}. Using NearestNeighbors instead.")
            # Fallback: Use NearestNeighbors
            nn = NearestNeighbors(n_neighbors=min(data.top_n, len(X_ref)), metric='euclidean')
            nn.fit(X_ref)
            
            # Find nearest neighbors
            distances, indices = nn.kneighbors(query_array)
            
            # Convert distances to similarity scores
            similarities = 1 / (1 + distances[0])
            
            # Get the indices of the nearest neighbors
            neighbor_indices = indices[0]
        
        # Create recommendation items
        recommendations = []
        for i, idx in enumerate(neighbor_indices):
            row = df_reference.iloc[idx]
            recommendation = RecommendationItem(
                formation=row['Formation_Recommandée'],
                similarity_score=float(similarities[i]),
                competences=final_competences,  # Use the optimally ordered competences
                centre_interet=row['Centre_Intérêt'],
                duree=int(row['Durée_Formation']),
                note=float(row['Note_Formation'])
            )
            recommendations.append(recommendation)
        
        return recommendations

    except Exception as e:
        print(f"Error during recommendation: {e}")
        print(traceback.format_exc())
        raise

# --- Initialize models at startup ---
try:
    print("Attempting to load artifacts at startup...")
    load_artifacts()
    load_svm_artifacts()
except Exception as e:
    print(f"Failed to load artifacts at startup, but continuing: {e}")
    print(traceback.format_exc())