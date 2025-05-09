import joblib
import pickle
import numpy as np
import pandas as pd
import traceback
import os
from typing import Dict, Any, List
from sklearn.neighbors import NearestNeighbors
from .models import CompetencesInput, RecommendationItem

# --- Artifact Paths ---
MODEL_PATH = "data/your_knn_model.joblib"
ENCODERS_PATH = "data/label_encoders.pkl"
SCALER_PATH = "data/standard_scaler.pkl"
REF_DATA_PATH = "data/df_reference.csv"  # Add path to the saved reference data

# --- Global Variables to Hold Loaded Artifacts ---
knn_model = None
label_encoders: Dict[str, Any] = {}
standard_scaler = None
df_reference = None

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

def get_recommendations(data: CompetencesInput) -> List[RecommendationItem]:
    """
    Get top N recommendations based on a list of competences using KNN.
    """
    global knn_model, label_encoders, standard_scaler, df_reference

    if label_encoders is None or df_reference is None:
        print("Artifacts not loaded. Cannot predict.")
        if not load_artifacts():  # Attempt to load them
            raise ValueError("Error: Model or preprocessing artifacts not available.")

    try:
        # Extract input competences and top_n
        input_competences = data.competences
        top_n = min(data.top_n, 20)  # Limit to 20 max recommendations
        
        print(f"Processing request for competences: {input_competences}, top_n: {top_n}")
        
        # Calculate how many unique formations we have in the dataset
        unique_formation_count = len(df_reference['Formation_Recommandée'].unique())
        
        # Request more neighbors initially to account for duplicates
        # We'll request either 3x the requested number or the total unique formations, whichever is smaller
        initial_neighbors = min(top_n * 3, unique_formation_count)
        
        # Initialize a nearest neighbors model for finding similar formations
        nearest_neighbors = NearestNeighbors(n_neighbors=initial_neighbors, metric='cosine')
        
        # Create a matrix of all competences from the reference data
        all_competences = set()
        for comp_col in ['Compétence_1', 'Compétence_2', 'Compétence_3']:
            all_competences.update(df_reference[comp_col].unique())
            
        # Build a binary feature vector for the input competences
        competence_vector = np.zeros(len(all_competences))
        all_competences_list = list(all_competences)
        
        for competence in input_competences:
            if competence in all_competences_list:
                idx = all_competences_list.index(competence)
                competence_vector[idx] = 1
                
        # Build feature vectors for all records in the reference data
        reference_vectors = np.zeros((len(df_reference), len(all_competences)))
        
        for i, row in df_reference.iterrows():
            for comp_col in ['Compétence_1', 'Compétence_2', 'Compétence_3']:
                competence = row[comp_col]
                if competence in all_competences_list:
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
            # Courses around 20 hours get higher boost
            duration_factor = 1.0 - min(1.0, abs(record['Durée_Formation'] - 20) / 10)
            final_score += 0.1 * duration_factor
            
            # 4. Diversity bonus - to encourage varied results (10% weight)
            # This would ideally look at previously shown recommendations to avoid similar ones
            # For now, we'll just use a small random factor
            diversity_factor = np.random.uniform(0, 1)
            final_score += 0.1 * diversity_factor
            
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
                
        # If we didn't get enough unique recommendations, we might need to request more from KNN
        if len(recommendations) < top_n and len(recommendations) < len(df_reference['Formation_Recommandée'].unique()):
            # We could add more complex logic here to get additional recommendations
            # For now we'll just return what we have
            pass
            
        return recommendations
            
    except Exception as e:
        print(f"Error during recommendation: {e}")
        print(traceback.format_exc())
        raise ValueError(f"Error processing recommendation: {str(e)}")

# --- Initialize models at startup ---
try:
    print("Attempting to load artifacts at startup...")
    load_artifacts()
except Exception as e:
    print(f"Failed to load artifacts at startup, but continuing: {e}")
    print(traceback.format_exc())