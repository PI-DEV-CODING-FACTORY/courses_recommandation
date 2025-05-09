from pydantic import BaseModel, Field
from typing import List, Any, Optional

class CompetencesInput(BaseModel):
    competences: List[str] = Field(..., 
        example=["Operating System", "MySQL", "Angular"],
        description="List of competences to match against the training data")
    top_n: int = Field(5, ge=1, le=20, 
        description="Number of recommendations to return (between 1 and 20)")

class RecommendationItem(BaseModel):
    formation: str = Field(..., example="Laravel")
    similarity_score: float = Field(..., example=0.291)
    competences: List[str] = Field(..., 
        example=["PHP", "Web Development", "MySQL"])
    centre_interet: str = Field(..., example="Web Dev")
    duree: int = Field(..., example=16)
    note: float = Field(..., example=4.1)

class RecommendationOutput(BaseModel):
    recommendations: List[RecommendationItem] = Field(...)

class InputFeatures(BaseModel):
    competence_1: str = Field(..., example="C++")
    competence_2: str = Field(..., example="Game Development")
    competence_3: str = Field(..., example="Unity")
    formation_suivie: str = Field(..., example="Game Development Intermédiaire")
    centre_interet: str = Field(..., example="Game Dev")
    duree_formation: float = Field(..., example=18.0) # Or int, ensure consistency with training data
    note_formation: float = Field(..., example=4.8)

class PredictionOutput(BaseModel):
    # This will output the string name of the recommended course
    recommended_formation: str = Field(..., example="Unity 3D")
    # You could add more details if needed, e.g.,
    # input_features: InputFeatures