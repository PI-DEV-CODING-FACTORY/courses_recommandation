from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .models import InputFeatures, PredictionOutput, CompetencesInput, RecommendationOutput
from .services import get_recommendations, load_artifacts

# Attempt to load artifacts when the app starts.
# The services.py already calls load_artifacts() at import time.
# You could add an explicit startup event if preferred:
# app.add_event_handler("startup", load_artifacts)

app = FastAPI(
    title="Formation Recommendation API",
    description="API for recommending courses based on competences and interests.",
    version="0.1.0",
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to the Formation Recommendation API! Visit /docs for API documentation."}

# Keep the original endpoint for backward compatibility
@app.post("/predict/", response_model=PredictionOutput, tags=["Legacy"])
async def get_recommendation(data: InputFeatures): 
    raise HTTPException(
        status_code=501, 
        detail="This endpoint is deprecated. Please use /recommendations/ instead."
    )
        
# New recommendations endpoint with the updated format
@app.post("/recommendations/", response_model=RecommendationOutput, tags=["Recommendations"])
async def get_multiple_recommendations(data: CompetencesInput):
    try:
        recommendations = get_recommendations(data)
        return RecommendationOutput(recommendations=recommendations)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# If you want a dedicated endpoint to check/reload artifacts (for admin/debug)
@app.post("/admin/reload-artifacts/", tags=["Admin"], status_code=200)
async def reload_all_artifacts():
    if load_artifacts():
        return {"message": "Artifacts reloaded successfully."}
    else:
        raise HTTPException(status_code=500, detail="Failed to reload artifacts.")
