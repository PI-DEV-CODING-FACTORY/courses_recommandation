from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.models import InputFeatures, PredictionOutput, CompetencesInput, RecommendationOutput
from app.services import get_recommendations, load_artifacts, get_svm_recommendation, load_svm_artifacts
import traceback

# Create a dependency to ensure artifacts are loaded before handling requests
async def ensure_artifacts_loaded():
    try:
        if not load_artifacts():
            print("Warning: KNN artifacts failed to load")
        if not load_svm_artifacts():
            print("Warning: SVM artifacts failed to load")
    except Exception as e:
        print(f"Error loading artifacts: {e}")
        print(traceback.format_exc())

app = FastAPI(
    title="Formation Recommendation API",
    description="API for recommending courses based on competences and interests.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add startup event to load artifacts
@app.on_event("startup")
async def startup_event():
    try:
        print("Loading artifacts on startup...")
        knn_success = load_artifacts()
        svm_success = load_svm_artifacts()
        
        if knn_success and svm_success:
            print("All artifacts loaded successfully on startup")
        elif knn_success:
            print("KNN artifacts loaded successfully, but SVM artifacts failed")
        elif svm_success:
            print("SVM artifacts loaded successfully, but KNN artifacts failed")
        else:
            print("Failed to load artifacts on startup")
    except Exception as e:
        print(f"Error loading artifacts on startup: {e}")
        print(traceback.format_exc())

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
async def get_multiple_recommendations(data: CompetencesInput, _=Depends(ensure_artifacts_loaded)):
    try:
        recommendations = get_recommendations(data)
        return RecommendationOutput(recommendations=recommendations)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Error in recommendations endpoint: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/recommendations/svm/", response_model=RecommendationOutput, tags=["Recommendations"])
async def get_svm_recommendations(data: CompetencesInput, _=Depends(ensure_artifacts_loaded)):
    try:
        recommendations = get_svm_recommendation(data)
        return RecommendationOutput(recommendations=recommendations)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Error in SVM recommendations endpoint: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# If you want a dedicated endpoint to check/reload artifacts (for admin/debug)
@app.post("/admin/reload-artifacts/", tags=["Admin"])
async def reload_all_artifacts():
    try:
        knn_success = load_artifacts()
        svm_success = load_svm_artifacts()
        
        if knn_success and svm_success:
            return JSONResponse(status_code=200, content={"message": "All artifacts reloaded successfully."})
        elif knn_success:
            return JSONResponse(status_code=207, content={"message": "KNN artifacts reloaded successfully, but SVM artifacts failed."})
        elif svm_success:
            return JSONResponse(status_code=207, content={"message": "SVM artifacts reloaded successfully, but KNN artifacts failed."})
        else:
            return JSONResponse(status_code=500, content={"message": "Failed to reload artifacts."})
    except Exception as e:
        print(f"Error reloading artifacts: {e}")
        print(traceback.format_exc())
        return JSONResponse(status_code=500, content={"message": f"Error reloading artifacts: {str(e)}"})
