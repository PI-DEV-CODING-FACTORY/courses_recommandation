from fastapi.testclient import TestClient
from app.main import app # Import your FastAPI app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the KNN Model API! Visit /docs for API documentation."}

def test_predict_endpoint_no_model():
    """
    Tests the predict endpoint.
    NOTE: This test will likely fail if your_knn_model.pkl doesn't exist
    or if the placeholder predict_knn function isn't robust enough yet.
    You'll need to adapt this test once your model is integrated.
    """
    # Example input data - adjust to match your model's expected features
    test_data = {"features": [1.0, 2.0, 3.0, 4.5]}
    
    response = client.post("/predict/", json=test_data)
    
    # Check that the request was successful (or handle expected errors)
    assert response.status_code == 200 
    
    # Check the structure of the response
    response_json = response.json()
    assert "predictions" in response_json

    # If the model is not loaded, the service might return an error message
    # This part of the test depends on how services.py handles a missing model
    if isinstance(response_json["predictions"], dict) and "error" in response_json["predictions"]:
        print(f"Prediction endpoint returned an error (as expected if model is missing): {response_json['predictions']['error']}")
    else:
        # Add more specific assertions here based on your model's expected output
        # For example, if it predicts a class:
        # assert isinstance(response_json["predictions"], list)
        # assert response_json["predictions"][0] == "ExpectedClass" # Or whatever your model should predict
        print(f"Prediction successful: {response_json['predictions']}")

# To run tests (from the root knn_fastapi_project directory):
# Ensure your virtual environment is activated.
# Install pytest: pip install pytest
# Run: pytest
