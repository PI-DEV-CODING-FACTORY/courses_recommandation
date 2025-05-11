# Course Recommendation API

This project is a FastAPI-based application that provides course recommendations using a K-Nearest Neighbors (KNN) model. This project also contains the implementation of the KNN model and a comparison with other models : in the comparison folder.

## Features

*   Course recommendations based on input features/competences.
*   Scalable API built with FastAPI.
*   Containerized with Docker for easy deployment.

## Technologies Used

*   Python 3.9+
*   FastAPI
*   Uvicorn
*   Scikit-learn
*   Docker

## Project Structure

```
.
├── app/
│   ├── __init__.py
│   ├── main.py     # FastAPI application, endpoints
│   ├── models.py   # Pydantic models for request/response
│   └── ml_model/   # KNN model and related logic (example path)
│       └── knn_model.pkl # Trained model file
├── requirements.txt    # Python dependencies
├── Dockerfile          # Docker build instructions
├── .dockerignore       # Files to ignore in Docker build
└── README.md           # This file
```

## Setup and Installation (Local)

1.  **Clone the repository (if you haven't already):**
    ```bash
    git clone https://github.com/PI-DEV-CODING-FACTORY/courses_recommandation.git
    cd courses_recommandation
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application (Local)

To run the FastAPI application locally using Uvicorn:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
(Assuming your FastAPI instance in `app/main.py` is named `app`. Adjust the port as needed.)
You can then access the API at `http://localhost:8000` and the auto-generated documentation at `http://localhost:8000/docs`.

## Building and Running with Docker

1.  **Build the Docker image:**
    ```bash
    docker build -t courses-recommandation-app .
    ```

2.  **Run the Docker container:**
    ```bash
    docker run -p 80:80 courses-recommandation-app
    ```
    The application will be accessible at `http://localhost:8000`.

## API Endpoints

*   `POST /predict`: Endpoint for getting course predictions.
    *   Request body: (Define your `InputFeatures` model here)
    *   Response body: (Define your `PredictionOutput` model here)
*   `POST /recommend_competences`: Endpoint for getting recommendations based on competences.
    *   Request body: (Define your `CompetencesInput` model here)
    *   Response body: (Define your `RecommendationOutput` model here)

(Please update the "Project Structure" and "API Endpoints" sections to accurately reflect your project.)
