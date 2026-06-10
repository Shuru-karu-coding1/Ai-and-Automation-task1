# ==========================================================
# SEC 10-K Classification API using FastAPI
# ==========================================================

# -----------------------------
# Imports
# -----------------------------
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# ==========================================================
# Create FastAPI application
# Swagger UI will automatically be available at:
# http://127.0.0.1:8000/docs
# ==========================================================
app = FastAPI(
    title="SEC 10-K Risk Classification API",
    description="Predicts risk level from SEC 10-K text",
    version="1.0"
)

# ==========================================================
# Load model and vectorizer at startup
# ==========================================================
try:
    model = joblib.load("xgboost_model.joblib")
    vectorizer = joblib.load("tfidf_vectorizer.joblib")

except Exception as e:
    print("Error loading model or vectorizer:")
    print(e)

    model = None
    vectorizer = None


# ==========================================================
# Input Schema
# Example:
# {
#     "text": "Company revenue declined significantly..."
# }
# ==========================================================
class TextInput(BaseModel):
    text: str


# ==========================================================
# Output Schema
# Example:
# {
#     "label": "high",
#     "confidence": 0.94
# }
# ==========================================================
class PredictionOutput(BaseModel):
    label: str
    confidence: float


# ==========================================================
# Health Check Endpoint
# GET /health
# ==========================================================
@app.get("/health")
def health_check():
    return {"status": "ok"}


# ==========================================================
# Prediction Endpoint
# POST /predict
# ==========================================================
@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: TextInput):

    # ------------------------------------------------------
    # Ensure model and vectorizer are loaded
    # ------------------------------------------------------
    if model is None or vectorizer is None:
        raise HTTPException(
            status_code=500,
            detail="Model or vectorizer not loaded."
        )

    # ------------------------------------------------------
    # Validate input text
    # ------------------------------------------------------
    if input_data.text is None or input_data.text.strip() == "":
        raise HTTPException(
            status_code=400,
            detail="Input text cannot be empty."
        )

    try:
        # --------------------------------------------------
        # Transform text using TF-IDF vectorizer
        # --------------------------------------------------
        X = vectorizer.transform([input_data.text])

        # --------------------------------------------------
        # Predict class
        # --------------------------------------------------
        prediction = model.predict(X)[0]

        # --------------------------------------------------
        # Predict probabilities
        # --------------------------------------------------
        probabilities = model.predict_proba(X)[0]

        # Highest probability = confidence
        confidence = float(np.max(probabilities))

        # --------------------------------------------------
        # Return response
        # --------------------------------------------------
        return PredictionOutput(
            label=str(prediction),
            confidence=round(confidence, 4)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )