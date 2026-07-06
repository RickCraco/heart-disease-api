import joblib
import pathlib
import pandas as pd
from api.schemas import PatientData, PredictionResponse
from fastapi import FastAPI
from src.logger import setup_logger

app = FastAPI()

MODEL_PATH = pathlib.Path("models/xgboost_pipeline.joblib")
model = joblib.load(MODEL_PATH)

LOGS_FILE = pathlib.Path("logs/prediction_api.log")
logger = setup_logger(LOGS_FILE, "prediction_api")

@app.post("/predict", response_model=PredictionResponse)
def make_prediction(patient_data: PatientData) -> PredictionResponse:
    """
    Predict the likelihood of heart disease for a single patient.

    Accepts validated clinical features, runs them through the trained
    preprocessing and classification pipeline, and returns the predicted
    class together with the estimated probability of heart disease.

    Args:
        patient_data (PatientData): Patient clinical measurements and
            categorical indicators used as model input.

    Returns:
        PredictionResponse: Binary prediction (0 or 1) and the associated
            probability score in the range [0, 1].
    """
    df = pd.DataFrame([patient_data.model_dump()])
    logger.info("Data loaded and validated correctly:\n%s", df)

    predicted_class = model.predict(df)[0]
    predicted_proba = model.predict_proba(df)[0][1]

    logger.info("Predicted class: %s, predicted probability: %s", predicted_class, predicted_proba)

    return PredictionResponse(prediction=int(predicted_class), predicted_proba=float(predicted_proba))