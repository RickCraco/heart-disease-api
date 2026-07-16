import joblib
import pathlib
import pandas as pd
from api.schemas import PatientData, PredictionResponse
from fastapi import FastAPI
from fastapi import Request
from fastapi import HTTPException
from fastapi import Depends, status
from fastapi.security import APIKeyHeader
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from src.logger import setup_logger
from api.config import settings

app = FastAPI()

# dedicated logger for prediction requests, separate from the training logs
LOGS_FILE = pathlib.Path("logs/prediction_api.log")
logger = setup_logger(LOGS_FILE, "prediction_api")

# API_KEY header
api_key_header = APIKeyHeader(name="X-API-Key")

# the trained pipeline (preprocessing + XGBoost model) is loaded once at
# startup and reused for every request, instead of reloading it per call
MODEL_PATH = pathlib.Path("models/xgboost_pipeline.joblib")
try:
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading the model: {e}")
    model = None


# Rate limit
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


def verify_api_key(api_key: str = Depends(api_key_header)):
    """
    Authenticate an incoming request via its API key header.

    Extracts the ``X-API-Key`` header and compares it against the configured
    key, allowing the request to proceed only when they match. Intended to be
    used as a FastAPI dependency to protect endpoints.

    Args:
        api_key (str): API key provided in the ``X-API-Key`` request header,
            injected by the ``api_key_header`` dependency.

    Raises:
        HTTPException: With status code 401 when the provided key does not
            match the configured API key.
    """
    if api_key != settings.api_key:
        logger.error("Authentication failed, user unauthorized")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API KEY")


@app.post("/predict", response_model=PredictionResponse, dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
def make_prediction(request: Request, patient_data: PatientData) -> PredictionResponse:
    """
    Predict the likelihood of heart disease for a single patient.

    Accepts validated clinical features, runs them through the trained
    preprocessing and classification pipeline, and returns the predicted
    class together with the estimated probability of heart disease.
    Requires a valid ``X-API-Key`` header and is rate-limited to 10
    requests per minute per client IP.

    Args:
        request (Request): Incoming HTTP request; required by the rate
            limiter to identify the client address.
        patient_data (PatientData): Patient clinical measurements and
            categorical indicators used as model input.

    Returns:
        PredictionResponse: Binary prediction (0 or 1) and the associated
            probability score in the range [0, 1].

    Raises:
        HTTPException: With status code 500 when the model fails during
            prediction.
    """
    # the model pipeline expects a DataFrame with one row per patient,
    # so the single validated payload is wrapped in a one-row DataFrame
    df = pd.DataFrame([patient_data.model_dump()])
    logger.info("Data loaded and validated correctly:\n%s", df)

    # predict() returns the class (0/1), predict_proba() returns [P(class=0), P(class=1)];
    # we only need the probability of class 1 (heart disease present)
    try:
        predicted_class = model.predict(df)[0]
        predicted_proba = model.predict_proba(df)[0][1]

        logger.info("Predicted class: %s, predicted probability: %s", predicted_class, predicted_proba)
    except Exception as e:
        logger.error(f"Error found: {e}")
        raise HTTPException(status_code=500, detail="Error during model prediction")

    # cast numpy types (int64/float64) to plain Python types for the response model
    return PredictionResponse(prediction=int(predicted_class), predicted_proba=float(predicted_proba))


@app.get("/health")
def check_model_status() -> dict[str, str]:
    """
    Report whether the prediction service is ready to serve requests.

    Checks that the trained pipeline was loaded successfully at startup,
    allowing clients and monitoring tools to verify service availability
    before sending prediction requests.

    Returns:
        dict[str, str]: A status payload ``{"status": "ok"}`` when the model
            is available.

    Raises:
        HTTPException: With status code 503 when the model failed to load and
            the service cannot serve predictions.
    """
    if model is not None:
        return {"status": "ok"}
    raise HTTPException(status_code=503, detail="Service unavailable")