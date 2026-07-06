from fastapi import FastAPI
import joblib
import pathlib
import pandas as pd
from api.schemas import PatientData, PredictionResponse

app = FastAPI()

MODEL_PATH = pathlib.Path("models/xgboost_pipeline.joblib")
model = joblib.load(MODEL_PATH)