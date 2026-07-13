from typing import Literal
from pydantic import BaseModel, Field

class PatientData(BaseModel):
    """
    Request schema for POST /predict: clinical data for a single patient.

    Pydantic validates every field automatically (types, ranges, and allowed
    values) before the request reaches the endpoint, so invalid payloads get
    rejected with a 422 response without running any prediction code.
    """
    Age: int = Field(ge=0, le=120, description="Age in years")
    Sex: Literal["M", "F"] = Field(description="Biological sex: M (male) or F (female)")
    ChestPainType: Literal["TA", "ATA", "NAP", "ASY"] = Field(
        description="Chest pain type: TA (typical angina), ATA (atypical angina), "
                     "NAP (non-anginal pain), ASY (asymptomatic)"
    )
    RestingBP: int = Field(ge=0, le=200, description="Resting blood pressure in mm Hg")
    Cholesterol: int = Field(ge=0, le=600, description="Serum cholesterol in mg/dl")
    FastingBS: Literal[1, 0] = Field(
        description="Fasting blood sugar > 120 mg/dl: 1 (true) or 0 (false)"
    )
    RestingECG: Literal["Normal", "ST", "LVH"] = Field(
        description="Resting electrocardiogram result: Normal, ST (ST-T wave "
                     "abnormality), or LVH (left ventricular hypertrophy)"
    )
    MaxHR: int = Field(ge=60, le=202, description="Maximum heart rate achieved")
    ExerciseAngina: Literal["Y", "N"] = Field(
        description="Exercise-induced angina: Y (yes) or N (no)"
    )
    Oldpeak: float = Field(
        ge=-3, le=7, description="ST depression induced by exercise relative to rest"
    )
    ST_Slope: Literal["Up", "Flat", "Down"] = Field(
        description="Slope of the peak exercise ST segment: Up, Flat, or Down"
    )


class PredictionResponse(BaseModel):
    """Response schema for POST /predict: the model's prediction for one patient."""
    prediction: Literal[1, 0] = Field(
        description="Predicted class: 1 (heart disease) or 0 (no heart disease)"
    )
    predicted_proba: float = Field(
        ge=0, le=1, description="Estimated probability of heart disease (class 1)"
    )