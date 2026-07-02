from typing import Literal
from pydantic import BaseModel, Field

class PatientData(BaseModel):
    Age: int = Field(ge=0, le=120)
    Sex: Literal["M","F"]
    ChestPainType: Literal["TA", "ATA", "NAP", "ASY"]
    RestingBP: int = Field(ge=0, le=200)
    Cholesterol: int = Field(ge=0, le=600)
    FastingBS: Literal[1,0]
    RestingECG: Literal["Normal", "ST", "LVH"]
    MaxHR: int = Field(ge=60, le=202)
    ExerciseAngina: Literal["Y", "N"]
    Oldpeak: float = Field(ge=-3, le=7)
    ST_Slope: Literal["Up", "Flat", "Down"]
