import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client():
    """HTTP client for testing the FastAPI app without starting a server."""
    return TestClient(app)


@pytest.fixture
def valid_patient():
    """A patient payload that passes Pydantic validation."""
    return {
        "Age": 58,
        "Sex": "M",
        "ChestPainType": "ASY",
        "RestingBP": 140,
        "Cholesterol": 289,
        "FastingBS": 1,
        "RestingECG": "ST",
        "MaxHR": 110,
        "ExerciseAngina": "Y",
        "Oldpeak": 2.0,
        "ST_Slope": "Flat",
    }


@pytest.fixture
def make_patient(valid_patient):
    """Return a copy of the valid payload with optional field overrides."""

    def _make(**overrides):
        return {**valid_patient, **overrides}

    return _make
