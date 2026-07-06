from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_predict_valid_patient():
    patient_data = {
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
        "ST_Slope": "Flat"
    }

    response = client.post("/predict", json=patient_data)
    data = response.json()

    assert response.status_code == 200

    assert data["prediction"] == 0 or data["prediction"] == 1

    assert 0 <= data["predicted_proba"] <= 1