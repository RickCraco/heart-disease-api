from unittest.mock import patch


def test_predict_valid_patient(client, valid_patient):
    """Return 200 and a valid prediction payload for correct input data."""
    response = client.post("/predict", json=valid_patient)
    data = response.json()

    assert response.status_code == 200
    assert data["prediction"] in (0, 1)
    assert 0 <= data["predicted_proba"] <= 1
    assert set(data.keys()) == {"prediction", "predicted_proba"}


def test_predict_age_below_minimum(client, make_patient):
    """Return 422 when Age is below the accepted minimum value."""
    # 422 Unprocessable Entity is the status code FastAPI/Pydantic use
    # automatically when request data fails schema validation
    response = client.post("/predict", json=make_patient(Age=-1))

    assert response.status_code == 422


def test_predict_invalid_sex(client, make_patient):
    """Return 422 when Sex is not one of the allowed literal values."""
    response = client.post("/predict", json=make_patient(Sex="X"))

    assert response.status_code == 422


def test_predict_resting_bp_above_max(client, make_patient):
    """Return 422 when RestingBP is above the maximum allowed value."""
    response = client.post("/predict", json=make_patient(RestingBP=250))

    assert response.status_code == 422


def test_predict_missing_required_field(client, valid_patient):
    """Return 422 when a required field is missing from the payload."""
    payload = {key: value for key, value in valid_patient.items() if key != "Age"}
    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_health_success(client):
    """Return 200 when the model is available."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@patch("api.main.model", None)
def test_health_fail(client):
    """Return 503 if the model is unavailable"""
    response = client.get("/health")

    assert response.status_code == 503


def test_wrong_api_key(auth_client, valid_patient):
    """Return 401 if the API KEY is wrong"""
    response = auth_client.post("/predict", json=valid_patient, headers={"X-API-Key":"wrong-key"})

    assert response.status_code == 401


def test_missing_api_key(auth_client, valid_patient):
    """Return 401 if the API Key is missing"""
    response = auth_client.post("/predict", json=valid_patient)

    assert response.status_code == 401


@patch("api.main.settings.api_key", "test-key")
def test_valid_api_key(auth_client, valid_patient):
    """Return 200 if the API Key is valid"""
    response = auth_client.post("/predict", json=valid_patient, headers={"X-API-Key": "test-key"})

    assert response.status_code == 200