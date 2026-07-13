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