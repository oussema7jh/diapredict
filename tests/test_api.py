"""API endpoint tests (uses FastAPI TestClient)."""

from __future__ import annotations


def test_health(test_client):
    r = test_client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["models_loaded"] == 4
    assert "not a medical diagnosis" in body["disclaimer"]


def test_models_status(test_client):
    r = test_client.get("/models/status")
    assert r.status_code == 200
    body = r.json()
    assert len(body["models"]) == 4
    assert body["all_loaded"] is True


def test_models_performance(test_client):
    r = test_client.get("/models/performance")
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 4
    for item in body:
        assert item["completed"] is True
        assert item["accuracy"] > 0.5
        assert item["roc_auc"] > 0.5
        assert set(item["confusion_matrix"].keys()) == {"tn", "fp", "fn", "tp"}


def test_predict_full(test_client, valid_patient):
    r = test_client.post("/predict", json=valid_patient.model_dump())
    assert r.status_code == 200
    body = r.json()
    assert body["assessment_id"] > 0
    assert 0 <= body["risk_percent"] <= 100
    assert len(body["model_predictions"]) == 4
    assert body["ensemble"]["risk_percent"] > 0
    assert body["explanation"] is not None
    assert len(body["explanation"]["top_features"]) <= 5
    assert "not a medical diagnosis" in body["disclaimer"]


def test_predict_ml_and_dl_and_ensemble(test_client, valid_patient):
    data = valid_patient.model_dump()
    ml = test_client.post("/predict/ml", json=data)
    assert ml.status_code == 200
    assert len(ml.json()) == 3

    dl = test_client.post("/predict/dl", json=data)
    assert dl.status_code == 200
    assert dl.json()["model_id"] == "neural_network"

    ens = test_client.post("/predict/ensemble", json=data)
    assert ens.status_code == 200
    assert ens.json()["model_id"] == "ensemble"


def test_invalid_input_rejected(test_client):
    """Impossible values must be rejected: negative age, bad insulin."""
    r = test_client.post(
        "/predict",
        json={
            "Pregnancies": -1,
            "Glucose": 145,
            "BloodPressure": 85,
            "SkinThickness": 32,
            "Insulin": 120,
            "BMI": 31.2,
            "DiabetesPedigreeFunction": 0.52,
            "Age": 45,
        },
    )
    assert r.status_code == 422

    r2 = test_client.post(
        "/predict",
        json={
            "Pregnancies": 2,
            "Glucose": 145,
            "BloodPressure": 85,
            "SkinThickness": 32,
            "Insulin": 0,
            "BMI": 31.2,
            "DiabetesPedigreeFunction": 0.52,
            "Age": 45,
        },
    )
    assert r2.status_code == 422


def test_models_stats(test_client):
    r = test_client.get("/models/stats")
    assert r.status_code == 200
    body = r.json()
    assert "total_assessments" in body
    assert "avg_risk_percent" in body
