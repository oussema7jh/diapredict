# DiaPredict AI — API Reference

Base URL (development): `http://localhost:8000`

Interactive docs (Swagger UI) are served at **`/docs`** once the backend runs.

All request/response bodies are JSON. The backend accepts the following
`PatientInput` shape and validates it (rejecting physiologically impossible
values with **`422 Unprocessable Content`**):

```json
{
  "Pregnancies": 2,
  "Glucose": 145,
  "BloodPressure": 85,
  "SkinThickness": 32,
  "Insulin": 120,
  "BMI": 31.2,
  "DiabetesPedigreeFunction": 0.52,
  "Age": 45
}
```

Validation ranges:

| Field                         | Min  | Max  |
|-------------------------------|------|------|
| `Pregnancies`                 | 0    | 20   |
| `Glucose`                     | 40   | 400  |
| `BloodPressure`               | 30   | 250  |
| `SkinThickness`               | 5    | 100  |
| `Insulin`                     | 5    | 900  |
| `BMI`                         | 10   | 90   |
| `DiabetesPedigreeFunction`    | 0.02 | 3.0  |
| `Age`                         | 1    | 120  |

---

## Endpoints

### `GET /health`
Liveness probe. Returns app status, version, number of loaded models and the
disclaimer.

### `GET /models/status`
Reports whether each trained model is loaded and ready.

**Response:**
```json
{
  "models": [
    {"model_id": "logistic_regression", "model_name": "Logistic Regression",
     "type": "ml", "loaded": true, "status": "ready", "file": "…", "message": null},
    {"model_id": "neural_network", "model_name": "Neural Network",
     "type": "dl", "loaded": true, "status": "ready", "file": "mlp.pt", "message": null}
  ],
  "all_loaded": true,
  "disclaimer": "…"
}
```

### `GET /models/performance`
Real validation metrics computed on the held-out test set.

**Response (array):**
```json
{
  "model_id": "random_forest",
  "model_name": "Random Forest",
  "accuracy": 0.7553,
  "precision": 0.7174,
  "recall": 0.5,
  "f1": 0.5893,
  "roc_auc": 0.8136,
  "confusion_matrix": {"tn": 109, "fp": 13, "fn": 33, "tp": 33},
  "trained_at": "2026-09-08T12:31:47.099343+00:00",
  "completed": true,
  "message": null
}
```

### `GET /models/curves`
ROC and precision-recall curve data for each model, computed on the test set
(used by the frontend charts).

### `GET /models/feature-importance`
Global mean-|SHAP| feature importance across the training background.

### `GET /models/dl-training`
Neural-network training / validation loss and validation AUC history.

### `GET /models/stats`
Aggregate dashboard statistics (total assessments, average predicted risk,
higher-risk count, recent assessments).

### `POST /predict` (full assessment)
Runs **all** models, the ensemble, SHAP explanation and persists the record.

**Response (`PredictionResponse`):**
```json
{
  "assessment_id": 7,
  "timestamp": "…",
  "input": { "Pregnancies": 2, "…": 45 },
  "input_pretty": [ {"feature": "Glucose", "label": "Glucose", "value": "145", "unit": "mg/dL"} ],
  "model_predictions": [
    {"model_id": "logistic_regression", "model_name": "Logistic Regression", "probability": 0.476, "risk_percent": 47.6, "status": "ok"},
    {"model_id": "random_forest", "model_name": "Random Forest", "probability": 0.53, "risk_percent": 53.0, "status": "ok"},
    {"model_id": "xgboost", "model_name": "XGBoost", "probability": 0.471, "risk_percent": 47.1, "status": "ok"},
    {"model_id": "neural_network", "model_name": "Neural Network", "probability": 0.503, "risk_percent": 50.3, "status": "ok"}
  ],
  "ensemble": {"model_id": "ensemble", "model_name": "Ensemble", "probability": 0.496, "risk_percent": 49.6, "status": "ok"},
  "risk_percent": 49.6,
  "risk_category": "Moderate predicted risk",
  "risk_category_explanation": "…",
  "explanation": {
    "method": "shap", "model": "xgboost", "base_value": 0.496, "patient_value": 0.496,
    "top_features": [
      {"feature": "Age", "label": "Age", "value": 0.872, "contribution": 0.213, "direction": "increases"}
    ],
    "summary_english": "The model assigned a … predicted risk mainly because …"
  },
  "disclaimer": "…"
}
```

### `POST /predict/ml`
Runs only the three machine-learning models. Returns an array of
`ModelPrediction`.

### `POST /predict/dl`
Runs only the neural network. Returns a single `ModelPrediction`.

### `POST /predict/ensemble`
Runs the weighted ensemble. Returns a single `ModelPrediction`.

### `POST /predict/explain`
Returns the SHAP local explanation for a given input (same body as `/predict`).

---

## Schemas

- `PatientInput` — the validated input.
- `ModelPrediction` — a single model's result.
- `PredictionResponse` — the full assessment result.
- `ModelPerformance` — one model's computed metrics.
- `ModelsStatusResponse` / `ModelStatus` — model loading status.
- `HealthResponse` — liveness info.

## Errors

- `422 Unprocessable Content` — invalid/impossible input (Pydantic
  validation).
- `503 Service Unavailable` — trained models missing (run
  `python scripts/train_all.py`).
- `200 OK` with `completed: false` / `message` in `/models/performance` when a
  model has no stored metrics.
