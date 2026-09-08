# DiaPredict AI — Architecture

## High-level diagram

```
                         ┌──────────────────────────────┐
        User              │        React Frontend        │
  ───────────────────────►│  TypeScript · Vite · Tailwind│
  patient measurements    └──────────────┬───────────────┘
                                         │ /api (JSON, REST)
                                         ▼
                         ┌──────────────────────────────┐
                         │        FastAPI Backend        │
                         │  Pydantic schemas + validation│
                         └──────────────┬───────────────┘
                                         │
                                         ▼
                                 ┌───────────────┐
                                 │  Preprocessing│  (scale + impute)
                                 └───────┬───────┘
                                         │
                ┌────────────────────────┼────────────────────────┐
                ▼                        ▼                        ▼
        ┌───────────────┐        ┌──────────────┐        ┌──────────────┐
        │  Logistic     │        │  Random      │        │  XGBoost     │
        │  Regression   │        │  Forest      │        │              │
        └───────┬───────┘        └──────┬───────┘        └──────┬───────┘
                └────────────────────────┼────────────────────────┘
                                         ▼
                                  ML Ensemble (0.7 × average)
                                         │
                                         ▼
                          ┌──────────────────────────────┐
                          │      PyTorch Neural Network   │
                          │        (0.3 weight)           │
                          └──────────────┬───────────────┘
                                         ▼
                         ┌──────────────────────────────┐
                         │      Final Risk Assessment    │
                         │   (weighted 0.7 / 0.3 blend)  │
                         └──────────────┬───────────────┘
                     ┌──────────────────┴──────────────────┐
                     ▼                                     ▼
        ┌──────────────────────────┐         ┌──────────────────────────┐
        │   SHAP Explanations      │         │   Model Metrics (test)   │
        │   local + global          │         │   ROC / PR / confusion   │
        └──────────────┬───────────┘         └──────────────┬───────────┘
                       └────────────────┬───────────────────┘
                                        ▼
                               Explanation + Metrics
                                        │
                                        ▼
                                   React UI
```

## Data flow

1. **User** enters anonymised measurements in the React form (5 steps, with
   validation) — `frontend/src/pages/Assessment.tsx`.
2. **Frontend** POSTs a `PatientInput` to the FastAPI backend via the Vite
   `/api` proxy — `frontend/src/api/client.ts`.
3. **Backend** validates the input (Pydantic `PatientInput` schema).
4. **Preprocessing** replays the same median-imputation + StandardScaler
   transforms used at training time — `backend/app/services.py`.
5. **Inference** runs all four loaded models and returns probabilities —
   `backend/app/inference.py`.
6. **Ensemble** blends the ML models (unweighted mean, weight `0.7`) with the
   neural network (weight `0.3`) — `backend/app/ensemble.py`.
7. **SHAP** explains the prediction via the XGBoost surrogate and converts
   contributions into probability space — `backend/app/explainability.py`.
8. **Persistence** stores an anonymous record in SQLite —
   `backend/app/database.py`.
9. **Response** is returned to the frontend and rendered —
   `frontend/src/components/Results.tsx`.

## Ensemble weighting (documented rationale)

| Step               | Weight | Rationale                                                                 |
|--------------------|--------|----------------------------------------------------------------------------|
| Logistic Regression| 1.0    | Linear baseline; low variance.                                              |
| Random Forest      | 1.0    | Bagged trees; different bias/variance profile.                              |
| XGBoost            | 1.0    | Boosted trees; strong but can overfit.                                      |
| **ML ensemble**    | 0.7    | Equal-weight averaging reduces variance of the individual ML estimators.    |
| **Neural network** | 0.3    | Non-linear representation from a different model family de-correlates errors.|

The final score is:

```
ML_ensemble = (P_logit + P_forest + P_xgb) / 3
Final_score = 0.7 * ML_ensemble + 0.3 * P_mlp
```

If the deep model is unavailable the final score falls back to `ML_ensemble`.
These weights are heuristics chosen for stability on this small dataset, not
learned values.

> **IMPORTANT:** The final score is a *model-defined probability*. It is **not**
> a medical diagnosis and must **never** be used to make medical decisions.

## Risk categories

The backend maps the 0–100 % score into three *model-defined* bands
(`backend/app/ensemble.py::risk_category`):

| Score       | Category               |
|-------------|------------------------|
| 0 – 30 %    | Lower predicted risk   |
| 31 – 60 %   | Moderate predicted risk|
| 61 – 100 %  | Higher predicted risk  |

These are **explanatory buckets derived from model output** — not clinical
diagnostic thresholds. The UI explicitly states this.

## Explainability (SHAP)

- **Surrogate**: XGBoost (exact, fast TreeSHAP; no background dataset needed).
- **Local**: `explainer.shap_values(row)` → per-feature SHAP values in log-odds
  space, converted to additive **probability-space** contributions so a value of
  `+0.21` reads as "this feature increased the predicted probability by ~0.21".
- **Global**: mean absolute SHAP over a background sample of the training set,
  used for the dataset-level feature importance chart.

## Persistence

SQLite (`data/diapredict.db`) stores, per assessment: `id`, `created_at`,
`input_features`, `model_predictions`, `ensemble_prob`, `risk_percent`,
`risk_category`, `model_used`. **No personally identifiable information is
stored** — only the anonymous auto-generated numeric `id`.

## Tech stack

- **Backend**: Python 3, FastAPI, Pydantic v2, scikit-learn, XGBoost, PyTorch,
  SHAP, SQLite (stdlib `sqlite3`).
- **Frontend**: React 18, TypeScript, Vite, Tailwind CSS, Recharts, Lucide
  icons.
- **Tests**: pytest + FastAPI `TestClient`.

## Design decisions

- `StandardScaler` and imputation medians are fitted **only on the training
  split** and reused at inference time to prevent **data leakage**.
- Models are loaded once into a singleton `ModelRegistry` to keep inference
  fast and consistent.
- Explainability is optional at the API level: a SHAP failure never blocks a
  prediction (defensive design).
- As a research prototype, CORS is permissive in dev; tighten `allow_origins`
  before any production use.
