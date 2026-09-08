# DiaPredict AI

### Explainable Diabetes Risk Prediction System

DiaPredict AI is an end-to-end, university-level **Applied AI / Machine
Learning** project that estimates a person's **diabetes risk** from structured
health measurements. It trains **three classical ML models and one deep
learning model**, combines them into an **ensemble**, and explains every
prediction with **SHAP** — all served through a modern **FastAPI + React**
web application.

> ⚠️ **Educational &amp; research prototype only.** DiaPredict AI provides
> machine-learning-based **risk estimates**, not a medical diagnosis. It must
> never be used to make medical decisions. Always consult a qualified
> healthcare professional for medical advice.

---

## Table of contents

1. [Overview](#overview)
2. [Features](#features)
3. [Architecture](#architecture)
4. [Machine Learning](#machine-learning)
5. [Deep Learning](#deep-learning)
6. [Explainable AI](#explainable-ai)
7. [Dataset](#dataset)
8. [Installation](#installation)
9. [Training models](#training-models)
10. [Running backend](#running-backend)
11. [Running frontend](#running-frontend)
12. [API documentation](#api-documentation)
13. [Testing](#testing)
14. [Project structure](#project-structure)
15. [Limitations](#limitations)
16. [Ethical considerations](#ethical-considerations)
17. [Future improvements](#future-improvements)

---

## Overview

A web application that takes anonymised patient measurements and returns a
**0–100 % risk score**, the **individual model predictions**, a **weighted
ensemble score**, and a **plain-language SHAP explanation** of what drove the
result.

| Input | Output |
|-------|--------|
| Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age | Risk %, risk category, per-model probabilities, ensemble score, SHAP explanations, comparison of model performance |

---

## Features

- **True ML**: Logistic Regression, Random Forest, XGBoost — actually trained,
  evaluated and saved (no fake predictions).
- **True DL**: a real PyTorch fully-connected network with training/validation
  loops, early stopping and checkpointing.
- **Ensemble**: weighted blend of the ML models and the neural network.
- **Explainable AI**: SHAP local (per-patient) and global (dataset-level)
  explanations, in probability space.
- **Model comparison**: real test-set metrics (accuracy, precision, recall,
  F1, ROC-AUC), confusion matrices, ROC & precision-recall curves, feature
  importance, and neural-network training curves.
- **Clean, professional UI**: React + TypeScript + Vite + Tailwind + Recharts
  + Lucide, with a 5-step validated assessment form.
- **Anonymous persistence**: SQLite (no PII).
- **Full test coverage**: pytest suite for preprocessing, models, validation
  and API endpoints.
- **Reproducible notebooks**: six Jupyter notebooks documenting the pipeline.

---

## Architecture

```
User → React Frontend → FastAPI Backend → Preprocessing
   → Logistic / Random Forest / XGBoost → ML Ensemble
   → PyTorch Neural Network → Final Risk Assessment
   → SHAP + Model Metrics → React UI
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full diagram,
data-flow walkthrough, and the documented ensemble weighting rationale.

---

## Machine Learning

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---:|---:|---:|---:|---:|---:|
| Logistic Regression | *calculated* | *calculated* | *calculated* | *calculated* | *calculated* |
| Random Forest | *calculated* | *calculated* | *calculated* | *calculated* | *calculated* |
| XGBoost | *calculated* | *calculated* | *calculated* | *calculated* | *calculated* |

> Every metric is computed from the **actual held-out test set** at training
> time (see `training/evaluation.py`) and served by `GET /models/performance`.
> Nothing is hardcoded. The trained models are saved with **`joblib`** to
> `models/ml/`.

Training is handled by `scripts/train_ml.py` and includes 5-fold stratified
cross-validation.

---

## Deep Learning

A PyTorch **MLP** architected as specified:

```
Input → Linear → ReLU → Dropout → Linear → ReLU → Dropout → Linear → Sigmoid
```

- Real `Dataset → Tensor → model` loaders (built directly from numpy arrays).
- Adam optimiser + `BCELoss`.
- **Early stopping** (patience 25) and **best-checkpoint** restoration.
- Training / validation / test loops with loss & AUC tracked per epoch.
- Checkpoint saved with `torch.save` to `models/dl/mlp.pt`.

See `scripts/train_dl.py` and `notebooks/04_deep_learning.ipynb`.

---

## Explainable AI

- **SHAP** via the XGBoost surrogate (`TreeExplainer`).
- **Local**: per-patient feature contributions, converted to additive
  **probability-space** shifts (so `+0.21` means "increased risk by ~0.21").
- **Global**: mean absolute SHAP across a training background sample.
- Every prediction page shows **"What influenced the prediction?"** with the
  top 5 features and a plain-language summary.

Example (illustrative):
```
Glucose     ████████████████  +0.31
BMI         ███████████       +0.18
Age         ███████           +0.12
```
> No treatment/medical recommendations are ever generated.

---

## Dataset

The public **Pima Indians Diabetes Dataset** (768 samples, 8 features +
binary `Outcome`). See [`data/README.md`](data/README.md) for source, schema,
preprocessing, and a manual-placement path for offline setup.

---

## Installation

### Prerequisites

- **Python 3.10+** (developed on 3.13)
- **Node.js 18+** and **npm**
- Recommended: a virtual environment

### 1. Backend

```bash
cd diapredict

# Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Frontend

```bash
cd frontend
npm install
```

---

## Training models

From the project root (with the venv active):

```bash
python scripts/train_all.py
```

Or train each in turn:

```bash
python scripts/train_ml.py   # trains Logistic Regression, Random Forest, XGBoost
python scripts/train_dl.py   # trains the PyTorch neural network
```

On success you'll find:

```
models/ml/logistic_regression.joblib
models/ml/random_forest.joblib
models/ml/xgboost.joblib
models/dl/mlp.pt
models/meta.json
data/processed/X_train.npy, scaler.joblib, medians.json …
```

> If `data/raw/` is empty and you're offline, place the dataset there
> (see `data/README.md`).

---

## Running backend

```bash
cd diapredict
# with the venv active
uvicorn backend.app.main:app --reload
```

- API base: `http://localhost:8000`
- Interactive Swagger docs: `http://localhost:8000/docs`

---

## Running frontend

```bash
cd diapredict/frontend
npm run dev
```

- App: `http://localhost:5173`
- The frontend calls the API through the Vite `/api` proxy, so no CORS setup
  is needed locally.

> **Note:** if the frontend and backend must run on different hosts (or you
> deploy a static build), set `VITE_API_URL` to the API URL (e.g.
> `http://localhost:8000`) and, for deployments to a different origin, allow
> that origin in `backend/app/main.py`.

---

## API documentation

The full, generated OpenAPI spec is at **`/docs`**. Key endpoints:

| Method | Path | Description |
|--------|------|-------------|
| POST | `/predict` | Full assessment (all models + ensemble + SHAP) |
| POST | `/predict/ml` | Machine-learning models only |
| POST | `/predict/dl` | Neural network only |
| POST | `/predict/ensemble` | Weighted ensemble only |
| POST | `/predict/explain` | SHAP local explanation for a given input |
| GET | `/models/performance` | Real test-set metrics per model |
| GET | `/models/status` | Which models are loaded/ready |
| GET | `/health` | Liveness probe |

See [`docs/API.md`](docs/API.md) for full schemas and examples.

---

## Testing

```bash
cd diapredict
python -m pytest tests/ -v
```

Covered: preprocessing (missing values, imputation, scaling, leak-free
split), model loading & inference, ensemble calculation, risk-category
thresholds, input validation, and all API endpoints.

---

## Project structure

```
diapredict/
├── backend/
│   └── app/
│       ├── main.py            # FastAPI app + CORS
│       ├── config.py          # paths, feature spec, risk bands, disclaimer
│       ├── schemas.py         # Pydantic models (input/response)
│       ├── preprocessing.py   # cleaning, imputation, scaling, split
│       ├── model_loader.py    # MLP + model registry (joblib / torch)
│       ├── inference.py       # run all loaded models
│       ├── ensemble.py        # ML + DL weighted ensemble & risk bands
│       ├── explainability.py  # SHAP local/global
│       ├── services.py        # orchestration of a full assessment
│       ├── database.py        # SQLite anonymous persistence
│       └── routers/           # predict, models, health
├── scripts/
│   ├── train_all.py           # train everything
│   ├── train_ml.py            # 3 ML models
│   ├── train_dl.py            # PyTorch MLP + early stopping
│   ├── build_notebooks.py     # regenerate notebooks/
│   └── publish_to_github.*    # create repo + push (sh / ps1)
├── .github/workflows/ci.yml   # GitHub Actions CI (tests + frontend build)
├── notebooks/                 # 6 runnable Jupyter notebooks
├── training/
│   └── evaluation.py          # metrics + meta.json persistence
├── tests/                     # pytest suite
├── models/                    # trained artifacts (gitignored)
├── data/                      # raw/processed data + README + SQLite
├── docs/                      # ARCHITECTURE.md, API.md
├── frontend/
│   └── src/
│       ├── api/client.ts      # typed API client
│       ├── components/        # Layout, StatCard, Results
│       ├── pages/             # Dashboard, Assessment, Models, About
│       ├── types/             # shared TS types
│       └── lib/config.ts      # API base URL / disclaimer
├── requirements.txt
└── README.md
```

---

## Limitations

- **Small, historical dataset** (768 samples) — benchmark-only.
- **Narrow population** (Pima Indian women, 21+) — does **not** generalise.
- **Class imbalance** — precision/recall/F1/AUC are more informative than
  accuracy.
- **Zero-as-missing assumption** and median imputation are pragmatic choices.
- Performance numbers are static snapshots from one seed; they will vary with
  retraining.
- This is **not** a clinical device or diagnostic tool.

---

## Ethical considerations

- **Not a diagnosis** — the system explicitly disclaims any diagnostic intent
  and never recommends treatments.
- **No PII** — only anonymous, auto-generated assessment ids are stored.
- **Explanation-first** — every prediction is accompanied by a SHAP
  explanation so users understand why the model produced it.
- **Risk categories** are clearly labelled as *model-defined*, not clinical
  thresholds.
- **Transparency** — metrics, architecture and limitations are fully
  documented; nothing is hardcoded or hidden.
- A qualified healthcare professional should always interpret any such model
  output.

---

## Publishing to GitHub

The repository is already initialised on the `main` branch with a clean commit
and a `.gitignore` that excludes all regenerable artifacts (models, datasets,
SQLite DB, `node_modules`, build output). The source + notebooks + docs + a
GitHub Actions CI workflow are tracked.

To publish, run the bundled helper (it creates the repo, sets the `origin` and
pushes `main`). Default name `diapredict`, default visibility `public`:

**Windows (PowerShell):**
```powershell
# One-time auth (choose either):  GitHub CLI  or  Personal Access Token
winget install GitHub.cli
gh auth login

.\scripts\publish_to_github.ps1                # diapredict / public
.\scripts\publish_to_github.ps1 diapredict private
```

**macOS / Linux / Git Bash:**
```bash
brew install gh        # or install from https://cli.github.com
gh auth login

./scripts/publish_to_github.sh                 # diapredict / public
./scripts/publish_to_github.sh diapredict private
```

**Manual (no CLI):**
1. Create an empty repo on [github.com](https://github.com/new) named
   `diapredict` (public).
2. Generate a [personal access token](https://github.com/settings/tokens) with
   the `repo` scope.
3. Push:
   ```bash
   git remote add origin https://github.com/<YOUR_USERNAME>/diapredict.git
   git push -u origin main
   ```
   (or, if using the token directly:
   `git push -u https://<YOUR_TOKEN>@github.com/<YOUR_USERNAME>/diapredict.git main`)

> Note: the automated CI on GitHub will download the dataset and train all
> four models itself, so you don't need to commit the trained artifacts.

---

## Future improvements

- **Cross-validation** across multiple seeds and report confidence intervals.
- **More features / richer data** (e.g. HbA1c, lifestyle, lab results).
- **Calibration** (Platt / isotonic) and Brier score reporting.
- **Fairness analysis** across demographic subgroups.
- **Hyperparameter search** (Optuna) for the ML and DL models.
- **Deployed, private inference** with strict authentication and audit logs.
- **Real-time monitoring** of model drift.
- **Interactive SHAP force plots / waterfall charts** in the UI.

---

## License / Disclaimer

Released for **educational and research** use. The software is provided “as
is”, without warranty. **DiaPredict AI is not a medical diagnosis and must not
be used to make medical decisions. Always consult a qualified healthcare
professional for medical advice.**
