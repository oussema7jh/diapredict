"""
DiaPredict AI — notebook builder.

Generates the runnable Jupyter notebooks under ``notebooks/``. Each notebook
contains real code (importable from the project) plus explanations and
visualisations. Run this script once to (re)generate them:

    python scripts/build_notebooks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

NB_DIR = PROJECT_ROOT / "notebooks"
NB_DIR.mkdir(parents=True, exist_ok=True)

CODE_LANG = "python"


def make_notebook(cells: list[tuple[str, str]], title: str, description: str) -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.13"},
        "title": title,
    }
    nb.cells = []
    nb.cells.append(
        nbf.v4.new_markdown_cell(
            f"# {title}\n\n{description}\n\n---\n\n"
            f"*DiaPredict AI — Explainable Diabetes Risk Prediction System.* "
            f"*Educational / research prototype, not a medical diagnosis.*"
        )
    )
    for kind, source in cells:
        if kind == "md":
            nb.cells.append(nbf.v4.new_markdown_cell(source))
        else:
            nb.cells.append(nbf.v4.new_code_cell(source))
    nbf.write(nb, NB_DIR / f"{title.split('.')[0].strip()}.ipynb")


# Common preamble that adds the project root to sys.path so imports work.
PREAMBLE = """\
import sys
from pathlib import Path

# Make the project root importable no matter where the notebook runs.
PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "backend").exists():
    PROJECT_ROOT = Path.cwd().parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110
print("Root:", PROJECT_ROOT)
"""

# --------------------------------------------------------------------------- #
# 01 data exploration
# --------------------------------------------------------------------------- #
cells_01 = [
    ("md", "## Objectives\n\nIn this notebook we explore the Pima Indians Diabetes dataset:\n- Load the raw data\n- Understand the schema and summary statistics\n- Visualise feature distributions and class balance\n- Motivate the preprocessing decisions used later (missing-value handling)."),
    ("code", PREAMBLE),
    ("md", "### 1. Load the dataset"),
    ("code", """\
from backend.app.preprocessing import load_raw_dataset
from backend.app.config import FEATURES, TARGET

df = load_raw_dataset()
print(f"Samples: {len(df)}, features: {len(FEATURES)}")
df.head()"""),
    ("md", "### 2. Summary statistics"),
    ("code", """\
df.describe().T"""),
    ("md", "\n> Notice that Glucose, BloodPressure, SkinThickness, Insulin and BMI have a minimum of `0`, which is physiologically impossible. In the original Pima dataset, `0` is used as a placeholder for a missing measurement. We handle this in preprocessing.\n"),
    ("md", "### 3. Class balance (target)"),
    ("code", """\
print(df[TARGET].value_counts())
df[TARGET].value_counts(normalize=True).plot(kind="bar", color=["#60a5fa", "#f87171"])
plt.title("Outcome distribution (0 = no diabetes, 1 = diabetes)")
plt.ylabel("Proportion")
plt.show()"""),
    ("md", "### 4. Feature distributions by class"),
    ("code", """\
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
for ax, col in zip(axes.ravel(), FEATURES):
    sns.histplot(data=df, x=col, hue=TARGET, kde=True, ax=ax, palette=["#60a5fa", "#f87171"])
    ax.set_title(col)
for ax in axes.ravel()[len(FEATURES):]:
    ax.axis("off")
plt.tight_layout()
plt.show()"""),
    ("md", "### 5. Correlation heat map"),
    ("code", """\
plt.figure(figsize=(11, 9))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar_kws={"shrink": .8})
plt.title("Feature correlation matrix")
plt.show()"""),
    ("md", "### 6. Observations\n\n- `Glucose` has the strongest single correlation with the outcome.\n- The dataset is **imbalanced** (roughly 35% positive), which is why evaluating precision / recall / ROC-AUC matters more than raw accuracy.\n- Several columns contain placeholder zeros that must be treated as missing values."),
]
make_notebook(cells_01, "01_data_exploration", "Load, inspect and visualise the Pima Indians Diabetes dataset.")

# --------------------------------------------------------------------------- #
# 02 preprocessing
# --------------------------------------------------------------------------- #
cells_02 = [
    ("md", "## Objectives\n- Demonstrate the full preprocessing pipeline\n- Convert physiologically impossible zeros to missing values\n- Impute with the training median and standardise features\n- Show how the transforms avoid data leakage"),
    ("code", PREAMBLE),
    ("md", "### 1. Cleaning — impossible zeros → NaN"),
    ("code", """\
import pandas as pd
from backend.app.preprocessing import clean_dataset, load_raw_dataset

raw = load_raw_dataset()
cleaned = clean_dataset(raw)
print(raw[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]].min())
print()
print("Rows with at least one missing value after cleaning:", cleaned.isnull().any(axis=1).sum())"""),
    ("md", "### 2. Missing-value imputation (median)"),
    ("code", """\
from backend.app.preprocessing import impute_median

imputed, medians = impute_median(cleaned)
print("Median used for imputation / feature:")
for k, v in medians.items():
    print(f"  {k:26s} {v:.2f}")
print("\\nRemaining NaNs:", imputed.isnull().sum().sum())"""),
    ("md", "### 3. Train / test split + scaling"),
    ("code", """\
from backend.app.preprocessing import make_train_test

X_train, y_train, X_test, y_test, scaler, medians = make_train_test(raw)
print("Train:", X_train.shape, "Test:", X_test.shape)
print("Train positives:", y_train.sum(), "/", len(y_train))
print("Test positives: ", y_test.sum(), "/", len(y_test))
print("Mean (train, scaled):", np.round(X_train.mean(axis=0), 2))
print("Std  (train, scaled):", np.round(X_train.std(axis=0), 2))"""),
    ("md", "### 4. Standardisation visualisation"),
    ("code", """\
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.hist(raw["Glucose"], bins=30, color="#60a5fa", edgecolor="white")
plt.title("Raw Glucose")
plt.subplot(1, 2, 2)
plt.hist(X_train[:, 1], bins=30, color="#f87171", edgecolor="white")
plt.title("Scaled Glucose (train split)")
plt.tight_layout()
plt.show()"""),
    ("md", "### 5. Why the scaler/medians must be fitted on training only\n\nThe `StandardScaler` and the imputation medians are computed **only on the training split** and reused verbatim on the test split / at inference time. This prevents **data leakage** — information from the test set must never influence how training data is scaled, otherwise the test metrics are optimistically biased."),
    ("md", "## Summary\n\nPreprocessing closes the loop from a messy raw dataset to a clean, standardised feature matrix ready for model training. The same transforms are replayed on a single patient record in the backend (`backend/app/preprocessing.py`, `prepare_feature_vector` in `services.py`)."),
]
make_notebook(cells_02, "02_preprocessing", "Missing-value handling, imputation, train/test split and feature scaling.")

# --------------------------------------------------------------------------- #
# 03 machine learning
# --------------------------------------------------------------------------- #
cells_03 = [
    ("md", "## Objectives\n- Train Logistic Regression, Random Forest and XGBoost\n- Evaluate with cross-validation and a held-out test set\n- Compare the models and inspect feature importance"),
    ("code", PREAMBLE),
    ("md", "### 1. Prepare data"),
    ("code", """\
from backend.app.preprocessing import load_raw_dataset, make_train_test

raw = load_raw_dataset()
X_train, y_train, X_test, y_test, scaler, medians = make_train_test(raw)
print(X_train.shape, X_test.shape)"""),
    ("md", "### 2. Train all three models"),
    ("code", """\
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced"),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42, n_estimators=300, learning_rate=0.05, max_depth=4),
}
for name, model in models.items():
    model.fit(X_train, y_train)
print("Training complete.")"""),
    ("md", "### 3. Evaluate on the held-out test set"),
    ("code", """\
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

results = []
for name, model in models.items():
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, proba),
    })
pd.DataFrame(results).round(3)"""),
    ("md", "### 4. Cross-validation (5-fold ROC-AUC)"),
    ("code", """\
from sklearn.model_selection import StratifiedKFold, cross_val_score

for name, model in models.items():
    cv = cross_val_score(model, X_train, y_train, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="roc_auc", n_jobs=-1)
    print(f"{name:20s} CV ROC-AUC: {cv.mean():.4f} +/- {cv.std():.4f}")"""),
    ("md", "### 5. Feature importance"),
    ("code", """\
from backend.app.config import FEATURES

rf = models["Random Forest"]
imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
imp.plot(kind="barh", color="#60a5fa")
plt.title("Random Forest feature importance")
plt.xlabel("Mean decrease in impurity")
plt.show()"""),
    ("md", "### 6. Confusion matrices"),
    ("code", """\
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, (name, model) in zip(axes, models.items()):
    pred = model.predict(X_test)
    sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
    ax.set_title(f"{name}\\nAccuracy {accuracy_score(y_test, pred):.3f}")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout()
plt.show()"""),
    ("md", "## Summary\n\nAll three models train in seconds and reach ROC-AUC in the 0.78-0.82 range on the test set. Random Forest and Logistic Regression are the strongest here; the full comparison (including the neural network) lives in `05_model_comparison`."),
]
make_notebook(cells_03, "03_machine_learning", "Train and evaluate the classical machine-learning models.")

# --------------------------------------------------------------------------- #
# 04 deep learning
# --------------------------------------------------------------------------- #
cells_04 = [
    ("md", "## Objectives\n- Build the PyTorch MLP\n- Train with a validation loop, Adam and early stopping\n- Inspect the training/validation curves and test-set metrics"),
    ("code", PREAMBLE),
    ("md", "### 1. Prepare data"),
    ("code", """\
from sklearn.model_selection import train_test_split
from backend.app.preprocessing import load_raw_dataset, make_train_test

raw = load_raw_dataset()
X_train, y_train, X_test, y_test, scaler, medians = make_train_test(raw)
X_tr, X_va, y_tr, y_va = train_test_split(X_train, y_train, test_size=0.20, random_state=42, stratify=y_train)
print("Train:", X_tr.shape, "Val:", X_va.shape, "Test:", X_test.shape)"""),
    ("md", "### 2. Build the network"),
    ("code", """\
import torch, torch.nn as nn
from backend.app.model_loader import MLP

net = MLP(input_dim=X_tr.shape[1])
print(net)
print("\\nTotal parameters:", sum(p.numel() for p in net.parameters()))"""),
    ("md", "### 3. Training loop with early stopping"),
    ("code", """\
from scripts.train_dl import Trainer

trainer = Trainer(input_dim=X_tr.shape[1], lr=1e-3, epochs=200, patience=25, batch_size=64)
trainer.fit(X_tr, y_tr, X_va, y_va)"""),
    ("md", "### 4. Training / validation curves"),
    ("code", """\
h = trainer.history
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].plot(h["train_loss"], label="train loss", color="#60a5fa")
axes[0].plot(h["val_loss"], label="val loss", color="#f87171")
axes[0].set_title("Loss curves"); axes[0].set_xlabel("Epoch"); axes[0].legend()
axes[1].plot(h["val_auc"], label="val ROC-AUC", color="#34d399")
axes[1].set_title("Validation AUC"); axes[1].set_xlabel("Epoch"); axes[1].legend()
plt.tight_layout(); plt.show()"""),
    ("md", "### 5. Test-set evaluation"),
    ("code", """\
metrics = trainer.evaluate(X_test, y_test)
for k, v in metrics.items():
    print(f"{k:20s} {v}")"""),
    ("md", "## Notes\n\nThe network uses two hidden layers (24 → 16 units) with ReLU activations, dropout regularisation and a sigmoid output. Early stopping picks the checkpoint with the best validation AUC, so the final weights are not the last epoch's — this guards against overfitting on this small dataset."),
]
make_notebook(cells_04, "04_deep_learning", "Train the PyTorch fully-connected deep network.")

# --------------------------------------------------------------------------- #
# 05 model comparison
# --------------------------------------------------------------------------- #
cells_05 = [
    ("md", "## Objectives\n- Compare all four models on the same test set\n- Plot ROC curves, precision-recall curves and confusion matrices\n- Identify the best-performing model"),
    ("code", PREAMBLE),
    ("md", "### 1. Prepare data and load trained models"),
    ("code", """\
from backend.app.preprocessing import load_raw_dataset, make_train_test
from backend.app.inference import ml_predict_proba
from backend.app.model_loader import get_registry

raw = load_raw_dataset()
X_train, y_train, X_test, y_test, scaler, medians = make_train_test(raw)

registry = get_registry()
print([m.model_id for m in registry.loaded()])"""),
    ("md", "### 2. Collect predictions and metrics"),
    ("code", """\
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

probs, labels, names = [], [], []
for m in registry.loaded():
    if m.kind == "dl":
        import torch
        x = torch.tensor(X_test, dtype=torch.float32)
        m.model.eval()
        with torch.no_grad():
            p = m.model(x).numpy()
    else:
        p = m.model.predict_proba(X_test)[:, 1]
    probs.append(p); labels.append(m.model_name); names.append(m.model_id)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
rows = []
for name, p in zip(names, probs):
    pred = (p >= 0.5).astype(int)
    rows.append({
        "model_id": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, p),
    })
pd.DataFrame(rows).set_index("model_id").round(3)"""),
    ("md", "### 3. ROC curves"),
    ("code", """\
plt.figure(figsize=(8, 6))
for name, p in zip(labels, probs):
    fpr, tpr, _ = roc_curve(y_test, p)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, p):.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.title("ROC curves — held-out test set"); plt.legend(); plt.tight_layout(); plt.show()"""),
    ("md", "### 4. Precision-recall curves"),
    ("code", """\
plt.figure(figsize=(8, 6))
for name, p in zip(labels, probs):
    prec, rec, _ = precision_recall_curve(y_test, p)
    plt.plot(rec, prec, label=f"{name} (AP={average_precision_score(y_test, p):.3f})")
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("Precision-recall curves"); plt.legend(); plt.tight_layout(); plt.show()"""),
    ("md", "## Interpretation\n\nThe ROC-AUC and precision-recall curves let us compare models independently of an arbitrary threshold. Because the dataset is imbalanced, the PR curve is the more informative view of the positive class. The chosen \"best\" model is the one with the best trade-off for a given use case — in our comparison this is typically Random Forest by F1 and Logistic Regression by ROC-AUC."),
]
make_notebook(cells_05, "05_model_comparison", "Compare all models: metrics, ROC and precision-recall curves.")

# --------------------------------------------------------------------------- #
# 06 explainable AI
# --------------------------------------------------------------------------- #
cells_06 = [
    ("md", "## Objectives\n- Load the trained XGBoost surrogate\n- Compute **global** feature importance (mean |SHAP|)\n- Compute a **local** explanation for a single patient\n- Understand what the SHAP values mean"),
    ("code", PREAMBLE),
    ("code", """\
import numpy as np
from backend.app.preprocessing import load_raw_dataset, make_train_test
from backend.app.model_loader import get_registry

raw = load_raw_dataset()
X_train, y_train, X_test, y_test, scaler, medians = make_train_test(raw)
print("Background matrix:", X_train.shape)"""),
    ("md", "### 1. Global feature importance (mean |SHAP|)"),
    ("code", """\
from backend.app.explainability import get_explainer

explainer = get_explainer()
global_imp = explainer.global_feature_importance(X_train[:200])
import pandas as pd
gdf = pd.DataFrame(global_imp)
print(gdf)
gdf.plot(kind="barh", x="label", y="mean_abs_shap", legend=False, color="#60a5fa", figsize=(9, 5))
plt.title("Global feature importance (mean |SHAP|)"); plt.gca().invert_yaxis(); plt.tight_layout(); plt.show()"""),
    ("md", "### 2. Local explanation for one patient"),
    ("code", """\
from backend.app.schemas import PatientInput

patient = PatientInput(
    Pregnancies=6, Glucose=148, BloodPressure=72, SkinThickness=35,
    Insulin=155, BMI=33.6, DiabetesPedigreeFunction=0.627, Age=50)

# Scale the patient's features with the persisted scaler/medians.
from backend.app.services import prepare_feature_vector
X_patient = prepare_feature_vector(patient)
base_prob = 0.71  # illustrative; the backend computes this live

local = explainer.local_explanation(X_patient, base_prob)
print("Summary:", local["summary_english"])
print()
print(f"{'Feature':28s}{'SHAP':>10s}{'Contribution':>14s}")
for b in local["top_features"]:
    print(f"{b['label']:28s}{b['value']:>10.3f}{b['contribution']:>14.3f}")"""),
    ("md", "### 3. Waterfall-style plot of contributions"),
    ("code", """\
bars = local["top_features"]
features = [b["label"] for b in bars][::-1]
contrib = [b["contribution"] for b in bars][::-1]
colors = ["#f87171" if c > 0 else "#60a5fa" for c in contrib]
plt.figure(figsize=(9, 5))
plt.barh(features, contrib, color=colors)
plt.axvline(0, color="black", lw=1)
plt.xlabel("Contribution to predicted probability")
plt.title("What influenced this prediction?")
plt.tight_layout(); plt.show()"""),
    ("md", "### What does a SHAP value mean?\n\nA SHAP value measures the contribution of a feature to the difference between the model's prediction for *this* patient and the model's average prediction. Positive values push the prediction higher (towards diabetes); negative values push it lower. Shown in **probability space**, a value of `+0.2` means that feature increased the predicted probability by about 0.2.\n\n> **Why explain?** Explainability is what makes a medical-risk model auditable and trustworthy. It tells a user which factors drove the estimate, so a clinician can interpret and sanity-check the output — which is a key requirement for any AI system used in a health context."),
]
make_notebook(cells_06, "06_explainable_ai", "SHAP global and local explanations.")

print("Notebooks generated in", NB_DIR)
