# Dataset

This project uses the **Pima Indians Diabetes Dataset**, a widely used public
benchmark for binary classification of diabetes. The data is **de-identified**
and contains only numeric physiological measurements — no names, addresses,
dates of birth or any personally identifiable information.

> **Note on privacy:** even though the source data is de-identified, the app
> never stores raw patient input as PII. Assessment records are stored
> anonymously by numeric, auto-generated `assessment_id` only (see
> `backend/app/database.py`).

---

## Source

- The dataset was originally published by the **National Institute of Diabetes
  and Digestive and Kidney Diseases** (NIDDK) and is available from the UCI
  Machine Learning Repository and Kaggle.
- A copy is available from several open sources; this project can load either
  of these two drops, or any CSV with the standard 8 features + `Outcome`.

## Features (input variables)

| Feature                     | Units             | Description                                          |
|-----------------------------|-------------------|------------------------------------------------------|
| `Pregnancies`               | count             | Number of times pregnant                             |
| `Glucose`                   | mg/dL             | Plasma glucose concentration (2-hour OGTT)           |
| `BloodPressure`             | mmHg              | Diastolic blood pressure                             |
| `SkinThickness`             | mm                | Triceps skin fold thickness                          |
| `Insulin`                   | mu U/mL           | 2-hour serum insulin                                 |
| `BMI`                       | kg/m²             | Body mass index                                      |
| `DiabetesPedigreeFunction`  | score             | Diabetes pedigree function (family history)          |
| `Age`                       | years             | Age                                                  |

## Target (label)

`Outcome` — binary: `1` = tested positive for diabetes, `0` = tested negative.

## Number of samples

- **768** rows total.
- 500 (65.1%) `Outcome = 0`, 268 (34.9%) `Outcome = 1` → class-imbalanced.

## Preprocessing applied

1. **Invalid-value handling**: in the original file `0` is used as a
   placeholder for a *missing* measurement in `Glucose`, `BloodPressure`,
   `SkinThickness`, `Insulin` and `BMI`. These impossible zeros are converted
   to `NaN`. (`Pregnancies = 0` and `Age` are legitimate and left untouched.)
2. **Imputation**: missing values are filled with the **median** computed on
   the *training* split only (robust to skew).
3. **Split**: stratified **75/25** train/test split (`random_state = 42`).
4. **Scaling**: features are standardised (z-score) with a `StandardScaler`
   fitted on the *training* split only, then applied to the test split and to
   every new patient record at inference time.

All of this lives in `backend/app/preprocessing.py`, and the exact same
transforms are replayed on a single incoming patient record by
`backend/app/services.py::prepare_feature_vector`.

## Auto-download

If you have internet access, the data is fetched automatically by the project
(recommended source listed below). `scripts/train_ml.py` calls
`load_raw_dataset()`, which looks in `data/raw/` first.

## Manual placement (no internet)

If internet access is unavailable during setup, drop the dataset CSV into this
folder:

```
data/raw/diabetes.csv
```

The CSV must contain a header row with columns:

```
Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
```

or a no-header version with the same 9 comma-separated columns (the loader
assigns names automatically). Both formats are accepted.

Recommended URL (if online):

```
https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv
```

## Limitations

- **Small sample** (768) — results are benchmark-only, not clinically
  validated.
- **Specific population** (Pima Indian women over 21) — the models do **not**
  generalise to other demographics.
- **Historical data** — collected decades ago; diagnostic criteria and
  population health have changed since.
- **Class imbalance** — accuracy can be misleading; metrics such as precision,
  recall, F1 and ROC-AUC are emphasised.
- **Zero-as-missing** encoding is an assumption; imputation with the median is
  a pragmatic choice, not a measured ground truth.
- This is an **educational prototype**. It cannot and must not be used to make
  medical decisions.
