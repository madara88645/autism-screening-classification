# Autism Screening Classification

A clean, beginner-friendly machine learning project that predicts autism screening outcomes from questionnaire and demographic features.

This repository started as a Kaggle notebook and was refactored into a more structured, GitHub-ready project with reusable Python modules, a training script, a portfolio notebook, local model saving, and MLflow experiment tracking.

## Why This Project Matters

This project is designed to show practical machine learning workflow skills in a simple and honest way:

- turning a notebook into a structured repository
- cleaning and preprocessing real tabular data
- training a baseline classification model with scikit-learn
- evaluating the model with standard classification metrics
- saving artifacts locally
- logging experiments with MLflow and optional DagsHub integration

It is intentionally not presented as a production system. The goal is clarity, reproducibility, and strong project presentation.

## Project Objective

Build an end-to-end binary classification pipeline that predicts the `Class/ASD` label from the `Autism.csv` dataset while keeping the code easy to read, easy to run, and suitable for a public portfolio.

## Dataset

- File: `data/Autism.csv`
- Target column: `Class/ASD`
- Source: dataset used in the original Kaggle notebook workflow

Notes:

- The raw CSV is stored unchanged in the repository.
- Data cleaning happens in code for reproducibility.
- The original dataset column name `contry_of_res` is preserved to match the source data exactly.
- The `age_desc` column is dropped during preprocessing because it is constant and does not add useful signal.

## What The Notebook Is For

The notebook in `notebooks/autism_mlops_portfolio.ipynb` is the portfolio version of the project.

It follows a clean top-to-bottom flow:

1. imports
2. config
3. data loading
4. quick EDA
5. preprocessing
6. train/test split
7. model training
8. evaluation
9. MLflow logging
10. next steps

The notebook reuses the same helper functions as `train.py`, so the notebook and script stay aligned instead of becoming two separate implementations.

## Repository Structure

```text
autismmlops/
|- data/
|  |- Autism.csv
|- models/
|- notebooks/
|  |- autism_mlops_portfolio.ipynb
|- outputs/
|- src/
|  |- config.py
|  |- data/
|  |  |- load_data.py
|  |- evaluation/
|  |  |- metrics.py
|  |- features/
|  |  |- preprocessing.py
|  |- models/
|  |  |- train_model.py
|  |- tracking/
|     |- mlflow_utils.py
|- tests/
|  |- test_pipeline.py
|- .gitignore
|- LICENSE
|- README.md
|- requirements.txt
`- train.py
```

## Tech Stack

- Python
- pandas
- numpy
- scikit-learn
- MLflow
- joblib
- pytest
- Jupyter Notebook

## How The Training Pipeline Works

`train.py` performs the following steps:

1. load `data/Autism.csv`
2. replace `?` values with missing values
3. clean quoted text values and whitespace
4. drop `age_desc`
5. map `Class/ASD` from `NO/YES` to `0/1`
6. split features and target
7. build a preprocessing pipeline
8. train a baseline `RandomForestClassifier`
9. evaluate the model with accuracy, precision, recall, and f1
10. save the trained pipeline and metrics locally
11. log parameters, metrics, and model artifacts to MLflow

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run Locally

```powershell
python train.py
```

This will create:

- `models/autism_rf_pipeline.joblib`
- `outputs/metrics.json`
- a local MLflow run in `mlruns/` if DagsHub credentials are not set

## Connect DagsHub MLflow

This project supports optional remote MLflow tracking with DagsHub.

Set these environment variables before running the training script:

```powershell
$env:DAGSHUB_REPO_OWNER="<your-owner>"
$env:DAGSHUB_REPO_NAME="<your-repo>"
$env:DAGSHUB_USERNAME="<your-dagshub-username>"
$env:DAGSHUB_TOKEN="<your-dagshub-token>"
python train.py
```

Behavior:

- If all four variables are available, MLflow uses `https://dagshub.com/<owner>/<repo>.mlflow`
- If they are missing, the project falls back to local tracking

## Run Tests

```powershell
pytest -q
```

## Open The Notebook

```powershell
jupyter notebook
```

Then open:

- `notebooks/autism_mlops_portfolio.ipynb`

## What This Repository Demonstrates

- ML project cleanup and refactoring
- reusable code structure for tabular ML
- beginner-friendly preprocessing for mixed numeric and categorical data
- experiment tracking with MLflow
- public portfolio presentation without unnecessary infrastructure

## Scope Note

This is a portfolio and learning project.

It does include:

- reusable training code
- saved model artifacts
- test coverage for core pipeline behavior
- optional remote experiment tracking

It does not include:

- model deployment
- API serving
- CI/CD
- orchestration tools
- advanced feature engineering
- hyperparameter tuning workflows

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
