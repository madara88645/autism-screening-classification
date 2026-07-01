# Autism Screening Classification

A beginner-friendly machine learning and MLOps practice project that predicts an autism screening label from questionnaire and demographic features.

This project is not intended for medical diagnosis. The goal is to demonstrate a clean ML workflow using a simple tabular screening dataset, including preprocessing, model training, evaluation, artifact saving, and MLflow experiment tracking.

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

## Dataset and Cleaning Decisions

- File: `data/Autism.csv`
- Target column: `Class/ASD`
- Source: dataset used in the original Kaggle notebook workflow

Notes:

* The raw CSV file is kept unchanged in the repository so the original source format is preserved.
* Missing values represented as `?` are treated as missing data and handled during preprocessing.
* Text values are cleaned by removing unnecessary quotes and extra whitespace.
* Numeric missing values are imputed using the median, while categorical missing values are imputed using the most frequent value.
* Categorical features are one-hot encoded so they can be used by scikit-learn models.
* The target column `Class/ASD` is mapped from `NO/YES` values to `0/1`.
* The original column name `contry_of_res` is preserved to stay consistent with the source dataset.
* The `age_desc` column is dropped because it is constant and does not add useful predictive signal.

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
autism-screening-classification/
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
|  |- test_data_loading.py
|  |- test_evaluation.py
|  |- test_pipeline.py
|  |- test_tracking.py
|  |- test_train_helpers.py
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

## Why RandomForestClassifier?

RandomForestClassifier was chosen as a strong and practical first baseline for this tabular classification task.

It works well with mixed numeric and one-hot encoded categorical features, does not require heavy feature scaling, and is more stable than a single decision tree because it combines multiple trees.

This choice is not meant to claim that Random Forest is the best possible model. The goal is to use a reliable baseline model while focusing on the full ML and MLOps workflow.

## Evaluation and Baseline Comparison

The model is evaluated with accuracy, precision, recall, f1 score and training also saves a confusion matrix with tn/fp/fn/tp values.

Accuracy alone can be misleading on an imbalanced dataset because a model can achieve a reasonable score by mostly predicting the majority class. For this reason, precision, recall, and f1 are also used to better understand performance on both classes.

The main training script trains and logs the `RandomForestClassifier` pipeline. The portfolio notebook includes an additional `DummyClassifier` with the `most_frequent` strategy as a naive majority-class baseline.

This notebook-only baseline comparison helps interpret the Random Forest results more honestly by showing whether the model is only relying on class imbalance or whether it appears to learn useful feature signals.

## Dataset Limitations

This dataset is small and limited, so the results should be interpreted carefully.

The target class is imbalanced, which means accuracy alone may give an overly optimistic view of performance.

The dataset may not represent all demographic groups equally, so the model may not generalize well to real-world populations.

Near-perfect performance on this dataset does not prove real-world robustness or clinical reliability. The dataset may be relatively easy, limited in diversity, or contain very strong predictive signals. Potential leakage cannot be fully ruled out without deeper validation.

## What This Project Is Not

This project is not a medical diagnosis tool and should not be used to make clinical decisions.

It is not a production machine learning system, and it does not include deployment, API serving, monitoring, or clinical validation.

The results in this repository should be interpreted as a portfolio-level machine learning workflow, not as evidence of real-world medical reliability or robustness.

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

This trains the pipeline, saves local artifacts, and creates an MLflow run.

## Saved Outputs

Running `python train.py` creates:

- `models/autism_rf_pipeline.joblib`
- `outputs/metrics.json`
- a local MLflow run in `mlruns/` if DagsHub credentials are not set

The `metrics.json` file includes:

- evaluation metrics: accuracy, precision, recall, f1
- confusion matrix values: tn, fp, fn, tp
- model/config metadata: model type, test size, random state, number of estimators, test sample count
- run/artifact metadata: timestamp, MLflow run ID, tracking URI, experiment name, saved model path, saved metrics path

## Connect DagsHub MLflow

This project supports optional remote MLflow tracking with DagsHub.

Set these environment variables before running the training script:

The example below uses PowerShell. On macOS/Linux, use your shell’s environment variable syntax.

```powershell
$env:DAGSHUB_REPO_OWNER="<your-owner>"
$env:DAGSHUB_REPO_NAME="<your-repo>"
$env:DAGSHUB_USERNAME="<your-dagshub-username>"
$env:DAGSHUB_TOKEN="<your-dagshub-token>"
python train.py
```

Behavior:

- If all four variables are available, MLflow uses `https://dagshub.com/<owner>/<repo>.mlflow`
- If none of the DagsHub variables are set, the project falls back to local MLflow tracking
- If only some DagsHub variables are set, the project raises a clear error instead of silently falling back to local tracking

## Run Tests

```powershell
pytest -q
```

The tests are organized by responsibility:

- `tests/test_pipeline.py` checks the main pipeline-level behavior.
- `tests/test_data_loading.py` checks raw data cleaning decisions such as missing values, quote/whitespace cleanup, target mapping, and unexpected target labels.
- `tests/test_evaluation.py` checks evaluation helper contracts, including metric keys and confusion matrix values.
- `tests/test_tracking.py` checks MLflow/DagsHub tracking configuration behavior without making real network calls.
- `tests/test_train_helpers.py` checks helper functions used by `train.py`, such as filtering numeric MLflow metrics and building the metrics report.

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
- medical diagnosis
- clinical validation
- real-world healthcare decision making

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
