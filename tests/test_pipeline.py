from pathlib import Path

import pandas as pd

from src.data.load_data import clean_raw_dataframe, load_dataset
from src.evaluation.metrics import evaluate_model
from src.features.preprocessing import build_preprocessor, split_features_target
from src.models.train_model import build_model, build_pipeline, train_pipeline
from src.tracking.mlflow_utils import configure_mlflow


def test_clean_raw_dataframe_handles_missing_quotes_and_target_mapping():
    raw = pd.DataFrame(
        {
            "A1_Score": [1, 0],
            "age": ["26", "?"],
            "ethnicity": ["'Middle Eastern '", "?"],
            "relation": ["'Health care professional'", "?"],
            "age_desc": ["'18 and more'", "'18 and more'"],
            "Class/ASD": ["YES", "NO"],
        }
    )

    cleaned = clean_raw_dataframe(raw)

    assert "age_desc" not in cleaned.columns
    assert cleaned["Class/ASD"].tolist() == [1, 0]
    assert cleaned.loc[0, "ethnicity"] == "Middle Eastern"
    assert pd.isna(cleaned.loc[1, "ethnicity"])
    assert pd.isna(cleaned.loc[1, "age"])
    assert cleaned.loc[0, "relation"] == "Health care professional"


def test_training_pipeline_runs_end_to_end_on_dataset():
    dataset_path = Path("data/Autism.csv")

    df = load_dataset(dataset_path)
    X, y = split_features_target(df)
    preprocessor = build_preprocessor(X)
    model = build_model()
    pipeline = build_pipeline(preprocessor, model)

    X_train = X.iloc[:500]
    X_test = X.iloc[500:]
    y_train = y.iloc[:500]
    y_test = y.iloc[500:]

    trained_pipeline = train_pipeline(pipeline, X_train, y_train)
    metrics = evaluate_model(trained_pipeline, X_test, y_test)

    assert set(metrics) == {"accuracy", "precision", "recall", "f1"}
    for value in metrics.values():
        assert 0.0 <= value <= 1.0


def test_configure_mlflow_uses_file_uri_without_dagshub_credentials(monkeypatch):
    monkeypatch.delenv("DAGSHUB_REPO_OWNER", raising=False)
    monkeypatch.delenv("DAGSHUB_REPO_NAME", raising=False)
    monkeypatch.delenv("DAGSHUB_USERNAME", raising=False)
    monkeypatch.delenv("DAGSHUB_TOKEN", raising=False)

    tracking_uri = configure_mlflow()

    assert tracking_uri.startswith("file:///")
