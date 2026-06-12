import json
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_PATH,
    EXPERIMENT_NAME,
    METRICS_OUTPUT_PATH,
    MODEL_OUTPUT_PATH,
    N_ESTIMATORS,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.data.load_data import load_dataset
from src.evaluation.metrics import evaluate_model,get_confusion_matrix
from src.features.preprocessing import build_preprocessor, split_features_target
from src.models.train_model import build_dummy_model, build_model, build_pipeline, train_pipeline
from src.tracking.mlflow_utils import configure_mlflow, log_training_run


def save_metrics(metrics: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)


def main() -> None:
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = load_dataset(DATA_PATH)
    X, y = split_features_target(df, target_col=TARGET_COLUMN)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = build_preprocessor(X)
    model = build_model(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
    dummy_model = build_dummy_model(strategy="most_frequent")
    pipeline = build_pipeline(preprocessor, model)
    pipeline_dummy = build_pipeline(preprocessor, dummy_model)
    trained_pipeline = train_pipeline(pipeline, X_train, y_train)
    trained_pipeline_dummy = train_pipeline(pipeline_dummy, X_train, y_train)

    metrics = evaluate_model(trained_pipeline, X_test, y_test)
    dummy_metrics = evaluate_model(trained_pipeline_dummy, X_test, y_test)

    joblib.dump(trained_pipeline, MODEL_OUTPUT_PATH)
    save_metrics(metrics, METRICS_OUTPUT_PATH)

    tracking_uri = configure_mlflow(EXPERIMENT_NAME)
    run_id = log_training_run(
        trained_pipeline,
        params={
            "model_type": "RandomForestClassifier",
            "n_estimators": N_ESTIMATORS,
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "target_column": TARGET_COLUMN,
        },
        metrics=metrics,
    )
    
    
    dummy_model_path = MODEL_OUTPUT_PATH.parent / "dummy_model.joblib"
    dummy_metrics_path = METRICS_OUTPUT_PATH.parent / "dummy_metrics.json"

    joblib.dump(trained_pipeline_dummy, dummy_model_path)
    save_metrics(dummy_metrics, dummy_metrics_path)
    
    tracking_uri_dummy = configure_mlflow(EXPERIMENT_NAME)
    run_id_dummy = log_training_run(
        trained_pipeline_dummy,
        params={
            "model_type": "DummyClassifier",
            "strategy": "most_frequent",
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "target_column": TARGET_COLUMN,
        },
        metrics=dummy_metrics,
    )
    

    print(f"Tracking URI: {tracking_uri}")
    print(f"MLflow run ID: {run_id}")
    print("Evaluation metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    print(f"Saved model: {MODEL_OUTPUT_PATH}")
    print(f"Saved metrics: {METRICS_OUTPUT_PATH}")
    
    print(f"Tracking URI: {tracking_uri_dummy}")
    print(f"MLflow run ID: {run_id_dummy}")
    print("Dummy Model Evaluation metrics:")
    for name, value in dummy_metrics.items():
        print(f"  {name}: {value:.4f}")
    print(f"Saved Dummy model: {dummy_model_path}")
    print(f"Saved Dummy Model metrics: {dummy_metrics_path}")

    print("Confusion Matrix for Random Forest:")
    print(get_confusion_matrix(trained_pipeline, X_test, y_test))

if __name__ == "__main__":
    main()

