import json
from datetime import datetime
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
from src.evaluation.metrics import evaluate_model, get_confusion_matrix
from src.features.preprocessing import build_preprocessor, split_features_target
from src.models.train_model import build_model, build_pipeline, train_pipeline
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
    pipeline = build_pipeline(preprocessor, model)
    trained_pipeline = train_pipeline(pipeline, X_train, y_train)

    metrics = evaluate_model(trained_pipeline, X_test, y_test)
    metrics["confusion_matrix"] = get_confusion_matrix(trained_pipeline, X_test, y_test)
    metrics["model_type"] = model.__class__.__name__
    metrics["test_size"] = TEST_SIZE
    metrics["timestamp"] = datetime.now().isoformat()
    metrics["random_state"] = RANDOM_STATE
    metrics["n_estimators"] = N_ESTIMATORS
    metrics["test_samples"] = len(X_test)
    mlflow_metrics = {
        name: value
        for name, value in metrics.items()
        if isinstance(value, int | float)
    }
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
        metrics=mlflow_metrics,
    )
    metrics["run_id"] = run_id
    metrics["tracking_uri"] = tracking_uri
    metrics["experiment_name"] = EXPERIMENT_NAME
    metrics["saved_model"] = str(MODEL_OUTPUT_PATH)
    metrics["saved_metrics"] = str(METRICS_OUTPUT_PATH)

    eval_keys = ["accuracy", "precision", "recall", "f1","confusion_matrix"]
    config_keys = ["model_type", "n_estimators", "random_state", "test_size", "test_samples"]
    artifact_keys = ["run_id", "tracking_uri","timestamp","experiment_name","saved_model","saved_metrics"]

    joblib.dump(trained_pipeline, MODEL_OUTPUT_PATH)
    save_metrics(metrics, METRICS_OUTPUT_PATH)

    print("Evaluation metrics:")
    for key in eval_keys:
        if key == "confusion_matrix":
            print(f"true negatives: {metrics[key]['tn']}\nfalse positives: {metrics[key]['fp']}\nfalse negatives: {metrics[key]['fn']}\ntrue positives: {metrics[key]['tp']}")
        else:
            print(f"{key}: {metrics[key]:.4f}")
    print("\n")
    print("Configuration parameters:")
    for key in config_keys:
        print(f"{key}: {metrics[key]}")
    print("\n")
    print("Artifacts:")
    for key in artifact_keys:
        print(f"{key}: {metrics[key]}")



if __name__ == "__main__":
    main()
