import json

from src.config import (
    EXPERIMENT_NAME,
    METRICS_OUTPUT_PATH,
    MODEL_OUTPUT_PATH,
    N_ESTIMATORS,
    RANDOM_STATE,
    TEST_SIZE,
)
from train import (
    add_artifact_info,
    build_metrics_report,
    get_mlflow_metrics,
    print_training_summary,
    save_metrics,
)


class FakeModel:
    pass


def test_build_metrics_report_adds_model_config_and_test_metadata():
    evaluation_metrics = {
        "accuracy": 0.9,
        "precision": 0.8,
        "recall": 0.7,
        "f1": 0.75,
    }

    metrics = build_metrics_report(
        evaluation_metrics=evaluation_metrics,
        model=FakeModel(),
        test_samples=25,
    )

    assert metrics["accuracy"] == 0.9
    assert metrics["precision"] == 0.8
    assert metrics["recall"] == 0.7
    assert metrics["f1"] == 0.75
    assert metrics["model_type"] == "FakeModel"
    assert metrics["test_size"] == TEST_SIZE
    assert metrics["random_state"] == RANDOM_STATE
    assert metrics["n_estimators"] == N_ESTIMATORS
    assert metrics["test_samples"] == 25
    assert "timestamp" in metrics


def test_get_mlflow_metrics_returns_only_numeric_values():
    metrics = {
        "accuracy": 0.9,
        "test_samples": 100,
        "confusion_matrix": {"tn": 1},
        "model_type": "RandomForestClassifier",
        "model_tree_based": True,
    }

    assert get_mlflow_metrics(metrics) == {
        "accuracy": 0.9,
        "test_samples": 100,
    }
    assert "model_tree_based" not in get_mlflow_metrics(metrics)


def test_add_artifact_info():
    metrics = {}
    metrics["accuracy"] = 0.9
    metrics["precision"] = 0.8
    metrics["recall"] = 0.7
    metrics["f1"] = 0.75

    run_id = "mock_run_id"
    tracking_uri = "file:///tmp/mlruns"

    metrics = add_artifact_info(metrics, run_id=run_id, tracking_uri=tracking_uri)

    assert metrics["run_id"] == "mock_run_id"
    assert metrics["tracking_uri"] == "file:///tmp/mlruns"
    assert metrics["experiment_name"] == str(EXPERIMENT_NAME)
    assert metrics["saved_model"] == str(MODEL_OUTPUT_PATH)
    assert metrics["saved_metrics"] == str(METRICS_OUTPUT_PATH)


def test_print_training_summary(capsys):
    metrics = {
        "accuracy": 0.9,
        "precision": 0.85,
        "recall": 0.8,
        "f1": 0.82,
        "confusion_matrix": {
            "tn": 45,
            "fp": 5,
            "fn": 10,
            "tp": 40,
        },
        "model_type": "RandomForestClassifier",
        "n_estimators": N_ESTIMATORS,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "test_samples": 100,
        "run_id": "mock_run_id",
        "tracking_uri": "file:///tmp/mlruns",
        "timestamp": "2026-07-11T08:00:00",
        "experiment_name": str(EXPERIMENT_NAME),
        "saved_model": str(MODEL_OUTPUT_PATH),
        "saved_metrics": str(METRICS_OUTPUT_PATH),
    }

    print_training_summary(metrics=metrics)
    captured = capsys.readouterr()

    assert "Evaluation metrics:" in captured.out
    assert "true negatives: 45" in captured.out
    assert "false positives: 5" in captured.out
    assert "false negatives: 10" in captured.out
    assert "true positives: 40" in captured.out
    assert "Configuration parameters:" in captured.out
    assert "Artifacts:" in captured.out
    assert "run_id" in captured.out
    assert "saved_model" in captured.out
    assert "tracking_uri" in captured.out


def test_save_metrics(tmp_path):
    output_path = tmp_path / "nested" / "outputs" / "metrics.json"

    metrics = {
        "accuracy": 0.9,
        "test_samples": 100,
        "confusion_matrix": {"tn": 1},
        "model_type": "RandomForestClassifier",
    }

    save_metrics(metrics, output_path)

    with open(output_path, "r", encoding="utf-8") as f:
        loaded_metrics = json.load(f)

    assert output_path.parent.exists()
    assert loaded_metrics == metrics
