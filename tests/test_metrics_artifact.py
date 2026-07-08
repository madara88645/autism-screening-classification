from datetime import datetime

from train import add_artifact_info, build_metrics_report

from src.config import (
    EXPERIMENT_NAME,
    METRICS_OUTPUT_PATH,
    MODEL_OUTPUT_PATH,
    N_ESTIMATORS,
    RANDOM_STATE,
    TEST_SIZE,
)


class FakeModel:
    pass


def test_metrics_artifact_contains_expected_evaluation_config_and_run_metadata():
    evaluation_metrics = {
        "accuracy": 0.95,
        "precision": 0.9,
        "recall": 0.85,
        "f1": 0.87,
        "confusion_matrix": {
            "tn": 10,
            "fp": 1,
            "fn": 2,
            "tp": 8,
        },
    }

    metrics = build_metrics_report(
        evaluation_metrics=evaluation_metrics,
        model=FakeModel(),
        test_samples=21,
    )
    metrics = add_artifact_info(
        metrics=metrics,
        run_id="test-run-id",
        tracking_uri="file:///tmp/mlruns",
    )

    expected_keys = {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "confusion_matrix",
        "model_type",
        "test_size",
        "random_state",
        "n_estimators",
        "test_samples",
        "run_id",
        "tracking_uri",
        "timestamp",
        "experiment_name",
        "saved_model",
        "saved_metrics",
    }

    assert set(metrics) == expected_keys
    assert metrics["confusion_matrix"] == {
        "tn": 10,
        "fp": 1,
        "fn": 2,
        "tp": 8,
    }
    assert metrics["model_type"] == "FakeModel"
    assert metrics["test_size"] == TEST_SIZE
    assert metrics["random_state"] == RANDOM_STATE
    assert metrics["n_estimators"] == N_ESTIMATORS
    assert metrics["test_samples"] == 21
    assert metrics["run_id"] == "test-run-id"
    assert metrics["tracking_uri"] == "file:///tmp/mlruns"
    assert metrics["experiment_name"] == EXPERIMENT_NAME
    assert metrics["saved_model"] == str(MODEL_OUTPUT_PATH)
    assert metrics["saved_metrics"] == str(METRICS_OUTPUT_PATH)
    assert isinstance(metrics["timestamp"], str)
    assert datetime.fromisoformat(metrics["timestamp"])
