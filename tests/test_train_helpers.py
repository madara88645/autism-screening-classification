from src.config import N_ESTIMATORS, RANDOM_STATE, TEST_SIZE
from train import build_metrics_report, get_mlflow_metrics


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
    }

    assert get_mlflow_metrics(metrics) == {
        "accuracy": 0.9,
        "test_samples": 100,
    }