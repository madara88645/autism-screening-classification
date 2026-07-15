import os
from unittest.mock import MagicMock, patch

import pytest

from src.config import LOCAL_MLFLOW_URI
from src.tracking.mlflow_utils import configure_mlflow, log_training_run


def test_configure_mlflow_uses_dagshub_when_all_env_vars_are_set(monkeypatch):
    monkeypatch.setenv("DAGSHUB_REPO_OWNER", "test-owner")
    monkeypatch.setenv("DAGSHUB_REPO_NAME", "test-repo")
    monkeypatch.setenv("DAGSHUB_USERNAME", "test-user")
    monkeypatch.setenv("DAGSHUB_TOKEN", "test-token")

    expected_uri = "https://dagshub.com/test-owner/test-repo.mlflow"
    experiment_name = "test-experiment"

    with patch(
        "src.tracking.mlflow_utils.mlflow.set_tracking_uri"
    ) as mock_set_tracking_uri, patch(
        "src.tracking.mlflow_utils.mlflow.set_experiment"
    ) as mock_set_experiment:
        tracking_uri = configure_mlflow(experiment_name=experiment_name)

    assert tracking_uri == expected_uri
    assert os.environ["MLFLOW_TRACKING_USERNAME"] == "test-user"
    assert os.environ["MLFLOW_TRACKING_PASSWORD"] == "test-token"
    mock_set_tracking_uri.assert_called_once_with(expected_uri)
    mock_set_experiment.assert_called_once_with(experiment_name)


def test_configure_mlflow_raises_error_when_partial_dagshub_configuration_is_detected(monkeypatch):
    monkeypatch.setenv("DAGSHUB_REPO_OWNER", "test-owner")
    monkeypatch.setenv("DAGSHUB_REPO_NAME", "test-repo")
    monkeypatch.setenv("DAGSHUB_USERNAME", "test-user")
    monkeypatch.delenv("DAGSHUB_TOKEN", raising=False)

    with pytest.raises(ValueError) as error:
        configure_mlflow(experiment_name="test-experiment")
    assert "Partial DagsHub configuration detected" in str(error.value)
    assert "DAGSHUB_TOKEN" in str(error.value)


def test_configure_mlflow_local_fallback_when_no_dagshub_configuration_is_detected(monkeypatch):
    monkeypatch.delenv("DAGSHUB_REPO_OWNER", raising=False)
    monkeypatch.delenv("DAGSHUB_REPO_NAME", raising=False)
    monkeypatch.delenv("DAGSHUB_USERNAME", raising=False)
    monkeypatch.delenv("DAGSHUB_TOKEN", raising=False)
    monkeypatch.delenv("MLFLOW_ALLOW_FILE_STORE", raising=False)

    experiment_name = "test-experiment"

    with patch(
        "src.tracking.mlflow_utils.mlflow.set_tracking_uri"
    ) as mock_set_tracking_uri, patch(
        "src.tracking.mlflow_utils.mlflow.set_experiment"
    ) as mock_set_experiment:
        tracking_uri = configure_mlflow(experiment_name=experiment_name)

    assert tracking_uri == LOCAL_MLFLOW_URI
    assert os.environ["MLFLOW_ALLOW_FILE_STORE"] == "true"
    mock_set_tracking_uri.assert_called_once_with(LOCAL_MLFLOW_URI)
    mock_set_experiment.assert_called_once_with(experiment_name)


def test_log_training_run():
    mock_trained_pipeline = object()
    params = {
        "model_type": "RandomForestClassifier",
        "n_estimators": 100,
        "random_state": 42,
        "test_size": 0.2,
        "target_column": "Class/ASD",
    }
    metrics = {
        "accuracy": 0.9,
        "precision": 0.85,
        "recall": 0.8,
        "f1": 0.82,
        "test_samples": 100,
        "random_state": 42,
        "n_estimators": 100,
        "test_size": 0.2,
    }

    fake_run = MagicMock()
    fake_run.info.run_id = "test-run-id"

    mock_start_run = MagicMock()
    mock_start_run.__enter__.return_value = fake_run
    mock_start_run.__exit__.return_value = None

    with patch(
        "src.tracking.mlflow_utils.mlflow.start_run",
        return_value=mock_start_run,
    ), patch(
        "src.tracking.mlflow_utils.mlflow.log_params",
    ) as mock_log_params, patch(
        "src.tracking.mlflow_utils.mlflow.log_metrics",
    ) as mock_log_metrics, patch(
        "src.tracking.mlflow_utils.mlflow.sklearn.log_model",
    ) as mock_log_model:
        result = log_training_run(mock_trained_pipeline, params, metrics)

    assert result == "test-run-id"
    mock_log_params.assert_called_once_with(params)
    mock_log_metrics.assert_called_once_with(metrics)
    mock_log_model.assert_called_once_with(
        mock_trained_pipeline,
        name="model",
    )
