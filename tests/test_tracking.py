import os
from unittest.mock import patch

from src.tracking.mlflow_utils import configure_mlflow


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