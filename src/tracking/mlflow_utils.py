import os

import mlflow
import mlflow.sklearn

from src.config import EXPERIMENT_NAME, LOCAL_MLFLOW_URI


def configure_mlflow(experiment_name: str = EXPERIMENT_NAME) -> str:
    repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
    repo_name = os.getenv("DAGSHUB_REPO_NAME")
    username = os.getenv("DAGSHUB_USERNAME")
    token = os.getenv("DAGSHUB_TOKEN")

    if all([repo_owner, repo_name, username, token]):
        tracking_uri = f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"
        os.environ["MLFLOW_TRACKING_USERNAME"] = username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    else:
        tracking_uri = LOCAL_MLFLOW_URI
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return tracking_uri


def log_training_run(
    pipeline,
    params: dict,
    metrics: dict,
    artifact_path: str = "model",
) -> str:
    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, artifact_path=artifact_path)
        return run.info.run_id
