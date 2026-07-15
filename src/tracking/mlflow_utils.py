import os

import mlflow
import mlflow.sklearn

from src.config import EXPERIMENT_NAME, LOCAL_MLFLOW_URI


def configure_mlflow(experiment_name: str = EXPERIMENT_NAME) -> str:
    repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
    repo_name = os.getenv("DAGSHUB_REPO_NAME")
    username = os.getenv("DAGSHUB_USERNAME")
    token = os.getenv("DAGSHUB_TOKEN")

    dagshub_env_vars = {
        "DAGSHUB_REPO_OWNER": repo_owner,
        "DAGSHUB_REPO_NAME": repo_name,
        "DAGSHUB_USERNAME": username,
        "DAGSHUB_TOKEN": token,
    }

    if all(dagshub_env_vars.values()):
        tracking_uri = f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"
        os.environ["MLFLOW_TRACKING_USERNAME"] = username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    elif any(dagshub_env_vars.values()):
        missing_values = [
            name for name, value in dagshub_env_vars.items() if not value
        ]
        raise ValueError(
            "Partial DagsHub configuration detected. "
            f"Missing environment variables: {missing_values}"
        )
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
        mlflow.sklearn.log_model(pipeline, name=artifact_path)
        return run.info.run_id
