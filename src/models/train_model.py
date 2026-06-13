import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.config import N_ESTIMATORS, RANDOM_STATE


def build_model(
    n_estimators: int = N_ESTIMATORS, random_state: int = RANDOM_STATE
) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
    )


def build_pipeline(preprocessor, model) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def train_pipeline(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    pipeline.fit(X_train, y_train)
    return pipeline
