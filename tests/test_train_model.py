from sklearn.ensemble import RandomForestClassifier

from src.config import N_ESTIMATORS, RANDOM_STATE
from src.models.train_model import build_model


def test_build_model_returns_configured_random_forest_classifier():
    model = build_model()

    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == N_ESTIMATORS
    assert model.random_state == RANDOM_STATE