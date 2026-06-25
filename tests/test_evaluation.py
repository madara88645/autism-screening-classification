from src.evaluation.metrics import evaluate_model, get_confusion_matrix


class FakeModel:
    def predict(self, X):
        return [0, 1, 0, 1]


def test_evaluate_model_returns_expected_metric_keys_and_values():
    X_test = ["sample-1", "sample-2", "sample-3", "sample-4"]
    y_test = [0, 0, 1, 1]

    metrics = evaluate_model(FakeModel(), X_test, y_test)

    assert set(metrics) == {"accuracy", "precision", "recall", "f1"}
    assert metrics == {
        "accuracy": 0.5,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    }


def test_get_confusion_matrix_returns_expected_tn_fp_fn_tp_values():
    X_test = ["sample-1", "sample-2", "sample-3", "sample-4"]
    y_test = [0, 0, 1, 1]

    confusion_matrix = get_confusion_matrix(FakeModel(), X_test, y_test)

    assert confusion_matrix == {
        "tn": 1,
        "fp": 1,
        "fn": 1,
        "tp": 1,
    }