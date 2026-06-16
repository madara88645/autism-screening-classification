from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def evaluate_model(model, X_test, y_test) -> dict[str, float]:
    predictions = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
    }


def get_confusion_matrix(model, X_test, y_test) -> dict[str, int]:
    predictions = model.predict(X_test)
    cm = confusion_matrix(y_test, predictions)
    tn, fp, fn, tp = cm.ravel()

    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
