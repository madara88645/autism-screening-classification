from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


def evaluate_model(model, X_test, y_test) -> dict[str, float]:
    predictions = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
    }

def get_confusion_matrix(model,X_test,y_test):
    predictions = model.predict(X_test)
    return confusion_matrix(y_test, predictions)
