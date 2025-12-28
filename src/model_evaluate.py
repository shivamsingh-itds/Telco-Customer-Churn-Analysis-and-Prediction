from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True
        )
    }

    return metrics
