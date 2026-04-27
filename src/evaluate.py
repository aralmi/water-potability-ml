import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


def evaluate_model(model, X, y) -> pd.Series:
    """
    Evaluate classification model.

    Parameters
    ----------
    model : estimator
        Trained sklearn-compatible model.
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        True labels.

    Returns
    -------
    pd.Series
        Evaluation metrics.
    """
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y, y_proba),
        "average_precision": average_precision_score(y, y_proba),
    }

    return pd.Series(metrics)