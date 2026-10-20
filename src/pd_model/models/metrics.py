import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true, probabilities, threshold=0.5):
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)
    if y_true.ndim != 1 or probabilities.shape != y_true.shape:
        raise ValueError("y и probabilities должны быть одномерными и одинаковой длины")
    if set(y_true.tolist()) != {0, 1}:
        raise ValueError("Для оценки ROC-AUC нужны оба бинарных класса")
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0) | (probabilities > 1)
    ):
        raise ValueError("Вероятности должны быть конечными числами от 0 до 1")
    if not 0 <= threshold <= 1:
        raise ValueError("Порог должен быть от 0 до 1")
    predictions = (probabilities >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "threshold": float(threshold),
        "rows": int(len(y_true)),
        "positive_rate": float(y_true.mean()),
    }
