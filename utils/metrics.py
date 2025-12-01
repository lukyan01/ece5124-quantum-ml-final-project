from typing import Dict, Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


def compute_classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        average: str = "macro",
) -> Dict[str, Any]:
    """
    Returns accuracy, precision, recall, F1 and per-class metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    precision_per, recall_per, f1_per, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "precision_per_class": precision_per,
        "recall_per_class": recall_per,
        "f1_per_class": f1_per,
        "support_per_class": support,
        "confusion_matrix": cm,
    }
