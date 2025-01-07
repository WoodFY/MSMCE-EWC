import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc
)


def compute_accuracy(y, predicts):
    """
    Compute accuracy

    Args:
        y: true labels
        predicts: predicted labels

    Returns:
        accuracy score
    """
    return accuracy_score(y, predicts)


def compute_precision(y, predicts, average='macro', zero_division=0):
    """
    Compute precision

    Args:
        y: true labels
        predicts: predicted labels
        average: averaging strategy

    Returns:
        precision score
    """
    return precision_score(y, predicts, average=average, zero_division=zero_division)


def compute_recall(y, predicts, average='macro'):
    """
    Compute recall

    Args:
        y: true labels
        predicts: predicted labels
        average: averaging strategy

    Returns:
        recall score
    """
    return recall_score(y, predicts, average=average)


def compute_f1(y, predicts, average='macro'):
    """
    Compute f1 score

    Args:
        y: true labels
        predicts: predicted labels
        average: averaging strategy

    Returns:
        f1 score
    """
    return f1_score(y, predicts, average=average)


