from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def choose_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    candidates = np.unique(scores)
    if len(candidates) > 250:
        candidates = np.quantile(scores, np.linspace(0.01, 0.99, 250))

    best_threshold = 0.5
    best_mcc = -np.inf
    for threshold in candidates:
        pred = (scores >= threshold).astype(int)
        mcc = matthews_corrcoef(y_true, pred)
        if mcc > best_mcc:
            best_mcc = mcc
            best_threshold = float(threshold)
    return best_threshold


def safe_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int | None = None) -> float:
    if k is None:
        k = int(max(1, y_true.sum()))
    k = min(k, len(y_true))
    top_idx = np.argsort(scores)[::-1][:k]
    return float(np.mean(y_true[top_idx])) if k else 0.0


def recall_at_precision(y_true: np.ndarray, scores: np.ndarray, target_precision: float = 0.8) -> float:
    precision, recall, _ = precision_recall_curve(y_true, scores)
    valid = recall[precision >= target_precision]
    return float(valid.max()) if len(valid) else 0.0


def evaluate_scores(
    y_train: np.ndarray,
    train_scores: np.ndarray,
    y_test: np.ndarray,
    test_scores: np.ndarray,
) -> dict[str, float | int]:
    threshold = choose_threshold(y_train, train_scores)
    y_pred = (test_scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    return {
        "threshold": float(threshold),
        "aupr": float(average_precision_score(y_test, test_scores)),
        "auroc": safe_auroc(y_test, test_scores),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_test, y_pred)),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "precision_at_k": precision_at_k(y_test, test_scores),
        "recall_at_precision_80": recall_at_precision(y_test, test_scores, 0.8),
        "brier": float(brier_score_loss(y_test, np.clip(test_scores, 0, 1))),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }
