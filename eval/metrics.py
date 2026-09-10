"""
eval/metrics.py
Standard evaluation metrics for Intent Classification and Escalation Gate.
Computes Accuracy, Macro-F1, Per-Class F1, and Confusion Matrices.
"""

from typing import List, Dict, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)
from pipeline.intents import INTENTS


def evaluate_intent_classification(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    """Calculate classification accuracy, macro-F1, weighted-F1, and per-class report."""
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    report = classification_report(
        y_true,
        y_pred,
        labels=INTENTS,
        target_names=INTENTS,
        output_dict=True,
        zero_division=0
    )
    
    matrix = confusion_matrix(y_true, y_pred, labels=INTENTS)

    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "classification_report": report,
        "confusion_matrix": matrix.tolist()
    }


def evaluate_escalation_gate(y_true: List[bool], y_pred: List[bool]) -> Dict[str, Any]:
    """Calculate escalation precision, recall, F1, and accuracy."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4)
    }
