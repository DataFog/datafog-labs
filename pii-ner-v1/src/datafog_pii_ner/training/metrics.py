"""Evaluation metrics for PII NER using seqeval."""

import numpy as np
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from ..data.label_schema import ID_TO_LABEL, TIER_MAP


def compute_metrics(eval_pred) -> dict:
    """Compute entity-level metrics from HF Trainer predictions.

    Args:
        eval_pred: EvalPrediction with predictions and label_ids.
            predictions: (batch, seq_len) predicted tag IDs (from CRF decode).
            label_ids: (batch, seq_len) ground truth tag IDs.

    Returns:
        Dict of metrics for wandb logging.
    """
    predictions, labels = eval_pred

    # If predictions come as logits (shouldn't with CRF, but handle it)
    if predictions.ndim == 3:
        predictions = np.argmax(predictions, axis=-1)

    # Convert IDs to label strings, filtering out padding (-100)
    true_labels = []
    pred_labels = []

    for pred_seq, label_seq in zip(predictions, labels):
        true_sent = []
        pred_sent = []
        for p, l in zip(pred_seq, label_seq):
            if l == -100:
                continue
            true_sent.append(ID_TO_LABEL.get(int(l), "O"))
            pred_sent.append(ID_TO_LABEL.get(int(p), "O"))
        true_labels.append(true_sent)
        pred_labels.append(pred_sent)

    # Overall metrics
    results = {
        "overall_f1": f1_score(true_labels, pred_labels),
        "overall_precision": precision_score(true_labels, pred_labels),
        "overall_recall": recall_score(true_labels, pred_labels),
    }

    # Per-type metrics from classification report
    report = classification_report(true_labels, pred_labels, output_dict=True)
    for entity_type, metrics in report.items():
        if isinstance(metrics, dict) and entity_type not in ("micro avg", "macro avg", "weighted avg"):
            safe_name = entity_type.replace("-", "_").lower()
            results[f"type_{safe_name}_f1"] = metrics["f1-score"]
            results[f"type_{safe_name}_recall"] = metrics["recall"]

    # Tier-level recall aggregation
    tier_recalls = {1: [], 2: [], 3: [], 4: []}
    for entity_type, metrics in report.items():
        if isinstance(metrics, dict) and entity_type not in ("micro avg", "macro avg", "weighted avg"):
            tier = TIER_MAP.get(entity_type)
            if tier is not None:
                tier_recalls[tier].append(metrics["recall"])

    for tier, recalls in tier_recalls.items():
        if recalls:
            results[f"tier_{tier}_recall"] = np.mean(recalls)

    return results
