"""Tests for metric computation logic."""

import numpy as np

from datafog_pii_ner.data.label_schema import LABEL_TO_ID
from datafog_pii_ner.training.metrics import compute_metrics


class FakeEvalPred:
    """Mimics HF EvalPrediction for testing."""

    def __init__(self, predictions, label_ids):
        self.predictions = predictions
        self.label_ids = label_ids

    def __iter__(self):
        return iter((self.predictions, self.label_ids))


def _ids(*labels):
    """Convert label strings to IDs."""
    return [LABEL_TO_ID[label] for label in labels]


def test_perfect_predictions():
    """Perfect predictions should give F1 = 1.0."""
    seq = _ids("O", "B-PERSON", "I-PERSON", "O", "B-EMAIL", "O")
    predictions = np.array([seq])
    labels = np.array([seq])

    metrics = compute_metrics(FakeEvalPred(predictions, labels))
    assert metrics["overall_f1"] == 1.0
    assert metrics["overall_precision"] == 1.0
    assert metrics["overall_recall"] == 1.0


def test_all_wrong_predictions():
    """Completely wrong predictions should give F1 = 0.0."""
    true = _ids("O", "B-PERSON", "I-PERSON", "O")
    pred = _ids("O", "O", "O", "O")
    predictions = np.array([pred])
    labels = np.array([true])

    metrics = compute_metrics(FakeEvalPred(predictions, labels))
    assert metrics["overall_f1"] == 0.0
    assert metrics["overall_recall"] == 0.0


def test_padding_ignored():
    """Labels with -100 should be filtered out, not affect metrics."""
    true = _ids("B-PERSON", "I-PERSON") + [-100, -100]
    pred = _ids("B-PERSON", "I-PERSON", "O", "O")
    predictions = np.array([pred])
    labels = np.array([true])

    metrics = compute_metrics(FakeEvalPred(predictions, labels))
    assert metrics["overall_f1"] == 1.0


def test_3d_predictions_argmaxed():
    """3D predictions (logits) should be argmax'd to 2D."""
    true = _ids("O", "B-SSN", "I-SSN", "O")
    num_labels = max(true) + 10
    # Create logits where the correct class has highest score
    logits = np.zeros((1, len(true), num_labels))
    for i, t in enumerate(true):
        logits[0, i, t] = 10.0

    labels = np.array([true])
    metrics = compute_metrics(FakeEvalPred(logits, labels))
    assert metrics["overall_f1"] == 1.0


def test_tier_recall_aggregation():
    """Tier recalls should be computed for entity types present in predictions."""
    # Create a sequence with Tier 1 (SSN) and Tier 2 (PERSON) entities
    true = _ids("B-SSN", "I-SSN", "O", "B-PERSON", "I-PERSON")
    pred = _ids("B-SSN", "I-SSN", "O", "B-PERSON", "I-PERSON")
    predictions = np.array([pred])
    labels = np.array([true])

    metrics = compute_metrics(FakeEvalPred(predictions, labels))
    # SSN is Tier 1, PERSON is Tier 2
    assert "tier_1_recall" in metrics
    assert metrics["tier_1_recall"] == 1.0
    assert "tier_2_recall" in metrics
    assert metrics["tier_2_recall"] == 1.0


def test_all_o_sequence():
    """A sequence with no entities should still return valid metrics."""
    seq = _ids("O", "O", "O", "O")
    predictions = np.array([seq])
    labels = np.array([seq])

    metrics = compute_metrics(FakeEvalPred(predictions, labels))
    # No entities -> F1 is 0 (seqeval returns 0 when no entities exist)
    assert "overall_f1" in metrics
    assert isinstance(metrics["overall_f1"], float)


def test_partial_entity_match():
    """Partial entity match (wrong boundary) should not count as correct."""
    true = _ids("B-PERSON", "I-PERSON", "I-PERSON", "O")
    # Predict entity starting one token late
    pred = _ids("O", "B-PERSON", "I-PERSON", "O")
    predictions = np.array([pred])
    labels = np.array([true])

    metrics = compute_metrics(FakeEvalPred(predictions, labels))
    # seqeval uses strict entity matching, so partial overlap = 0 F1
    assert metrics["overall_f1"] == 0.0


def test_multiple_sequences():
    """Metrics should work across multiple sequences in a batch."""
    true1 = _ids("B-EMAIL", "O", "O")
    pred1 = _ids("B-EMAIL", "O", "O")
    true2 = _ids("O", "B-PHONE", "O")
    pred2 = _ids("O", "O", "O")

    predictions = np.array([pred1, pred2])
    labels = np.array([true1, true2])

    metrics = compute_metrics(FakeEvalPred(predictions, labels))
    # 1 out of 2 entities correct
    assert metrics["overall_precision"] == 1.0  # 1 predicted, 1 correct
    assert metrics["overall_recall"] == 0.5  # 2 true, 1 found
