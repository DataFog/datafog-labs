"""Tests for the composed PII NER model."""

import torch
from datafog_pii_ner.model.pii_model import PiiNerConfig, PiiNerModel


def _make_model(num_labels: int = 5) -> PiiNerModel:
    config = PiiNerConfig(
        backbone="microsoft/deberta-v3-xsmall",
        num_labels=num_labels,
        char_vocab_size=258,
    )
    return PiiNerModel(config)


def test_forward_with_labels():
    """Model returns loss and predictions when labels are provided."""
    model = _make_model()
    model.eval()

    batch_size, seq_len, max_char = 2, 16, 20
    inputs = {
        "input_ids": torch.randint(1, 1000, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        "char_ids": torch.randint(0, 258, (batch_size, seq_len, max_char)),
        "labels": torch.randint(0, 5, (batch_size, seq_len)),
    }

    with torch.no_grad():
        output = model(**inputs)

    assert output.loss is not None
    assert output.loss.dim() == 0  # scalar
    assert output.predictions.shape == (batch_size, seq_len)


def test_forward_without_labels():
    """Model returns predictions (no loss) when labels are omitted."""
    model = _make_model()
    model.eval()

    batch_size, seq_len, max_char = 2, 16, 20
    inputs = {
        "input_ids": torch.randint(1, 1000, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        "char_ids": torch.randint(0, 258, (batch_size, seq_len, max_char)),
    }

    with torch.no_grad():
        output = model(**inputs)

    assert output.loss is None
    assert output.predictions.shape == (batch_size, seq_len)


def test_forward_without_char_ids():
    """Model falls back to context-only when char_ids is None."""
    model = _make_model()
    model.eval()

    batch_size, seq_len = 2, 16
    inputs = {
        "input_ids": torch.randint(1, 1000, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        "labels": torch.randint(0, 5, (batch_size, seq_len)),
    }

    with torch.no_grad():
        output = model(**inputs)

    assert output.loss is not None
    assert output.predictions.shape == (batch_size, seq_len)


def test_predictions_are_valid_label_ids():
    """All predicted IDs should be within the label range."""
    num_labels = 5
    model = _make_model(num_labels=num_labels)
    model.eval()

    inputs = {
        "input_ids": torch.randint(1, 1000, (2, 16)),
        "attention_mask": torch.ones(2, 16, dtype=torch.long),
        "char_ids": torch.randint(0, 258, (2, 16, 20)),
    }

    with torch.no_grad():
        output = model(**inputs)

    assert output.predictions.min() >= 0
    assert output.predictions.max() < num_labels
