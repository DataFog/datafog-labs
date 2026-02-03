"""Integration test: exercises the full pipeline from raw data through model forward pass.

Loads real examples from AI4Privacy, tokenizes, collates, and runs through the model.
Requires network access to download data from HuggingFace Hub.
"""

import pytest
import torch
from transformers import AutoTokenizer

from datafog_pii_ner.data.collator import PiiDataCollator
from datafog_pii_ner.data.dataset import load_single_dataset
from datafog_pii_ner.data.label_schema import NUM_LABELS
from datafog_pii_ner.model.pii_model import PiiNerConfig, PiiNerModel
from datafog_pii_ner.training.metrics import compute_metrics

pytestmark = pytest.mark.integration

BACKBONE = "microsoft/deberta-v3-xsmall"
NUM_EXAMPLES = 10


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(BACKBONE)


@pytest.fixture(scope="module")
def dataset(tokenizer):
    return load_single_dataset(
        dataset_name="ai4privacy",
        tokenizer=tokenizer,
        max_seq_len=256,
        max_char_len=20,
        max_examples=NUM_EXAMPLES,
    )


@pytest.fixture(scope="module")
def model():
    config = PiiNerConfig(backbone=BACKBONE, num_labels=NUM_LABELS)
    m = PiiNerModel(config)
    m.eval()
    return m


@pytest.fixture(scope="module")
def collator(tokenizer):
    return PiiDataCollator(tokenizer=tokenizer, max_char_len=20)


def test_dataset_loads_correct_count(dataset):
    """Loading with max_examples=10 produces 10 examples."""
    assert len(dataset) == NUM_EXAMPLES


def test_dataset_has_required_columns(dataset):
    """Processed dataset has the 4 required columns."""
    required = {"input_ids", "attention_mask", "labels", "char_ids"}
    assert required.issubset(set(dataset.column_names))


def test_collator_produces_correct_batch(dataset, collator):
    """Collator pads a batch and produces correct tensor shapes."""
    batch_features = [dataset[i] for i in range(min(4, len(dataset)))]
    batch = collator(batch_features)

    assert batch["input_ids"].ndim == 2
    assert batch["attention_mask"].ndim == 2
    assert batch["labels"].ndim == 2
    assert batch["char_ids"].ndim == 3

    bs = batch["input_ids"].shape[0]
    seq_len = batch["input_ids"].shape[1]
    assert batch["attention_mask"].shape == (bs, seq_len)
    assert batch["labels"].shape == (bs, seq_len)
    assert batch["char_ids"].shape == (bs, seq_len, 20)


def test_full_forward_pass(dataset, model, collator):
    """Full pipeline: real data -> collate -> model -> loss + predictions."""
    batch_features = [dataset[i] for i in range(min(4, len(dataset)))]
    batch = collator(batch_features)

    with torch.no_grad():
        output = model(**batch)

    assert output.loss is not None
    assert output.loss.dim() == 0
    assert output.loss.item() > 0
    assert output.predictions.shape[0] == batch["input_ids"].shape[0]
    assert output.predictions.shape[1] == batch["input_ids"].shape[1]


def test_predictions_valid_range(dataset, model, collator):
    """All predictions should be valid label IDs."""
    batch_features = [dataset[i] for i in range(min(4, len(dataset)))]
    batch = collator(batch_features)

    with torch.no_grad():
        output = model(**batch)

    assert output.predictions.min() >= 0
    assert output.predictions.max() < NUM_LABELS


def test_compute_metrics_on_real_predictions(dataset, model, collator):
    """compute_metrics works on predictions from the full pipeline."""
    batch_features = [dataset[i] for i in range(min(4, len(dataset)))]
    batch = collator(batch_features)

    with torch.no_grad():
        output = model(**batch)

    # Simulate EvalPrediction namedtuple
    class FakeEvalPred:
        def __init__(self, preds, labels):
            self.predictions = preds
            self.label_ids = labels

        def __iter__(self):
            return iter((self.predictions, self.label_ids))

    eval_pred = FakeEvalPred(
        output.predictions.cpu().numpy(),
        batch["labels"].cpu().numpy(),
    )

    metrics = compute_metrics(eval_pred)
    assert "overall_f1" in metrics
    assert "overall_precision" in metrics
    assert "overall_recall" in metrics
    assert 0.0 <= metrics["overall_f1"] <= 1.0
