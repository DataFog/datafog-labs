"""Benchmark evaluation utilities."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from ..training.metrics import compute_metrics_from_labels
from .utils import spans_to_bio


def _batch_iter(dataset, batch_size: int):
    batch = []
    for example in dataset:
        batch.append(example)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def evaluate_dataset(
    dataset,
    adapter,
    batch_size: int = 8,
    return_preds: bool = False,
):
    true_labels: list[list[str]] = []
    pred_labels: list[list[str]] = []

    by_source_true: dict[str, list[list[str]]] = defaultdict(list)
    by_source_pred: dict[str, list[list[str]]] = defaultdict(list)

    pred_records = []

    for batch in _batch_iter(dataset, batch_size):
        texts = [ex["text"] for ex in batch]
        preds = adapter.predict(texts)

        for ex, spans in zip(batch, preds):
            gold = ex["bio_tags"]
            pred = spans_to_bio(ex["tokens"], spans)
            true_labels.append(gold)
            pred_labels.append(pred)

            source = ex.get("source", "unknown")
            by_source_true[source].append(gold)
            by_source_pred[source].append(pred)

            if return_preds:
                pred_records.append(
                    {
                        "example_id": ex.get("example_id"),
                        "source": source,
                        "text": ex.get("text"),
                        "tokens": ex.get("tokens"),
                        "gold_tags": gold,
                        "pred_tags": pred,
                        "pred_spans": spans,
                    }
                )

    metrics = compute_metrics_from_labels(true_labels, pred_labels)
    metrics_by_source = {
        source: compute_metrics_from_labels(by_source_true[source], by_source_pred[source])
        for source in sorted(by_source_true.keys())
    }

    return metrics, metrics_by_source, pred_records
