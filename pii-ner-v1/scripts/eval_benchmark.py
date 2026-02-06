#!/usr/bin/env python3
"""Evaluate PII NER models on standardized benchmark splits."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from datafog_pii_ner.eval import evaluate_dataset, get_adapter, load_eval_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _normalize_metrics(metrics: dict) -> dict:
    return {k: float(v) for k, v in metrics.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PII-NER benchmark evaluation")
    parser.add_argument("--model", default="datafog", help="Model adapter name")
    parser.add_argument(
        "--model-path",
        default="DataFog/pii-ner-v1",
        help="Model path or HF ID for DataFog adapter",
    )
    parser.add_argument("--device", default=None, help="Device for model inference")
    parser.add_argument("--dataset", default="combined", help="combined|ai4privacy|nemotron|gretel")
    parser.add_argument("--split", default="test", help="train|validation|test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--max-char-len", type=int, default=20)
    parser.add_argument("--tokenizer", default="microsoft/deberta-v3-xsmall")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-examples-per-dataset", type=int, default=None)
    parser.add_argument("--output", default=None, help="Write metrics JSON to this path")
    parser.add_argument("--preds", default=None, help="Write per-example predictions JSONL")
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("Loading tokenizer for truncation...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    logger.info("Preparing evaluation dataset...")
    dataset = load_eval_dataset(
        dataset_name=args.dataset,
        tokenizer=tokenizer,
        split=args.split,
        seed=args.seed,
        max_seq_len=args.max_seq_len,
        max_examples=args.max_examples,
        max_examples_per_dataset=args.max_examples_per_dataset,
    )
    logger.info(f"Eval dataset: {len(dataset)} examples")

    if args.model.lower() in {"datafog", "datafog-pii", "datafog_pii"}:
        adapter = get_adapter(
            args.model,
            model_path=args.model_path,
            device=args.device,
            max_seq_len=args.max_seq_len,
            max_char_len=args.max_char_len,
        )
    else:
        adapter = get_adapter(args.model)

    logger.info(f"Running evaluation with adapter: {adapter.name}")
    metrics, metrics_by_source, pred_records = evaluate_dataset(
        dataset,
        adapter,
        batch_size=args.batch_size,
        return_preds=bool(args.preds),
    )

    metrics = _normalize_metrics(metrics)
    metrics_by_source = {k: _normalize_metrics(v) for k, v in metrics_by_source.items()}

    result = {
        "model": {
            "adapter": adapter.name,
            "model_path": args.model_path if adapter.name == "datafog" else None,
        },
        "dataset": {
            "name": args.dataset,
            "split": args.split,
            "seed": args.seed,
            "max_seq_len": args.max_seq_len,
            "tokenizer": args.tokenizer,
            "num_examples": len(dataset),
        },
        "metrics": metrics,
        "metrics_by_source": metrics_by_source,
    }

    print("=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Model:  {result['model']['adapter']}")
    print(f"Data:   {args.dataset} / {args.split} ({len(dataset)} examples)")
    print(f"F1:     {metrics.get('overall_f1', 0):.4f}")
    print(f"Prec:   {metrics.get('overall_precision', 0):.4f}")
    print(f"Recall: {metrics.get('overall_recall', 0):.4f}")
    for tier in [1, 2, 3, 4]:
        key = f"tier_{tier}_recall"
        if key in metrics:
            print(f"Tier {tier} recall: {metrics[key]:.4f}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        logger.info(f"Wrote metrics to {out_path}")

    if args.preds:
        preds_path = Path(args.preds)
        preds_path.parent.mkdir(parents=True, exist_ok=True)
        with preds_path.open("w") as f:
            for record in pred_records:
                f.write(json.dumps(record) + "\n")
        logger.info(f"Wrote predictions to {preds_path}")


if __name__ == "__main__":
    main()
