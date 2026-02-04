#!/usr/bin/env python3
"""DataFog PII-NER v1 — Local Training Runner.

Self-contained training script designed for CLI-driven iteration.
Run with: python scripts/run_training.py --config configs/rtx3090.yaml

Features:
  - Auto precision selection (BF16 → FP32 fallback)
  - Preflight check before committing to full training
  - Structured logging to file + console
  - Status file (output/status.json) for programmatic monitoring
  - Resume from checkpoint support
"""

import argparse
import gc
import json
import logging
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import Dataset
from transformers import AutoTokenizer, TrainingArguments

# Add src to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from datafog_pii_ner.data.collator import PiiDataCollator
from datafog_pii_ner.data.dataset import load_pii_datasets
from datafog_pii_ner.data.label_schema import NUM_LABELS
from datafog_pii_ner.model.pii_model import PiiNerConfig, PiiNerModel
from datafog_pii_ner.training.metrics import compute_metrics
from datafog_pii_ner.training.train import PiiTrainer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(output_dir: str) -> logging.Logger:
    """Set up dual logging: console (INFO) + file (DEBUG)."""
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_{timestamp}.log")

    logger = logging.getLogger("pii_ner_training")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(fh)

    logger.info(f"Log file: {log_file}")
    return logger


# ---------------------------------------------------------------------------
# Status tracking
# ---------------------------------------------------------------------------

class StatusTracker:
    """Writes status.json for programmatic monitoring."""

    def __init__(self, output_dir: str):
        self.path = os.path.join(output_dir, "status.json")
        os.makedirs(output_dir, exist_ok=True)
        self.data = {
            "stage": "initializing",
            "started_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "precision": None,
            "epoch": 0,
            "total_epochs": 0,
            "train_loss": None,
            "eval_loss": None,
            "eval_f1": None,
            "best_f1": 0.0,
            "error": None,
        }
        self._write()

    def update(self, **kwargs):
        self.data.update(kwargs)
        self.data["updated_at"] = datetime.now().isoformat()
        self._write()

    def _write(self):
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Preflight check
# ---------------------------------------------------------------------------

def run_preflight(config: dict, logger: logging.Logger) -> dict:
    """Auto-select precision: try BF16 → FP32 fallback.

    Returns dict with 'fp16' and 'bf16' keys.
    """
    logger.info("=" * 60)
    logger.info("PREFLIGHT CHECK — auto-selecting precision")
    logger.info("=" * 60)

    batch_size = config["training"]["batch_size"]
    seq_len = config["data"]["max_seq_len"]
    max_char_len = config["model"]["max_char_len"]
    backbone = config["model"]["backbone"]
    lr_backbone = config["training"]["lr_backbone"]
    lr_head = config["training"]["lr_head"]

    def _test_precision(fp16: bool, bf16: bool) -> tuple[bool, float]:
        label = "BF16" if bf16 else "FP16" if fp16 else "FP32"
        logger.info(f"Testing {label} (batch_size={batch_size}, seq_len={seq_len})...")

        tokenizer = AutoTokenizer.from_pretrained(backbone)
        n_samples = batch_size * 2

        fake_data = {
            "input_ids": np.random.randint(1, 1000, (n_samples, seq_len)).tolist(),
            "attention_mask": np.ones((n_samples, seq_len), dtype=int).tolist(),
            "labels": np.random.randint(0, NUM_LABELS, (n_samples, seq_len)).tolist(),
            "char_ids": np.random.randint(0, 100, (n_samples, seq_len, max_char_len)).tolist(),
        }
        ds = Dataset.from_dict(fake_data)
        collator = PiiDataCollator(tokenizer=tokenizer, max_char_len=max_char_len)
        model_config = PiiNerConfig(backbone=backbone, num_labels=NUM_LABELS)
        model = PiiNerModel(model_config)

        args = TrainingArguments(
            output_dir=os.path.join(config["training"]["output_dir"], "_preflight"),
            num_train_epochs=1,
            per_device_train_batch_size=batch_size,
            learning_rate=lr_backbone,
            fp16=fp16,
            bf16=bf16,
            report_to="none",
            logging_steps=1,
            max_steps=5,
            remove_unused_columns=False,
            save_strategy="no",
        )

        trainer = PiiTrainer(
            model=model, args=args, train_dataset=ds, data_collator=collator,
            lr_backbone=lr_backbone, lr_head=lr_head,
        )

        try:
            result = trainer.train()
            loss = result.training_loss
        except Exception as e:
            logger.warning(f"  {label} CRASHED: {type(e).__name__}: {e}")
            del model, trainer, ds, collator, args
            gc.collect()
            torch.cuda.empty_cache()
            return False, float("nan")

        passed = True

        if math.isnan(loss) or math.isinf(loss):
            logger.warning(f"  {label} FAIL: loss is {loss}")
            passed = False
        else:
            logger.info(f"  {label} loss = {loss:.4f}")

        nan_params = [n for n, p in model.named_parameters() if torch.isnan(p).any()]
        if nan_params:
            logger.warning(f"  {label} FAIL: NaN in {len(nan_params)} param tensors")
            passed = False

        log_losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
        if log_losses and any(math.isnan(l) for l in log_losses):
            logger.warning(f"  {label} FAIL: NaN in step losses")
            passed = False
        elif len(log_losses) >= 2:
            if log_losses[-1] < log_losses[0]:
                logger.info(f"  {label} loss decreasing: {log_losses[0]:.1f} → {log_losses[-1]:.1f}")

        status = "PASS" if passed else "FAIL"
        logger.info(f"  {label}: {status}")

        del model, trainer, ds, collator, args
        gc.collect()
        torch.cuda.empty_cache()
        return passed, loss

    # Try BF16 first (if GPU supports it)
    if torch.cuda.is_available():
        cc = torch.cuda.get_device_properties(0).major
        if cc >= 8:
            ok, loss = _test_precision(fp16=False, bf16=True)
            if ok:
                logger.info(f"Selected: BF16 (preflight loss={loss:.4f})")
                return {"fp16": False, "bf16": True}
            logger.info("BF16 failed, trying FP32...")

    # FP32 fallback
    ok, loss = _test_precision(fp16=False, bf16=False)
    if ok:
        logger.info(f"Selected: FP32 (preflight loss={loss:.4f})")
        return {"fp16": False, "bf16": False}

    raise RuntimeError("Preflight failed on FP32 — something is fundamentally broken")


# ---------------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------------

def train(config: dict, precision: dict, logger: logging.Logger, status: StatusTracker,
          resume_from: str | None = None):
    """Run full training pipeline."""
    output_dir = config["training"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # --- Tokenizer ---
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["backbone"])

    # --- Data ---
    logger.info("Loading datasets (may take a few minutes on first run)...")
    status.update(stage="loading_data")
    datasets = load_pii_datasets(
        tokenizer=tokenizer,
        max_seq_len=config["data"]["max_seq_len"],
        max_char_len=config["model"]["max_char_len"],
        val_ratio=config["data"]["val_ratio"],
        test_ratio=config["data"]["test_ratio"],
        seed=config["data"].get("seed", 42),
    )
    logger.info(f"Train: {len(datasets['train']):,}  Val: {len(datasets['validation']):,}  "
                f"Test: {len(datasets['test']):,}")

    # --- Model ---
    logger.info("Initializing model...")
    status.update(stage="initializing_model")
    pii_config = PiiNerConfig(
        backbone=config["model"]["backbone"],
        num_labels=NUM_LABELS,
        char_vocab_size=config["model"].get("char_vocab_size", 256),
        char_embed_dim=config["model"].get("char_embed_dim", 50),
        char_cnn_filters=config["model"].get("char_cnn_filters", [50, 50, 50]),
        char_cnn_widths=config["model"].get("char_cnn_widths", [3, 4, 5]),
        max_char_len=config["model"]["max_char_len"],
        dropout=config["model"].get("dropout", 0.1),
    )
    model = PiiNerModel(pii_config)

    param_count = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters: {param_count:,} total, {trainable:,} trainable")

    # --- Collator ---
    collator = PiiDataCollator(tokenizer=tokenizer, max_char_len=config["model"]["max_char_len"])

    # --- Training args ---
    tcfg = config["training"]
    prec_label = "BF16" if precision["bf16"] else "FP16" if precision["fp16"] else "FP32"
    logger.info(f"Precision: {prec_label}")
    logger.info(f"Backbone LR: {tcfg['lr_backbone']}, Head LR: {tcfg['lr_head']}")
    logger.info(f"Batch: {tcfg['batch_size']} x {tcfg['gradient_accumulation_steps']} = "
                f"{tcfg['batch_size'] * tcfg['gradient_accumulation_steps']} effective")

    wandb_cfg = config.get("wandb", {})
    report_to = "wandb" if wandb_cfg.get("enabled", False) else "none"

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=tcfg["epochs"],
        per_device_train_batch_size=tcfg["batch_size"],
        per_device_eval_batch_size=tcfg["batch_size"],
        gradient_accumulation_steps=tcfg.get("gradient_accumulation_steps", 1),
        learning_rate=tcfg["lr_backbone"],
        warmup_ratio=tcfg.get("warmup_ratio", 0.1),
        weight_decay=tcfg.get("weight_decay", 0.01),
        fp16=precision["fp16"],
        bf16=precision["bf16"],
        eval_strategy=tcfg.get("eval_strategy", "epoch"),
        save_strategy=tcfg.get("save_strategy", "epoch"),
        metric_for_best_model=tcfg.get("metric_for_best_model", "overall_f1"),
        load_best_model_at_end=True,
        report_to=report_to,
        run_name=tcfg.get("run_name", "pii-ner-v1"),
        logging_steps=tcfg.get("logging_steps", 50),
        remove_unused_columns=False,
        dataloader_num_workers=tcfg.get("dataloader_num_workers", 0),
        save_total_limit=tcfg.get("save_total_limit", 3),
    )

    # --- Trainer ---
    trainer = PiiTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=collator,
        compute_metrics=compute_metrics,
        lr_backbone=tcfg["lr_backbone"],
        lr_head=tcfg["lr_head"],
    )

    # --- Train ---
    total_epochs = tcfg["epochs"]
    status.update(stage="training", total_epochs=total_epochs, precision=prec_label)
    logger.info(f"Starting training for {total_epochs} epochs...")
    t0 = time.time()

    train_result = trainer.train(resume_from_checkpoint=resume_from)

    elapsed = time.time() - t0
    logger.info(f"Training complete in {elapsed/3600:.1f}h. Final loss: {train_result.training_loss:.4f}")

    # --- Evaluate test set ---
    logger.info("Evaluating on held-out test set...")
    status.update(stage="evaluating")
    test_results = trainer.evaluate(datasets["test"])

    overall_f1 = test_results.get("eval_overall_f1", 0)
    overall_p = test_results.get("eval_overall_precision", 0)
    overall_r = test_results.get("eval_overall_recall", 0)

    logger.info("=" * 60)
    logger.info("TEST SET RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Overall F1:        {overall_f1:.4f}")
    logger.info(f"  Overall Precision: {overall_p:.4f}")
    logger.info(f"  Overall Recall:    {overall_r:.4f}")

    for tier in [1, 2, 3, 4]:
        key = f"eval_tier_{tier}_recall"
        if key in test_results:
            target = {1: 0.98, 2: 0.95, 3: 0.90, 4: 0.85}[tier]
            actual = test_results[key]
            result_str = "PASS" if actual >= target else "FAIL"
            logger.info(f"  Tier {tier} recall: {actual:.4f}  (target >= {target})  [{result_str}]")

    # Per-type F1 (top 20)
    type_metrics = [(k, v) for k, v in test_results.items()
                    if k.startswith("eval_type_") and k.endswith("_f1")]
    type_metrics.sort(key=lambda x: x[1], reverse=True)
    if type_metrics:
        logger.info("Per-entity F1 (top 20):")
        for k, v in type_metrics[:20]:
            name = k.replace("eval_type_", "").replace("_f1", "")
            logger.info(f"  {name:30s} {v:.4f}")

    # --- Save ---
    save_path = os.path.join(output_dir, "best_model")
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    logger.info(f"Model saved to {save_path}")

    with open(os.path.join(save_path, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)

    with open(os.path.join(save_path, "test_results.json"), "w") as f:
        json.dump({k: float(v) if hasattr(v, "__float__") else v
                    for k, v in test_results.items()}, f, indent=2)

    status.update(
        stage="complete",
        eval_f1=overall_f1,
        best_f1=overall_f1,
        train_loss=train_result.training_loss,
    )
    logger.info("Done.")
    return test_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="DataFog PII-NER v1 Training Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_training.py --config configs/rtx3090.yaml
  python scripts/run_training.py --config configs/rtx3090.yaml --preflight-only
  python scripts/run_training.py --config configs/rtx3090.yaml --resume output/checkpoint-1234
  python scripts/run_training.py --config configs/rtx3090.yaml --no-wandb
        """,
    )
    parser.add_argument("--config", type=str, default="configs/rtx3090.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--preflight-only", action="store_true",
                        help="Run preflight check only, don't train")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint directory")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable WandB logging")
    parser.add_argument("--fp32", action="store_true",
                        help="Force FP32 (skip BF16 preflight)")
    args = parser.parse_args()

    # Load config
    config_path = os.path.join(str(PROJECT_ROOT), args.config)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Resolve output dir relative to project root
    output_dir = config["training"]["output_dir"]
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(str(PROJECT_ROOT), output_dir)
    config["training"]["output_dir"] = output_dir
    os.makedirs(output_dir, exist_ok=True)

    if args.no_wandb:
        config.setdefault("wandb", {})["enabled"] = False

    # Setup
    logger = setup_logging(output_dir)
    status = StatusTracker(output_dir)

    logger.info("=" * 60)
    logger.info("DataFog PII-NER v1 — Training Runner")
    logger.info("=" * 60)
    logger.info(f"Config: {config_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        logger.info(f"VRAM: {props.total_memory / 1e9:.1f} GB")
        logger.info(f"Compute capability: {props.major}.{props.minor}")
    else:
        logger.error("No GPU found")
        sys.exit(1)

    # Preflight
    if args.fp32:
        precision = {"fp16": False, "bf16": False}
        logger.info("Forced FP32 mode (--fp32 flag)")
    else:
        precision = run_preflight(config, logger)

    status.update(precision="BF16" if precision["bf16"] else "FP32")

    if args.preflight_only:
        logger.info("Preflight complete (--preflight-only). Exiting.")
        status.update(stage="preflight_complete")
        return

    # Train
    try:
        train(config, precision, logger, status, resume_from=args.resume)
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        status.update(stage="interrupted")
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        status.update(stage="failed", error=str(e))
        raise


if __name__ == "__main__":
    main()
