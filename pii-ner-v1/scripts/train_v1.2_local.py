"""Local training script for DataFog PII-NER v1.2 (backbone freezing).

Run on RTX 3090 while v1.1 finishes on Colab A100.
Identical hyperparameters to v1.1 + backbone freeze after epoch 4.

Usage:
    python scripts/train_v1.2_local.py
"""

import json
import logging
import math
import os
import sys

import torch
from transformers import AutoTokenizer, TrainerCallback, TrainingArguments

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from datafog_pii_ner.data.collator import PiiDataCollator
from datafog_pii_ner.data.dataset import load_pii_datasets
from datafog_pii_ner.data.label_schema import NUM_LABELS, build_label_weights
from datafog_pii_ner.model.pii_model import PiiNerConfig, PiiNerModel
from datafog_pii_ner.training.metrics import compute_metrics
from datafog_pii_ner.training.train import PiiTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---- Configuration ----

CONFIG = {
    # Model
    "backbone": "microsoft/deberta-v3-xsmall",
    "max_seq_len": 256,
    "max_char_len": 20,
    "dropout": 0.1,
    # Training (identical to v1.1)
    "epochs": 10,  # 4 full + 6 head-only
    "batch_size": 8,  # RTX 3090 24GB — smaller batch
    "gradient_accumulation_steps": 4,  # effective batch = 32
    "lr_backbone": 1e-5,
    "lr_head": 1e-3,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 500,
    "weight_decay": 0.01,
    # v1.2: backbone freezing
    "freeze_backbone_after_epoch": 4,
    # Tier-weighted CRF loss
    "tier_weights": {1: 3.0, 2: 2.0, 3: 1.5, 4: 1.0},
    # Oversampling
    "oversample_tiers": [1],
    "oversample_factor": 3,
    # Precision
    "bf16": True,  # RTX 3090 = compute 8.6
    "fp16": False,
    # Data
    "val_ratio": 0.1,
    "test_ratio": 0.1,
    "seed": 42,
    # Output
    "output_dir": os.path.join(os.path.dirname(__file__), "..", "output-v1.2"),
    "run_name": "pii-ner-v1.2-freeze-backbone-rtx3090",
}


# ---- Backbone Freezing Callback ----


class FreezeBackboneCallback(TrainerCallback):
    """Freezes DeBERTa backbone parameters after a specified epoch."""

    def __init__(self, freeze_after_epoch: int):
        self.freeze_after_epoch = freeze_after_epoch
        self.frozen = False

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        current_epoch = int(state.epoch)
        if not self.frozen and current_epoch >= self.freeze_after_epoch:
            frozen_count = 0
            for name, param in model.named_parameters():
                if "deberta" in name:
                    param.requires_grad = False
                    frozen_count += 1
            self.frozen = True

            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            logger.info("=" * 60)
            logger.info(f"BACKBONE FROZEN after epoch {current_epoch}")
            logger.info(f"Froze {frozen_count} DeBERTa parameter tensors")
            logger.info(f"Trainable: {trainable:,} / {total:,} parameters")
            logger.info(f"Continuing with: CharCNN + GatingFusion + CRF")
            logger.info("=" * 60)


# ---- Main ----


def main():
    # GPU check
    assert torch.cuda.is_available(), "No GPU found"
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"GPU: {gpu_name} ({gpu_mem:.0f}GB)")
    logger.info(f"Effective batch size: {CONFIG['batch_size'] * CONFIG['gradient_accumulation_steps']}")

    output_dir = os.path.abspath(CONFIG["output_dir"])
    os.makedirs(output_dir, exist_ok=True)

    # Load tokenizer and data
    logger.info("Loading tokenizer and datasets...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["backbone"])

    datasets = load_pii_datasets(
        tokenizer=tokenizer,
        max_seq_len=CONFIG["max_seq_len"],
        max_char_len=CONFIG["max_char_len"],
        val_ratio=CONFIG["val_ratio"],
        test_ratio=CONFIG["test_ratio"],
        seed=CONFIG["seed"],
        oversample_tiers=CONFIG.get("oversample_tiers"),
        oversample_factor=CONFIG.get("oversample_factor", 2),
    )

    logger.info(f"Train: {len(datasets['train']):,}")
    logger.info(f"Val:   {len(datasets['validation']):,}")
    logger.info(f"Test:  {len(datasets['test']):,}")

    # Initialize model
    logger.info("Initializing model...")
    config = PiiNerConfig(
        backbone=CONFIG["backbone"],
        num_labels=NUM_LABELS,
        dropout=CONFIG["dropout"],
    )
    model = PiiNerModel(config)

    # Set tier-weighted CRF loss
    tier_weights_cfg = CONFIG.get("tier_weights")
    if tier_weights_cfg:
        weights = build_label_weights(tier_weights_cfg)
        model.crf_head.set_label_weights(torch.tensor(weights, dtype=torch.float32))
        logger.info(f"Tier-weighted CRF loss: {tier_weights_cfg}")

    param_count = sum(p.numel() for p in model.parameters())
    backbone_params = sum(p.numel() for n, p in model.named_parameters() if "deberta" in n)
    head_params = param_count - backbone_params
    logger.info(f"Parameters: {param_count:,} total ({backbone_params:,} backbone, {head_params:,} head)")
    logger.info(f"Backbone freezes after epoch {CONFIG['freeze_backbone_after_epoch']}")

    # Training setup
    collator = PiiDataCollator(tokenizer=tokenizer, max_char_len=CONFIG["max_char_len"])

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=CONFIG["epochs"],
        per_device_train_batch_size=CONFIG["batch_size"],
        per_device_eval_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        learning_rate=CONFIG["lr_backbone"],
        lr_scheduler_type=CONFIG["lr_scheduler_type"],
        warmup_steps=CONFIG["warmup_steps"],
        weight_decay=CONFIG["weight_decay"],
        fp16=CONFIG["fp16"],
        bf16=CONFIG["bf16"],
        eval_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="overall_f1",
        load_best_model_at_end=True,
        report_to="wandb",
        run_name=CONFIG["run_name"],
        logging_steps=50,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        save_total_limit=3,
    )

    freeze_callback = FreezeBackboneCallback(
        freeze_after_epoch=CONFIG["freeze_backbone_after_epoch"]
    )

    trainer = PiiTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=collator,
        compute_metrics=compute_metrics,
        lr_backbone=CONFIG["lr_backbone"],
        lr_head=CONFIG["lr_head"],
        callbacks=[freeze_callback],
    )

    # Train
    logger.info(f"Starting v1.2 training: {CONFIG['epochs']} epochs")
    logger.info(f"  Epochs 1-{CONFIG['freeze_backbone_after_epoch']}: full model")
    logger.info(f"  Epochs {CONFIG['freeze_backbone_after_epoch']+1}-{CONFIG['epochs']}: head only")
    train_result = trainer.train()
    logger.info(f"Training complete. Final loss: {train_result.training_loss:.4f}")

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate(datasets["test"])

    logger.info("=" * 60)
    logger.info("TEST SET RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Overall F1:        {test_results.get('eval_overall_f1', 0):.4f}")
    logger.info(f"  Overall Precision: {test_results.get('eval_overall_precision', 0):.4f}")
    logger.info(f"  Overall Recall:    {test_results.get('eval_overall_recall', 0):.4f}")

    for tier in [1, 2, 3, 4]:
        key = f"eval_tier_{tier}_recall"
        if key in test_results:
            target = {1: 0.98, 2: 0.95, 3: 0.90, 4: 0.85}[tier]
            actual = test_results[key]
            status = "PASS" if actual >= target else "FAIL"
            logger.info(f"  Tier {tier} recall: {actual:.4f}  (target >= {target})  [{status}]")

    # Save
    save_path = os.path.join(output_dir, "best_model")
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    logger.info(f"Model saved to {save_path}")

    with open(os.path.join(save_path, "training_config.json"), "w") as f:
        json.dump(CONFIG, f, indent=2, default=str)

    with open(os.path.join(save_path, "test_results.json"), "w") as f:
        json.dump(
            {k: float(v) if hasattr(v, "__float__") else v for k, v in test_results.items()},
            f,
            indent=2,
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
