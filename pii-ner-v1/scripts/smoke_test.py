"""Smoke test: overfit on a small data slice to verify model wiring.

Success criteria:
- Eval loss < 10 (CRF NLL is sequence-level, not per-token)
- F1 on the 100 examples > 0.90
- Runs in under 10 minutes on any GPU

Usage: python -m scripts.smoke_test
"""

import logging
import sys

import torch
from transformers import AutoTokenizer, TrainingArguments

from datafog_pii_ner.data.collator import PiiDataCollator
from datafog_pii_ner.data.dataset import load_single_dataset
from datafog_pii_ner.data.label_schema import ID_TO_LABEL, NUM_LABELS
from datafog_pii_ner.model.pii_model import PiiNerConfig, PiiNerModel
from datafog_pii_ner.training.metrics import compute_metrics
from datafog_pii_ner.training.train import PiiTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NUM_EXAMPLES = 100
NUM_EPOCHS = 50
BACKBONE = "microsoft/deberta-v3-xsmall"

# Differential learning rates: low for pretrained backbone, high for random head
LR_BACKBONE = 2e-5
LR_HEAD = 1e-3


def main():
    logger.info(f"=== SMOKE TEST: Overfit on {NUM_EXAMPLES} examples ===")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)

    # Load a small slice from one dataset
    logger.info("Loading data...")
    dataset = load_single_dataset(
        dataset_name="ai4privacy",
        tokenizer=tokenizer,
        max_seq_len=256,
        max_char_len=20,
        max_examples=NUM_EXAMPLES,
    )
    logger.info(f"Loaded {len(dataset)} examples")

    # Model
    config = PiiNerConfig(
        backbone=BACKBONE,
        num_labels=NUM_LABELS,
    )
    model = PiiNerModel(config)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {param_count:,}")

    # Collator
    collator = PiiDataCollator(tokenizer=tokenizer, max_char_len=20)

    # Auto-detect mixed precision: bf16 for A100+, fp16 for T4, none for CPU
    use_bf16 = False
    use_fp16 = False
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_mem >= 20:
            use_bf16 = True
        else:
            use_fp16 = True

    # Training args — overfit mode with differential LRs
    training_args = TrainingArguments(
        output_dir="outputs/smoke_test",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,  # effective batch size = 16
        learning_rate=LR_BACKBONE,
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=use_fp16,
        bf16=use_bf16,
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",
        logging_steps=10,
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    logger.info(
        f"Backbone LR: {LR_BACKBONE}, Head LR: {LR_HEAD} ({LR_HEAD / LR_BACKBONE:.0f}x)"
    )
    logger.info(f"Mixed precision: {'bf16' if use_bf16 else 'fp16' if use_fp16 else 'none'}")

    # PiiTrainer handles differential learning rates internally via create_optimizer().
    # This avoids passing a custom optimizer through optimizers=(), which bypasses
    # Accelerate's optimizer wrapping and causes FP16/BF16 gradient scaler issues.
    trainer = PiiTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        data_collator=collator,
        compute_metrics=compute_metrics,
        lr_backbone=LR_BACKBONE,
        lr_head=LR_HEAD,
    )

    # Train
    logger.info(f"Training for {NUM_EPOCHS} epochs...")
    train_result = trainer.train()

    # Evaluate
    logger.info("Evaluating...")
    eval_results = trainer.evaluate()

    # Report
    eval_loss = eval_results.get("eval_loss", float("inf"))
    f1 = eval_results.get("eval_overall_f1", 0.0)

    logger.info("=" * 60)
    logger.info("SMOKE TEST RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Final training loss: {train_result.training_loss:.4f}")
    logger.info(f"  Eval loss:           {eval_loss:.4f}")
    logger.info(f"  F1 on training data: {f1:.4f}")
    logger.info(f"  Overall precision:   {eval_results.get('eval_overall_precision', 0):.4f}")
    logger.info(f"  Overall recall:      {eval_results.get('eval_overall_recall', 0):.4f}")

    # Tier recalls
    for tier in [1, 2, 3, 4]:
        key = f"eval_tier_{tier}_recall"
        if key in eval_results:
            logger.info(f"  Tier {tier} recall:    {eval_results[key]:.4f}")

    # Print sample predictions
    logger.info("\nSample predictions:")
    predictions = trainer.predict(dataset.select(range(min(5, len(dataset)))))
    for i in range(min(5, len(dataset))):
        pred_ids = predictions.predictions[i]
        label_ids = predictions.label_ids[i]
        input_ids = dataset[i]["input_ids"]

        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        logger.info(f"\n  Example {i + 1}:")
        for tok, pred, label in zip(tokens, pred_ids, label_ids):
            if label == -100:
                continue
            pred_label = ID_TO_LABEL.get(int(pred), "O")
            true_label = ID_TO_LABEL.get(int(label), "O")
            if true_label != "O" or pred_label != "O":
                marker = "  " if pred_label == true_label else "!!"
                logger.info(f"    {marker} {tok:20s}  pred={pred_label:20s}  true={true_label}")

    # Pass/fail — CRF NLL is sequence-level so loss thresholds differ from cross-entropy
    logger.info("\n" + "=" * 60)
    loss_ok = eval_loss < 10
    f1_ok = f1 > 0.90
    logger.info(f"  Loss < 10:   {'PASS' if loss_ok else 'FAIL'} ({eval_loss:.4f})")
    logger.info(f"  F1 > 0.90:   {'PASS' if f1_ok else 'FAIL'} ({f1:.4f})")

    if loss_ok and f1_ok:
        logger.info("\n  SMOKE TEST PASSED -- model is wired correctly")
        return 0
    else:
        logger.info("\n  SMOKE TEST FAILED -- check model wiring")
        return 1


if __name__ == "__main__":
    sys.exit(main())
