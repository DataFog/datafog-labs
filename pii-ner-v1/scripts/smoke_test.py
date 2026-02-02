"""Smoke test: overfit on a small data slice to verify model wiring.

Success criteria:
- Training loss < 0.1 after 50 epochs
- F1 on the 100 examples > 0.95
- Runs in under 5 minutes on any GPU

Usage: python -m scripts.smoke_test
"""

import logging
import sys

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

    # Training args — overfit mode
    training_args = TrainingArguments(
        output_dir="outputs/smoke_test",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=5e-4,
        warmup_steps=0,
        weight_decay=0.0,
        fp16=False,
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",
        logging_steps=10,
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    # Use same data for train and eval (intentional overfitting)
    trainer = PiiTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    # Train
    logger.info(f"Training for {NUM_EPOCHS} epochs...")
    train_result = trainer.train()

    # Evaluate
    logger.info("Evaluating...")
    eval_results = trainer.evaluate()

    # Report
    final_loss = train_result.training_loss
    f1 = eval_results.get("eval_overall_f1", 0.0)

    logger.info("=" * 60)
    logger.info("SMOKE TEST RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Final training loss: {final_loss:.4f}")
    logger.info(f"  F1 on training data:  {f1:.4f}")
    logger.info(f"  Overall precision:    {eval_results.get('eval_overall_precision', 0):.4f}")
    logger.info(f"  Overall recall:       {eval_results.get('eval_overall_recall', 0):.4f}")

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

    # Pass/fail
    logger.info("\n" + "=" * 60)
    loss_ok = final_loss < 0.1
    f1_ok = f1 > 0.95
    logger.info(f"  Loss < 0.1:  {'PASS' if loss_ok else 'FAIL'} ({final_loss:.4f})")
    logger.info(f"  F1 > 0.95:   {'PASS' if f1_ok else 'FAIL'} ({f1:.4f})")

    if loss_ok and f1_ok:
        logger.info("\n  SMOKE TEST PASSED -- model is wired correctly")
        return 0
    else:
        logger.info("\n  SMOKE TEST FAILED -- check model wiring")
        return 1


if __name__ == "__main__":
    sys.exit(main())
