"""Training entry point for DataFog PII-NER v1."""

import argparse
import logging

import yaml
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from ..data.collator import PiiDataCollator
from ..data.dataset import load_pii_datasets
from ..data.label_schema import NUM_LABELS
from ..model.pii_model import PiiNerConfig, PiiNerModel
from .metrics import compute_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PiiTrainer(Trainer):
    """Custom Trainer with differential learning rates and CRF-aware predictions.

    Supports two parameter groups (backbone vs head) with separate learning rates.
    Pass lr_backbone and lr_head via keyword arguments.
    """

    def __init__(self, *args, lr_backbone: float | None = None, lr_head: float | None = None, **kwargs):
        self.lr_backbone = lr_backbone
        self.lr_head = lr_head
        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        """Create AdamW with differential learning rates for backbone vs head.

        The backbone group uses eps=1.0 to dampen AdamW's adaptive per-parameter
        scaling. Without this, AdamW's bias-corrected update at step 1 applies
        ~lr per weight regardless of gradient magnitude, which produces NaN in
        DeBERTa on PyTorch 2.9+. Setting eps=1.0 makes backbone updates behave
        like SGD (update ≈ lr * grad), which is numerically safe and standard
        for fine-tuning pretrained transformers.
        """
        if self.lr_backbone is not None and self.lr_head is not None:
            from torch.optim import AdamW

            backbone_params = [p for n, p in self.model.named_parameters() if "deberta" in n and p.requires_grad]
            head_params = [p for n, p in self.model.named_parameters() if "deberta" not in n and p.requires_grad]
            self.optimizer = AdamW(
                [
                    {"params": backbone_params, "lr": self.lr_backbone, "eps": 1.0},
                    {"params": head_params, "lr": self.lr_head},
                ],
                weight_decay=self.args.weight_decay,
            )
        else:
            super().create_optimizer()
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        import torch

        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs.loss.detach() if outputs.loss is not None else None
        if prediction_loss_only:
            return (loss, None, None)

        predictions = outputs.predictions.detach()
        labels = inputs.get("labels").detach()

        return (loss, predictions, labels)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def train(config_path: str = "configs/default.yaml"):
    config = load_config(config_path)

    model_cfg = config["model"]
    data_cfg = config["data"]
    train_cfg = config["training"]

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["backbone"])

    # Data
    logger.info("Loading datasets...")
    datasets = load_pii_datasets(
        tokenizer=tokenizer,
        max_seq_len=data_cfg["max_seq_len"],
        max_char_len=model_cfg["max_char_len"],
        val_ratio=data_cfg["val_split"],
        test_ratio=data_cfg["test_split"],
    )
    logger.info(
        f"Train: {len(datasets['train'])}, Val: {len(datasets['validation'])}, "
        f"Test: {len(datasets['test'])}"
    )

    # Model
    pii_config = PiiNerConfig(
        backbone=model_cfg["backbone"],
        num_labels=NUM_LABELS,
        char_vocab_size=model_cfg["char_vocab_size"],
        char_embed_dim=model_cfg["char_embed_dim"],
        char_cnn_filters=model_cfg["char_cnn_filters"],
        char_cnn_widths=model_cfg["char_cnn_widths"],
        max_char_len=model_cfg["max_char_len"],
        dropout=model_cfg["dropout"],
    )
    model = PiiNerModel(pii_config)

    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {param_count:,}")
    logger.info(f"Trainable parameters: {trainable_count:,}")

    # Data collator
    collator = PiiDataCollator(
        tokenizer=tokenizer,
        max_char_len=model_cfg["max_char_len"],
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["epochs"],
        per_device_train_batch_size=train_cfg["batch_size"],
        per_device_eval_batch_size=train_cfg["batch_size"],
        learning_rate=train_cfg["lr_backbone"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        fp16=train_cfg["fp16"],
        eval_strategy=train_cfg["eval_strategy"],
        save_strategy=train_cfg["save_strategy"],
        metric_for_best_model=train_cfg["metric_for_best_model"],
        load_best_model_at_end=True,
        report_to="wandb",
        run_name="pii-ner-v1",
        logging_steps=50,
        remove_unused_columns=False,
    )

    # Trainer — PiiTrainer handles differential learning rates internally
    # via create_optimizer(), avoiding Accelerate gradient scaler issues
    trainer = PiiTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=collator,
        compute_metrics=compute_metrics,
        lr_backbone=train_cfg["lr_backbone"],
        lr_head=train_cfg["lr_head"],
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Final evaluation on test set
    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate(datasets["test"])
    logger.info(f"Test results: {test_results}")

    # Save
    trainer.save_model(f"{train_cfg['output_dir']}/best_model")
    tokenizer.save_pretrained(f"{train_cfg['output_dir']}/best_model")
    logger.info(f"Model saved to {train_cfg['output_dir']}/best_model")

    return test_results


def main():
    parser = argparse.ArgumentParser(description="Train DataFog PII-NER v1")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to training config YAML",
    )
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
