# Design: DataFog PII-NER v1

## Summary

A 22.7M parameter PII detection model combining DeBERTa-v3-xsmall with a character CNN encoder, sigmoid gating fusion, and CRF output layer. Trained on 360K+ open-licensed PII examples using HuggingFace Trainer with wandb experiment tracking. Targets >=0.90 F1 on PIILO, >=0.98 Tier 1 recall, <50ms P99 CPU latency.

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Location | `datafog-labs/pii-ner-v1/` | Keep research and implementation together during R&D |
| Framework | PyTorch + HuggingFace Trainer | DeBERTa comes from HF, token classification is first-class |
| Experiment tracking | Weights & Biases (wandb) | Industry standard, free tier, visual run comparison |
| Training compute | Cloud GPU (on-demand) | Flexibility, pay-per-use |
| First milestone | Smoke test (overfit 100 examples) | Validates wiring before investing in full training |

## Project Structure

```
pii-ner-v1/
├── pyproject.toml
├── configs/
│   └── default.yaml
├── src/
│   └── datafog_pii_ner/
│       ├── __init__.py
│       ├── model/
│       │   ├── __init__.py
│       │   ├── char_cnn.py
│       │   ├── gating_fusion.py
│       │   ├── crf.py
│       │   └── pii_model.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── dataset.py
│       │   ├── label_schema.py
│       │   └── char_vocab.py
│       └── training/
│           ├── __init__.py
│           ├── train.py
│           ├── metrics.py
│           └── collator.py
├── scripts/
│   ├── download_data.py
│   └── smoke_test.py
├── tests/
│   ├── test_char_cnn.py
│   ├── test_model.py
│   └── test_data.py
└── notebooks/
    └── exploration.ipynb
```

## Model Architecture

### Component 1: Character CNN Encoder (`char_cnn.py`)
- Input: character IDs per token, padded to max 20 characters
- Character embedding: 256 vocab -> 50-dim vectors
- Three parallel Conv1D filters (widths 3, 4, 5), 50 filters each
- Max-pool over character sequence -> 150-dim output per token
- Parameters: ~0.3M

### Component 2: Gating Fusion (`gating_fusion.py`)
- Projects 150-dim char features to 384-dim (match DeBERTa hidden size)
- Sigmoid gate: `g = sigmoid(W @ [char_proj; context] + b)`
- Output: `g * char_proj + (1 - g) * context`
- Parameters: ~0.1M

### Component 3: CRF Layer (`crf.py`)
- Uses `pytorch-crf` library
- Learns transition scores between BIO tags
- Viterbi decoding at inference
- Parameters: ~0.2M

### Component 4: Composed Model (`pii_model.py`)
- Loads `microsoft/deberta-v3-xsmall` (~22M params)
- Forward: input -> DeBERTa + CharCNN (parallel) -> GatingFusion -> Linear -> CRF
- Returns CRF negative log-likelihood (training) or decoded tag sequences (inference)
- Inherits from `PreTrainedModel` for HF Trainer compatibility
- Total: ~22.7M parameters

## Data Pipeline

### Label Schema (`label_schema.py`)
- Canonical list of ~50 PII entity types across 4 sensitivity tiers
- Mapping tables from each dataset's naming convention to canonical names
- Generates full BIO label set (O, B-SSN, I-SSN, B-PERSON, I-PERSON, ...)

### Dataset Loading (`dataset.py`)
- Three source datasets from HuggingFace Hub:
  - AI4Privacy (~200K, Apache 2.0)
  - NVIDIA Nemotron-PII (~100K, CC-BY-4.0)
  - Gretel PII Masking (~70K, Apache 2.0)
- Unified preprocessing: apply label mapping, tokenize with DeBERTa tokenizer
- Output columns: `input_ids`, `labels` (BIO-aligned to subwords), `char_ids`
- Subword alignment: B on first subword, I on continuation subwords
- Split: 80/10/10 train/val/test, stratified by source dataset

### Batch Collation (`collator.py`)
- Custom collator for the extra `char_ids` 3D tensor (batch, seq_len, char_len)
- Pads all tensors to batch max lengths

### Character Vocabulary (`char_vocab.py`)
- Fixed 256-entry byte-level mapping
- PAD=0, UNK=1, then raw byte values

## Training Configuration

```yaml
# configs/default.yaml
model:
  backbone: microsoft/deberta-v3-xsmall
  char_embed_dim: 50
  char_vocab_size: 256
  char_cnn_filters: [50, 50, 50]
  char_cnn_widths: [3, 4, 5]
  max_char_len: 20
  dropout: 0.1

data:
  max_seq_len: 256
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1

training:
  epochs: 10
  batch_size: 32
  lr_backbone: 2e-5
  lr_head: 1e-3
  warmup_ratio: 0.1
  weight_decay: 0.01
  fp16: true
  eval_strategy: epoch
  save_strategy: epoch
  metric_for_best_model: overall_f1
```

## Evaluation (`metrics.py`)
- Entity-level F1, precision, recall via `seqeval`
- Per-type breakdown (logged to wandb)
- Tier-level aggregation (Tier 1 recall, Tier 2 recall, etc.)

## Milestone 1: Smoke Test

`scripts/smoke_test.py`:
1. Load 100 examples from training set
2. Initialize model with random CharCNN/gate/CRF weights + pretrained DeBERTa
3. Train for 50 epochs
4. Assert: training loss < 0.1, F1 on those 100 examples > 0.95
5. Print sample predictions vs ground truth
6. Should complete in under 5 minutes on any GPU

## Future Milestones (not in scope for this design)

- Milestone 2: Full training on 360K examples, benchmark vs GLiNER2
- Milestone 3: ONNX export + INT8 quantization + latency benchmarks
- Milestone 4: PII-specific pretraining (NuNER paradigm)
- Milestone 5: Adversarial evaluation, curriculum learning, production integration
