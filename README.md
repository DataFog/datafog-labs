# DataFog Labs

Open research and development for lightweight PII detection models. This repo contains the full training code, experiment history, and research behind [DataFog](https://datafog.ai)'s PII-NER model family.

**Latest checkpoint:** [DataFog/pii-small-en](https://huggingface.co/DataFog/pii-small-en) on HuggingFace (v1.3, Apache 2.0)

## PII-NER v1

A 22.7M parameter model for detecting 44 types of personally identifiable information in English text. Combines a pretrained DeBERTa-v3-xsmall backbone with a character CNN encoder, adaptive gating fusion, and CRF output layer.

```
Input Text
    |
[Tokenization + Word-to-Char mapping]
    |
DeBERTa-v3-xsmall (22M)  +  CharCNN (0.3M)
    |                            |
    +-------> Gating Fusion <----+
                  |
             CRF Head (0.2M)
                  |
         BIO Tag Predictions (Viterbi decode)
                  |
         Span-level PII Entities
```

The gating fusion dynamically weights character-level features (for structured PII like SSNs and credit cards) against contextual features (for soft PII like names and addresses) on a per-token basis.

### Results

Best results from v1.3 training on H100 (20 hours, 10 epochs):

| Metric | V1.0 | V1.1 | V1.2 | V1.3 |
|--------|------|------|------|------|
| **Overall F1** | 0.904 | 0.901 | 0.901 | **0.907** |
| Precision | 0.907 | 0.906 | 0.905 | 0.898 |
| Recall | 0.902 | 0.895 | 0.896 | **0.916** |
| Tier 1 Recall (SSN, Credit Card, ...) | 0.722 | 0.771 | **0.841** | 0.823 |
| Tier 2 Recall (Person, Email, Phone, ...) | 0.934 | 0.933 | 0.936 | **0.945** |
| Tier 3 Recall (Username, Date, Location, ...) | 0.919 | 0.908 | 0.911 | **0.930** |
| Tier 4 Recall (Employee ID, IBAN, ...) | 0.866 | 0.844 | 0.845 | **0.868** |

V1.3 has the best overall F1 and recall. V1.2 is better for Tier 1-critical deployments (0.841 vs 0.823).

### Top entity F1 scores (v1.3)

| Entity | F1 | Entity | F1 |
|--------|------|--------|------|
| URL | 0.994 | License Plate | 0.952 |
| Biometric | 0.992 | Gender | 0.946 |
| IP Address | 0.988 | Employee ID | 0.940 |
| Date of Birth | 0.981 | IBAN | 0.935 |
| Vehicle ID | 0.976 | Username | 0.930 |
| Email | 0.968 | SSN | 0.930 |
| Phone | 0.966 | Location | 0.929 |

### Quick start

```python
from datafog_pii_ner.inference import PiiPipeline

pipeline = PiiPipeline.from_pretrained("DataFog/pii-small-en")
entities = pipeline("My SSN is 123-45-6789 and email is john@example.com")
# [PiiEntity(text='123-45-6789', label='SSN', start=10, end=21, tier=1),
#  PiiEntity(text='john@example.com', label='EMAIL', start=32, end=48, tier=2)]
```

### Setup

```bash
cd pii-ner-v1
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check src/ tests/ scripts/

# Smoke test (requires GPU)
python -m scripts.smoke_test

# Full training
python scripts/train_v1.3.py --config configs/h100-v1.3.yaml
```

### Evaluation

```bash
python scripts/eval_benchmark.py \
  --model datafog \
  --model-path DataFog/pii-small-en \
  --dataset combined \
  --split test
```

See [eval_benchmark.md](pii-ner-v1/docs/eval_benchmark.md) for flags and options.

## Training data

| Dataset | Size | License |
|---------|------|---------|
| [AI4Privacy](https://huggingface.co/datasets/ai4privacy/pii-masking-200k) | ~200K examples | Apache 2.0 |
| [NVIDIA Nemotron-PII](https://huggingface.co/datasets/nvidia/Nemotron-PII) | ~100K examples | CC-BY-4.0 |
| [Gretel Synthetic PII Finance](https://huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual) | ~56K examples | Apache 2.0 |

Combined: ~169K English examples after filtering and dedup. 44 canonical entity types across 4 sensitivity tiers, unified into 89 BIO labels. The dataset has a 323x frequency imbalance (DATE: 170K occurrences vs PASSPORT: 526), which drives many of the training innovations below.

## Documentation

| Document | Description |
|----------|-------------|
| [Training Chronicle](pii-ner-v1/docs/training_chronicle.md) | Full narrative of the ML journey: 4 NaN sources, backbone instability, tier-weighted loss, freezing experiments |
| [Smoke Test Walkthrough](pii-ner-v1/docs/smoke_test_walkthrough.md) | Why differential learning rates are essential for pretrained+CRF architectures |
| [Evaluation Harness](pii-ner-v1/docs/eval_benchmark.md) | Head-to-head model comparison on the same test split |
| [Design Document](docs/plans/2026-02-01-pii-ner-v1-design.md) | Original architecture decisions and project structure |

### Research

The [RESEARCH/](RESEARCH/pii-model-architecture/) directory contains the pre-implementation research: 8 reports surveying 26 architectures, 9 PII datasets, and the competitive landscape. Includes a [29-slide interactive architecture guide](RESEARCH/pii-model-architecture/architecture-guide.html).

Key finding: no published work combines differentiable character-level pattern recognition with contextual transformers specifically for PII detection.

## Development log

| Date | Version | What changed |
|------|---------|-------------|
| 2026-02-07 | **v1.3** | Best F1 (0.907). Early backbone freeze (epoch 3) + progressive tier weight reduction. Discovered training spikes originate in head components, not backbone. |
| 2026-02-05 | v1.2 | Best Tier 1 recall (0.841). Backbone freezing after epoch 4. Epoch 3 identified as consistent sweet spot. |
| 2026-02-04 | v1.1 | Tier-weighted CRF loss (3x for Tier 1), rare entity oversampling, inference pipeline. Tier 1 recall +4.9pts. |
| 2026-02-04 | — | Training chronicle, entity frequency audit (323x imbalance discovered). |
| 2026-02-03 | v1.0 | First full training on A100. F1=0.904 on 360K examples. Model uploaded to HuggingFace. |
| 2026-02-03 | — | NaN gauntlet: 4 distinct NaN sources identified and fixed (CRF overflow, AdamW bias-correction, BF16 mantissa, FP16 gradient scaler). |
| 2026-02-02 | — | Smoke test passed (F1=0.947 on 100 examples). Differential learning rates proven essential. |
| 2026-02-01 | — | Architecture design. Research phase complete (2,800+ lines across 8 reports). |

## Key technical findings

1. **Differential learning rates are non-negotiable.** A flat LR across pretrained backbone + random CRF head produces F1=0.000. A 50x ratio (backbone 2e-5, head 1e-3) is needed.

2. **AdamW eps=1.0 for pretrained backbones.** Standard eps=1e-8 makes effective updates ~±lr regardless of gradient magnitude, causing NaN on DeBERTa with PyTorch 2.9+. Setting eps=1.0 restores gradient-proportional updates.

3. **The training spike is a head problem, not backbone.** V1.3 proved this definitively: the spike occurred at epoch 5 with the backbone already frozen since epoch 3. The CharCNN/GatingFusion/CRF destabilize under continued training.

4. **Epoch 3 is consistently the best checkpoint.** Across v1.2 and v1.3, the model peaks at epoch 3 then destabilizes. Earlier representations generalize better.

5. **Tier-weighted loss works but amplifies instability.** 3x weight + 3x oversampling = ~9x gradient signal for Tier 1, which accelerates learning but accumulates damage.

## Open problems

- **Tier 1 recall gap**: 0.823 vs 0.98 target. Passport number (0.426 F1) has only 526 training examples.
- **16 zero-occurrence entity types**: NATIONALITY, ETHNICITY, RELIGION, etc. exist in the taxonomy but no training data covers them.
- **Head instability**: Root cause of the epoch 3+ training spike is unknown. Gradient clipping, per-component LR decay, or early stopping are candidate fixes.
- **ONNX export**: CRF Viterbi decode doesn't export cleanly; needs pure-PyTorch reimplementation.

## Project structure

```
datafog-labs/
├── pii-ner-v1/
│   ├── src/datafog_pii_ner/      # Model, data pipeline, training, inference
│   ├── scripts/                   # Training runners, evaluation, data download
│   ├── configs/                   # YAML configs per GPU/version
│   ├── tests/                     # Unit + integration tests (6 modules)
│   ├── notebooks/                 # Experiment notebooks (Colab/local)
│   └── docs/                      # Training chronicle, eval docs
├── RESEARCH/                      # Pre-implementation research (8 reports)
├── docs/plans/                    # Design documents
└── .github/workflows/ci.yml      # Lint + test CI
```

## License

Apache 2.0
