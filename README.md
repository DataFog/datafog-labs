# DataFog Labs

Research and development for DataFog's PII detection models.

## Projects

### [PII-NER v1](pii-ner-v1/)

A 22.7M parameter PII detection model combining DeBERTa-v3-xsmall with a character CNN encoder, sigmoid gating fusion, and CRF output layer. Trained on 360K+ open-licensed PII examples.

**Architecture:** DeBERTa-v3-xsmall (contextual) + CharCNN (character patterns) → Gating Fusion → CRF (constrained BIO decoding)

**Key features:**
- 163 BIO tags covering ~81 PII entity types across 4 sensitivity tiers
- Adaptive gating between character-level patterns (for structured PII like SSNs) and contextual features (for soft PII like names)
- CRF output layer enforcing valid BIO sequences
- Differential learning rates for pretrained backbone vs randomly initialized head

**Status:** Smoke test complete (F1=0.947 on 100 examples). Full training pending.

```bash
cd pii-ner-v1

# Install
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check src/ tests/ scripts/

# Smoke test (requires GPU)
python -m scripts.smoke_test
```

### [Research](RESEARCH/)

Architecture survey, training data catalog, evaluation framework, and gap analysis for PII-NER model design. Includes a 29-slide interactive architecture guide.

## Data Sources

| Dataset | Size | License |
|---------|------|---------|
| [AI4Privacy](https://huggingface.co/datasets/ai4privacy/pii-masking-200k) | ~200K | Apache 2.0 |
| [NVIDIA Nemotron-PII](https://huggingface.co/datasets/nvidia/Nemotron-PII) | ~100K | CC-BY-4.0 |
| [Gretel Synthetic PII Finance](https://huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual) | ~56K | Apache 2.0 |

## License

Apache 2.0
