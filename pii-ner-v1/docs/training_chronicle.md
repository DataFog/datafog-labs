# DataFog PII-NER: Training Chronicle

A record of building a 22.7M parameter PII detection model from architecture research through first production training run and v1.1 improvements.

---

## Phase 0: Research & Architecture Design

**Commits:** `755f034` → `889a36e`

We started with an extensive research phase — 2,800+ lines of reports across 8 documents surveying 26 model architectures, 9 PII datasets, and the competitive landscape (GLiNER2, Presidio, Microsoft). The research identified three genuine gaps in the PII NER literature:

1. No existing model combines character-level pattern recognition with contextual embeddings for PII
2. No tiered evaluation framework prioritizes critical PII (SSN, credit cards) over general entities
3. No small (<50M param) model specifically targets PII at production latency (<50ms)

**Architecture decision:** DeBERTa-v3-xsmall (22M params, pretrained) + CharCNN (0.3M) + Gating Fusion (0.1M) + CRF Head (0.2M). The key insight: structured PII like SSNs and credit cards have rigid character patterns (XXX-XX-XXXX), while soft PII like names and addresses need contextual understanding. The gating fusion lets the model dynamically weight character features vs. contextual features per token.

**Data:** 360K examples from three open-licensed HuggingFace datasets — AI4Privacy (200K, Apache 2.0), NVIDIA Nemotron-PII (100K, CC-BY-4.0), and Gretel Synthetic Finance (56K, Apache 2.0). Unified into a 4-tier taxonomy of 44 canonical entity types (89 BIO labels).

---

## Phase 1: Smoke Test (Overfit Validation)

**Commits:** `9bea601` → `9922e44`

The first milestone was proving the model wiring works by overfitting 100 examples for 50 epochs on a free Colab T4.

### Run 1: Flat Learning Rate — Total Failure

Used a single 5e-4 learning rate for the entire model. After 50 epochs: F1 = 0.000. Not a single entity predicted. The model learned to predict "O" (not-an-entity) for every token — a local minimum that reduces loss because ~85% of tokens are "O".

**Root cause:** A flat LR simultaneously destroys the pretrained DeBERTa weights (too high for fine-tuning) and learns the CRF head too slowly (too low for random initialization). The pretrained/random asymmetry demands different learning rates.

### Run 2: Differential Learning Rates — Success

Set backbone LR to 2e-5 (gentle fine-tuning) and head LR to 1e-3 (fast learning from scratch) — a 50x ratio. Added 10% warmup and weight decay 0.01.

The training curve was dramatic: 11 epochs of zero F1 (CRF learning internal transition probabilities), then a phase transition at epoch 12 where the first entities appeared. By epoch 50: **F1 = 0.947**, near-perfect overfit.

Entities emerged in order of frequency and pattern distinctiveness: Person (epoch 12) → Location (15) → Email/Credit Card (17) → Date (18) → Phone (20) → SSN (30). This order made intuitive sense — common contextual entities first, then pattern-based entities.

**Lesson learned:** Differential learning rates aren't optional when combining pretrained and randomly initialized components. This is well-known in NLP but the failure mode with CRF (silent all-O predictions with decreasing loss) is uniquely deceptive.

We also had to adjust success thresholds. CRF loss operates on a fundamentally different scale than cross-entropy (sequence-level partition function over 89^seq_len paths). A CRF loss of 5.3 at convergence means ~45% probability on the exact correct sequence — excellent, but nowhere near 0.1.

### Colab Integration Issues

Spent 10 commits fixing Colab environment issues: GPU memory attribute names changed between PyTorch versions, pip install bracket escaping, AI4Privacy column name detection, HuggingFace `List` feature type breaking `getattr`, and FP16 numerical instability in the CRF head.

---

## Phase 2: The NaN Gauntlet (Full Training Debugging)

**Commits:** `8da5cba` → `c43788b`

Scaling from 100 examples to 360K revealed three distinct sources of NaN, each requiring a different fix. This was the hardest part of the project.

### NaN Source 1: CRF Emission Overflow

**Symptom:** Occasional NaN loss during training.
**Cause:** The CRF's log-sum-exp forward algorithm overflows when emission values are extreme.
**Fix:** Clamp emissions to [-100, 100]. Safe because the CRF only uses relative differences between scores.

### NaN Source 2: AdamW Bias-Correction Amplification

**Symptom:** Training loss appears as 0.000000 from step 2 onward (HF Trainer v5.0 renders NaN as 0). All metrics zero.
**Cause:** AdamW's adaptive scaling makes the effective update ≈ ±lr × sign(gradient), independent of gradient magnitude. With standard eps=1e-8, every backbone weight changes by ±2e-5 per step — enough to push DeBERTa's internal representations into NaN territory on PyTorch 2.9+.

We confirmed this through systematic ablation:
- Default AdamW: NaN at step 2
- Reference AdamW (no CUDA fusion): NaN at step 2 — not a CUDA kernel bug
- Manual SGD (p -= lr * grad): OK — raw updates are safe
- Frozen DeBERTa (head only): OK — head learns without backbone
- **AdamW with eps=1.0 for backbone: OK — fix confirmed**

**Fix:** Set eps=1.0 in the backbone parameter group. This dampens adaptive scaling so updates are proportional to gradient magnitude (SGD-like) rather than a fixed ±lr. The head retains standard eps=1e-8 because it benefits from adaptive scaling as a randomly initialized component.

### NaN Source 3: BF16 Mantissa Precision

**Symptom:** Identical to NaN #2 — training loss 0.000000, all metrics zero. Even with the eps=1.0 fix present.
**Cause:** BF16 has only 7 bits of mantissa (vs 10 for FP16, 23 for FP32). DeBERTa v3's disentangled attention with relative position encodings involves large attention scores before softmax, exponentiation, and layer norm with small denominators. Under BF16, these operations lose enough precision to produce NaN — especially with real text at full sequence length, where attention patterns are more concentrated than synthetic random data.

**Why the preflight didn't catch it:** The original validation used batch_size=4 and seq_len=32 with random data. Smaller sequences and random tokens produce diffuse attention patterns that don't trigger BF16 overflow.

### NaN Source 4: FP16 Gradient Scaler Crash

**Symptom:** `ValueError: Attempting to unscale FP16 gradients` — Accelerate's gradient scaler crashes with custom optimizer parameter groups.
**Cause:** HF Accelerate 1.x+ changed how it handles mixed precision with custom optimizers. The gradient scaler expected specific parameter group structure.

### The Final Precision Strategy

After fighting all four NaN sources, we landed on a robust auto-selection strategy:

1. Try BF16 first (no gradient scaler, simpler) — works on RTX 3090 (compute 8.6)
2. Fall back to FP32 if BF16 fails
3. Skip FP16 entirely (gradient scaler incompatibility is unfixable without forking Accelerate)
4. Always force FP32 for CRF layer via `fused.float()` + `autocast(enabled=False)`

The preflight check was upgraded to use full batch size/sequence length, 5 steps, explicit NaN checks on loss and all parameters, and loss-decreasing verification.

---

## Phase 3: V1 Full Training

**Commit:** `1ebfb3e`

Trained on A100 (Colab) with BF16, 10 epochs, effective batch size 32.

### Training Curve

| Epoch | Val Loss | F1 | Notes |
|-------|----------|------|-------|
| 1 | 13.96 | 0.818 | Strong start — pretrained backbone carrying |
| 3 | 7.86 | 0.889 | Steady improvement |
| 5 | 6.17 | **0.903** | **Best checkpoint** |
| 6 | 10.66 | 0.886 | Regression begins |
| 7 | 10.24 | 0.886 | Train loss spiked 2.96 → 5.81 |

Best model restored from epoch 5 via `load_best_model_at_end=True`.

### Test Set Results

| Metric | Score |
|--------|-------|
| Overall F1 | 0.904 |
| Overall Precision | 0.907 |
| Overall Recall | 0.902 |

### Tier Recall vs Targets

| Tier | Recall | Target | Status |
|------|--------|--------|--------|
| Tier 1 (SSN, Credit Card, Passport, ...) | 0.722 | >= 0.98 | **FAIL** |
| Tier 2 (Person, Email, Phone, ...) | 0.934 | >= 0.95 | **FAIL** |
| Tier 3 (Username, Date, Location, ...) | 0.919 | >= 0.90 | PASS |
| Tier 4 (Medical Record, Employee ID, ...) | 0.866 | >= 0.85 | PASS |

### What We Learned

**Overall F1 of 0.904 is strong for a 22.7M param model** — competitive with much larger models (Knowledgator GLiNER-PII best F1: 0.81). But the tier breakdown revealed a critical gap: the entities that matter most (financial, identity documents) have the worst recall.

**Late-epoch regression** at epochs 6-7 indicated the backbone LR of 2e-5 was too aggressive for extended training. The cosine schedule or lower LR was needed.

Model uploaded to HuggingFace Hub under the DataFog org.

---

## Phase 4: Diagnosing the Tier 1 Gap

**Commit:** `53e656f`

### Entity Frequency Audit

Ran a comprehensive audit across all 360K training examples. The results explained everything:

| Entity Type | Count | % of Total | F1 |
|-------------|-------|-----------|------|
| DATE | 169,799 | 19.1% | 0.960 |
| PERSON | 163,418 | 18.4% | 0.951 |
| LOCATION | 97,834 | 11.0% | 0.918 |
| ... | | | |
| BANK_ACCOUNT | 1,001 | 0.11% | 0.791 |
| PASSPORT_NUMBER | 526 | **0.06%** | **0.469** |

**323x imbalance** between the most common entity (DATE: 170K) and the rarest (PASSPORT_NUMBER: 526). Direct correlation between frequency and model performance. The model isn't architecturally weak on Tier 1 — it just doesn't see enough examples.

Additionally, **16 entity types have zero training examples** (NATIONALITY, ETHNICITY, RELIGION, MARITAL_STATUS, and 12 others). These exist in our taxonomy but none of the three datasets annotate them.

---

## Phase 5: V1.1 Improvements

**Commits:** `53e656f` → `350a1b1`

Three targeted fixes based on the audit findings:

### 1. Tier-Weighted CRF Loss

Sequences containing critical PII now get proportionally higher loss weight:
- Tier 1 entities: 3x weight (SSN, Credit Card, Passport, etc.)
- Tier 2 entities: 2x weight (Person, Email, Phone, etc.)
- Tier 3 entities: 1.5x weight
- Tier 4 entities: 1x weight (baseline)

Implementation: CRF returns per-sequence log-likelihoods (reduction="none"), each weighted by the max label weight in that sequence. This means a batch with one SSN example and seven date-only examples will allocate 3x more gradient signal to the SSN example.

### 2. Oversampling Rare Entities

Examples containing Tier 1 entities are duplicated 3x in the training set. Applied after train/val/test split to avoid data leakage. This doesn't create new data but ensures the model sees critical PII examples ~4x more often per epoch.

### 3. Training Dynamics

- Backbone LR halved: 2e-5 → 1e-5 (prevents the epoch 6-7 regression)
- Cosine LR schedule (smoother decay than linear)
- 15 epochs (from 10) with 500-step warmup
- Best checkpoint selection by `overall_f1`

### Preflight Verification

Ran on RTX 3090: BF16 loss 1166.0 → 1150.6 (decreasing). All changes verified compatible with the precision strategy.

### Inference Pipeline

Built `PiiPipeline` — a simple API for PII detection:

```python
from datafog_pii_ner.inference import PiiPipeline

pipeline = PiiPipeline.from_pretrained("DataFog/pii-ner-v1")
entities = pipeline("My SSN is 123-45-6789")
# [PiiEntity(text='123-45-6789', label='SSN', start=10, end=21, tier=1)]
```

Supports HuggingFace Hub loading, character-level offsets, BIO→span decoding, and batch input. 9 tests, all passing (40 total across the project).

---

## Commit Log Summary

| Phase | Commits | Key Outcome |
|-------|---------|-------------|
| Research | 2 | Architecture decision, design doc |
| Smoke Test | 18 | F1=0.947 on 100 examples, Colab env fixes |
| NaN Debugging | 5 | Four distinct NaN sources identified and fixed |
| Precision Strategy | 3 | Auto-select BF16→FP32 fallback |
| V1 Full Training | 1 | F1=0.904 on 360K examples, model on HuggingFace |
| V1.1 Improvements | 3 | Tier-weighted loss, oversampling, inference pipeline |
| **Total** | **32** | |

---

## Open Questions for V1.1+

1. **Will tier-weighted loss + oversampling close the Tier 1 gap?** The 323x imbalance is severe. Oversampling helps but doesn't create genuinely new examples. If Tier 1 recall stays below 0.90, we may need synthetic data augmentation.

2. **16 zero-occurrence entity types** — NATIONALITY, ETHNICITY, RELIGION, MARITAL_STATUS, MEDICAL_RECORD, STUDENT_ID, DEVICE_ID, CRYPTO_WALLET, INSURANCE_NUMBER, SALARY, CRIMINAL_RECORD, POLITICAL_AFFILIATION, SEXUAL_ORIENTATION, HEALTH_CONDITION, GENETIC_DATA, TRADE_UNION. These can't be learned without new data sources. Should we drop them from the taxonomy or find additional datasets?

3. **ONNX export** — The CRF Viterbi decode doesn't export cleanly from pytorch-crf. May need a pure-PyTorch reimplementation (~50 lines) for the ONNX path.

4. **Backbone scaling** — DeBERTa-v3-xsmall (22M) was chosen for latency. If we need more capacity for Tier 1, DeBERTa-v3-small (44M) or DeBERTa-v3-base (86M) are the next steps, with corresponding latency trade-offs.
