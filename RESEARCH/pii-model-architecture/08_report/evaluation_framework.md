# Comprehensive PII Detection Evaluation Framework

## For DataFog's Lightweight PII Detection Model (<50M params)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Standard NER Evaluation Metrics](#2-standard-ner-evaluation-metrics)
3. [PII-Specific Evaluation Considerations](#3-pii-specific-evaluation-considerations)
4. [Existing PII Benchmarks and Datasets](#4-existing-pii-benchmarks-and-datasets)
5. [Latency and Throughput Benchmarking](#5-latency-and-throughput-benchmarking)
6. [Cross-Domain Generalization Evaluation](#6-cross-domain-generalization-evaluation)
7. [Adversarial and Robustness Evaluation](#7-adversarial-and-robustness-evaluation)
8. [Recommended Evaluation Protocol for DataFog](#8-recommended-evaluation-protocol-for-datafog)
9. [Sources](#9-sources)

---

## 1. Executive Summary

This document presents a comprehensive evaluation framework for DataFog's novel lightweight PII detection model. Standard NER evaluation (entity-level F1 via seqeval) is necessary but insufficient for PII systems. PII detection is a safety-critical task where **missing sensitive data carries far greater cost than false positives**, which demands recall-oriented metrics, per-entity-type analysis across sensitivity tiers, adversarial robustness testing, cross-domain generalization evaluation, and rigorous latency benchmarking.

We recommend a **multi-tier evaluation approach** that combines:

- **Tier 1 (Core Accuracy)**: Entity-level F1 via seqeval (strict mode, IOB2), plus nervaluate four-schema evaluation (strict/exact/partial/type), plus the recall-oriented F-beta scores (F2 and F5).
- **Tier 2 (PII-Specific)**: Per-entity-type recall stratified by sensitivity level, false positive rate analysis, context-dependent PII evaluation, and weighted aggregate metrics.
- **Tier 3 (Operational)**: Latency benchmarking (CPU and GPU, single and batch), throughput (tokens/sec and docs/sec), ONNX Runtime performance, and memory footprint.
- **Tier 4 (Robustness)**: Cross-domain generalization (CrossNER + domain-specific PII), adversarial perturbation resilience, edge case handling (nested, partial, multilingual, obfuscated PII).

---

## 2. Standard NER Evaluation Metrics

### 2.1 Entity-Level F1: Strict vs. Relaxed Matching

The standard approach for NER evaluation is **entity-level F1**, where a predicted entity is only counted as a true positive if both the entity type and span boundaries match the gold standard exactly. This is the convention used by the CoNLL shared tasks and implemented in the `seqeval` library.

However, strict entity-level F1 has a well-documented problem: it **double-penalizes partial matches**. If a model predicts "James Earle" instead of "James Earle Jones", strict evaluation counts this as one false positive AND one false negative, despite the prediction being substantially correct. Chris Manning noted this issue as early as 2006, arguing that F1 alone is a poor optimization target for NER.

The `nervaluate` library (based on SemEval'13 metrics) addresses this by providing **four evaluation schemas**:

| Schema | Entity Type Must Match | Boundaries Must Match | Partial Credit |
|--------|----------------------|----------------------|----------------|
| **Strict** | Yes | Exact | No |
| **Exact** | No (ignored) | Exact | No |
| **Partial** | No (ignored) | Overlap sufficient | Yes (0.5 credit) |
| **Type** | Yes | Overlap sufficient | Yes (0.5 credit) |

For partial/relaxed matching, precision and recall are computed as:
- Precision = (COR + 0.5 x PAR) / ACT
- Recall = (COR + 0.5 x PAR) / POS

where COR = correct, PAR = partial match, ACT = actual predictions, POS = possible (gold entities).

**Recommendation**: Report all four nervaluate schemas. Strict F1 is the primary comparison metric (matches community convention), but partial and type metrics reveal whether errors are boundary issues vs. type confusion issues -- a critical distinction for PII redaction where even an approximate boundary capture may be sufficient for privacy protection.

### 2.2 Token-Level F1 (BIO-Based)

Token-level evaluation assesses each token independently: is it correctly tagged as B-PERSON, I-PERSON, O, etc.? This is simpler to compute but has two weaknesses:

1. It inflates scores because O-tagged tokens vastly outnumber entity tokens (often 95%+ of tokens are O), leading to high accuracy that masks poor entity detection.
2. It does not capture whether multi-token entities are detected as coherent spans.

Token-level metrics are still useful for **debugging** (identifying which token positions are most error-prone) and for **training diagnostics** (monitoring per-token loss), but should not be the primary reported metric.

**Recommendation**: Track token-level F1 during development for diagnostic purposes, but report entity-level metrics in all comparisons and publications.

### 2.3 seqeval Library and Its Evaluation Modes

`seqeval` is the de facto standard Python library for sequence labeling evaluation. It supports two modes:

**Default mode** (conlleval-compatible): More lenient. If the gold label is `B-PER` but the prediction is `I-PER` (wrong prefix, right type), default mode still counts it as correct. This simulates the original Perl `conlleval` script behavior.

**Strict mode**: Requires exact prefix matching according to the specified tagging scheme. With `mode='strict'` and `scheme=IOB2`, a prediction of `I-PER` where `B-PER` was expected counts as incorrect.

Supported tagging schemes: IOB1, IOB2, IOE1, IOE2, IOBES, BILOU.

```python
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2

# Strict evaluation with IOB2 scheme
report = classification_report(y_true, y_pred, mode='strict', scheme=IOB2)
```

**Recommendation**: Always use `mode='strict'` with `scheme=IOB2` for reported results. The lenient default mode can mask tagging errors that indicate the model has not properly learned entity boundary detection.

### 2.4 CoNLL Evaluation Script Conventions

The original CoNLL `conlleval` Perl script established the conventions used throughout the NER community:

- **Input format**: Tab-separated columns with token, gold tag, and predicted tag; sentences separated by blank lines.
- **Evaluation granularity**: Phrase-based (entity-level), not token-based. An entity is correct only if the entire span matches.
- **Reported metrics**: Per-entity-type precision, recall, and F1 (FB1), plus overall micro-averaged metrics.
- **Output format**: "processed N tokens with M phrases; found: K phrases; correct: J" followed by per-type breakdown.

Python re-implementations exist (e.g., `conlleval.py` by spyysalo) that produce identical output. The `seqeval` library in default mode is designed to reproduce `conlleval` behavior.

**Recommendation**: Use `seqeval` for all evaluations (it subsumes `conlleval` behavior) and report results in the standard per-entity-type table format.

### 2.5 Span-Based Evaluation Summary

| Evaluation Type | What It Measures | Tool | When to Use |
|----------------|-----------------|------|-------------|
| **Token-level** | Per-token tag accuracy | sklearn, seqeval | Debugging, training monitoring |
| **Entity-level strict** | Exact span + type match | seqeval (strict), nervaluate (strict) | Primary reported metric |
| **Entity-level partial** | Overlapping span, type match | nervaluate (type schema) | Understanding boundary errors |
| **Boundary-only** | Exact span, type ignored | nervaluate (exact schema) | Evaluating span detection ability |
| **Overlap-only** | Any overlap, type ignored | nervaluate (partial schema) | Most lenient; useful for recall analysis |

---

## 3. PII-Specific Evaluation Considerations

### 3.1 Why Standard NER F1 Is Insufficient for PII

Standard NER F1 treats all entity types equally and balances precision and recall equally. Both assumptions are wrong for PII detection:

1. **Asymmetric error costs**: Missing a Social Security Number (false negative) in a dataset meant for public release can lead to identity theft and regulatory fines up to 20M EUR under GDPR or 4% of global turnover. A false positive (flagging "John" as a name when it is a common noun) merely causes unnecessary redaction. The cost asymmetry is extreme.

2. **Non-uniform entity importance**: An SSN or credit card number leaking is catastrophically worse than leaking a generic date. Standard micro-averaged F1 treats a missed DATE the same as a missed SSN.

3. **Context dependence**: "Dr. Lee" may be PII in a medical record (PHI under HIPAA) but not in a news article about a public figure. Standard NER evaluation does not capture this distinction.

4. **Boundary tolerance**: For privacy purposes, redacting "John Smith Jr." as "[REDACTED] Jr." is a privacy failure even though it partially detected the name. Standard partial-match metrics might give credit here, but the privacy objective requires complete coverage.

### 3.2 Per-Entity-Type Recall by Sensitivity Level

We propose stratifying PII types into three sensitivity tiers and setting minimum recall thresholds for each:

**Tier 1 -- Critical (target recall >= 0.98)**:
- Social Security Numbers / Government IDs
- Credit card numbers / Financial account numbers
- Passwords / API keys / Credentials
- Biometric identifiers
- Medical record numbers (PHI under HIPAA)

**Tier 2 -- High (target recall >= 0.95)**:
- Full names (especially when combined with other identifiers)
- Email addresses
- Phone numbers
- Physical addresses
- Dates of birth
- IP addresses

**Tier 3 -- Moderate (target recall >= 0.90)**:
- Usernames / Online handles
- URLs (personal)
- Generic dates (when not DOB)
- Organization names (when they serve as indirect identifiers)
- Partial identifiers (age, gender, nationality when contextually identifying)

**Recommendation**: Report per-entity-type recall separately for each tier. The model's overall PII readiness should be judged by whether it meets the minimum recall threshold for every Tier 1 entity type. A model that achieves 0.99 F1 overall but only 0.85 recall on SSNs is not production-ready for privacy applications.

### 3.3 Recall-Oriented Metrics: F-Beta Scores

The F-beta score generalizes the F1 score by weighting recall more heavily when beta > 1:

```
F_beta = (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall)
```

| Metric | Beta | Recall Weight | Use Case |
|--------|------|--------------|----------|
| F1 | 1 | Equal to precision | Standard NER comparison |
| F2 | 2 | 4x precision | General PII detection (Presidio recommendation) |
| F5 | 5 | 25x precision | High-stakes PII (Kaggle PII competition metric) |

The 2024 Kaggle PII Data Detection competition used **micro F5** as its evaluation metric, meaning recall was weighted 25x more heavily than precision. The winning solution (ensemble of DeBERTa-v3-large models) achieved 0.953 F5 on the hidden test set.

Microsoft Presidio's documentation recommends **F2** for general PII evaluation, noting that "recall is often more important than precision, as we'd like to avoid missing any PII."

**Recommendation**: Report F1 (for comparability with NER literature), F2 (for general PII assessment), and F5 (for high-stakes privacy comparison with Kaggle baselines). F2 should be the primary optimization target during training.

### 3.4 False Positive Rate and Its Impact on Usability

While recall is paramount for safety, excessively low precision creates usability problems:

- **Over-redaction**: Aggressive false positives render documents unreadable. If 30% of flagged text is not actually PII, users lose trust in the system and may disable it.
- **Manual review burden**: Each false positive requires human review in regulated workflows, creating operational cost.
- **Downstream task degradation**: If PII-redacted text is used for training or analytics, excessive redaction removes signal.

**Recommendation**: Track and report:
- **Entity-level false positive rate**: FP / (FP + TN), measured at the entity level.
- **PII-to-text false positive ratio**: Number of false positive entity spans / total tokens in the document. Private AI calls this "PII missed (% of all words)" and its inverse is equally informative.
- **Per-entity-type precision**: Some types (e.g., DATE, ORG) are notorious for false positives; track these separately.

A practical target: precision should not drop below 0.80 for any entity type, even while optimizing for recall.

### 3.5 Weighted Metrics Based on PII Sensitivity Levels

To produce a single summary metric that accounts for sensitivity differences, we propose a **sensitivity-weighted F-beta score**:

```
Weighted_F2 = sum(w_i * F2_i * support_i) / sum(w_i * support_i)
```

Where:
- `w_i` = sensitivity weight for entity type i (Tier 1: 3.0, Tier 2: 2.0, Tier 3: 1.0)
- `F2_i` = F2 score for entity type i
- `support_i` = number of instances of entity type i in the evaluation set

This ensures that poor performance on SSNs (Tier 1, weight 3.0) drags down the aggregate score more than poor performance on generic dates (Tier 3, weight 1.0).

### 3.6 How Microsoft Presidio Evaluates Its Models

The `presidio-evaluator` package (from `presidio-research`) provides:

1. **Token-based evaluation**: Assesses per-token PII classification. Useful for understanding which parts of multi-token entities the model detects.
2. **Entity-level evaluation**: Standard precision/recall/F1 at the entity span level.
3. **Confusion matrix analysis**: Which PII types are confused with each other (e.g., PERSON vs. ORGANIZATION).
4. **Synthetic data generation**: Presidio includes a PII data generator for creating evaluation sets with controlled PII distributions.
5. **Configurable boosting**: Notebook 5 in presidio-research shows how to boost F-score by ~30% through configuration tuning (confidence thresholds, context enhancers, deny lists).

An Australian clinical study using Presidio reported F1 of 0.8980 (relaxed) and 0.8471 (strict), with recall of 0.9039 (relaxed) and 0.8064 (strict). This reveals the significant gap between relaxed and strict evaluation in real-world deployments.

### 3.7 Evaluating Context-Dependent PII Detection

Context-dependent PII is among the hardest evaluation challenges. Key approaches:

**PII-Bench (2025)** introduced a two-stage evaluation: (1) PII Recognition (Desc-F1) measuring raw entity detection, and (2) Query-Relevant PII Detection (Query-F1) measuring whether detected PII is actually sensitive in context. State-of-the-art LLMs achieved >0.90 Desc-F1 but <0.63 Query-F1, indicating that contextual relevance assessment is far harder than raw detection.

**NIST SP 800-122 guidelines** recommend evaluating PII sensitivity based on:
- The identifiability factor (how easily can this PII identify someone?)
- The context of use (what purpose is the data serving?)
- The combination factor (benign data that becomes identifying when combined)

**Presidio's ContextAwareEnhancer** uses surrounding text to adjust confidence scores. Evaluation should test both with and without context enhancement to measure its impact.

**Recommendation**: Include a context-dependent evaluation split in the test set where the same surface form (e.g., "Washington") appears as both PII (person name) and non-PII (location/common noun) and measure whether the model correctly disambiguates.

---

## 4. Existing PII Benchmarks and Datasets

### 4.1 Established PII Detection Benchmarks

There is **no single universally accepted PII detection leaderboard** comparable to GLUE/SuperGLUE for NLU. However, several benchmarks serve as de facto evaluation standards:

| Benchmark | Domain | Size | PII Types | Metric | Access |
|-----------|--------|------|-----------|--------|--------|
| **Kaggle PII 2024 (PIILO)** | Education (student essays) | ~22K essays | 7 types (NAME, EMAIL, USERNAME, ID_NUM, PHONE, ADDRESS, URL) | Micro F5 | CC BY 4.0 |
| **i2b2/n2c2 2014 De-id** | Clinical notes | 1,304 records | HIPAA PHI (25+ types) | Micro F1 | DUA required |
| **TAB (Text Anonymization Benchmark)** | Legal (ECHR court cases) | 1,268 cases | Comprehensive PII + disclosure risk | Privacy-oriented metrics | MIT License |
| **PII-Bench** | General (multi-scenario) | 2,842 samples | 55 fine-grained categories | Desc-F1 + Query-F1 | arXiv (2025) |
| **CrossNER** | 5 domains (AI, Lit, Music, Politics, Science) | ~100-200 per domain | Domain-specific entity types | Entity F1 | Open (AAAI 2021) |
| **BigCode PII** | Source code | 12,099 samples, 31 languages | PII in code (names, emails, keys, IPs) | Entity F1 | Hugging Face |
| **CoNLL-2003** | News (Reuters) | 22K sentences | PER, LOC, ORG, MISC | Entity F1 | Open |
| **Private AI benchmark** | Mixed domains | ~45K words | 50+ entity types | Recall-focused | Request required |
| **Protecto benchmark** | Mixed | Various | Multiple PII types | F1, recall, precision | Whitepaper |

### 4.2 Kaggle PII Data Detection Competition (2024)

The Learning Agency Lab's Kaggle competition is the most significant recent PII benchmark:

- **Dataset**: ~22,000 student essays (PIILO dataset) annotated in BIO format.
- **PII types**: NAME_STUDENT, EMAIL, USERNAME, ID_NUM, PHONE_NUM, URL_PERSONAL, STREET_ADDRESS.
- **Evaluation metric**: Micro F-beta with beta=5 (recall weighted 25x over precision).
- **Top result**: 0.953 F5 on hidden test set (1st place: ensemble of 5 DeBERTa-v3-large models).
- **Key insight**: The extreme recall weighting (beta=5) reflects the competition organizers' view that in educational data, missing PII is far worse than over-flagging.
- **License**: CC BY 4.0, freely available for research.

### 4.3 i2b2/n2c2 De-Identification Shared Tasks

The i2b2 (now n2c2) shared tasks are the gold standard for clinical de-identification:

- **2006 task**: De-identification of discharge summaries (HIPAA PHI categories).
- **2014 task**: De-identification of longitudinal clinical narratives with broader entity types than HIPAA alone.
- **Evaluation**: Micro F1 as primary metric, with both token-based and entity-based evaluation. The top 2014 system achieved F1 of 0.936 (strict).
- **95% threshold**: It has been suggested that 95% recall is the minimum for safe de-identification. In 2014, the top 4 systems met this threshold using token-based evaluation on HIPAA-PHI entities.
- **Access**: Available through the n2c2 portal under a Data Use Agreement.

### 4.4 Text Anonymization Benchmark (TAB)

TAB stands apart from traditional NER/de-identification benchmarks:

- **Corpus**: 1,268 ECHR court cases with comprehensive annotations.
- **Key innovation**: Annotated based on actual anonymization need (what must be masked to prevent re-identification of a specific individual), not merely semantic category.
- **Evaluation metrics**: Specifically designed for measuring both privacy protection level and text utility preservation.
- **Annotation layers**: Semantic category, identifier type, confidential attributes, and co-reference relations.
- **Growing adoption**: Citations increased from 4 (2022) to 12 (2024), indicating growing community interest.
- **License**: MIT License, available on GitHub and Hugging Face.

### 4.5 Summary of Recommended Evaluation Datasets

For DataFog's model, we recommend evaluating on the following combination:

| Dataset | Purpose | Priority |
|---------|---------|----------|
| **PIILO (Kaggle 2024)** | Primary accuracy benchmark; compare with competition baselines | Must-have |
| **CoNLL-2003** | Standard NER baseline; comparability with all NER literature | Must-have |
| **CrossNER** | Cross-domain generalization evaluation | Must-have |
| **TAB** | Privacy-oriented evaluation on legal text | Should-have |
| **i2b2 2014** | Clinical de-identification (if DUA obtainable) | Should-have |
| **BigCode PII** | PII in code domain | Nice-to-have |
| **PII-Bench** | Context-dependent PII evaluation | Nice-to-have |
| **Custom adversarial set** | Robustness testing (see Section 7) | Must-have |

---

## 5. Latency and Throughput Benchmarking

### 5.1 Why Latency Matters for DataFog

DataFog's value proposition centers on a model that is "dramatically smaller, faster, and more accurate" than GLiNER2 (205M params). Latency and throughput benchmarks are as important as accuracy metrics -- a model that is 2% better on F1 but 10x slower is not a practical improvement for production PII detection.

### 5.2 Key Metrics to Track

| Metric | Unit | What It Measures |
|--------|------|-----------------|
| **Single-document latency** | ms | End-to-end time for one document (P50, P90, P95, P99) |
| **Batch throughput** | docs/sec | Documents processed per second at optimal batch size |
| **Token throughput** | tokens/sec | Tokens processed per second |
| **First-token latency** | ms | Time to produce the first entity prediction |
| **Memory footprint** | MB | Peak RAM/VRAM usage during inference |
| **Model load time** | ms | Time from cold start to first inference |

### 5.3 CPU vs. GPU Benchmarking Protocol

**CPU benchmarking**:
- Report the exact CPU model (e.g., Intel Xeon Gold 6248R, Apple M2 Ultra).
- Fix the number of threads (test at 1, 4, and optimal).
- Use ONNX Runtime with CPU execution provider.
- Test with and without quantization (FP32, FP16, INT8).
- Report VNNI/AVX-512 availability (impacts ONNX Runtime performance significantly).

**GPU benchmarking**:
- Report GPU model, VRAM, and CUDA version.
- Test with CUDA execution provider.
- Warm up with 100 forward passes before timing.
- Test multiple batch sizes to find the throughput-optimal batch size.
- Report both latency (batch=1) and throughput (optimal batch).

**Critical protocol requirements**:
- Run at least 5 independent timing runs per configuration; report mean and standard deviation.
- Use fixed input lengths (test at 128, 256, and 512 tokens) for worst-case analysis.
- Use maximum input length for worst-case throughput numbers (as Microsoft did for Hypefactors' NER model).
- Flush caches between runs.
- Document all software versions (PyTorch, ONNX Runtime, CUDA, Python).

### 5.4 Tokens per Second vs. Documents per Second

Both metrics are important but measure different things:

- **Tokens/sec**: Pure model throughput, independent of document length. Best for comparing raw model speed.
- **Docs/sec**: Includes preprocessing (tokenization, windowing for long documents) and postprocessing (span assembly, deduplication). More representative of real-world performance.

For PII detection, documents vary enormously in length (a tweet vs. a legal contract), so report both metrics and specify document length distributions.

### 5.5 Batch vs. Single-Document Latency

| Scenario | Relevant Metric | Why |
|----------|----------------|-----|
| Real-time API (single request) | Single-doc latency (P99) | User is waiting |
| Batch processing (data pipeline) | Batch throughput (docs/sec) | Total processing time matters |
| Streaming (log ingestion) | Single-doc latency + throughput | Both matter |

Test at batch sizes: 1, 8, 16, 32, 64 (CPU) and 1, 16, 32, 64, 128, 256 (GPU).

### 5.6 ONNX Runtime Benchmarking Best Practices

Based on Microsoft's production NER deployment experience (Hypefactors case study):

1. **Convert PyTorch model to ONNX**: Use `torch.onnx.export()` with `opset_version=14` or later.
2. **Apply graph optimizations**: ONNX Runtime performs constant folding, node fusion, and redundant node elimination automatically. Use `onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL`.
3. **Quantize**: INT8 quantization via `onnxruntime.quantization` typically provides 2-4x speedup with <1% accuracy loss for NER models.
4. **Benchmark with `onnxruntime_perf_test`**: The official tool reports average inference time, throughput, and latency percentiles.
5. **Match optimization to target hardware**: Graph optimizations are hardware-specific; optimize on the same hardware type you will deploy on.
6. **Use `timeit.repeat()`**: For custom benchmarks, use `timeit.repeat(lambda: inference(), repeat=50, number=1)` and report min, mean, and P95.

Expected speedups (based on published results):
- ONNX Runtime vs. PyTorch (CPU, batch=1): ~20-25% faster (24.17ms vs. 30.39ms in published benchmarks).
- INT8 quantization on VNNI CPUs: Additional 2-4x speedup.
- FP16 on GPU: 1.5-2x speedup over FP32.

### 5.7 How GLiNER2 Benchmarked Latency

GLiNER2's Table 4 measured inference latency in milliseconds for text classification across varying numbers of labels:

- **Key finding**: GLiNER2 processes all labels in a single forward pass, maintaining consistent latency regardless of label count. DeBERTa requires a separate forward pass per label (6.8x slower with 20 labels).
- **CPU performance**: GLiNER2 achieves ~2.6x speedup over GPT-4o while running on standard CPU hardware.
- **Evaluation protocol**: All models evaluated on CPU except GPT-4o (API).
- **NER performance**: On CrossNER, GLiNER2 (205M params) achieves 0.590 overall F1 vs. GPT-4o's 0.599.

**Recommendation for DataFog**: Replicate GLiNER2's benchmarking protocol exactly (same hardware, same input lengths, same label counts) for direct comparison. Additionally, test with PII-specific label sets (7-16 PII types) since the number of labels directly impacts GLiNER-style models' latency.

### 5.8 Recommended Latency Comparison Table Format

```
| Model | Params | Hardware | Batch | Seq Len | Latency (ms) | Throughput (tok/s) |
|-------|--------|----------|-------|---------|--------------|--------------------|
| DataFog | <50M | CPU (spec) | 1 | 256 | XX | XX |
| DataFog | <50M | CPU (spec) | 32 | 256 | XX | XX |
| DataFog-ONNX | <50M | CPU (spec) | 1 | 256 | XX | XX |
| DataFog-INT8 | <50M | CPU (spec) | 1 | 256 | XX | XX |
| GLiNER2 | 205M | CPU (spec) | 1 | 256 | XX | XX |
| DeBERTa-v3-large | 304M | CPU (spec) | 1 | 256 | XX | XX |
| Presidio (default) | N/A | CPU (spec) | 1 | 256 | XX | XX |
```

---

## 6. Cross-Domain Generalization Evaluation

### 6.1 Why Cross-Domain Evaluation Matters for PII

PII appears in radically different contexts: medical records, legal contracts, financial documents, social media posts, source code, and student essays. A model trained on one domain may fail catastrophically on another. Published results show that models achieving 0.98 F1 on civilian datasets can drop to 0.41 F1 on clinical text, a >50% performance collapse.

### 6.2 CrossNER Benchmark

CrossNER (AAAI 2021) is the standard benchmark for cross-domain NER generalization:

- **Domains**: Politics, Natural Science, Music, Literature, Artificial Intelligence.
- **Design**: Small training sets (100-200 examples per domain) simulating low-resource conditions. Reuters News (CoNLL-2003) serves as the source domain.
- **Key finding**: Even the best models achieve <70% F1 on average, highlighting the difficulty of cross-domain transfer.
- **Evaluation protocol**: Train on source domain (CoNLL-2003), optionally fine-tune on small target domain set, evaluate on target domain test set.

GLiNER2 reports the following CrossNER results (zero-shot):

| Domain | GLiNER2 | GPT-4o |
|--------|---------|--------|
| AI | 0.547 | 0.526 |
| Literature | 0.564 | 0.561 |
| Music | 0.533 | 0.632 |
| Politics | 0.679 | 0.712 |
| Science | 0.627 | 0.563 |
| **Average** | **0.590** | **0.599** |

### 6.3 Domain-Specific PII Evaluation Protocol

Beyond CrossNER (which tests general NER, not PII specifically), we need domain-specific PII evaluation:

| Domain | Key PII Types | Data Source | Challenge |
|--------|--------------|-------------|-----------|
| **Medical** | PHI (patient names, MRNs, dates of treatment) | i2b2/n2c2 | Domain-specific vocabulary, abbreviations |
| **Legal** | Party names, case numbers, addresses | TAB (ECHR) | Long documents, nested entity references |
| **Financial** | Account numbers, SSNs, transaction IDs | Synthetic (Gretel, Mendeley) | Structured formats mixed with narrative |
| **Education** | Student names, IDs, emails | PIILO/Kaggle | Informal writing, varied formats |
| **Code** | API keys, passwords, emails in comments | BigCode PII | Code syntax interference with NER |
| **Social Media** | Usernames, handles, phone numbers in informal text | Custom (to build) | Abbreviations, emoji, code-switching |

### 6.4 Zero-Shot vs. Few-Shot Evaluation for New PII Types

**Zero-shot protocol**:
1. Train the model on a base set of PII types (e.g., the 7 Kaggle types).
2. Evaluate on held-out PII types never seen during training (e.g., CREDIT_CARD, SSN, MEDICAL_RECORD_NUMBER).
3. Provide only the type name/description as input (critical for GLiNER-style models).
4. Measure per-type F1 on the unseen types.

**Few-shot protocol** (progressive evaluation):
1. Start with zero-shot evaluation.
2. Provide 1, 5, 10, 50, 100 labeled examples of the new type.
3. Measure F1 at each shot count to produce a learning curve.
4. Compare with fine-tuning baselines (full retraining vs. adapter-based).

**Key finding from literature**: Zero-shot NER on biomedical entities achieves ~35% F1, 1-shot reaches ~50%, 10-shot reaches ~70%, and 100-shot reaches ~80%. The steepness of this curve is a critical measure of a model's sample efficiency.

**Recommendation**: Report both zero-shot and 10-shot performance on new PII types. A DataFog model that achieves good zero-shot PII detection is significantly more valuable than one requiring per-domain fine-tuning.

---

## 7. Adversarial and Robustness Evaluation

### 7.1 Adversarial Threat Model for PII Detection

PII detection faces both **unintentional** and **intentional** adversarial inputs:

**Unintentional** (common in real data):
- Typos and misspellings in names ("Jonh Smth")
- Non-standard formatting of structured PII ("123 456 7890" vs. "(123) 456-7890")
- OCR errors ("0" vs. "O", "1" vs. "l")
- Code-switching and multilingual names
- Abbreviations and informal text

**Intentional** (adversarial evasion):
- Homoglyph substitution (Cyrillic "a" for Latin "a")
- Zero-width characters inserted into PII
- Leetspeak and character substitution ("5ive" for "five", "4ddress" for "address")
- Context manipulation (embedding PII in misleading contexts)
- Whitespace manipulation (splitting tokens: "J o h n S m i t h")

### 7.2 The "Unmasking the Reality" Framework

Singh & Narayanan (2025) developed a novel evaluation framework testing PII masking models across **six dimensions**:

1. **Basic Entity Recognition**: Standard PII detection in clean text.
2. **Contextual Entity Disambiguation**: Same surface form in PII vs. non-PII contexts.
3. **NER in Noisy & Real-World Data**: Typos, abbreviations, informal text.
4. **Evolving & Novel Entities Detection**: New entity formats (cryptocurrency addresses, new social media handles).
5. **Cross-Lingual / Multilingual NER**: PII in non-English text or code-switched text.
6. **Adversarial Context**: Deliberately crafted inputs using homoglyphs, perturbations, and contextual camouflage.

They tested MS Presidio, Piiranha, and Starpii on 17K semi-synthetic sentences across 16 PII types from India, UK, and US jurisdictions. All three models showed significant vulnerabilities, especially in dimensions 3-6.

### 7.3 Specific Adversarial Tests to Include

**Character-level perturbations**:
- Homoglyph substitution: Replace Latin characters with visually identical Unicode characters.
- Zero-width character injection: Insert U+200B (zero-width space) within PII tokens.
- Diacritical addition: Add unnecessary diacritics to ASCII names.
- Case manipulation: ALL CAPS, aLtErNaTiNg CaSe, lowercase names.

**Format perturbations**:
- Phone: "+1-555-123-4567" vs. "5551234567" vs. "(555) 123-4567" vs. "555.123.4567"
- SSN: "123-45-6789" vs. "123 45 6789" vs. "123456789"
- Email: standard vs. plus-addressing vs. unusual TLDs
- Dates: "01/15/1990" vs. "January 15, 1990" vs. "15-Jan-90" vs. "1990.01.15"

**Context perturbations**:
- Embed PII in code comments, JSON, XML
- PII in tables and structured formats within free text
- PII mentioned indirectly ("same person as mentioned in paragraph 3")
- Negated PII ("this is NOT the patient's name: John Smith")

**Multilingual perturbations**:
- Names with non-Latin scripts (Arabic, Chinese, Hindi, Japanese, Korean)
- Transliterated names ("Muhammad" vs. "Mohammed" vs. "Mohamed")
- Mixed-script text (Latin text with Cyrillic names)

### 7.4 Robustness Metrics

For each adversarial dimension, report:

- **Attack Success Rate (ASR)**: Fraction of adversarial inputs where a previously-detected PII entity is no longer detected. Lower is better.
- **Robustness Delta**: F1 (clean) - F1 (adversarial). Smaller is better.
- **Per-perturbation-type breakdown**: Which perturbation types cause the most degradation?

**Recommendation**: Target a robustness delta of <5% for character-level perturbations and <10% for context perturbations. Any adversarial dimension where recall drops below the tier-specific thresholds (Section 3.2) should be flagged as a critical vulnerability.

### 7.5 Edge Cases Requiring Special Attention

| Edge Case | Example | Why It Is Hard |
|-----------|---------|---------------|
| **Partial PII** | Phone number with last 4 digits masked: "xxx-xxx-4567" | May still be identifying when combined with other data |
| **Nested PII** | "Dr. John Smith, MD" contains TITLE + NAME + CREDENTIAL | Overlapping spans require resolution strategy |
| **Indirect identifiers** | "the 42-year-old female CEO of the company based in Zurich" | Combination of quasi-identifiers |
| **PII in structured formats** | `"name": "John Smith"` in JSON | Model must handle markup alongside text |
| **Very long PII spans** | Full mailing addresses spanning 3+ lines | Boundary detection across line breaks |
| **Adjacent PII** | "John Smith john.smith@email.com 555-1234" | Entity boundary disambiguation |
| **PII as part of larger text** | "Contact john.smith@email.com for details" | Must extract PII from surrounding text |

---

## 8. Recommended Evaluation Protocol for DataFog

### 8.1 Complete Evaluation Pipeline

```
Phase 1: CORE ACCURACY
  |-- seqeval strict F1 (IOB2) on PIILO test set
  |-- seqeval strict F1 (IOB2) on CoNLL-2003 test set
  |-- nervaluate 4-schema analysis on PIILO test set
  |-- Per-entity-type P/R/F1/F2/F5 table
  |-- Sensitivity-weighted F2 aggregate
  |
Phase 2: PII-SPECIFIC
  |-- Per-sensitivity-tier recall (must meet thresholds)
  |-- False positive rate analysis
  |-- Context-dependent disambiguation accuracy
  |-- Confusion matrix (which PII types are confused)
  |
Phase 3: GENERALIZATION
  |-- CrossNER zero-shot evaluation (5 domains)
  |-- Domain-specific PII evaluation (medical, legal, code)
  |-- Zero-shot evaluation on held-out PII types
  |-- Few-shot learning curve (1/5/10/50/100 shots)
  |
Phase 4: ROBUSTNESS
  |-- Character-level perturbation tests
  |-- Format variation tests
  |-- Context manipulation tests
  |-- Multilingual robustness tests
  |-- Attack Success Rate per dimension
  |
Phase 5: LATENCY & EFFICIENCY
  |-- CPU single-doc latency (P50/P90/P95/P99)
  |-- CPU batch throughput (optimal batch size)
  |-- GPU single-doc latency (P50/P90/P95/P99)
  |-- GPU batch throughput (optimal batch size)
  |-- ONNX Runtime benchmarks (FP32/FP16/INT8)
  |-- Memory footprint (peak RAM/VRAM)
  |-- Model size on disk (FP32/FP16/INT8)
  |-- Comparison table vs. GLiNER2, DeBERTa-v3-large, Presidio
```

### 8.2 Minimum Viability Thresholds

For the DataFog model to be considered production-ready for PII detection:

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| Overall entity F1 (strict) | >= 0.90 on PIILO | Competitive with published baselines |
| Overall F2 | >= 0.92 on PIILO | Recall-oriented target |
| Overall F5 | >= 0.93 on PIILO | Kaggle competition winners at 0.953 |
| Tier 1 entity recall | >= 0.98 each | Safety-critical PII types |
| Tier 2 entity recall | >= 0.95 each | High-sensitivity PII types |
| Tier 3 entity recall | >= 0.90 each | Moderate-sensitivity PII types |
| Per-entity-type precision | >= 0.80 each | Usability floor |
| CrossNER average F1 (zero-shot) | >= 0.55 | Match or exceed GLiNER2's 0.590 |
| Adversarial robustness delta | < 5% (char), < 10% (context) | Practical robustness |
| CPU single-doc latency (256 tokens) | < 50ms (P99) | Real-time API feasibility |
| CPU batch throughput (256 tokens) | > 100 docs/sec | Batch processing feasibility |
| Parameter count | < 50M | Per research contract |
| Model size (INT8, ONNX) | < 60MB | Edge deployment feasibility |

### 8.3 Comparison Baselines

Every metric should be reported alongside these baselines:

| Baseline | Why Include |
|----------|-------------|
| **GLiNER2 (205M)** | Primary comparison target from research contract |
| **DeBERTa-v3-large** | Kaggle competition winning architecture |
| **Microsoft Presidio** | Most widely-used open-source PII tool |
| **Starpii** | Popular Hugging Face PII model (580K+ downloads) |
| **Piiranha** | Popular Hugging Face PII model |
| **spaCy NER (en_core_web_trf)** | Common NLP pipeline baseline |
| **Regex-only baseline** | Floor performance for structured PII |

### 8.4 Reporting Format

The final evaluation report should include:

1. **Summary table**: One-row-per-model comparison with key metrics (F1, F2, F5, Tier 1 recall, latency, params).
2. **Per-entity-type table**: Full P/R/F1/F2 breakdown for each PII type, with sensitivity tier annotations.
3. **Error analysis**: Top 20 most common error patterns, categorized by type (boundary error, type confusion, false positive, false negative).
4. **nervaluate radar chart**: Strict/Exact/Partial/Type F1 scores visualized.
5. **Cross-domain heatmap**: F1 scores across all evaluation domains.
6. **Robustness dashboard**: ASR per adversarial dimension.
7. **Latency plots**: Latency vs. batch size curves for CPU and GPU, comparing all baselines.
8. **Efficiency Pareto frontier**: Plot F1 vs. latency for all models, highlighting the Pareto-optimal frontier.

### 8.5 Tools and Libraries

| Tool | Purpose | Installation |
|------|---------|-------------|
| `seqeval` | Entity-level P/R/F1, strict and default modes | `pip install seqeval` |
| `nervaluate` | Four-schema evaluation (strict/exact/partial/type) | `pip install nervaluate` |
| `presidio-evaluator` | Presidio-compatible evaluation, confusion matrices | `pip install presidio-evaluator` |
| `onnxruntime` | ONNX model inference and benchmarking | `pip install onnxruntime` |
| `onnxruntime_perf_test` | Official ONNX benchmarking tool | Build from source |
| `datasets` (HF) | Load PIILO, CrossNER, BigCode PII datasets | `pip install datasets` |
| Custom scripts | Adversarial test generation, sensitivity-weighted metrics | Build in-house |

### 8.6 Evaluation Cadence

| When | What to Evaluate |
|------|------------------|
| **Every training run** | Token-level F1 (training diagnostic), entity F1 on validation set |
| **Every milestone** | Full Phase 1 + Phase 2 on PIILO, latency spot-check |
| **Pre-release** | Complete Phase 1-5 evaluation, full comparison table |
| **Monthly (post-deployment)** | Precision/recall drift on labeled validation set, latency monitoring |

---

## 9. Sources

### NER Evaluation Metrics
- [nervaluate: SemEval'13 entity-level evaluation (GitHub)](https://github.com/MantisAI/nervaluate)
- [seqeval: sequence labeling evaluation framework (GitHub)](https://github.com/chakki-works/seqeval)
- [David Batista -- Named-Entity Evaluation Metrics (Blog)](https://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/)
- [conlleval.py: Python CoNLL evaluation script (GitHub)](https://github.com/sighsmile/conlleval)
- [CoNLL-2003 Shared Task (CLIPS, University of Antwerp)](https://www.clips.uantwerpen.be/conll2003/ner/)
- [Skeptric -- How not to Evaluate NER Systems](https://skeptric.com/ner-evaluate/)
- [Hugging Face Evaluate: Sequence Labeling Metrics (DeepWiki)](https://deepwiki.com/huggingface/evaluate/5.3-sequence-labeling-metrics)

### PII-Specific Evaluation
- [Microsoft Presidio Evaluation Documentation](https://microsoft.github.io/presidio/evaluation/)
- [Microsoft Presidio-Research (GitHub)](https://github.com/microsoft/presidio-research)
- [Private AI -- How to Benchmark PII Detection Solutions](https://www.private-ai.com/en/blog/pii-dectection-benchmark)
- [Protecto -- Benchmarking PII Identification in Unstructured Text (PDF)](https://protecto.ai/wp-content/uploads/2024/07/6646f1564c513545cbf9d2f9_Quantitative-Benchmark-Study-PII-Identification-1.pdf)
- [NIST SP 800-122: Guide to Protecting PII Confidentiality](https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-122.pdf)

### PII Benchmarks and Datasets
- [Kaggle PII Data Detection Competition (2024)](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data)
- [1st Place Solution: PII Detection (GitHub)](https://github.com/bogoconic1/pii-detection-1st-place)
- [PIILO Dataset (The Learning Agency Lab)](https://the-learning-agency-lab.com/learning-exchange/piilo-dataset/)
- [i2b2/n2c2 2014 De-Identification Shared Task (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4989908/)
- [n2c2 Data Sets Portal](https://n2c2.dbmi.hms.harvard.edu/data-sets)
- [Text Anonymization Benchmark -- TAB (MIT Press)](https://direct.mit.edu/coli/article/48/4/1053/112770/The-Text-Anonymization-Benchmark-TAB-A-Dedicated)
- [TAB GitHub Repository](https://github.com/NorskRegnesentral/text-anonymization-benchmark)
- [PII-Bench: Evaluating Query-Aware Privacy Protection Systems (arXiv)](https://arxiv.org/abs/2502.18545)
- [BigCode PII Dataset (Hugging Face)](https://huggingface.co/datasets/bigcode/bigcode-pii-dataset)
- [CrossNER: Evaluating Cross-Domain NER (arXiv, AAAI 2021)](https://arxiv.org/abs/2012.04373)
- [CrossNER GitHub Repository](https://github.com/zliucr/CrossNER)

### Latency and Throughput Benchmarking
- [Microsoft -- Scaling Up PyTorch Inference with ONNX Runtime (Blog)](https://opensource.microsoft.com/blog/2022/04/19/scaling-up-pytorch-inference-serving-billions-of-daily-nlp-inferences-with-onnx-runtime)
- [ONNX Runtime Performance Documentation](https://onnxruntime.ai/docs/performance/)
- [ONNX Runtime Benchmark Tool (GitHub)](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/transformers/benchmark.py)
- [NVIDIA -- LLM Inference Benchmarking Fundamental Concepts](https://developer.nvidia.com/blog/llm-benchmarking-fundamental-concepts/)
- [GLiNER2: An Efficient Multi-Task IE System (arXiv)](https://arxiv.org/html/2507.18546v1)
- [GLiNER2 Paper (ACL Anthology PDF)](https://aclanthology.org/2025.emnlp-demos.10.pdf)

### Cross-Domain Generalization
- [Recent Advances in NER: A Comprehensive Survey (arXiv)](https://arxiv.org/html/2401.10825v3)
- [IBM ZShot: Zero and Few-Shot NER (GitHub)](https://github.com/IBM/zshot)
- [From Zero to Hero: Transformers for Biomedical NER (arXiv)](https://arxiv.org/html/2305.04928v5)
- [Few-Shot Domain Adaptation for NER (arXiv)](https://arxiv.org/html/2412.00426)

### Adversarial and Robustness Evaluation
- [Unmasking the Reality of PII Masking Models (arXiv)](https://arxiv.org/abs/2504.12308)
- [RECAP: Hybrid Methods for Multilingual PII Detection (arXiv)](https://arxiv.org/abs/2510.07551)
- [Resilience of NER Models under Adversarial Attack (ACL Anthology)](https://aclanthology.org/2022.dadc-1.1/)
- [Context-Aware Adversarial Attack on NER (arXiv)](https://arxiv.org/html/2309.08999v2)
- [Adaptive PII Mitigation Framework for LLMs (arXiv)](https://arxiv.org/abs/2501.12465)
- [Comparing Feature-based and Context-aware PII Approaches (arXiv)](https://arxiv.org/html/2407.02837v1)
