# Executive Summary: Novel Lightweight PII Detection Model Architecture

**Research Program**: DataFog Next-Generation PII Detection
**Date**: February 2026
**Classification**: Type C (Analysis) -- Multi-domain technical evaluation

---

## 1. Research Question and Scope

**Core Question**: What is the optimal architecture for a PII-specific detection model that is dramatically smaller (<50M params), faster, and more accurate on PII than general-purpose models like GLiNER2 (205M params)?

**Working Hypothesis**: A hybrid character-contextual architecture that jointly learns character-level patterns (structured PII like SSNs, credit card numbers, phone numbers) and contextual representations (soft PII like names, addresses, indirect identifiers) in a single differentiable model can achieve state-of-the-art PII detection at <50M parameters.

**Scope**: All NER architectures relevant to PII (2020--2025 literature plus key foundational work), PII-specific datasets and benchmarks, evaluation frameworks, efficient model design, character-level and contextual approaches. Excluded: general LLM-based approaches (>1B params), non-English-first approaches, vision-based PII detection.

Five parallel research tracks were executed:

1. **Architecture Survey** -- Comprehensive catalog of NER/PII detection architectures and their tradeoffs
2. **Training Data Catalog** -- PII-specific datasets, general NER corpora, synthetic generation pipelines, and licensing analysis
3. **Evaluation Framework** -- PII-specific metrics, benchmarks, latency protocols, adversarial robustness testing
4. **Design Space Exploration** -- Efficient backbones, character encoders, fusion mechanisms, span methods, inference optimization
5. **Gap Analysis** -- Systematic identification of genuine research gaps and novel contribution opportunities

---

## 2. Key Findings Across Research Tracks

### Track 1: Architecture Survey
The PII detection landscape spans from regex-based systems (Presidio) through traditional NER (Flair, spaCy) to modern transformer-based approaches (GLiNER2, Piiranha, Roblox PII Classifier). No existing system jointly optimizes for character-level pattern recognition and contextual understanding in a single lightweight model. The closest production systems are Piiranha (280M params, 98.27% recall but large) and Roblox's XLM-RoBERTa-Large classifier (94% F1 at 370K RPS but 560M params). Lightweight models like TinyBERT-NER (~14.5M params) exist but are not PII-specialized and sacrifice significant accuracy.

### Track 2: Training Data
A commercially viable training corpus of 360K+ PII-specific examples can be assembled from three permissively licensed sources: NVIDIA Nemotron-PII (~100K examples, CC-BY-4.0), Gretel PII Masking EN v1 (~70K examples, Apache 2.0), and AI4Privacy datasets (200K--500K examples, multiple releases, Apache 2.0). Additional general NER data (CoNLL-2003, OntoNotes, WikiANN) provides foundational entity recognition capability. Synthetic data generation via Faker+LLM pipelines can fill gaps in underrepresented PII types (financial identifiers, medical record numbers, API keys).

### Track 3: Evaluation Framework
Standard NER F1 is insufficient for PII -- the asymmetric error costs (missing an SSN vs. false-flagging a common noun) demand recall-oriented metrics. The recommended protocol uses a four-tier evaluation: (1) core accuracy via seqeval strict F1 + nervaluate four-schema analysis, (2) PII-specific per-entity-type recall stratified by sensitivity tier with F2/F5 metrics, (3) cross-domain generalization via CrossNER + domain-specific PII benchmarks, and (4) adversarial robustness testing across six perturbation dimensions. Minimum viability: >=0.90 strict F1 on PIILO, >=0.98 recall on Tier 1 (critical) PII types, <50ms P99 latency on CPU.

### Track 4: Design Space
DeBERTa-v3-xsmall (22M params) emerges as the optimal backbone -- it uses disentangled attention (content + position), ELECTRA-style replaced token detection pretraining, and achieves strong NLU performance at minimal size. Character-level features via a Character CNN (Ma & Hovy 2016 style, ~0.5M params) capture the orthographic patterns that distinguish structured PII. Gating fusion (learned weighted combination) outperforms simple concatenation for combining character and contextual features. A CRF output layer adds ~0.2M params but enforces valid BIO tag sequences and improves entity boundary detection by 1--3 F1 points.

### Track 5: Gap Analysis
Five genuine research gaps were identified, with no existing published work addressing them. The top three are: (1) Joint character-contextual models specifically designed for PII -- no one has combined character CNNs with efficient transformers in a PII-specialized architecture; (2) PII-specific pretraining using the NuNER paradigm (entity-centric self-supervised learning) applied to PII-dense corpora; (3) Learned regex / differentiable pattern matching (DeepDFA) applied to NER/PII detection -- this has never been attempted despite the natural fit.

---

## 3. Recommended Architecture

**DataFog PII-NER v1**: ~22.7M parameters total

| Component | Parameters | Role |
|-----------|-----------|------|
| DeBERTa-v3-xsmall backbone | ~22.0M | Contextual token representations via disentangled attention |
| Character CNN encoder | ~0.3M | Character-level orthographic features (3 filter widths: 3, 4, 5) |
| Gating fusion layer | ~0.1M | Learned weighted combination of contextual + character features |
| Token classification head | ~0.1M | Linear projection to BIO tag space |
| CRF output layer | ~0.2M | Constrained decoding enforcing valid BIO sequences |

**Key design decisions**:
- **Token classification (BIO) over span-based**: Token classification is simpler, faster at inference, and empirically competitive for PII where entities rarely nest. BIO with CRF captures the vast majority of PII entity structures.
- **DeBERTa-v3-xsmall over alternatives**: MiniLM and ELECTRA-small were considered. DeBERTa-v3-xsmall's disentangled attention provides superior position-aware representations critical for structured PII (SSNs, phone numbers where digit position matters).
- **Character CNN over Character LSTM**: CNNs are 3--5x faster than LSTMs for character encoding, and the parallel computation maps well to GPU/ONNX optimization. Ma & Hovy (2016) demonstrated CNNs capture morphological features as effectively as LSTMs for NER.
- **Gating fusion over concatenation**: Gating allows the model to dynamically weight character features higher for structured PII (emails, SSNs) and contextual features higher for soft PII (names in context). This is the key architectural novelty.

**Expected performance targets**: >=0.92 F1 on PII benchmarks, >=0.98 recall on critical PII types, <30ms latency on CPU (ONNX INT8), 9x smaller than GLiNER2.

---

## 4. Top 3 Novel Contribution Opportunities (Ranked)

### Rank 1: Joint Character-Contextual Architecture for PII Detection
**Gap**: No published work combines character-level pattern recognition with efficient transformer contextual understanding in a single model specifically designed for PII. Existing PII models either use transformers alone (Piiranha, GLiNER-PII) or regex patterns alone (Presidio). The character CNN + DeBERTa fusion with gating is novel.
**Impact**: High -- directly addresses the dual nature of PII (structured patterns + contextual soft entities).
**Feasibility**: High -- all components are well-understood; the novelty is in their PII-specific combination and the gating mechanism.

### Rank 2: PII-Specific Pretraining via Entity-Centric Self-Supervision
**Gap**: NuNER (2024) demonstrated that entity-centric pretraining (predicting entity spans in unlabeled text) dramatically improves NER. No one has applied this to PII-dense corpora (privacy policies, financial documents, medical records). PII-specific pretraining could teach the model PII-relevant patterns before any labeled fine-tuning.
**Impact**: High -- could yield 3--5 F1 points improvement based on NuNER's results on general NER.
**Feasibility**: Medium -- requires curating a PII-dense unlabeled pretraining corpus and implementing the NuNER objective.

### Rank 3: Differentiable Pattern Matching for Structured PII
**Gap**: DeepDFA (2024, JCLB) learns differentiable finite automata that can approximate regex-like pattern matching within a neural network. This has never been applied to NER or PII detection, despite structured PII (SSNs, credit cards, phone numbers) being inherently regular-language patterns.
**Impact**: Medium-High -- could dramatically improve recall on structured PII types while maintaining the end-to-end differentiability of the model.
**Feasibility**: Medium -- requires adapting DeepDFA's architecture for token-level NER output rather than sequence classification.

**Recommended paper strategy**: Combine Ranks 1 + 2 + 3 into a single contribution -- "A Lightweight Hybrid Architecture for PII Detection with Character-Contextual Fusion, PII-Specific Pretraining, and Differentiable Pattern Matching." This positions the work as a comprehensive, novel approach rather than an incremental improvement.

---

## 5. Recommended Training Data Strategy

**Phase 1 -- Core PII Training Corpus (360K+ examples)**:

| Dataset | Size | License | PII Types | Priority |
|---------|------|---------|-----------|----------|
| NVIDIA Nemotron-PII | ~100K examples | CC-BY-4.0 | 15+ types | Must-have |
| Gretel PII Masking EN v1 | ~70K examples | Apache 2.0 | 12+ types | Must-have |
| AI4Privacy (200K release) | ~200K examples | Apache 2.0 | 50+ types | Must-have |

**Phase 2 -- General NER Foundation**:
- CoNLL-2003 (22K sentences, PER/LOC/ORG/MISC) for foundational entity recognition
- OntoNotes 5.0 (76K sentences, 18 entity types) for broader entity coverage
- WikiANN (cross-lingual NER) for multilingual generalization

**Phase 3 -- Synthetic Augmentation**:
- Faker+LLM pipeline (Gretel approach) for underrepresented PII types
- NVIDIA NeMo Data Designer for domain-specific synthetic data
- Targeted augmentation for financial PII (account numbers, routing numbers) and medical PII (MRNs, DEA numbers)

**Phase 4 -- Domain-Specific Fine-Tuning**:
- i2b2/n2c2 for clinical de-identification (requires DUA)
- TAB for legal text anonymization
- BigCode PII for code domain

---

## 6. Recommended Evaluation Protocol Summary

**Primary Metrics**: Entity-level strict F1 (seqeval, IOB2), F2 (general PII), F5 (high-stakes PII comparison)

**Evaluation Tiers**:
- **Tier 1 (Core Accuracy)**: seqeval strict F1, nervaluate four-schema, per-entity-type P/R/F1/F2/F5
- **Tier 2 (PII-Specific)**: Per-sensitivity-tier recall, false positive rate, context-dependent disambiguation
- **Tier 3 (Operational)**: CPU/GPU latency (P50/P90/P95/P99), throughput (tokens/sec, docs/sec), ONNX Runtime benchmarks, memory footprint
- **Tier 4 (Robustness)**: Cross-domain generalization (CrossNER), adversarial perturbation resilience, edge case handling

**Minimum Viability Thresholds**: >=0.90 F1 on PIILO, >=0.98 Tier 1 recall, >=0.95 Tier 2 recall, <50ms P99 CPU latency, <50M params

**Comparison Baselines**: GLiNER2 (205M), DeBERTa-v3-large, Presidio, Starpii, Piiranha, spaCy NER, regex-only

Full evaluation protocol details are in [evaluation_framework.md](./evaluation_framework.md).

---

## 7. Next Steps

1. **Architecture Implementation** (Weeks 1--2): Implement the DeBERTa-v3-xsmall + Character CNN + Gating Fusion + CRF architecture in PyTorch. Validate forward pass, parameter count, and single-batch inference latency.

2. **Data Pipeline Construction** (Weeks 1--3): Download and preprocess Nemotron-PII + Gretel + AI4Privacy datasets. Unify entity type taxonomy across sources. Build BIO-tagged training/validation/test splits.

3. **Baseline Training** (Weeks 3--4): Train the base architecture on the unified PII corpus. Establish baseline F1/F2/F5 metrics on PIILO and CoNLL-2003.

4. **PII-Specific Pretraining Experiment** (Weeks 4--6): Curate a PII-dense unlabeled corpus. Implement NuNER-style entity-centric self-supervised pretraining. Measure improvement over the baseline.

5. **Differentiable Pattern Matching Integration** (Weeks 5--7): Adapt DeepDFA for token-level PII detection. Integrate as a parallel pathway alongside the Character CNN. Measure recall improvement on structured PII types.

6. **Full Evaluation and Ablation** (Weeks 7--8): Run the complete four-tier evaluation protocol. Ablation studies on each component (character CNN, gating, CRF, pretraining, differentiable patterns). Latency benchmarking with ONNX Runtime INT8 quantization.

7. **Paper Writing** (Weeks 8--10): Draft the research paper combining all three novel contributions. Target venue: EMNLP 2026 or ACL 2026 (Industry Track).

---

*This executive summary synthesizes findings from five parallel research tracks conducted for DataFog's next-generation PII detection model. Full details are available in the individual track reports.*
