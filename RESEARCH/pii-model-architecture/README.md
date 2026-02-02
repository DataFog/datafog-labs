# Research: Novel Lightweight PII Detection Model Architecture

## Overview

This research program investigates the optimal architecture for a PII-specific detection model that is dramatically smaller (<50M params), faster, and more accurate on PII than general-purpose models like GLiNER2 (205M params).

**Working Hypothesis**: A hybrid character-contextual architecture that jointly learns character-level patterns (structured PII) and contextual representations (soft PII) in a single differentiable model can achieve SOTA PII detection at <50M parameters.

**Recommended Architecture**: ~22.7M parameters
- DeBERTa-v3-xsmall backbone (22M params)
- Character CNN encoder (0.3M params)
- Gating fusion module (0.1M params)
- Token classification head + CRF (0.3M params)

## Documents

### Research Contract
- [`00_research_contract.md`](./00_research_contract.md) -- Scope, hypotheses, and research questions

### Reports

| Document | Description |
|----------|-------------|
| [`08_report/00_executive_summary.md`](./08_report/00_executive_summary.md) | 2-page executive summary with key findings and recommendations |
| [`08_report/01_architecture_survey.md`](./08_report/01_architecture_survey.md) | Comprehensive survey of all NER/PII detection architectures |
| [`08_report/02_training_data_catalog.md`](./08_report/02_training_data_catalog.md) | PII datasets, general NER data, synthetic generation, licensing |
| [`08_report/03_design_space.md`](./08_report/03_design_space.md) | Backbone selection, character encoders, fusion, output heads, optimization |
| [`08_report/04_gap_analysis.md`](./08_report/04_gap_analysis.md) | Novel contribution opportunities ranked by impact and feasibility |
| [`08_report/evaluation_framework.md`](./08_report/evaluation_framework.md) | Evaluation protocol: metrics, benchmarks, latency, adversarial testing |
| [`08_report/05_recency_check.md`](./08_report/05_recency_check.md) | Late 2025/early 2026 literature sweep validating architecture decisions |
| [`08_report/06_gliner2_comparison.md`](./08_report/06_gliner2_comparison.md) | Head-to-head comparison: GLiNER2 (205M) vs DataFog PII-NER v1 (22.7M) |

### Visual Guide
- [`architecture-guide.html`](./architecture-guide.html) -- Interactive 29-slide visual presentation covering ML fundamentals, architecture comparisons, and the design recommendation. Clone the repo and open in a browser to navigate with arrow keys.

## Key Findings

### Top Novel Contributions (Ranked)
1. **Joint character-contextual architecture for PII** -- No published work combines differentiable structural pattern learning with contextual transformers for PII (GENUINELY NOVEL)
2. **PII-specific pretraining via NuNER paradigm** -- Entity-biased contrastive pretraining on PII-dense corpora (GENUINELY NOVEL)
3. **Cross-domain PII benchmark** -- First standardized multi-domain PII NER benchmark (NOVEL with caveats)

### Training Data Strategy
- **360K+ commercially-licensed PII examples**: Nemotron-PII (100K, CC BY 4.0) + Gretel (60K, Apache 2.0) + AI4Privacy (200K, Apache 2.0)
- Pre-train on Few-NERD (188K, CC BY-SA 4.0) for general entity recognition

### Evaluation Targets
- >= 0.90 strict F1 on PIILO
- >= 0.98 recall on Tier 1 PII (SSN, credit cards)
- < 50ms P99 CPU latency (256 tokens)
- < 50M parameters

### Comparison Baselines
GLiNER2 (205M), Piiranha (280M), Presidio, Starpii, spaCy NER, regex-only

## Research Classification
- **Type**: C (Analysis) -- requires judgment across multiple technical domains
- **Tracks**: 5 parallel research tracks executed
- **Sources**: 100+ academic papers, technical blogs, and GitHub repositories surveyed

---

*Research conducted February 2026 for DataFog's next-generation PII detection model.*
