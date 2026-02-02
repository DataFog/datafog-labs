# Recency Check: Late 2025 -- Early 2026 Literature Sweep

**Purpose**: Verify that architecture decisions from the initial research are not outdated given the fast-moving NER/PII landscape.

**Method**: 18+ targeted web searches across efficient NER architectures, PII-specific models, character-level approaches, training data, backbone models, and differentiable pattern matching — all filtered to publications from mid-2025 through early 2026.

---

## 1. Backbone Validation: DeBERTa-v3-xsmall Remains Best at ~22M

- **ModernBERT vs DeBERTaV3 (April 2025)**: Controlled study ([arXiv:2504.08716](https://arxiv.org/html/2504.08716v1)) confirmed DeBERTaV3's disentangled attention gives it an edge on token-level tasks like NER. ModernBERT trains faster but DeBERTaV3 wins on sample efficiency and final NER performance. ModernBERT has no xsmall variant.
- **NeoBERT (Feb 2025)**: 250M params, strong MTEB/GLUE scores, 46.7% faster than ModernBERT on long sequences. No TokenClassification head at launch. 11x larger than our target. ([arXiv:2502.19587](https://arxiv.org/abs/2502.19587))
- **mmBERT (Sep 2025)**: 1800+ languages, ModernBERT-based, small variant is 140M (42M non-embedding). Explicitly noted as "behind on NER" due to tokenizer prefix-space issue. ([arXiv:2509.06888](https://arxiv.org/abs/2509.06888))
- **ModernBERT**: Only base (~150M) and large sizes. First base-size to beat DeBERTaV3 on GLUE, but DeBERTaV3 retains NER edge. Early NER experiments had tokenizer issues with `is_split_into_words`.
- **SmolLM2/SmolLM3 (2025-2026)**: 135M-3B params, decoder-only, designed for on-device use. No NER benchmarks published. Decoder architecture is fundamentally less suited for token classification.

**Verdict**: DeBERTa-v3-xsmall remains the best encoder backbone at ~22M parameters for NER. No 2025-2026 model has displaced it. **Our choice is validated.**

---

## 2. PII Model Landscape Update

- **BetterData AI PII Model (May 2025)**: Qwen2-0.5B decoder, 29 PII classes, 7 languages, "CPU-friendly." At 500M params, it is 22x larger than our target. Decoder architecture is less efficient for token classification. ([HuggingFace](https://huggingface.co/betterdataai/PII_DETECTION_MODEL))
- **Piiranha v1**: Surpassed 1.1M downloads, no v2 released. Still mDeBERTa-v3-base, 17 PII types, 256 token context.
- **GLiNER-PII Ecosystem**: NVIDIA gliner-PII, Knowledgator gliner-pii-base (F1 ~81%), urchade/gliner_multi_pii-v1. LOGICAL paper (Oct 2025, [arXiv:2510.19346](https://arxiv.org/abs/2510.19346)) showed fine-tuned GLiNER achieved F1 0.980 on clinical PII, outperforming Gemini-Pro-2.5 (0.845).
- **GLiNER2 (EMNLP 2025)**: Multi-task IE, ~500M params, CPU-efficient. Matches GPT-4o on CrossNER (0.590 vs 0.599 F1). Strong generalist but not a lightweight specialist.
- **RECAP (NeurIPS 2025 Workshop)**: Hybrid regex + LLM for PII, 300+ entity types, 13 locales. Outperforms fine-tuned NER by 82% weighted F1. Requires runtime LLM inference calls. ([arXiv:2510.07551](https://arxiv.org/abs/2510.07551))
- **T5-small vs Mistral-7B for PII (Dec 2025)**: T5 excels as lightweight pattern matcher but fails on OOD inputs. Mistral provides robustness via world knowledge but needs GPUs. Directly validates our hybrid pattern+context approach. ([arXiv:2512.18608](https://arxiv.org/abs/2512.18608))

**Verdict**: No lightweight encoder-based PII model with character-level features exists. Closest competitors are all larger (280M-560M) or use decoder architectures. **Our approach fills a genuine gap.**

---

## 3. Architecture Novelty: Confirmed

- **Subword-Character Multi-Scale Transformer (July 2025)**: Built for machine translation, not NER. Validates the multi-granularity architectural pattern. ([Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1002/eng2.70287))
- **Multi-Granularity Embedding Fusion for NER (Nature, May 2025)**: Applied to psychomedical NER. Validates multi-granularity for specialized NER domains. ([Nature](https://www.nature.com/articles/s41598-025-90939-8))
- **SynthoNER (Dec 2025)**: Distillation from 200B-param LLMs to 10MB spaCy models. Complementary data generation technique.
- **Hybrid Rule-Based + ML for Financial PII (Nature, July 2025)**: 94.7% precision, 89.4% recall on financial docs. Notes transformer models are impractical for rapid iteration. ([Nature](https://www.nature.com/articles/s41598-025-04971-9))
- **"Unmasking PII Masking Models" (April 2025)**: Benchmark across 5 dimensions revealing significant gaps in all current PII models. ([arXiv:2504.12308](https://arxiv.org/abs/2504.12308))

**Verdict**: No paper implements joint character-contextual gating for PII. The closest works validate individual components but not the combination. **Our novelty claim holds.**

---

## 4. New Training Data Available

| Dataset | Published | Size | License | Action |
|---------|-----------|------|---------|--------|
| AI4Privacy 500K | 2025 | 500K examples | Llama Community License | Evaluate licensing terms |
| AI4Privacy 400K | Feb 2025 | 407K examples | Custom | Contact for commercial terms |
| SPY Dataset | NAACL 2025 | Synthetic PII | Research | Use for evaluation |
| PANORAMA | 2025 | 384K samples from 9.6K profiles | Research | Use for evaluation |
| Patronus EnterprisePII | 2025 | 3K examples, enterprise-specific | Unknown | Evaluate for enterprise PII classes |
| UnPII | ICSE 2026 (April) | Synthetic for unlearning | Research | Monitor |

**Verdict**: Training data landscape has expanded. AI4Privacy 500K and SPY are notable additions. Core commercially-licensed strategy (Nemotron-PII + Gretel + AI4Privacy 200K) remains valid with potential to expand.

---

## 5. DeepDFA / Differentiable Pattern Matching

- DeepDFA (ECAI 2024, popularized June 2025): No NER application published. Multi-layer versions and RL applications planned.
- Neuro-Symbolic Framework (IJCAI 2025): DFAs via logic tensor networks for sequence classification, not NER.

**Verdict**: Differentiable pattern matching for NER/PII remains unexplored. Our character CNN approach is the more practical route for now, with DeepDFA as a future research direction.

---

## 6. Recommended Updates to Research

Based on this sweep, the following updates should be incorporated:

### Additional Baselines
- RECAP (NeurIPS 2025) -- hybrid regex+LLM approach
- GLiNER fine-tuned on clinical PII (LOGICAL, Oct 2025) -- 0.98 F1
- BetterData AI Qwen2-0.5B PII model

### Additional Evaluation
- Adopt "Unmasking PII Masking Models" 6-dimension evaluation framework
- Consider Patronus EnterprisePII for enterprise PII coverage testing

### Additional Training Data
- AI4Privacy 500K (if licensing permits)
- SPY dataset (NAACL 2025) for synthetic PII evaluation

### Key Validation Points
- Dec 2025 T5-vs-Mistral study confirms the pattern-vs-context tradeoff our gating mechanism addresses
- NAACL 2025 SPY paper confirms general NER models underperform on PII-specific tasks
- April 2025 DeBERTaV3-vs-ModernBERT study confirms our backbone choice

---

## Sources

1. [ModernBERT vs DeBERTaV3 (April 2025)](https://arxiv.org/html/2504.08716v1)
2. [NeoBERT (Feb 2025)](https://arxiv.org/abs/2502.19587)
3. [mmBERT (Sep 2025)](https://arxiv.org/abs/2509.06888)
4. [GLiNER2 (EMNLP 2025)](https://arxiv.org/abs/2507.18546)
5. [BetterData AI PII Model](https://huggingface.co/betterdataai/PII_DETECTION_MODEL)
6. [Piiranha v1](https://huggingface.co/iiiorg/piiranha-v1-detect-personal-information)
7. [LOGICAL -- GLiNER for Clinical PII (Oct 2025)](https://arxiv.org/abs/2510.19346)
8. [RECAP (NeurIPS 2025)](https://arxiv.org/abs/2510.07551)
9. [Lightweight LMs for PII Masking (Dec 2025)](https://arxiv.org/abs/2512.18608)
10. [Subword-Character Multi-Scale Transformer (July 2025)](https://onlinelibrary.wiley.com/doi/abs/10.1002/eng2.70287)
11. [Multi-Granularity Embedding Fusion NER (May 2025)](https://www.nature.com/articles/s41598-025-90939-8)
12. [Hybrid PII for Financial Documents (July 2025)](https://www.nature.com/articles/s41598-025-04971-9)
13. [Unmasking PII Masking Models (April 2025)](https://arxiv.org/abs/2504.12308)
14. [SynthoNER-Trainer (Dec 2025)](https://medium.com/@mail.fede.cesarini/ner-in-the-llm-era-how-i-used-giant-models-to-train-tiny-ones-03b4b46a5b3c)
15. [SPY Dataset (NAACL 2025)](https://aclanthology.org/2025.naacl-srw.23/)
16. [PANORAMA Dataset (2025)](https://arxiv.org/html/2505.12238v1)
17. [AI4Privacy 500K](https://huggingface.co/datasets/ai4privacy/open-pii-masking-500k-ai4privacy)
18. [Patronus EnterprisePII](https://www.patronus.ai/announcements/patronus-ai-launches-enterprisepii-the-industrys-first-llm-dataset-for-detecting-business-sensitive-information)
19. [UnPII (ICSE 2026)](https://arxiv.org/html/2601.01786)
20. [DeepDFA (ECAI 2024)](https://arxiv.org/abs/2408.08622)
21. [Knowledgator GLiNER-PII](https://huggingface.co/knowledgator/gliner-pii-base-v1.0)
22. [ModernBERT Blog](https://huggingface.co/blog/modernbert)
23. [Clinical ModernBERT (April 2025)](https://arxiv.org/html/2504.03964v1)
24. [SmolLM2](https://arxiv.org/html/2502.02737v1)

---

*Recency sweep conducted February 2026. All architecture recommendations from the initial research remain valid.*
