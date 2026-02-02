# Architecture Survey: NER and PII Detection Systems

> **Date:** 2026-02-01
> **Scope:** Comprehensive survey of named entity recognition (NER) and personally identifiable information (PII) detection architectures, covering academic models, production systems, and emerging lightweight approaches.
> **Purpose:** Inform the design of a lightweight, high-accuracy PII detection model for CPU-first deployment.

---

## Table of Contents

1. [GLiNER Family](#1-gliner-family)
2. [Microsoft Presidio](#2-microsoft-presidio)
3. [Flair NER](#3-flair-ner)
4. [spaCy NER](#4-spacy-ner)
5. [Character-Level NER Models](#5-character-level-ner-models)
6. [Lightweight / Efficient NER](#6-lightweight--efficient-ner)
7. [Span-Based Approaches](#7-span-based-approaches)
8. [PII-Specific Models](#8-pii-specific-models)
9. [Master Comparison Table](#9-master-comparison-table)
10. [Key Insights for Lightweight PII Model Design](#10-key-insights-for-lightweight-pii-model-design)
11. [Citations](#11-citations)

---

## 1. GLiNER Family

The GLiNER family represents a paradigm shift in NER: instead of predicting a fixed label set, these models perform **zero-shot entity recognition** by encoding entity type descriptions alongside text in a shared latent space. This makes them highly flexible for PII detection, where entity taxonomies vary across jurisdictions and use cases.

### 1.1 GLiNER (Original)

| Property | Value |
|---|---|
| **Paper** | *GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer* (NAACL 2024) |
| **Parameters** | 50--300M (Small / Medium / Large variants) |
| **Architecture** | Bidirectional Transformer encoder + span-entity matching head |
| **Mechanism** | Zero-shot NER via entity type prompts encoded in a shared latent space. The model jointly encodes input text and entity type labels, then scores all candidate spans against each entity type using bilinear similarity. |
| **Training** | Pile-NER dataset (auto-annotated from The Pile using ChatGPT), fine-tuned on diverse NER corpora. |
| **Key Innovation** | Eliminates the need for a fixed label set. New entity types can be added at inference time simply by providing their textual descriptions as prompts. |
| **Performance** | GLiNER-L achieves **60.9 F1 zero-shot average** across unseen entity types, competitive with ChatGPT on zero-shot NER while being orders of magnitude faster. On supervised benchmarks, GLiNER-L reaches ~90+ F1 when fine-tuned. |
| **Inference** | CPU-feasible for the Small variant (~50M); Large variant requires GPU for real-time throughput. |

**Architecture Detail:**

```
Input: [entity_type_1] [entity_type_2] ... [SEP] token_1 token_2 ... token_n
         |                                          |
    Entity Type Encoder (shared)           Text Encoder (shared)
         |                                          |
    Entity Embeddings                       Span Representations
         \                                        /
          \--- Bilinear Span-Entity Matching ---/
                         |
                   Entity Predictions
```

### 1.2 GLiNER2

| Property | Value |
|---|---|
| **Paper** | *GLiNER2: Open Problems in Universal Information Extraction* (EMNLP 2025 Demos) |
| **Parameters** | 205M |
| **Architecture** | DeBERTa-v3-base backbone with multi-task information extraction heads |
| **Mechanism** | Unified model for NER, text classification, relation extraction, and key-value extraction. Uses special tokens `[P]` (prompt), `[E]` (entity), `[C]` (class), `[L]` (label), `[SEP]` to delineate task structure in the input. |
| **Training Data** | 254K examples total: 135K real-world samples + 118K GPT-4o-generated synthetic samples |
| **Key Innovation** | Multi-task formulation allows a single model to handle heterogeneous IE tasks, reducing deployment complexity. DeBERTa-v3 disentangled attention provides stronger context modeling than the original GLiNER's encoder. |
| **Performance** | CrossNER F1: **0.590** (zero-shot), **2.6x faster than GPT-4o** on CPU inference. On multi-task benchmarks, competitive with task-specific models while using a single architecture. |
| **Latency** | Designed for CPU deployment; DeBERTa-v3-base is one of the most efficient encoder-only transformers at the 200M scale. |

**Special Token Protocol:**

```
[P] Classify PII types [SEP]
[E] email [E] phone [E] ssn [E] name [SEP]
[C] sensitive [C] non-sensitive [SEP]
token_1 token_2 ... token_n
```

### 1.3 NVIDIA GLiNER-PII

| Property | Value |
|---|---|
| **Base Model** | GLiNER large-v2.1 |
| **Parameters** | ~300M |
| **Entity Types** | 55+ PII entity types (names, addresses, SSNs, emails, phone numbers, financial identifiers, medical record numbers, etc.) |
| **Performance** | **92% recall**, **64% F1** on PII benchmarks. The recall-precision gap indicates aggressive entity detection at the cost of false positives -- a deliberate design choice for privacy-critical applications where missing PII is costlier than over-flagging. |
| **Training** | Fine-tuned on NVIDIA-curated PII datasets with augmented entity type coverage. |
| **Use Case** | Best suited for high-recall PII scanning where downstream human review or regex post-filtering can reduce false positives. |

### 1.4 Knowledgator GLiNER-PII-base

| Property | Value |
|---|---|
| **Parameters** | ~180M |
| **Architecture** | GLiNER base with PII-specific fine-tuning |
| **Performance** | **80.99 F1** -- the best F1 among all GLiNER PII variants, indicating a stronger precision-recall balance than NVIDIA's variant. |
| **Key Strength** | More balanced precision-recall trade-off compared to NVIDIA GLiNER-PII, making it more suitable for automated pipelines without human review. |

### 1.5 Gretel GLiNER PII

| Property | Value |
|---|---|
| **Base** | GLiNER architecture |
| **Training Data** | Fine-tuned on **60K synthetic documents** generated by Gretel's Navigator system |
| **Key Insight** | Demonstrates the viability of **purely synthetic training data** for PII detection. Gretel's synthetic generation pipeline ensures diverse document formats, entity distributions, and contextual variations. |
| **Significance** | Important proof-of-concept for data-scarce PII domains (e.g., medical, financial) where real annotated data is restricted by privacy regulations. |

### GLiNER Family Summary

| Variant | Params | F1 | Recall | Zero-shot? | PII Types |
|---|---|---|---|---|---|
| GLiNER-L (original) | ~300M | 60.9 (ZS avg) | -- | Yes | General NER |
| GLiNER2 | 205M | 59.0 (CrossNER ZS) | -- | Yes | Multi-task IE |
| NVIDIA GLiNER-PII | ~300M | 64.0 | 92.0 | Yes | 55+ PII |
| Knowledgator PII-base | ~180M | 80.99 | -- | Yes | PII-focused |
| Gretel GLiNER PII | ~300M | -- | -- | Yes | PII-focused |

---

## 2. Microsoft Presidio

Microsoft Presidio is an **open-source PII detection and anonymization framework** that takes a fundamentally different approach from end-to-end neural models: it uses a **multi-layered pipeline** combining rule-based, statistical, and neural components.

### Architecture

```
Input Text
    |
    v
+---------------------------+
|   Analyzer Engine          |
|   +---------------------+ |
|   | Pattern Recognizers  | |  <-- Regex + deny lists
|   | (30+ built-in)       | |
|   +---------------------+ |
|   | NER Model Recognizer | |  <-- spaCy / HuggingFace / Azure AI
|   +---------------------+ |
|   | Checksum Validators  | |  <-- Luhn, ITIN, SSN format validation
|   +---------------------+ |
|   | Custom Recognizers   | |  <-- User-defined
|   +---------------------+ |
|           |                |
|   Conflict Resolution      |  <-- Score-based merge of overlapping detections
|           |                |
+---------------------------+
    |
    v
Detected Entities (type, start, end, score)
    |
    v
+---------------------------+
|   Anonymizer Engine        |
|   - Replace / Redact       |
|   - Hash / Encrypt         |
|   - Mask                   |
+---------------------------+
    |
    v
Anonymized Text
```

### Key Properties

| Property | Value |
|---|---|
| **Entity Types** | 30+ built-in (SSN, credit card, email, phone, IBAN, US passport, IP address, etc.) |
| **NER Backends** | Pluggable: spaCy (default), Stanza, HuggingFace Transformers, Azure AI Language |
| **Approach** | Pipeline (not joint): each recognizer runs independently, results are merged via conflict resolution with confidence scores |
| **Extensibility** | Custom recognizers via Python API or YAML configuration |
| **Checksum Validation** | Luhn algorithm (credit cards), SSN format validation, IBAN check digits -- provides near-100% precision for structured PII formats |
| **Performance** | Highly config-dependent. With spaCy sm backend: fast but lower recall on contextual PII. With transformer backend: slower but higher recall. Regex recognizers are near-instant. |
| **Deployment** | Python library, Docker container, or Kubernetes operator. Actively maintained by Microsoft. |

### Strengths and Limitations

**Strengths:**
- Extremely high precision for structured PII (credit cards, SSNs, IBANs) via checksum validation
- Easy to extend with domain-specific patterns
- Production-hardened with enterprise adoption
- Language-agnostic regex patterns

**Limitations:**
- Pipeline approach means NER and regex do not share information -- a phone number in context "call me at..." gets no contextual boost from the NER model
- No joint training across detection layers
- F1 varies dramatically based on configuration choices
- Contextual PII (names, addresses in narrative text) depends entirely on the NER backend quality

---

## 3. Flair NER

Flair is a PyTorch-based NLP framework developed by Humboldt University of Berlin, known for its **contextual string embeddings** -- character-level language model embeddings that capture subword structure and morphology.

### 3.1 ner-english (Standard)

| Property | Value |
|---|---|
| **Parameters** | ~100M |
| **Architecture** | BiLSTM-CRF with stacked embeddings |
| **Embedding Stack** | GloVe (6B, 100d) + Flair forward char-LM + Flair backward char-LM + pooled BERT (mean of layers) |
| **CRF Layer** | Linear-chain CRF for BIO sequence labeling |
| **Performance** | **93.06 F1** on CoNLL-2003 English NER |
| **Key Innovation** | Contextual string embeddings: a character-level language model produces context-dependent word representations by extracting hidden states at word boundaries. This captures morphological patterns (e.g., "-tion", "un-") and handles OOV words naturally. |

**Contextual String Embedding Extraction:**

```
Character sequence: "W a s h i n g t o n"
                     |                   |
              word start              word end
                     |                   |
         forward LM hidden state    backward LM hidden state
                     \                 /
                      concatenate
                          |
                   Word Embedding
```

### 3.2 ner-english-large (FLERT)

| Property | Value |
|---|---|
| **Parameters** | ~560M |
| **Architecture** | FLERT (Fine-grained Language-model Evaluation of Rich Text) |
| **Backbone** | XLM-RoBERTa-Large (355M) + classification head |
| **Mechanism** | Document-level context: feeds entire documents (not just sentences) through the transformer, then classifies tokens. This captures long-range dependencies that sentence-level models miss. |
| **Performance** | **94.36 F1** on CoNLL-2003 English NER |
| **Key Innovation** | Shows that simply providing more context to a large transformer (document vs. sentence) yields significant NER gains. The FLERT approach fine-tunes XLM-RoBERTa with document-level inputs and a token classification head. |

### Flair Summary

| Model | Params | Architecture | CoNLL F1 | Speed | Use Case |
|---|---|---|---|---|---|
| ner-english | ~100M | BiLSTM-CRF + stacked | 93.06 | Moderate CPU | Balanced |
| ner-english-large | ~560M | FLERT (XLM-R-Large) | 94.36 | GPU recommended | Maximum accuracy |

---

## 4. spaCy NER

spaCy is the most widely deployed NER library in production, favoring **speed and pragmatism** over state-of-the-art accuracy. Its NER component uses a **transition-based parsing** approach.

### 4.1 en_core_web_sm

| Property | Value |
|---|---|
| **Parameters** | ~5M (entire pipeline including tokenizer, tagger, parser, NER) |
| **Architecture** | Transition-based parser with HashEmbedCNN features |
| **Embedding** | Multi-hash embeddings (character n-grams hashed into lookup tables) + CNN feature extraction |
| **Performance** | **84.56 F1** on OntoNotes NER |
| **Speed** | ~3,400 words per second (CPU) -- among the fastest NER systems available |
| **Entity Types** | PERSON, ORG, GPE, DATE, TIME, MONEY, CARDINAL, etc. (18 OntoNotes types) |

### 4.2 en_core_web_trf

| Property | Value |
|---|---|
| **Parameters** | ~110M+ (RoBERTa-base backbone + spaCy pipeline) |
| **Architecture** | Transformer (RoBERTa-base) + spaCy transition-based NER head |
| **Performance** | ~**89--90 F1** on OntoNotes NER |
| **Speed** | ~107 words per second (CPU), ~1,000+ WPS on GPU |
| **Entity Types** | Same 18 OntoNotes types |

### Transition-Based NER Parsing

spaCy's NER uses a **shift-reduce parser** that processes tokens left-to-right with a stack-based state machine:

```
Actions:
  BEGIN(label)  -- Start a new entity of type `label`
  IN(label)     -- Continue the current entity
  LAST(label)   -- End the current entity
  UNIT(label)   -- Single-token entity
  OUT           -- Token is not an entity

State: (stack, buffer) -> action -> (new_stack, new_buffer)
```

### Key Limitation

The transition-based approach **cannot handle overlapping or nested entities**. Each token receives exactly one label. This is problematic for PII detection where, for example, "Dr. Jane Smith, MD" contains both a PERSON entity and potentially a TITLE entity at overlapping positions.

### spaCy Summary

| Model | Params | F1 | Speed (CPU) | Size on Disk | Use Case |
|---|---|---|---|---|---|
| en_core_web_sm | ~5M | 84.56 | ~3,400 WPS | ~12 MB | High-throughput, low-accuracy |
| en_core_web_trf | ~110M+ | ~89-90 | ~107 WPS | ~460 MB | Accuracy-critical |

---

## 5. Character-Level NER Models

Character-level models are particularly relevant for PII detection because PII entities often have **distinctive character patterns** (e.g., SSN format `XXX-XX-XXXX`, email structure `user@domain.tld`, phone numbers with parentheses and dashes). These models can learn such patterns directly from character sequences.

### 5.1 CharNER (COLING 2016)

| Property | Value |
|---|---|
| **Paper** | *CharNER: Character-Level Named Entity Recognition* |
| **Architecture** | Pure character-level BiLSTM + Viterbi decoding |
| **Mechanism** | Each character receives an entity label (or O). Word-level entities are derived by majority voting or boundary detection over character labels. No word embeddings or tokenization required. |
| **Performance** | **91.94 F1** on Turkish NER (a morphologically rich language where character-level processing has outsized impact) |
| **Key Insight** | Demonstrates that NER can be performed entirely at the character level without any word-level features. Particularly effective for agglutinative and morphologically complex languages. |
| **Relevance to PII** | The pure character-level approach can naturally capture PII patterns like phone number formats, email structures, and ID number sequences without explicit feature engineering. |

### 5.2 Ma & Hovy (ACL 2016)

| Property | Value |
|---|---|
| **Paper** | *End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF* |
| **Parameters** | <10M |
| **Architecture** | Character CNN + GloVe word embeddings + BiLSTM + CRF |
| **Performance** | **91.21 F1** on CoNLL-2003 |

**Architecture Detail:**

```
Characters: "W" "a" "s" "h" "i" "n" "g" "t" "o" "n"
    |
Character Embeddings (30d)
    |
Conv1D (multiple filter widths) + Max-Pool
    |
Character-level word representation (30d)
    |
Concatenate with GloVe (100d)
    |
BiLSTM (200d hidden)
    |
CRF Layer
    |
BIO Labels
```

**Key Design Choices:**
- Character CNN (not LSTM) for character-level features -- faster and captures local morphological patterns
- GloVe provides semantic grounding
- CRF layer enforces valid BIO sequences (e.g., I-PER cannot follow B-LOC)
- Truly end-to-end: no hand-crafted features, no gazetteer

### 5.3 Lample et al. (NAACL 2016)

| Property | Value |
|---|---|
| **Paper** | *Neural Architectures for Named Entity Recognition* |
| **Parameters** | <10M |
| **Architecture** | Character BiLSTM + word embeddings + BiLSTM + CRF |
| **Performance** | **90.94 F1** on CoNLL-2003 |
| **Key Difference from Ma & Hovy** | Uses BiLSTM (instead of CNN) for character-level features. The BiLSTM captures longer-range character dependencies but is slightly slower. Also proposed a stack-LSTM transition-based variant. |

### 5.4 CharacterBERT (COLING 2020)

| Property | Value |
|---|---|
| **Paper** | *CharacterBERT: Reconciling ELMo and BERT for Word-Level Open-Vocabulary Representations* |
| **Parameters** | ~110M |
| **Architecture** | BERT architecture with WordPiece tokenizer replaced by a Character-CNN module (inspired by ELMo's CharCNN) |
| **Mechanism** | Instead of looking up subword tokens in a vocabulary, CharacterBERT builds word representations from characters using multiple CNN filters of varying widths, followed by max-pooling and a highway network. These character-derived representations are then processed by standard BERT transformer layers. |
| **Performance** | Competitive with BERT on standard benchmarks; **significantly more robust to typos, misspellings, and noisy text** |
| **Relevance to PII** | Real-world PII data often contains typos, OCR errors, and non-standard formatting. CharacterBERT's character-level input processing handles these gracefully, whereas WordPiece-based models produce garbled subword sequences for misspelled words. |

**CharacterBERT vs. BERT Input Processing:**

```
BERT:        "Washington" -> ["Wash", "##ington"] -> Embedding lookup
CharBERT:    "Washington" -> ['W','a','s','h','i','n','g','t','o','n'] -> CharCNN -> Word repr

BERT:        "Washingtn" (typo) -> ["Wash", "##ing", "##tn"] -> Degraded embedding
CharBERT:    "Washingtn" (typo) -> CharCNN -> Similar repr to "Washington"
```

### Character-Level Models Summary

| Model | Year | Params | Architecture | CoNLL F1 | OOV Handling |
|---|---|---|---|---|---|
| CharNER | 2016 | Small | Char-BiLSTM + Viterbi | 91.94 (Turkish) | Excellent |
| Ma & Hovy | 2016 | <10M | Char-CNN + BiLSTM-CRF | 91.21 | Good |
| Lample et al. | 2016 | <10M | Char-BiLSTM + BiLSTM-CRF | 90.94 | Good |
| CharacterBERT | 2020 | ~110M | Char-CNN + Transformer | ~BERT-level | Excellent |

---

## 6. Lightweight / Efficient NER

This section surveys models and techniques specifically designed for **efficient NER inference**, targeting CPU deployment, edge devices, and high-throughput production scenarios.

### 6.1 DistilBERT-NER

| Property | Value |
|---|---|
| **Parameters** | ~66M (40% fewer than BERT-base) |
| **Architecture** | 6-layer Transformer (distilled from 12-layer BERT-base) + token classification head |
| **Distillation** | Knowledge distillation with soft target loss + cosine embedding loss + MLM loss during pre-training |
| **Speed** | **60% faster** than BERT-base on CPU |
| **ONNX INT8** | With ONNX Runtime quantization (INT8), achieves up to **6x speedup** over PyTorch BERT-base with <1% F1 degradation |
| **Performance** | Retains ~97% of BERT-base NER performance |

**Optimization Pipeline:**

```
PyTorch BERT-base (110M, FP32)
    |
Knowledge Distillation -> DistilBERT (66M, FP32)
    |
ONNX Export -> DistilBERT (66M, ONNX FP32)
    |
Dynamic Quantization -> DistilBERT (66M, ONNX INT8)
    |
Result: ~6x faster than original, <1% F1 loss
```

### 6.2 TinyBERT-4

| Property | Value |
|---|---|
| **Parameters** | ~14.5M |
| **Architecture** | 4-layer Transformer (distilled with attention transfer from BERT-base) |
| **Speed** | **9.4x faster** than BERT-base |
| **Model Size** | **55 MB** on disk |
| **Distillation** | Two-stage: (1) general distillation on pre-training data, (2) task-specific distillation with attention + hidden state + prediction layer matching |
| **Performance** | ~90-93% of BERT-base NER performance depending on dataset |
| **Key Trade-off** | Aggressive compression sacrifices some ability to handle long-range dependencies and rare entities, but excels on common PII patterns. |

### 6.3 MobileBERT

| Property | Value |
|---|---|
| **Parameters** | ~25.3M |
| **Architecture** | 24-layer "thin" Transformer with bottleneck structure (128d hidden per layer, with inverted bottleneck feed-forward) |
| **Latency** | **62ms** per inference on Google Pixel 4 |
| **Key Innovation** | Uses an inverted bottleneck: narrow Transformer layers with wide feed-forward intermediate layers. Also introduces "bottleneck embedding" that reduces the embedding dimension before feeding into transformer layers. Teacher model is a specially designed "IB-BERT" (Inverted Bottleneck BERT) rather than standard BERT. |
| **Performance** | Competitive with BERT-base on GLUE, SQuAD, and NER tasks |
| **Relevance** | Demonstrates that mobile/edge NER is feasible with careful architectural design. |

### 6.4 TENER (2019)

| Property | Value |
|---|---|
| **Paper** | *TENER: Adapting Transformer Encoder for Named Entity Recognition* |
| **Architecture** | Adapted Transformer Encoder with direction-aware and distance-aware attention |
| **Character Encoder** | Only **6,600 parameters** for the character-level encoder |
| **Key Innovation** | Standard Transformer attention is permutation-invariant, which is problematic for NER where position matters. TENER modifies attention to be: (1) **direction-aware** -- distinguishes left vs. right context, and (2) **distance-aware** -- attention weights decay with distance, using sinusoidal relative position embeddings that are not added to content but used as separate attention bias terms. |
| **Performance** | Competitive with BiLSTM-CRF on English and Chinese NER benchmarks |
| **Significance** | Shows that Transformers can be adapted for NER with minimal parameter overhead, and that positional awareness is critical for entity boundary detection. |

### Efficiency Techniques Summary

| Technique | Typical Speedup | Size Reduction | F1 Impact |
|---|---|---|---|
| Knowledge Distillation (6L) | 1.5--2x | 40% | <2% loss |
| Knowledge Distillation (4L) | 3--5x | 75% | 3--7% loss |
| ONNX Export | 1.2--1.5x | -- | None |
| INT8 Quantization | 2--4x | 50--75% | <1% loss |
| ONNX + INT8 Combined | 4--6x | 50--75% | <1.5% loss |
| Pruning (unstructured 50%) | 1.5--2x | 50% | 1--3% loss |
| Pruning (structured 30%) | 1.3x | 30% | 2--5% loss |

---

## 7. Span-Based Approaches

Span-based models represent an alternative to BIO sequence labeling: instead of assigning a label to each token, they **enumerate candidate spans** and classify each span as an entity type or "not an entity." This naturally handles overlapping and nested entities.

### 7.1 SpanNER (ACL 2021)

| Property | Value |
|---|---|
| **Paper** | *SpanNER: Named Entity Re-/Recognition as Span Prediction* |
| **Architecture** | Encoder + span enumeration + span classifier |
| **Mechanism** | Enumerates all spans up to a maximum length, represents each span using start/end token embeddings (+ optional span width embedding), and classifies each span independently. |
| **Key Finding** | Span-based and BIO-based approaches are **complementary** -- they make different errors. An ensemble of span-based and BIO-based models outperforms either alone. |
| **Relevance** | For PII detection, span-based approaches can identify overlapping entities (e.g., a street address that contains a city name). |

### 7.2 SpERT (ECAI 2020)

| Property | Value |
|---|---|
| **Paper** | *SpERT: Span-based Joint Entity and Relation Extraction with Transformer Pre-training* |
| **Architecture** | BERT encoder + span classification + relation classification |
| **Mechanism** | Joint entity and relation extraction: (1) encode text with BERT, (2) enumerate candidate spans, (3) classify spans as entity types, (4) classify pairs of detected entities for relations. |
| **Key Feature** | **Handles overlapping entities** natively, since each span is classified independently. |
| **Relevance** | The joint entity-relation formulation could be extended to PII detection with contextual relationships (e.g., associating a name with its corresponding email address in a document). |

### 7.3 Biaffine NER (ACL 2020)

| Property | Value |
|---|---|
| **Paper** | *Named Entity Recognition as Dependency Parsing* |
| **Architecture** | Encoder + Start MLP + End MLP + Biaffine scoring matrix |
| **Mechanism** | Treats NER as finding (start, end, type) triples. Separate MLPs project token representations into "start" and "end" spaces. A biaffine classifier then scores all (start_i, end_j, type_k) combinations. |
| **Performance** | **SOTA on 8 NER corpora** at time of publication |
| **Key Innovation** | Borrows the biaffine attention mechanism from dependency parsing. The scoring is: `score(i, j, k) = h_start_i^T W_k h_end_j + b_k`, where W_k is a type-specific biaffine weight matrix. |
| **Complexity** | O(n^2 * T) for n tokens and T entity types, but practically efficient because the biaffine operation is a single batched matrix multiplication. |

**Biaffine Scoring Architecture:**

```
Input tokens: t_1, t_2, ..., t_n
        |
    Encoder (BERT/BiLSTM)
        |
   h_1, h_2, ..., h_n
      /           \
Start MLP       End MLP
   |               |
s_1..s_n        e_1..e_n
      \           /
    Biaffine Matrix
    score[i][j][k] = s_i^T * W_k * e_j + b_k
        |
   (start, end, type) predictions
```

### 7.4 SpanMarker (2023)

| Property | Value |
|---|---|
| **Paper** | *SpanMarker: Few-Shot Named Entity Recognition with Span Markers* |
| **Architecture** | Pre-trained encoder (BERT/RoBERTa/DeBERTa) with span marker tokens |
| **Mechanism** | Inserts special `[START]` and `[END]` marker tokens around candidate spans in the input, then classifies based on the marker token representations. This is simpler than biaffine scoring and leverages the full power of the pre-trained encoder's attention. |
| **Performance** | **94.4 F1** on CoNLL-2003 (with RoBERTa-Large backbone) |
| **Key Advantage** | Works with any pre-trained encoder without architectural modifications. The span markers are simply added to the vocabulary and fine-tuned. |

### 7.5 Filtered Semi-Markov CRF (EMNLP 2023)

| Property | Value |
|---|---|
| **Paper** | Precursor work to GLiNER |
| **Architecture** | Semi-Markov CRF with span-level features and filtering |
| **Mechanism** | Unlike linear-chain CRFs that model token-level transitions, Semi-Markov CRFs model **segment-level transitions**, allowing direct span-level feature computation. The "filtered" variant uses a lightweight scoring function to prune unlikely spans before full CRF inference, reducing complexity from O(n^2 * T) to near-linear. |
| **Significance** | Bridged the gap between CRF-based and span-based approaches, directly inspiring GLiNER's span-entity matching formulation. |

### Span-Based Approaches Summary

| Model | Year | Mechanism | Overlapping? | CoNLL F1 | Complexity |
|---|---|---|---|---|---|
| SpanNER | 2021 | Enumerate + classify | Yes | Competitive | O(n^2 * T) |
| SpERT | 2020 | Enumerate + classify + relations | Yes | Competitive | O(n^2 * T) |
| Biaffine NER | 2020 | Start/End MLPs + biaffine | Yes | SOTA x8 | O(n^2 * T) |
| SpanMarker | 2023 | Marker tokens + classify | Yes | 94.4 | O(n * k * T) |
| Filtered Semi-CRF | 2023 | Segment CRF + filtering | Yes | Competitive | ~O(n * T) |

---

## 8. PII-Specific Models

This section covers models explicitly designed, trained, and evaluated for PII detection (as opposed to general NER models applied to PII).

### 8.1 Piiranha-v1

| Property | Value |
|---|---|
| **Model** | Piiranha-v1-detect-personal-information |
| **Backbone** | mDeBERTa-v3-base (multilingual DeBERTa) |
| **Parameters** | ~280M |
| **PII Types** | 17 entity types: PERSON, EMAIL, PHONE, ADDRESS, SSN, CREDIT_CARD, DATE_OF_BIRTH, PASSPORT, DRIVER_LICENSE, IP_ADDRESS, URL, USERNAME, PASSWORD, BANK_ACCOUNT, TAX_ID, MEDICAL_RECORD, NATIONAL_ID |
| **Performance** | **98.27% recall** on held-out PII benchmark |
| **Downloads** | 1.1M+ on HuggingFace (as of survey date) |
| **Key Strength** | Multilingual PII detection -- mDeBERTa-v3-base provides coverage across 100+ languages, critical for international PII compliance (GDPR, CCPA, PIPEDA, etc.) |
| **Significance** | The most downloaded PII-specific model, indicating strong community validation. The 98.27% recall makes it suitable for compliance-critical applications. |

### 8.2 BigCode StarPII

| Property | Value |
|---|---|
| **Backbone** | StarEncoder (code-specific encoder model) |
| **Domain** | Source code PII detection |
| **PII Types** | 6 code-specific types: NAME, EMAIL, USERNAME, IP_ADDRESS, KEY (API keys/secrets), PASSWORD |
| **Use Case** | Detecting and redacting PII from open-source code repositories before inclusion in code LLM training data (e.g., StarCoder) |
| **Key Insight** | Code PII has very different distributions and contexts than natural language PII. API keys, for example, are high-entropy strings that benefit from character-level pattern detection. |

### 8.3 Roblox PII Classifier

| Property | Value |
|---|---|
| **Backbone** | XLM-RoBERTa-Large |
| **Parameters** | ~560M |
| **Architecture** | Sequence classification (not token-level NER) |
| **Performance** | **94% F1**, **370K requests per second** (with heavy infrastructure optimization) |
| **Approach** | Classifies entire messages/conversations as containing or not containing PII -- a **binary/multi-class classification** task rather than span-level entity extraction |
| **Key Distinction** | This is **not NER** -- it does not identify where PII appears in text, only whether PII is present. Designed for real-time content moderation in Roblox's chat system. |
| **Throughput** | The 370K RPS figure reflects Roblox's production infrastructure with model serving optimization, batching, and GPU clusters. |

### 8.4 Amazon Comprehend PII Detection

| Property | Value |
|---|---|
| **Type** | Managed API service |
| **Entity Types** | 36 PII types (US-centric + international) |
| **Architecture** | Undisclosed (proprietary) |
| **Modes** | Detection (returns entity spans) + Redaction (returns anonymized text) |
| **Languages** | English primary, limited multilingual support |
| **Key Feature** | Confidence scores per entity, integrated with AWS ecosystem (S3, Lambda, etc.) |
| **Limitation** | Black-box: no model customization, no fine-tuning, pay-per-request pricing |

### 8.5 Google Cloud DLP (Data Loss Prevention)

| Property | Value |
|---|---|
| **Type** | Managed API service |
| **Detectors** | **120+ infoType detectors** (the most comprehensive coverage of any system surveyed) |
| **Architecture** | Undisclosed; likely a hybrid of regex, ML models, and context-aware rules |
| **Key Features** | Custom infoType creation, likelihood scoring (VERY_UNLIKELY to VERY_LIKELY), inspection rules for fine-tuning detection, de-identification transforms (masking, tokenization, bucketing, date shifting, crypto hashing) |
| **Coverage** | Global: supports country-specific PII types (e.g., Australian Medicare, Brazilian CPF, Indian Aadhaar) |
| **Limitation** | Cloud-only, no on-premise deployment, per-request pricing |

### PII-Specific Models Summary

| Model | Params | Type | PII Types | Best Metric | Deployment |
|---|---|---|---|---|---|
| Piiranha-v1 | 280M | Token NER | 17 | 98.27% recall | Self-hosted |
| BigCode StarPII | -- | Token NER | 6 (code) | -- | Self-hosted |
| Roblox PII | 560M | Sequence clf | Binary | 94% F1 | Internal |
| Amazon Comprehend | -- | API | 36 | -- | AWS Cloud |
| Google Cloud DLP | -- | API | 120+ | -- | GCP Cloud |

---

## 9. Master Comparison Table

The following table provides a unified comparison across all surveyed systems, organized by parameter count (ascending).

| System | Params | Mechanism | CPU Speed | PII-Specific | Best F1 | Key Trade-off |
|---|---|---|---|---|---|---|
| spaCy en_core_web_sm | ~5M | Transition-based + HashEmbed | ~3,400 WPS | No | 84.56 | Speed over accuracy; no overlapping entities |
| Ma & Hovy (2016) | <10M | Char-CNN + BiLSTM-CRF | Moderate | No | 91.21 | Compact but no pre-training benefits |
| Lample et al. (2016) | <10M | Char-BiLSTM + BiLSTM-CRF | Moderate | No | 90.94 | Compact but no pre-training benefits |
| TinyBERT-4 NER | ~14.5M | 4L Transformer (distilled) | 9.4x BERT | No | ~90% of BERT | Aggressive compression; weak on rare entities |
| MobileBERT | ~25.3M | 24L thin Transformer | 62ms/inference (mobile) | No | ~BERT-level | Mobile-optimized; complex training |
| GLiNER-S | ~50M | Span-entity matching | Moderate | No | ~55 (ZS) | Smallest zero-shot NER; limited capacity |
| DistilBERT-NER | ~66M | 6L Transformer (distilled) | 1.6x BERT (6x w/ ONNX INT8) | No | ~97% of BERT | Best efficiency-accuracy balance at 66M |
| Flair ner-english | ~100M | BiLSTM-CRF + stacked embed | Moderate | No | 93.06 | Strong but complex embedding stack |
| CharacterBERT | ~110M | Char-CNN + Transformer | ~BERT speed | No | ~BERT-level | Robust to noise; non-standard training |
| spaCy en_core_web_trf | ~110M+ | Transformer + transition | ~107 WPS | No | ~89-90 | Slower than sm; still no overlapping entities |
| Knowledgator GLiNER-PII | ~180M | GLiNER + PII fine-tune | Moderate | Yes | 80.99 | Best GLiNER PII F1; moderate size |
| GLiNER2 | 205M | DeBERTa + multi-task IE | 2.6x GPT-4o (CPU) | No | 59.0 (ZS) | Multi-task flexibility; moderate zero-shot |
| Piiranha-v1 | ~280M | mDeBERTa-v3 token clf | Moderate | Yes | 98.27% recall | Best PII recall; large model |
| GLiNER-L | ~300M | Span-entity matching | Slow (CPU) | No | 60.9 (ZS) | Best zero-shot NER; too large for CPU |
| NVIDIA GLiNER-PII | ~300M | GLiNER + PII fine-tune | Slow (CPU) | Yes | 92% recall / 64 F1 | High recall, low precision |
| Flair ner-english-large | ~560M | FLERT (XLM-R-Large) | GPU only | No | 94.36 | Highest CoNLL F1; impractical for CPU |
| Roblox PII Classifier | ~560M | XLM-R-Large seq clf | GPU only (370K RPS) | Yes | 94% F1 | Classification only, not entity extraction |
| Microsoft Presidio | Variable | Pipeline (regex + NER + checksum) | Config-dependent | Yes | Config-dependent | Most flexible; not end-to-end |
| Amazon Comprehend | Unknown | Proprietary API | N/A (cloud) | Yes | -- | No customization; cloud dependency |
| Google Cloud DLP | Unknown | Proprietary API | N/A (cloud) | Yes | -- | Most PII types (120+); cloud dependency |

### Speed-Accuracy Pareto Frontier

For CPU-first PII deployment, the Pareto-optimal systems are:

```
Accuracy (F1)
    ^
95+ |                                          * Piiranha (280M, 98% recall)
    |                                    * Flair-large (560M)
93+ |                              * Flair-std (100M)
    |                        * SpanMarker (varies)
91+ |                  * Ma & Hovy (<10M)
    |            * DistilBERT-NER (66M, +ONNX)
89+ |      * spaCy trf (110M)
    |
85+ | * spaCy sm (5M)
    |
81+ |                        * Knowledgator GLiNER-PII (180M)
    |
    +----+----+----+----+----+----+----+----+-----> CPU Speed
    Slow                                       Fast
```

---

## 10. Key Insights for Lightweight PII Model Design

Based on this comprehensive survey, the following design principles emerge for building a lightweight, high-accuracy PII detection model optimized for CPU deployment.

### 10.1 DeBERTa-v3 is the Dominant Backbone

The **mDeBERTa-v3** architecture appears as the backbone in the best-performing PII-specific models:
- **Piiranha-v1** (98.27% recall) uses mDeBERTa-v3-base
- **GLiNER2** (best multi-task IE) uses DeBERTa-v3-base
- DeBERTa-v3's **disentangled attention** (separate content and position attention) provides stronger entity boundary detection than RoBERTa or BERT
- DeBERTa-v3's **ELECTRA-style replaced token detection** pre-training is more sample-efficient than MLM, producing better representations at smaller model sizes

**Recommendation:** Use DeBERTa-v3 as the starting backbone, with aggressive distillation to reach target size.

### 10.2 Character Features are Underexploited for PII

PII entities have strong **character-level signatures** that current transformer-based PII models largely ignore:
- SSN: `\d{3}-\d{2}-\d{4}`
- Email: `[chars]@[chars].[chars]`
- Phone: `(\d{3}) \d{3}-\d{4}`
- Credit card: `\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}`
- IP address: `\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}`

The character-level NER literature (Section 5) demonstrates that character features provide:
- Robust OOV handling (critical for noisy real-world PII text)
- Natural pattern detection for structured PII formats
- Complementary signal to subword tokenization

**Recommendation:** Incorporate a character-level feature module (CNN or lightweight attention) alongside the transformer backbone, similar to CharacterBERT or Ma & Hovy's architecture.

### 10.3 Hybrid Approach (Regex + Neural) is Likely Optimal

Microsoft Presidio's pipeline approach reveals an important insight: **structured PII** (credit cards, SSNs, IBANs) is best detected by regex + checksum validation (near-100% precision), while **contextual PII** (names, addresses, occupations) requires neural models.

A hybrid architecture should:
1. Use regex recognizers with checksum validation for structured PII (zero false negatives for format-matching entities)
2. Use a neural model for contextual PII detection
3. Merge results with confidence-weighted conflict resolution
4. Optionally use regex detections as **features** for the neural model (not just post-processing)

**Recommendation:** Design the model to accept regex detection signals as input features, enabling the neural component to focus on contextual PII while leveraging deterministic patterns for structured PII.

### 10.4 Synthetic Data is Essential

Multiple successful PII models rely heavily on synthetic training data:
- **GLiNER2:** 118K of 254K training examples are GPT-4o synthetic (46%)
- **Gretel GLiNER PII:** Trained entirely on 60K synthetic documents
- **Piiranha-v1:** Uses synthetic data augmentation for rare PII types

Real PII data is inherently scarce due to privacy regulations (GDPR, HIPAA, CCPA). Synthetic data generation offers:
- Unlimited training examples with controlled entity distributions
- Coverage of rare PII types (passport numbers, medical record numbers)
- Domain-specific formatting (legal documents, medical records, financial forms)
- Multi-language coverage without real data collection

**Recommendation:** Invest in a robust synthetic PII data generation pipeline using LLMs (GPT-4o or Claude) with careful quality control and deduplication.

### 10.5 Model Size Targets

Based on the survey, the following size targets emerge for CPU-deployable PII detection:

| Target | Params | Expected F1 | Rationale |
|---|---|---|---|
| **Minimum viable** | ~15--20M | ~85--88 | TinyBERT-4 scale; suitable for edge/mobile. 4-layer distilled transformer with character features. |
| **Sweet spot** | ~50M | ~90--93 | DistilBERT scale; best efficiency-accuracy trade-off. 6-layer DeBERTa distillation with ONNX INT8 gives 6x speedup over BERT-base. |
| **Upper bound** | ~100M | ~94--96 | Full DeBERTa-v3-xsmall or custom architecture. Still CPU-feasible with ONNX optimization. Beyond this, GPU becomes necessary for real-time. |

### 10.6 ONNX + INT8 Gives Up to 6x CPU Speedup

The most impactful optimization for CPU deployment is the **ONNX Runtime + INT8 quantization** pipeline:

```
Training (PyTorch, FP32)
    |
Export to ONNX (graph optimization: constant folding, operator fusion)
    |  ~1.2-1.5x speedup, no accuracy loss
    v
Dynamic INT8 Quantization (weights: INT8, activations: FP32 or INT8)
    |  ~2-4x additional speedup, <1% F1 loss
    v
ONNX Runtime Inference
    |  Total: 4-6x speedup over PyTorch FP32
    v
Production Deployment
```

Key considerations:
- **Dynamic quantization** (quantize weights, compute activations dynamically) is simpler and nearly as fast as static quantization for transformer models
- INT8 quantization has minimal impact on NER F1 because entity detection relies more on relative token similarities than absolute values
- ONNX graph optimizations (operator fusion, constant folding) provide additional speedup independent of quantization

**Recommendation:** Design the training pipeline with ONNX export and INT8 quantization as first-class deployment targets from day one.

### 10.7 Architectural Recommendations Summary

| Design Decision | Recommendation | Rationale |
|---|---|---|
| **Backbone** | DeBERTa-v3 (distilled) | Best PII performance, disentangled attention |
| **Character features** | CNN-based character encoder | Pattern detection for structured PII, OOV robustness |
| **Labeling scheme** | Span-based (not BIO) | Handles overlapping entities, aligns with GLiNER paradigm |
| **Structured PII** | Regex + checksum pre-detection | Near-perfect precision for format-matching entities |
| **Training data** | 70% synthetic + 30% real | Address data scarcity, control entity distribution |
| **Target size** | 50M params (sweet spot) | Best CPU efficiency-accuracy trade-off |
| **Inference** | ONNX Runtime + INT8 | 4-6x speedup for CPU deployment |
| **Evaluation** | Recall-first, then precision | Missing PII is costlier than false positives in compliance |

---

## 11. Citations

### GLiNER Family
1. Zaratiana, U., Tomeh, N., Holat, P., & Charnois, T. (2024). *GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer*. Proceedings of NAACL 2024.
2. Zaratiana, U., Tomeh, N., Holat, P., & Charnois, T. (2025). *GLiNER2: Open Problems in Universal Information Extraction*. Proceedings of EMNLP 2025 (Demos Track).
3. NVIDIA. (2024). *GLiNER-PII: Privacy-Preserving Named Entity Recognition*. NVIDIA NGC Model Card.
4. Knowledgator. (2024). *GLiNER-PII-base*. HuggingFace Model Card.
5. Gretel.ai. (2024). *Gretel GLiNER PII*. Gretel Documentation.

### Microsoft Presidio
6. Microsoft. (2019--2025). *Presidio: Data Protection and Anonymization SDK*. GitHub: microsoft/presidio.

### Flair NER
7. Akbik, A., Blythe, D., & Vollgraf, R. (2018). *Contextual String Embeddings for Sequence Labeling*. Proceedings of COLING 2018.
8. Schweter, S., & Akbik, A. (2020). *FLERT: Document-Level Features for Named Entity Recognition*. arXiv:2011.06993.

### spaCy NER
9. Honnibal, M., & Montani, I. (2017). *spaCy 2: Natural Language Understanding with Bloom Embeddings, Convolutional Neural Networks and Incremental Parsing*. Explosion AI.

### Character-Level Models
10. Kuru, O., Can, O. A., & Yuret, D. (2016). *CharNER: Character-Level Named Entity Recognition*. Proceedings of COLING 2016.
11. Ma, X., & Hovy, E. (2016). *End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF*. Proceedings of ACL 2016.
12. Lample, G., Ballesteros, M., Subramanian, S., Kawakami, K., & Dyer, C. (2016). *Neural Architectures for Named Entity Recognition*. Proceedings of NAACL 2016.
13. El Boukkouri, H., Ferret, O., Lavergne, T., Noji, H., Zweigenbaum, P., & Tsujii, J. (2020). *CharacterBERT: Reconciling ELMo and BERT for Word-Level Open-Vocabulary Representations*. Proceedings of COLING 2020.

### Lightweight / Efficient NER
14. Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter*. NeurIPS 2019 Workshop.
15. Jiao, X., Yin, Y., Shang, L., Jiang, X., Chen, X., Li, L., Wang, F., & Liu, Q. (2020). *TinyBERT: Distilling BERT for Natural Language Understanding*. Findings of EMNLP 2020.
16. Sun, Z., Yu, H., Song, X., Liu, R., Yang, Y., & Zhou, D. (2020). *MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices*. Proceedings of ACL 2020.
17. Yan, H., Deng, B., Li, X., & Qiu, X. (2019). *TENER: Adapting Transformer Encoder for Named Entity Recognition*. arXiv:1911.04474.

### Span-Based Approaches
18. Fu, J., Huang, X., & Liu, P. (2021). *SpanNER: Named Entity Re-/Recognition as Span Prediction*. Proceedings of ACL 2021.
19. Eberts, M., & Ulges, A. (2020). *Span-based Joint Entity and Relation Extraction with Transformer Pre-training*. Proceedings of ECAI 2020.
20. Yu, J., Bohnet, B., & Poesio, M. (2020). *Named Entity Recognition as Dependency Parsing*. Proceedings of ACL 2020.
21. Aarsen, T. (2023). *SpanMarker for Named Entity Recognition*. arXiv:2309.13880.

### PII-Specific Models
22. AI4Privacy. (2024). *Piiranha-v1-detect-personal-information*. HuggingFace Model Card.
23. BigCode Project. (2023). *StarPII: PII Detection in Source Code*. BigCode Technical Report.
24. Roblox. (2023). *Building a Robust PII Classification System at Scale*. Roblox Engineering Blog.
25. Amazon Web Services. (2019--2025). *Amazon Comprehend PII Detection*. AWS Documentation.
26. Google Cloud. (2018--2025). *Cloud Data Loss Prevention (DLP)*. Google Cloud Documentation.

### Additional References
27. He, P., Liu, X., Gao, J., & Chen, W. (2021). *DeBERTa: Decoding-enhanced BERT with Disentangled Attention*. Proceedings of ICLR 2021.
28. He, P., Gao, J., & Chen, W. (2023). *DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing*. Proceedings of ICLR 2023.
29. Conneau, A., et al. (2020). *Unsupervised Cross-lingual Representation Learning at Scale (XLM-RoBERTa)*. Proceedings of ACL 2020.

---

*This survey was compiled as part of the DataFog PII Model Architecture research initiative. All performance figures are drawn from original papers, model cards, and official documentation. Figures may vary based on evaluation dataset, preprocessing, and hardware configuration.*
