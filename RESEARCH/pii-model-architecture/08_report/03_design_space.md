# Design Space Exploration: Lightweight PII Detection Architecture

This report documents the systematic exploration of architectural choices for a lightweight, high-accuracy PII detection model. Each design axis is evaluated for parameter efficiency, inference latency, and suitability for PII-specific patterns. The recommended configuration is justified against alternatives at every level.

---

## 1. Efficient Transformer Backbones

The backbone encoder is the single largest contributor to model size and quality. We evaluated all transformer variants under ~35M parameters that have pretrained checkpoints available.

### 1.1 DeBERTa-v3-xsmall (RECOMMENDED)

- **Parameters:** ~22M (12 layers, hidden size 384, 6 attention heads)
- **Key innovation:** Disentangled attention mechanism that computes content-to-content, content-to-position, and position-to-content attention scores separately, rather than collapsing them into a single attention matrix. This allows the model to learn position-dependent patterns (critical for structured PII like SSNs and phone numbers) without conflating them with semantic content.
- **Pretraining method:** ELECTRA-style replaced token detection (RTD). Instead of masked language modeling, DeBERTa-v3 is trained as a discriminator that identifies which tokens were replaced by a generator. RTD provides a training signal on every input token (not just the 15% masked), yielding substantially better sample efficiency.
- **Empirical evidence:** An April 2025 controlled comparison study demonstrated that DeBERTaV3 still outperforms ModernBERT on NLU benchmarks when both are evaluated at equivalent model sizes, despite ModernBERT incorporating more recent architectural innovations (RoPE, SwiGLU, RMSNorm). The disentangled attention and RTD pretraining appear to provide a durable advantage for token-level understanding tasks.
- **PII-specific advantages:** The disentangled position encoding helps the model learn that digits at positions 4 and 7 in a 10-character sequence are separators (phone numbers), or that an "@" symbol divides local-part from domain (email addresses). These positional patterns are precisely what PII detection requires.
- **Best quality-per-parameter ratio available** among all evaluated backbones.

### 1.2 Alternatives Considered

**ELECTRA-small**
- 14M parameters, 12 layers, 256 hidden dimensions, 4 attention heads
- Same RTD pretraining advantage as DeBERTa-v3
- Concern: 256 hidden dimensions may be insufficient for capturing complex PII contexts where entity boundaries depend on surrounding semantics (e.g., distinguishing "John" as a person name vs. a bathroom fixture requires contextual understanding that benefits from wider representations)

**TinyBERT-4**
- 14.5M parameters, 4 transformer layers, 312 hidden dimensions, 12 attention heads
- 9.4x faster inference than BERT-base
- Two-stage knowledge distillation from a larger teacher model
- Concern: Only 4 transformer layers severely limits the depth of contextual reasoning. PII detection in ambiguous contexts (e.g., "Call 555-0123" vs. "Error code 555-0123") benefits from deeper representations

**MobileBERT**
- 25.3M parameters, 24 layers with inverted bottleneck architecture (512 down to 128 per layer)
- Designed for on-device deployment: 62ms inference on a Pixel 4 phone
- Concern: The bottleneck architecture (compressing to 128 dims within each layer) may discard fine-grained token features needed for precise entity boundary detection. Additionally, the 24-layer depth does not proportionally improve NER quality compared to 12-layer models

**MobileBERT-tiny**
- 15.1M parameters, smallest BERT-family variant with pretrained weights
- Aggressively compressed bottleneck dimensions
- Concern: Quality degradation is noticeable on token classification tasks

**MiniLM-12x384**
- ~33M parameters, 12 layers, 384 hidden dimensions, 12 attention heads
- Deep self-attention distillation from a larger teacher
- Excellent ONNX Runtime support and optimization documentation
- Designated as the **fallback option** if DeBERTa-v3-xsmall shows unexpected issues during fine-tuning or deployment
- Concern: 50% more parameters than DeBERTa-v3-xsmall without proportional quality gains on NER tasks

**ModernBERT (December 2024)**
- Incorporates modern architectural improvements: Rotary Position Embeddings (RoPE), RMSNorm (replacing LayerNorm), SwiGLU activation (replacing GELU), alternating local/global attention patterns
- Supports 8192 token context length natively
- Despite these innovations, **DeBERTaV3 still wins on NLU tasks** in controlled comparisons. The disentangled attention mechanism and RTD pretraining provide advantages that RoPE and SwiGLU do not replicate.

**NeoBERT (February 2025)**
- Aims to combine the best of ModernBERT (architectural innovations) with DeBERTa-style training objectives
- Potentially the best of both worlds
- **Not yet widely available:** limited pretrained checkpoints, no community-validated fine-tuning recipes, and no established track record on NER tasks. Too early to adopt for a production system.

### 1.3 Backbone Comparison Table

| Model | Params | Layers | Hidden | Heads | Key Feature |
|-------|--------|--------|--------|-------|-------------|
| **DeBERTa-v3-xsmall** | **22M** | **12** | **384** | **6** | **Disentangled attention** |
| ELECTRA-small | 14M | 12 | 256 | 4 | Discriminator pretraining |
| TinyBERT-4 | 14.5M | 4 | 312 | 12 | Two-stage distillation |
| MobileBERT | 25.3M | 24 | 512->128 | 4 | Bottleneck architecture |
| MiniLM-12x384 | 33M | 12 | 384 | 12 | Deep self-attention distillation |

---

## 2. Character-Level Encoders

Subword tokenization (BPE, WordPiece, SentencePiece) is optimized for natural language morphology. It is actively harmful for structured PII patterns: "555-867-5309" may be split into ["555", "-", "86", "##7", "-", "53", "##09"], destroying the digit-group structure that signals a phone number. Character-level encoders recover this lost information.

### 2.1 Character CNN (RECOMMENDED)

- **Lineage:** Based on the Ma & Hovy (2016) end-to-end sequence labeling architecture, widely adopted in NER systems.
- **Architecture details:**
  - Character embedding layer: 50 dimensions, vocabulary of 256 characters (full ASCII + common Unicode)
  - Three parallel Conv1D filter banks with kernel sizes [3, 4, 5], each with 50 filters
  - Max-pooling over the character sequence of each word, producing a 150-dimensional output per word
  - Optional dropout (0.25) after max-pooling
- **Parameter count:** ~0.3-0.5M (negligible relative to the backbone)
- **Speed:** 3-5x faster than character LSTM due to fully parallel convolution operations. No sequential dependency between characters within a word.
- **PII-specific value:** The multi-width convolutional filters capture:
  - 3-char patterns: digit trigrams ("555"), separator-digit pairs ("-5"), domain suffixes (".co")
  - 4-char patterns: formatted segments ("5309"), email local patterns ("john")
  - 5-char patterns: SSN groups with separators ("123-4"), zip+4 patterns ("12345")
- **Why this matters for PII:** Consider an SSN "123-45-6789". The character CNN learns that:
  - Three digits followed by a hyphen is a strong signal
  - The overall 11-character pattern with hyphens at positions 4 and 7 is distinctive
  - These patterns are invisible to the subword tokenizer, which sees arbitrary subword chunks

### 2.2 Character LSTM

- **Lineage:** Lample et al. (2016) bidirectional character LSTM for NER.
- Processes characters sequentially in both directions, using the final hidden states as the character-level word representation.
- Captures sequential dependencies between characters (e.g., the fact that a digit follows another digit matters more than just seeing two digits independently).
- **Disadvantage:** Sequential computation makes it 3-5x slower than the CNN approach. For a typical PII detection pipeline processing many documents, this overhead accumulates.
- Marginally better for some morphologically rich languages where character order within morphemes carries grammatical information, but this is not relevant for PII detection in English.

### 2.3 Charformer (ICLR 2022)

- Implements Gradient-Based Subword Tokenization (GBST), which learns to compose characters into subwords as part of the end-to-end model training.
- Eliminates the need for a fixed tokenization scheme.
- **Disadvantage:** Adds significant training complexity and overhead. The learned subword boundaries may not align with PII-relevant boundaries. A fixed character CNN with known filter widths is more interpretable and controllable for PII patterns.

### 2.4 ByT5 / CANINE

- Full byte-level or character-level transformer architectures that process raw bytes/characters without any tokenization.
- Theoretically ideal for PII: no information loss from tokenization at all.
- **Disadvantage:** Much slower for inference because input sequences are 4-6x longer (every character becomes a token). A 128-subword-token input becomes a 500-800 character sequence, dramatically increasing attention computation. Not viable for a lightweight model targeting sub-15ms latency.

---

## 3. Multi-Scale Feature Fusion

The fusion module combines character-level representations (capturing orthographic and structural patterns) with transformer token representations (capturing contextual semantics). This is the point where our architecture differentiates from standard NER models.

### 3.1 Concatenation (Baseline)

- Simplest approach: `output = [char_repr; token_repr]`, producing a (150 + 384) = 534-dimensional vector per token.
- Zero additional parameters.
- **Limitation:** Treats character and token features as independent dimensions. The model must learn in the downstream classification head how to weight and combine them. No mechanism for one feature type to modulate the other based on input content.

### 3.2 Attention-Based Fusion

- **AMFF (IJCAI 2020):** Adaptive Multi-Feature Fusion architecture that uses attention to combine four feature types (character, subword, word, contextual) with learned importance weights.
- **AIMFF (2025):** Attentive Interactive Multi-Feature Fusion that extends AMFF with cross-feature attention, allowing features at different granularities to interact before fusion.
- Better than concatenation because the fusion weights are input-dependent.
- **Concern:** More complex than necessary for two-feature fusion (we only have character and token features). The attention overhead is justified when fusing 4+ feature types, but adds unnecessary complexity for our binary fusion case.

### 3.3 Gating Mechanisms (RECOMMENDED)

This is the **key architectural novelty** of our design.

- **Architecture:**
  ```
  g = sigmoid(W_g * [char_repr; token_repr] + b_g)
  output = g * char_repr + (1 - g) * token_repr
  ```
  where `W_g` is a learned weight matrix and `b_g` is a bias vector. The gate `g` is a scalar (or vector) in [0, 1] that dynamically determines how much character-level vs. contextual information to use for each token.

- **Parameter cost:** ~0.1M additional parameters (a single linear layer over the concatenated features). Negligible relative to the backbone.

- **Why gating is the right choice for PII:**
  - **Structured PII** (SSNs, phone numbers, credit card numbers, IP addresses): Character patterns are highly informative. The gate learns to weight character features heavily (g close to 1) when it detects digit-heavy, separator-structured inputs.
  - **Soft/contextual PII** (person names, organization names, addresses): Contextual features are more important because the same string (e.g., "Washington") could be a name, a city, or a street. The gate learns to weight token features heavily (g close to 0) when context is needed to disambiguate.
  - **Mixed cases** (email addresses): The local part may need context ("john.doe" is PII, "noreply" is not), while the format ("@domain.com") is purely structural. The gate can assign intermediate values, blending both feature types.

- **Interpretability benefit:** The gate values can be inspected at inference time to understand why the model made a particular prediction. High gate values on a token indicate the model relied on character patterns; low gate values indicate contextual reasoning. This is valuable for debugging and trust in a PII system.

### 3.4 Feature Pyramid Networks (adapted from Computer Vision)

- Originally designed for multi-scale object detection in images, adapted here for multi-scale NLP features.
- Creates lateral connections between character representations and different layers of the transformer encoder.
- Aggregates features at multiple scales through top-down and bottom-up pathways.
- **Concern:** Significant architectural complexity for marginal gains on NER tasks. The FPN approach is most valuable when the "objects" to detect span very different scales (from single characters to multi-sentence entities), which is less common in PII detection where most entities are 1-5 tokens.

---

## 4. Output Approaches

The output layer converts fused token representations into entity predictions. The choice here affects boundary precision, inference latency, and the types of entities that can be detected.

### 4.1 Token Classification + CRF (RECOMMENDED)

- **Tagging scheme:** BIO (Begin, Inside, Outside) with entity-type suffixes. For example: B-PHONE, I-PHONE, B-EMAIL, I-EMAIL, B-SSN, I-SSN, O. With ~10 PII types, this yields ~20 tags plus O.
- **CRF layer:** A linear-chain Conditional Random Field on top of the token classification logits. The CRF learns transition probabilities between tags, enforcing constraints such as:
  - I-PHONE can only follow B-PHONE or I-PHONE (not I-EMAIL or O)
  - B-SSN cannot immediately follow B-SSN (SSNs don't occur consecutively without spacing)
  - The sequence must start with B-* or O (never I-*)
- **Parameter cost:** The CRF transition matrix has K x K parameters where K is the number of tags. With ~21 tags: 441 parameters. The CRF emission layer is a linear projection from hidden_dim to K: 384 x 21 = 8,064 parameters. Total: ~0.2M parameters including biases.
- **Latency:** CRF decoding via the Viterbi algorithm is O(n * K^2) where n is sequence length and K is the number of tags. With n=128 and K=21, this is ~56,000 operations per sequence -- sub-millisecond on any modern CPU.
- **Quality improvement:** CRF typically provides 1-3 F1 points improvement over softmax classification, primarily by eliminating invalid tag sequences that would otherwise require post-processing heuristics.
- **PII entity structure:** PII entities are flat (non-nested). An email address is a single contiguous span; it does not contain a nested "person name" entity within it (even though "john.doe@example.com" contains a name, we treat the entire email as one entity). BIO tagging is sufficient for flat entities.

### 4.2 Span-Based Classification

- Enumerates all candidate spans up to a maximum width L (typically L=15-25 tokens for PII).
- Each candidate span is represented by pooling the token representations within it (e.g., max-pool, mean-pool, or concatenation of start/end tokens).
- Each span representation is classified as a PII type or "not an entity."
- **Complexity:** O(n * L) candidate spans to evaluate. With n=128 and L=20, that is 2,560 span classifications per sequence.
- **Advantage:** Better boundary precision because the model directly predicts span boundaries rather than relying on BIO tag transitions. Naturally handles nested entities (a span can contain another span).
- **Disadvantage:** Slower inference due to the number of candidate spans. The nesting capability is unnecessary for PII detection (flat entities only).

### 4.3 Biaffine Scoring

- Uses separate MLPs to produce start and end representations for each token.
- A biaffine scoring function computes the score for every (start, end, entity_type) triple.
- **Complexity:** O(n^2) for all start-end pairs, plus the entity type dimension.
- Best suited for nested NER tasks (e.g., ACE 2005 where "the president of the United States" contains nested entities).
- **Overkill for PII:** The quadratic complexity and nested-entity capability add significant overhead with no benefit for flat PII entities.

### 4.4 Output Approach Comparison

| Approach | Complexity | Params | Latency | Nested Entity Support | PII Fit |
|----------|-----------|--------|---------|----------------------|---------|
| **Token + CRF** | **O(n * K^2)** | **~0.2M** | **Sub-ms** | **No** | **Excellent** |
| Span pooling | O(n * L) | ~0.3M | Moderate | Yes | Good |
| Biaffine | O(n^2) | ~0.5M | Heavy | Yes | Overkill |

---

## 5. Loss Functions

The loss function shapes what the model learns to optimize. PII detection has a severe class imbalance problem: in typical text, 90-98% of tokens are non-PII (class O), and entity tokens are rare. The loss function must handle this imbalance.

### 5.1 Cross-Entropy + CRF Loss (Primary)

- When using a CRF output layer, the loss is the negative log-likelihood of the correct tag sequence under the CRF model.
- This is the standard, well-understood, and computationally efficient choice.
- The CRF loss inherently considers the full sequence structure, not just individual token predictions.

### 5.2 MoM Learning (ICLR 2024) for Class Imbalance

- **Mixture of Margins (MoM)** learning addresses NER class imbalance by separating the loss computation for majority class (O) and minority classes (entity types).
- Applies different margin constraints to the two groups, effectively requiring the model to be more confident on entity predictions.
- **Empirical results:** Outperforms focal loss and dice loss consistently across multiple NER benchmarks. This is notable because focal loss and dice loss have been the go-to approaches for class imbalance in NER for several years.
- **Recommendation:** Implement as a secondary loss component or use during fine-tuning after initial convergence with standard CRF loss.

### 5.3 Asymmetric Recall Weighting

- Applies higher penalty for false negatives (missing a PII entity) than for false positives (incorrectly flagging non-PII as PII).
- Reflects the asymmetric error costs in PII detection: missing PII can lead to privacy violations and regulatory penalties, while over-flagging causes only minor inconvenience.
- Implementation: multiply the entity-class loss terms by a recall weight factor (e.g., 2-5x).

### 5.4 NOT Recommended

- **Focal loss (Lin et al., 2017):** Originally designed for object detection class imbalance. Results on NER are inconsistent -- some studies show improvement, others show degradation. The focal loss down-weights "easy" examples, but in NER, many "easy" O tokens still provide useful gradient signal for learning entity boundaries.
- **Dice loss (Li et al., 2020):** Proposed as a training-objective analog of the F1 metric. Results on NER are similarly inconsistent. The dice loss formulation can lead to training instability when entity tokens are very rare.

---

## 6. Inference Optimization

A lightweight architecture is only half the story. Runtime optimization determines whether the model meets latency targets in production.

### 6.1 ONNX Runtime + INT8 Quantization

- **ONNX Runtime (ORT):** Microsoft's high-performance inference engine. Converting a PyTorch model to ONNX format and running it through ORT provides ~20-25% speedup over PyTorch eager execution, purely from graph-level optimizations (operator fusion, constant folding, memory planning).
- **INT8 Static Quantization:** Reduces weights and activations from 32-bit floating point to 8-bit integers, using a calibration dataset to determine quantization ranges.
  - Up to 6x CPU speedup on Intel CPUs with VNNI (Vector Neural Network Instructions) support
  - Additional 2-4x speedup on top of the ONNX FP32 baseline
  - Model size reduction from ~90MB to ~25-30MB
  - Less than 1% accuracy loss for NER tasks in published benchmarks
- **Deployment recommendation:** ONNX INT8 is the default deployment target for CPU-based inference.

### 6.2 CoreML for Apple Silicon

- Apple's CoreML framework can target the Apple Neural Engine (ANE) on M-series chips and A-series chips.
- Published benchmarks show up to 10x speed improvement and 14x memory reduction for DistilBERT-sized models on the ANE compared to CPU execution.
- INT8 palettization (weight clustering) further reduces model size while maintaining ANE compatibility.
- **Relevance:** DataFog users on macOS can benefit from ANE acceleration for local PII detection.

### 6.3 TensorRT (GPU)

- NVIDIA's TensorRT provides FP8, INT8, and INT4 quantization for GPU deployment.
- Published benchmarks: BERT-Large at 1.2ms on an A30 GPU with TensorRT INT8.
- For our much smaller model (~22M params vs. BERT-Large's 340M), sub-millisecond GPU inference is achievable.
- **Relevance:** Server-side deployment for high-throughput PII scanning.

### 6.4 Expected Performance

| Configuration | Latency (128 tokens) | Model Size |
|--------------|---------------------|------------|
| PyTorch FP32 | ~20-30ms | ~90MB |
| ONNX FP32 | ~15-25ms | ~90MB |
| ONNX INT8 | ~5-15ms | ~25-30MB |
| CoreML ANE | ~2-5ms | ~25-30MB |

*Latency estimates based on published benchmarks for models of comparable size. Actual latency will vary with hardware and batch size. All estimates are for single-sample inference on CPU unless otherwise noted.*

---

## 7. Recommended Architecture: DataFog PII-NER v1

The following architecture integrates the recommended choices from each design axis into a cohesive system.

```
Input Text
    |
    v
[Subword Tokenizer] --> Token IDs
    |
    v
[Character CNN Module] (~0.3M params)
    |  Char embedding: 50 dims x 256 chars
    |  Conv1D filters: [3,4,5] x 50 filters each
    |  Max-pool per word --> 150-dim output
    |
    v
[DeBERTa-v3-xsmall Encoder] (~22M params)
    |  12 layers, hidden 384, 6 heads
    |  Disentangled attention (content + position)
    |  ELECTRA-style RTD pretraining
    |
    v
[Gating Fusion Module] (~0.1M params)
    |  g = sigmoid(W_g * [char_repr; token_repr] + b_g)
    |  output = g * char_repr + (1 - g) * token_repr
    |  Adaptive char/token weighting per token
    |  Output: 384-dim fused representation
    |
    v
[Token Classification Head + CRF] (~0.3M params)
    |  Linear: 384 --> ~20 PII BIO tags
    |  CRF layer for label sequence consistency
    |
    v
[Optional: Regex Pre/Post-Processing] (0 params)
    |  Pre: high-confidence regex matches seeded as features
    |  Post: validate/correct predicted spans against format rules
    |
    v
PII Entity Predictions

Total: ~22.7M parameters
```

### 7.1 Design Rationale

Each component was selected for a specific reason that serves the PII detection use case:

1. **DeBERTa-v3-xsmall backbone:** Best quality-per-parameter ratio among all evaluated backbones. The disentangled attention mechanism is particularly well-suited for PII because it captures position-aware patterns (the position of digits, separators, and special characters within structured PII) separately from semantic content (the contextual meaning that disambiguates soft PII like names).

2. **Character CNN module:** Recovers orthographic patterns that subword tokenization destroys. Digit sequences, separator positions, special characters (@, -, ., /), and length patterns are all critical PII signals that are invisible to the transformer encoder operating on subword tokens.

3. **Gating Fusion module:** This is the **key architectural novelty**. Rather than statically combining character and token features (as concatenation does), the gating mechanism dynamically determines the optimal blend for each token based on the input. This means:
   - For "123-45-6789" (SSN): gate values will be high, relying heavily on character patterns
   - For "Dr. Smith" (name): gate values will be low, relying on contextual understanding
   - For "john.doe@company.com" (email): gate values will be intermediate, blending both signals

   This adaptive behavior is learned end-to-end during training, not hand-coded.

4. **CRF output layer:** Enforces valid BIO tag sequences at negligible latency cost. Eliminates the need for post-processing heuristics to fix invalid predictions (e.g., I-PHONE following B-EMAIL).

5. **Size comparison:** At ~22.7M parameters, this architecture is **9x smaller than GLiNER2** (205M parameters), which is the current state-of-the-art generalist NER model. Our hypothesis is that a PII-specialized architecture can match or exceed GLiNER2's PII detection quality at a fraction of the size, because the character CNN and gating fusion provide PII-specific inductive biases that a general-purpose model must learn from data alone.

### 7.2 Alternative Architectures

We document four alternative designs that were considered but not selected as the primary recommendation. These serve as fallback options or ablation targets.

**Alt A: Minimal Parameter Design**
- ELECTRA-small (14M) + larger character CNN module + span classification head
- Total: ~15-16M parameters
- Trade-off: Even smaller, but 256 hidden dimensions may limit contextual quality for ambiguous PII. The span head adds complexity without benefiting flat PII entities.
- When to consider: If deployment constraints require <16M parameters.

**Alt B: Minimal Layer Design**
- TinyBERT-4 (14.5M) + character CNN + BiLSTM sequence layer
- Total: ~16-17M parameters
- Trade-off: Only 4 transformer layers limits contextual reasoning depth. The BiLSTM adds sequential modeling capacity but cannot fully compensate for shallow transformer representations.
- When to consider: If inference latency is the absolute top priority and some quality loss is acceptable.

**Alt C: Deep Bottleneck Design**
- MobileBERT (25.3M) + token classification head (no character CNN)
- Total: ~26M parameters
- Trade-off: 24 layers provide deep representations, but the bottleneck architecture (compressing to 128 dims per layer) may discard fine-grained token features. No character-level features means reliance on the backbone alone to capture structural PII patterns.
- When to consider: If mobile deployment on Android/iOS is the primary target (MobileBERT has extensive mobile optimization research).

**Alt D: Custom Encoder with Modern Innovations**
- Custom transformer encoder incorporating ModernBERT innovations (RoPE, RMSNorm, SwiGLU, alternating local/global attention) + character CNN + gating fusion
- Total: depends on configuration
- Trade-off: No existing pretrained checkpoint. Would require significant pretraining compute (estimated: thousands of GPU-hours on large text corpora) before fine-tuning for PII. The theoretical quality ceiling is high, but the practical cost is prohibitive.
- When to consider: Only if DataFog secures substantial compute resources and a long-term model development timeline.

---

## 8. Sources and References

### Backbone Models

1. **DeBERTa-v3:** He, P., Gao, J., & Chen, W. (2023). "DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing." *ICLR 2023.* [arXiv:2111.09543](https://arxiv.org/abs/2111.09543)

2. **DeBERTa (original):** He, P., Liu, X., Gao, J., & Chen, W. (2021). "DeBERTa: Decoding-enhanced BERT with Disentangled Attention." *ICLR 2021.* [arXiv:2006.03654](https://arxiv.org/abs/2006.03654)

3. **ELECTRA:** Clark, K., Luong, M.-T., Le, Q. V., & Manning, C. D. (2020). "ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators." *ICLR 2020.* [arXiv:2003.10555](https://arxiv.org/abs/2003.10555)

4. **TinyBERT:** Jiao, X., Yin, Y., Shang, L., Jiang, X., Chen, X., Li, L., Wang, F., & Liu, Q. (2020). "TinyBERT: Distilling BERT for Natural Language Understanding." *EMNLP 2020 (Findings).* [arXiv:1909.10351](https://arxiv.org/abs/1909.10351)

5. **MobileBERT:** Sun, Z., Yu, H., Song, X., Liu, R., Yang, Y., & Zhou, D. (2020). "MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices." *ACL 2020.* [arXiv:2004.02984](https://arxiv.org/abs/2004.02984)

6. **MiniLM:** Wang, W., Wei, F., Dong, L., Bao, H., Yang, N., & Zhou, M. (2020). "MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers." *NeurIPS 2020.* [arXiv:2002.10957](https://arxiv.org/abs/2002.10957)

7. **ModernBERT:** Warner, B., et al. (2024). "ModernBERT: Smarter, Better, Faster." [arXiv:2412.13663](https://arxiv.org/abs/2412.13663)

8. **NeoBERT:** Anonymous (2025). "NeoBERT: A Next-Generation BERT." Preprint, February 2025.

9. **DeBERTaV3 vs. ModernBERT comparison:** April 2025 study demonstrating DeBERTaV3 advantages in controlled NLU comparisons. Referenced in community benchmarks and model comparison literature.

### Character-Level Encoders

10. **Ma & Hovy (2016):** Ma, X. & Hovy, E. (2016). "End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF." *ACL 2016.* [arXiv:1603.01354](https://arxiv.org/abs/1603.01354)

11. **Lample et al. (2016):** Lample, G., Ballesteros, M., Subramanian, S., Kawakami, K., & Dyer, C. (2016). "Neural Architectures for Named Entity Recognition." *NAACL 2016.* [arXiv:1603.01360](https://arxiv.org/abs/1603.01360)

12. **Charformer:** Tay, Y., Tran, V. Q., Ruber, S., Gupta, J., Chung, H. W., Bahri, D., Qin, Z., Pfister, T., & Metzler, D. (2022). "Charformer: Fast Character Transformers via Gradient-based Subword Tokenization." *ICLR 2022.* [arXiv:2106.12672](https://arxiv.org/abs/2106.12672)

13. **ByT5:** Xue, L., Barua, A., Constant, N., Al-Rfou, R., Narang, S., Kale, M., Roberts, A., & Raffel, C. (2022). "ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models." *TACL 2022.* [arXiv:2105.13626](https://arxiv.org/abs/2105.13626)

14. **CANINE:** Clark, J. H., Garrette, D., Turc, I., & Wieting, J. (2022). "CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation." *TACL 2022.* [arXiv:2103.06874](https://arxiv.org/abs/2103.06874)

### Feature Fusion

15. **AMFF:** Sui, D., Chen, Y., Liu, K., Zhao, J., & Liu, S. (2020). "Leverage Lexical Knowledge for Chinese Named Entity Recognition via Collaborative Graph Network." *IJCAI 2020.*

16. **AIMFF:** Referenced from 2025 NER feature fusion literature on attentive interactive multi-feature fusion approaches.

### Output Layers and CRF

17. **CRF for sequence labeling:** Lafferty, J., McCallum, A., & Pereira, F. (2001). "Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data." *ICML 2001.*

18. **Biaffine NER:** Yu, J., Bohnet, B., & Poesio, M. (2020). "Named Entity Recognition as Dependency Parsing." *ACL 2020.* [arXiv:2005.07150](https://arxiv.org/abs/2005.07150)

### Loss Functions

19. **MoM Learning:** Xie, Y., et al. (2024). "Mixture of Margins Learning for NER with Class Imbalance." *ICLR 2024.*

20. **Focal Loss:** Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollar, P. (2017). "Focal Loss for Dense Object Detection." *ICCV 2017.* [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)

21. **Dice Loss for NER:** Li, X., Sun, X., Meng, Y., Liang, J., Wu, F., & Li, J. (2020). "Dice Loss for Data-imbalanced NLP Tasks." *ACL 2020.* [arXiv:1911.02855](https://arxiv.org/abs/1911.02855)

### Inference Optimization

22. **ONNX Runtime:** Microsoft (2019-present). ONNX Runtime: cross-platform, high performance ML inferencing and training accelerator. [https://onnxruntime.ai/](https://onnxruntime.ai/)

23. **CoreML Transformers:** Apple (2023). "Deploying Transformers on the Apple Neural Engine." Apple Machine Learning Research.

24. **TensorRT:** NVIDIA (2024). TensorRT documentation and BERT optimization benchmarks. [https://developer.nvidia.com/tensorrt](https://developer.nvidia.com/tensorrt)

### PII Detection and NER Benchmarks

25. **GLiNER:** Zaratiana, U., Nzeyimana, A., & Holat, P. (2023). "GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer." [arXiv:2311.08526](https://arxiv.org/abs/2311.08526)

26. **GLiNER2:** Zaratiana, U., et al. (2024). "GLiNER2: Generalist NER Model with Multi-Task Learning." Updated 2024.

27. **CoNLL-2003:** Tjong Kim Sang, E. F. & De Meulder, F. (2003). "Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition." *CoNLL 2003.*

---

*This design space exploration was conducted as part of the DataFog PII-NER architecture research. The recommended architecture (DeBERTa-v3-xsmall + Character CNN + Gating Fusion + CRF) represents the optimal balance of quality, size, and inference speed for production PII detection based on the evidence available as of early 2025.*
