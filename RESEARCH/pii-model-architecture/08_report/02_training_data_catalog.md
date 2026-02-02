# Training Data Catalog: PII Detection Model

This catalog inventories all datasets evaluated for training a production PII detection model. Datasets are organized by category, with licensing, quality assessments, and prioritized recommendations for a commercially viable training pipeline.

---

## Part 1: PII-Specific Datasets

### 1.1 NVIDIA Nemotron-PII

| Field | Details |
|---|---|
| **URL** | <https://huggingface.co/datasets/nvidia/Nemotron-PII> |
| **Size** | ~100K records across 50+ industries |
| **Entity Types** | 55+ categories (US + international) |
| **License** | CC BY 4.0 |
| **Quality** | High -- persona-grounded synthetic data from NeMo Data Designer, Census-grounded demographics |
| **Priority** | **MUST-HAVE** |

NVIDIA's Nemotron-PII dataset represents the current state of the art in synthetic PII data generation. Each record is grounded in a realistic persona drawn from Census-calibrated demographic distributions, ensuring diversity across age, gender, ethnicity, and geography. The NeMo Data Designer pipeline produces documents spanning 50+ industries (healthcare, finance, legal, education, technology, etc.) with naturally embedded PII that mirrors real-world density and co-occurrence patterns. The 55+ entity types cover both US-specific identifiers (SSN, state IDs, EIN) and international formats (IBAN, NHS numbers, Aadhaar).

### 1.2 Gretel PII Masking EN v1

| Field | Details |
|---|---|
| **URL** | <https://huggingface.co/datasets/gretelai/gretel-pii-masking-en-v1> |
| **Size** | 60K records (50K train / 5K val / 5K test) |
| **Entity Types** | 54 types across 47 document formats |
| **License** | Apache 2.0 |
| **Quality** | High -- LLM-as-a-Judge quality filtering, generated with Mistral-NEMO-2407 |
| **Priority** | **MUST-HAVE** |

Gretel's dataset was generated using a sophisticated multi-stage pipeline: a backend LLM (Mistral-NEMO-2407) generates realistic document contexts, the Faker library injects format-correct PII values, a NER labeler aligns token-level annotations, and an LLM-as-a-Judge step filters out low-quality samples. The 47 document formats include emails, contracts, medical notes, financial statements, chat logs, resumes, and more. The pre-split train/val/test partitions make it immediately usable for model development.

### 1.3 AI4Privacy pii-masking-200k

| Field | Details |
|---|---|
| **URL** | <https://huggingface.co/datasets/ai4privacy/pii-masking-200k> (Isotonic mirror) |
| **Size** | 200K examples, multilingual |
| **Entity Types** | 54 types |
| **License** | Apache 2.0 |
| **Quality** | Good -- used to train Piiranha (98.27% recall) |
| **Priority** | **MUST-HAVE** |

The AI4Privacy 200K dataset is the largest openly licensed PII dataset available. It includes multilingual coverage and was used to train Piiranha, which achieved 98.27% recall on PII detection benchmarks. The 54 entity types provide broad coverage of both personal and organizational identifiers. The Isotonic mirror on Hugging Face provides convenient access. While quality is slightly more variable than Nemotron-PII or Gretel (due to less aggressive filtering), the sheer volume and proven downstream performance make it indispensable.

### 1.4 AI4Privacy Extended (300k / 400k / 500k)

| Field | Details |
|---|---|
| **Size** | 300K / 400K / 500K examples (progressively larger) |
| **License** | Custom / Llama Community License |
| **Contact** | licensing@ai4privacy.com for commercial use |
| **Priority** | NICE-TO-HAVE (pending licensing) |

AI4Privacy offers progressively larger dataset variants beyond the 200K version. However, these larger datasets carry custom or Llama Community licenses that may impose restrictions on commercial use. Organizations interested in commercial deployment should contact licensing@ai4privacy.com to negotiate terms. If commercially licensable, the 500K variant would substantially increase training data volume.

### 1.5 Gretel Synthetic PII Finance Multilingual

| Field | Details |
|---|---|
| **URL** | <https://huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual> |
| **Size** | 55.9K records across 100 financial document formats |
| **Entity Types** | 29 types |
| **License** | Apache 2.0 |
| **Quality** | High -- domain-specific focus on financial services |
| **Priority** | RECOMMENDED for financial domain coverage |

This dataset fills a critical gap: domain-specific PII in financial contexts. The 100 financial document formats include loan applications, wire transfer confirmations, KYC forms, audit reports, investment prospectuses, and tax documents. The 29 entity types are tailored to finance (account numbers, routing numbers, SWIFT/BIC codes, tax IDs). Multilingual coverage adds international financial identifier formats. Essential for any deployment targeting financial services compliance (GLBA, PCI-DSS).

### 1.6 Kaggle PII / PIILO Dataset

| Field | Details |
|---|---|
| **Source** | 2024 Kaggle PII Detection competition |
| **Size** | 22K student essays, 7 PII types |
| **Entity Types** | NAME_STUDENT, EMAIL, USERNAME, ID_NUM, PHONE_NUM, URL_PERSONAL, STREET_ADDRESS |
| **License** | Kaggle competition license |
| **Metric** | F5 score; 1st place achieved 0.953 |
| **Priority** | SUPPLEMENTARY |

The Kaggle PII Detection competition dataset consists of student essays with naturally occurring PII. The competition's use of F5 score (heavily weighting recall over precision) aligns with production PII detection priorities where missing PII is far worse than false positives. The 1st-place solution achieving 0.953 F5 provides a strong benchmark. However, the narrow domain (student essays) and limited entity types (7) make this supplementary rather than primary. The Kaggle competition license may impose restrictions on non-competition use.

### 1.7 i2b2/n2c2 2014 Clinical De-identification

| Field | Details |
|---|---|
| **Size** | 1,304 records |
| **Entity Types** | PHI-specific: names, dates, locations, ages, contact info, IDs |
| **License** | Data Use Agreement (DUA) required, research only |
| **Priority** | RECOMMENDED for healthcare domain |

The i2b2 (now n2c2) 2014 de-identification shared task dataset is the gold standard for clinical text de-identification. It contains 1,304 longitudinal medical records with expert annotations of Protected Health Information (PHI) as defined by HIPAA. While small and restricted to research use, it is invaluable for evaluating model performance on clinical text, which has unique linguistic characteristics (abbreviations, fragmented sentences, specialized terminology). The DUA requirement means this dataset cannot be used for commercial model training directly, but it serves as an essential evaluation benchmark for healthcare deployments.

### 1.8 TAB (Text Anonymization Benchmark)

| Field | Details |
|---|---|
| **Source** | European Court of Human Rights (ECHR) court cases |
| **Size** | 1,268 cases |
| **Entity Types** | Standard PII + quasi-identifiers |
| **License** | MIT |
| **Priority** | RECOMMENDED for legal domain and quasi-identifier coverage |

TAB is unique in two respects. First, it is sourced from real legal documents (ECHR court cases), providing authentic legal language patterns. Second, it annotates quasi-identifiers -- attributes that are not PII in isolation but can re-identify individuals when combined (e.g., rare professions, specific medical conditions, unique family configurations). This quasi-identifier annotation is critical for building models that understand contextual privacy risk beyond simple entity matching. The MIT license makes it fully commercially usable.

### 1.9 BigCode PII Dataset

| Field | Details |
|---|---|
| **Entity Types** | Code-specific PII: names, emails, API keys, passwords, IP addresses, usernames |
| **License** | Gated access, PII detection purposes only |
| **Priority** | SUPPLEMENTARY for code-processing applications |

The BigCode PII dataset focuses on personally identifiable information found in source code repositories. It covers code-specific PII patterns like hardcoded API keys, database connection strings with credentials, email addresses in comments, and IP addresses in configuration files. Access is gated and restricted to PII detection purposes. Relevant primarily for organizations building code-aware PII detection (e.g., pre-commit hooks, CI/CD pipeline scanners).

---

## Part 2: General NER Datasets

General Named Entity Recognition datasets provide complementary training signal for entity boundary detection and type classification, even though they were not designed specifically for PII.

### 2.1 CoNLL-2003

| Field | Details |
|---|---|
| **Size** | 22K sentences |
| **Entity Types** | 4 types: PER, LOC, ORG, MISC |
| **License** | DUA required |
| **Priority** | SUPPLEMENTARY |

The de facto standard NER benchmark. While limited to 4 coarse entity types, CoNLL-2003 provides high-quality human annotations on newswire text. The PER (person) and LOC (location) types directly overlap with PII categories. Useful for pre-training or multi-task learning to improve general entity boundary detection. The DUA license restricts redistribution but permits research use.

### 2.2 OntoNotes 5.0

| Field | Details |
|---|---|
| **Size** | 76K sentences |
| **Entity Types** | 18 types (PERSON, ORG, GPE, DATE, MONEY, CARDINAL, etc.) |
| **License** | LDC license (paid) |
| **Priority** | RECOMMENDED for pre-training |

OntoNotes provides substantially richer entity type coverage than CoNLL-2003, with 18 types including several PII-relevant categories (PERSON, DATE, MONEY, CARDINAL/ORDINAL numbers). The multi-genre corpus (newswire, broadcast, web, telephone conversations) improves domain robustness. The LDC license requires institutional membership or individual purchase but permits research and many commercial uses.

### 2.3 Few-NERD

| Field | Details |
|---|---|
| **Size** | 188K sentences |
| **Entity Types** | 66 fine-grained types (8 coarse + 66 fine) |
| **License** | CC BY-SA 4.0 -- **COMMERCIALLY USABLE** |
| **Priority** | RECOMMENDED |

Few-NERD is particularly valuable for two reasons. First, its 66 fine-grained entity types provide granularity that transfers well to PII sub-typing (e.g., distinguishing person-artist from person-politician). Second, its CC BY-SA 4.0 license makes it one of the few large NER datasets that is unambiguously commercially usable. The dataset was designed for few-shot NER evaluation but serves equally well as pre-training data. Used by GLiNER2 as a benchmark, establishing a direct connection to the model architecture under consideration.

### 2.4 WikiANN (PAN-X)

| Field | Details |
|---|---|
| **Size** | Varies by language (typically 10K-100K per language) |
| **Languages** | 176 languages |
| **Entity Types** | 3 types: PER, LOC, ORG |
| **Quality** | Silver-standard (automatically generated from Wikipedia) |
| **Priority** | SUPPLEMENTARY for multilingual coverage |

WikiANN provides unmatched language coverage (176 languages) but at the cost of annotation quality -- labels are silver-standard, automatically projected from Wikipedia hyperlinks. Useful primarily for multilingual model training where no gold-standard PII data exists in the target language. Should be used with noise-aware training techniques.

### 2.5 CrossNER

| Field | Details |
|---|---|
| **Size** | ~5K sentences across 5 domains |
| **Domains** | Politics, Natural Science, Music, Literature, AI |
| **License** | Research license |
| **Priority** | SUPPLEMENTARY |

CrossNER evaluates cross-domain NER transfer, with domain-specific entity types (e.g., "algorithm" in AI, "protein" in science). Used by GLiNER2 as a benchmark for zero-shot cross-domain generalization. The domain-specific types are not PII-relevant, but the benchmark is useful for evaluating model robustness across text domains.

### 2.6 MultiNERD

| Field | Details |
|---|---|
| **Size** | 1.67M examples |
| **Entity Types** | 15 types |
| **Languages** | 10 languages |
| **License** | CC BY-NC-SA 4.0 |
| **Priority** | SUPPLEMENTARY (non-commercial license limits use) |

MultiNERD is the largest multilingual NER dataset by example count. The 15 entity types include several PII-adjacent categories (PERSON, LOCATION, ORGANIZATION). However, the CC BY-NC-SA 4.0 license prohibits commercial use, limiting its applicability to research and evaluation only. For commercial pipelines, Few-NERD or WikiANN are preferable multilingual alternatives.

---

## Part 3: Synthetic Data Generation Approaches

Given the inherent scarcity of real PII data (privacy regulations prevent large-scale collection), synthetic data generation is essential for building production PII detection models.

### 3.1 GLiNER2's Synthetic Data Pipeline

GLiNER2 used GPT-4o to generate 118K synthetic NER examples across diverse document types:

- **Email threads** with sender/recipient PII, signatures, CC lists
- **Text messages** with informal PII mentions (nicknames, partial phone numbers)
- **Professional documents** (contracts, invoices, HR forms) with structured PII
- **Social media posts** with usernames, handles, tagged locations
- **News articles** with public figure names, organization details

The pipeline demonstrates that LLM-generated synthetic data can effectively train entity recognition models, achieving state-of-the-art zero-shot NER performance. Key insight: diversity of document formats matters more than raw volume.

### 3.2 Gretel's Multi-Stage Pipeline

Gretel's approach separates concerns across four stages:

1. **Context Generation**: Backend LLM (Mistral-NEMO-2407) generates realistic document text with placeholder markers for PII
2. **PII Injection**: Faker library replaces placeholders with format-correct, locale-appropriate PII values
3. **NER Labeling**: Automated alignment of injected PII values with token-level BIO annotations
4. **Quality Filtering**: LLM-as-a-Judge evaluates each sample for naturalness, coherence, and annotation accuracy; low-quality samples are discarded

This pipeline achieves high quality by decoupling PII generation from context generation, preventing the LLM from generating memorized real PII while maintaining natural document flow.

### 3.3 NVIDIA NeMo Data Designer

NVIDIA's approach emphasizes demographic grounding:

- **Persona Generation**: Each synthetic document starts with a detailed persona (name, age, gender, ethnicity, occupation, location) sampled from Census-calibrated distributions
- **Industry Coverage**: 50+ industry templates ensure domain diversity
- **PII Density Control**: Configurable PII density per document type (e.g., medical intake forms are PII-dense; news articles are sparse)
- **International Formats**: PII values generated in locale-specific formats (US SSN vs. UK NIN vs. Indian Aadhaar)

### 3.4 Faker Library Integration

The Faker library is the backbone of PII value generation across multiple pipelines:

- **Locale Support**: 50+ locales with locale-specific PII formats
- **Provider Coverage**: Names, addresses, phone numbers, SSNs, credit cards, IBANs, company names, job titles, and more
- **Best Practices**:
  - **Placeholder approach**: Generate document text with placeholders (e.g., `[PERSON_NAME]`), then replace with Faker values -- avoids LLM memorization of real PII
  - **Gender consistency**: Use `Faker.profile()` to generate internally consistent personas (matching name gender to pronoun usage)
  - **Deterministic mapping**: Use seeded Faker instances for reproducibility; map specific placeholder IDs to consistent PII values within a document (same person referenced multiple times gets the same name)
  - **Format variation**: Randomly vary PII formats (e.g., phone as `(555) 123-4567` vs. `555.123.4567` vs. `+1-555-123-4567`) to improve model robustness

### 3.5 Data Augmentation Techniques

| Technique | Description | When to Use | Risk |
|---|---|---|---|
| **Mention Replacement** | Replace entity mentions with alternatives from the same type (e.g., swap one name for another) | Increasing entity diversity without new contexts | May break coreference or gender agreement |
| **Contextual Word Replacement** | Use masked LM to replace non-entity tokens with contextually appropriate alternatives | Increasing context diversity | May inadvertently alter entity boundaries |
| **Back-Translation** | Translate to another language and back to generate paraphrases | Increasing syntactic diversity | May lose or distort PII formatting |
| **LLM-Based Generation** | Prompt an LLM to generate new examples given entity types and constraints | Generating entirely new training examples | Potential for hallucinated or memorized real PII |
| **Template-Based** | Define document templates with PII slots, fill programmatically | High-precision PII placement | Limited naturalness and diversity |
| **Token-level Entity Replacement (TER)** | Replace entity tokens while preserving surrounding context and BIO tags | Augmenting existing annotated data | Requires careful re-alignment of token offsets |

---

## Part 4: Licensing Summary

### Commercially Usable (Open Licenses)

| Dataset | License | Size | PII Types | Notes |
|---|---|---|---|---|
| NVIDIA Nemotron-PII | CC BY 4.0 | ~100K | 55+ | Attribution required |
| Gretel PII Masking EN v1 | Apache 2.0 | 60K | 54 | Pre-split train/val/test |
| AI4Privacy pii-masking-200k | Apache 2.0 | 200K | 54 | Multilingual |
| Gretel Synthetic PII Finance | Apache 2.0 | 55.9K | 29 | Finance domain, multilingual |
| Few-NERD | CC BY-SA 4.0 | 188K sentences | 66 (fine-grained NER) | ShareAlike required |
| TAB | MIT | 1,268 cases | PII + quasi-identifiers | Legal domain |

**Combined commercially usable PII data: ~415K+ records with 55+ entity types.**

### Research-Only / Restricted

| Dataset | License | Restriction | Size | Workaround |
|---|---|---|---|---|
| AI4Privacy 300k/400k/500k | Custom / Llama Community | Commercial use requires negotiation | 300-500K | Contact licensing@ai4privacy.com |
| CoNLL-2003 | DUA | Redistribution prohibited | 22K sentences | Obtain DUA from LDC |
| OntoNotes 5.0 | LDC | Paid license required | 76K sentences | Institutional LDC membership |
| i2b2/n2c2 2014 | DUA | Research only, no redistribution | 1,304 records | Use as evaluation benchmark only |
| Kaggle PII | Competition license | Competition terms apply | 22K essays | Review Kaggle ToS for derivative works |
| BigCode PII | Gated | PII detection purposes only | Varies | Apply for access on Hugging Face |
| MultiNERD | CC BY-NC-SA 4.0 | Non-commercial only | 1.67M | Use for evaluation / research only |
| CrossNER | Research license | Research only | ~5K sentences | Use as evaluation benchmark only |

---

## Part 5: Recommended Entity Type Taxonomy

A unified entity type taxonomy is required to harmonize annotations across heterogeneous datasets. The following four-tier taxonomy balances coverage with practical detectability.

### Tier 1: Direct Identifiers (Highest Priority)

These entities can directly identify an individual on their own.

| Entity Type | Examples | Source Datasets |
|---|---|---|
| **FULL_NAME** | "John Smith", "Dr. Maria Garcia" | Nemotron, Gretel, AI4Privacy, CoNLL, OntoNotes |
| **SSN** | "123-45-6789" | Nemotron, Gretel, AI4Privacy |
| **PASSPORT_NUMBER** | "US12345678", "C1234567" | Nemotron, Gretel, AI4Privacy |
| **DRIVERS_LICENSE** | "D123-4567-8901" | Nemotron, Gretel, AI4Privacy |
| **EMAIL_ADDRESS** | "john.smith@example.com" | Nemotron, Gretel, AI4Privacy, BigCode |
| **PHONE_NUMBER** | "+1 (555) 123-4567" | Nemotron, Gretel, AI4Privacy, Kaggle |
| **DATE_OF_BIRTH** | "03/15/1990", "March 15, 1990" | Nemotron, Gretel, AI4Privacy, i2b2 |

### Tier 2: Contact and Financial Identifiers

These entities enable contact or financial access.

| Entity Type | Examples | Source Datasets |
|---|---|---|
| **STREET_ADDRESS** | "123 Main St, Apt 4B, Springfield, IL 62701" | Nemotron, Gretel, AI4Privacy, Kaggle |
| **CREDIT_CARD_NUMBER** | "4111-1111-1111-1111" | Nemotron, Gretel, AI4Privacy |
| **BANK_ACCOUNT_NUMBER** | "1234567890", IBAN formats | Nemotron, Gretel, Gretel Finance |
| **IP_ADDRESS** | "192.168.1.1", "2001:db8::1" | Nemotron, Gretel, AI4Privacy, BigCode |
| **ROUTING_NUMBER** | "021000021" | Gretel Finance |
| **SWIFT_BIC** | "CHASUS33XXX" | Gretel Finance |

### Tier 3: Digital and Professional Identifiers

These entities identify individuals in digital or organizational contexts.

| Entity Type | Examples | Source Datasets |
|---|---|---|
| **USERNAME** | "@johndoe", "jsmith42" | AI4Privacy, BigCode, Kaggle |
| **PASSWORD** | "P@ssw0rd123!" | BigCode |
| **URL_PERSONAL** | "https://johndoe.com" | AI4Privacy, Kaggle |
| **DEVICE_ID** | "IMEI: 353456789012345" | Nemotron |
| **VEHICLE_ID (VIN)** | "1HGCM82633A004352" | Nemotron, AI4Privacy |
| **EMPLOYEE_ID** | "EMP-2024-0042" | Nemotron, Gretel |
| **API_KEY** | "sk-abc123def456..." | BigCode |

### Tier 4: Domain-Specific Identifiers

These entities are relevant in specific regulatory or industry contexts.

| Entity Type | Examples | Source Datasets |
|---|---|---|
| **MEDICAL_RECORD_NUMBER** | "MRN-12345678" | i2b2, Nemotron |
| **BIOMETRIC_ID** | Fingerprint hash, facial recognition template ID | Nemotron |
| **CRYPTO_ADDRESS** | "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa" | AI4Privacy |
| **GPS_COORDINATES** | "37.7749, -122.4194" | AI4Privacy, Nemotron |
| **TAX_ID** | "EIN: 12-3456789", VAT numbers | Gretel Finance, Nemotron |
| **NATIONAL_ID** | Aadhaar, NHS Number, SIN | Nemotron, AI4Privacy |

### Taxonomy Mapping Notes

When harmonizing across datasets, apply these mapping rules:

- Nemotron-PII's 55+ types map cleanly to all four tiers; use as the primary schema reference
- Gretel's 54 types and AI4Privacy's 54 types overlap ~80%; create explicit mapping tables during preprocessing
- CoNLL/OntoNotes PER maps to FULL_NAME; LOC/GPE maps to STREET_ADDRESS (partial); ORG is not PII but useful for context
- i2b2 PHI types map to Tier 1 + Tier 4 medical categories
- Kaggle's 7 types are a strict subset of Tier 1 + Tier 3

---

## Part 6: Prioritized Recommendations

### Priority 1: Core Training Data (MUST-HAVE)

**Target: 360K+ commercially usable PII-annotated records**

| Dataset | Records | Entity Types | License | Action |
|---|---|---|---|---|
| NVIDIA Nemotron-PII | ~100K | 55+ | CC BY 4.0 | Download from Hugging Face |
| Gretel PII Masking EN v1 | 60K | 54 | Apache 2.0 | Download from Hugging Face |
| AI4Privacy pii-masking-200k | 200K | 54 | Apache 2.0 | Download from Hugging Face |

**Combined**: 360K records, 55+ entity types, all commercially usable. This combination provides sufficient volume and diversity for initial model training. The Gretel dataset's pre-defined train/val/test split can serve as the primary evaluation partition, with Nemotron-PII and AI4Privacy data mixed into training.

**Estimated effort**: 1-2 weeks for download, schema harmonization, and preprocessing.

### Priority 2: Domain-Specific Augmentation (RECOMMENDED)

**Target: Specialized coverage for high-value verticals**

| Dataset | Domain | Records | License | Action |
|---|---|---|---|---|
| Gretel Synthetic PII Finance | Financial services | 55.9K | Apache 2.0 | Download from Hugging Face |
| TAB | Legal / court documents | 1,268 | MIT | Download, extract quasi-identifiers |
| i2b2/n2c2 2014 | Clinical / healthcare | 1,304 | DUA (research) | Apply for DUA; use for evaluation only |

These datasets fill critical domain gaps. Financial services PII detection is a high-demand use case (GLBA, PCI-DSS compliance). Legal document anonymization is growing (GDPR, court filing redaction). Clinical de-identification is mandated by HIPAA. Even where licensing prevents training use (i2b2), these datasets are invaluable for domain-specific evaluation.

**Estimated effort**: 1 week for download and integration; 2-4 weeks for i2b2 DUA process.

### Priority 3: General NER Pre-training (RECOMMENDED)

**Target: Improved entity boundary detection via multi-task pre-training**

| Dataset | Size | Entity Types | License | Action |
|---|---|---|---|---|
| Few-NERD | 188K sentences | 66 fine-grained | CC BY-SA 4.0 | Download; use for pre-training or multi-task |
| OntoNotes 5.0 | 76K sentences | 18 types | LDC | Obtain LDC license; use for pre-training |

General NER data improves entity boundary detection, which transfers to PII tasks. Few-NERD's fine-grained types and commercial license make it the top choice. OntoNotes adds multi-genre coverage but requires an LDC license. Pre-training on NER data before fine-tuning on PII data is a proven strategy (used by GLiNER, Microsoft Presidio's models, and others).

**Estimated effort**: 1 week for integration; ongoing for LDC licensing.

### Priority 4: Custom Synthetic Data Pipeline (LONG-TERM)

**Target: Unlimited domain-specific PII data generation**

Build a custom synthetic data pipeline combining best practices from Gretel, NVIDIA, and GLiNER2:

1. **Document Template Library**: Define templates for target document types (customer support tickets, medical forms, financial applications, etc.)
2. **Persona Engine**: Census-grounded persona generation (following NVIDIA's approach) for demographic diversity
3. **PII Injection**: Faker-based PII value generation with locale-specific formatting and cross-reference consistency
4. **Context Generation**: LLM-based document generation with PII placeholders (following Gretel's separation-of-concerns approach)
5. **Quality Assurance**: LLM-as-a-Judge filtering + human spot-check validation
6. **Augmentation**: Apply TER, mention replacement, and back-translation to multiply effective training data

**Estimated effort**: 4-8 weeks for initial pipeline; ongoing maintenance and expansion.

---

## Sources

1. NVIDIA. "Nemotron-PII." Hugging Face. <https://huggingface.co/datasets/nvidia/Nemotron-PII>
2. Gretel.ai. "Gretel PII Masking EN v1." Hugging Face. <https://huggingface.co/datasets/gretelai/gretel-pii-masking-en-v1>
3. AI4Privacy. "PII Masking 200K." Hugging Face. <https://huggingface.co/datasets/ai4privacy/pii-masking-200k>
4. Gretel.ai. "Synthetic PII Finance Multilingual." Hugging Face. <https://huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual>
5. Kaggle. "The Learning Agency Lab - PII Data Detection." 2024. <https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data>
6. Stubbs, Amber, and Ozlem Uzuner. "Annotating longitudinal clinical narratives for de-identification: The 2014 i2b2/UTHealth corpus." Journal of Biomedical Informatics 58 (2015): S20-S29.
7. Pilan, Ildiko, et al. "The Text Anonymization Benchmark (TAB): A Dedicated Corpus and Evaluation Framework for Text Anonymization." Computational Linguistics 48.4 (2022): 1053-1101.
8. BigCode Project. "BigCode PII Dataset." Hugging Face. <https://huggingface.co/datasets/bigcode/pii-dataset>
9. Tjong Kim Sang, Erik F., and Fien De Meulder. "Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition." CoNLL 2003.
10. Weischedel, Ralph, et al. "OntoNotes Release 5.0." Linguistic Data Consortium (2013).
11. Ding, Ning, et al. "Few-NERD: A Few-shot Named Entity Recognition Dataset." ACL 2021.
12. Pan, Xiaoman, et al. "Cross-lingual Name Tagging and Linking for 282 Languages." ACL 2017. (WikiANN)
13. Liu, Zihan, et al. "CrossNER: Evaluating Cross-Domain Named Entity Recognition." AAAI 2021.
14. Tedeschi, Simone, and Roberto Navigli. "MultiNERD: A Multilingual, Multi-Genre and Fine-Grained Dataset for Named Entity Recognition (and Disambiguation)." NAACL 2022.
15. Zaratiana, Urchade, et al. "GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer." NAACL 2024.
16. GLiNER2 / NuNER. "NuNER: Entity Recognition Encoder Pre-training via LLM-Annotated Data." arXiv:2402.15343 (2024).
17. Faker Library. <https://faker.readthedocs.io/>
18. NVIDIA NeMo Data Designer. <https://developer.nvidia.com/nemo>
19. Piiranha Model (trained on AI4Privacy). <https://huggingface.co/lakshyakh93/deberta_finetuned_pii>
