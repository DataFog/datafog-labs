"""Dataset loading and preprocessing for PII NER training.

Loads AI4Privacy, Nemotron-PII, Gretel, NCBI Disease, MACCROBAT,
and synthetic datasets. Maps labels to canonical schema, tokenizes
with DeBERTa tokenizer, and generates character IDs for the CharCNN.

Supports three data formats:
- Token-based: pre-tokenized tokens + NER tags (AI4Privacy, NCBI, MACCROBAT)
- Span-based: raw text + character-offset spans (Nemotron, Gretel)
- Entity-based: raw text + entity values/types without offsets (Gretel v1)
"""

import ast
import logging
import re
from pathlib import Path

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from transformers import PreTrainedTokenizerFast

from .char_vocab import text_to_char_ids
from .label_schema import LABEL_TO_ID, TIER_1, TIER_2, map_label

logger = logging.getLogger(__name__)

# Path to synthetic data directory (relative to this file's package root)
_SYNTHETIC_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "synthetic"

# HuggingFace dataset identifiers
DATASET_CONFIGS = {
    "ai4privacy": {
        "path": "ai4privacy/pii-masking-200k",
        "split": "train",
        "format": "token",
    },
    "nemotron": {
        "path": "nvidia/Nemotron-PII",
        "split": "train",
        "format": "span",
        "text_col": "text",
        "spans_col": "spans",
        # Dataset is English-only (locale values: "us", "intl") — no filtering needed
    },
    "gretel": {
        "path": "gretelai/synthetic_pii_finance_multilingual",
        "split": "train",
        "format": "span",
        "text_col": "generated_text",
        "spans_col": "pii_spans",
        "language_col": "language",
        "language_value": "English",
    },
    "gretel_v1": {
        "path": "gretelai/gretel-pii-masking-en-v1",
        "split": "train",
        "format": "entity",
        "text_col": "text",
        "entities_col": "entities",
    },
    "ncbi": {
        "path": "ncbi/ncbi_disease",
        "splits": {"train": "train", "validation": "validation", "test": "test"},
        "format": "token",
        "tag_names": ["O", "B-Disease", "I-Disease"],
    },
    "maccrobat": {
        "path": "singh-aditya/MACCROBAT_biomedical_ner",
        "split": "train",
        "format": "token",
    },
    "synthetic": {
        "path": str(_SYNTHETIC_DATA_DIR),
        "split": "train",
        "format": "synthetic",
    },
}

# Regex for whitespace-aware word tokenization preserving character offsets
_WORD_RE = re.compile(r"\S+")


def _entities_to_spans(
    text: str,
    entities: list[dict] | str,
) -> list[dict]:
    """Convert entity-value format (type + value, no offsets) to character-offset spans.

    For datasets like Gretel v1 that provide entity values and types but not
    character offsets. Finds each entity's position in the text.

    Args:
        text: The raw document text.
        entities: List of dicts with 'value' and 'type' keys, or string repr.

    Returns:
        List of {"start": int, "end": int, "label": str} dicts.
    """
    try:
        if isinstance(entities, str):
            entities = ast.literal_eval(entities)
    except (ValueError, SyntaxError):
        return []

    if not isinstance(entities, list):
        return []

    # Collect (value, type) pairs, sort by value length descending so
    # longer matches take priority
    entity_list = []
    for ent in entities:
        if not isinstance(ent, dict):
            continue
        value = ent.get("value", "")
        etype = ent.get("type", ent.get("entity_type", ent.get("label", "")))
        if value and etype:
            entity_list.append((str(value), str(etype)))
    entity_list.sort(key=lambda x: len(x[0]), reverse=True)

    # Track which character positions are already claimed
    claimed = set()
    spans = []

    for value, etype in entity_list:
        # Try exact match first
        idx = text.find(value)
        if idx == -1:
            # Case-insensitive fallback
            pattern = re.escape(value)
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                idx = match.start()
            else:
                continue

        start, end = idx, idx + len(value)

        # Skip if any character position overlaps with an already-claimed span
        if any(pos in claimed for pos in range(start, end)):
            continue

        claimed.update(range(start, end))
        spans.append({"start": start, "end": end, "label": etype})

    return spans


def _spans_to_bio(
    text: str,
    spans_str: str,
    dataset_name: str,
) -> tuple[list[str], list[str]]:
    """Convert raw text + character-offset spans into word tokens + BIO tags.

    Args:
        text: The raw document text.
        spans_str: String representation of a list of span dicts,
            each with 'start', 'end', and 'label' keys.
        dataset_name: Name of the source dataset (for label mapping).

    Returns:
        (tokens, bio_tags) — word-level lists ready for _tokenize_and_align.
    """
    # Parse spans from string representation
    try:
        spans = ast.literal_eval(spans_str) if isinstance(spans_str, str) else spans_str
    except (ValueError, SyntaxError):
        spans = []

    if not isinstance(spans, list):
        spans = []

    # Whitespace-tokenize, tracking character offsets
    tokens = []
    token_offsets = []
    for match in _WORD_RE.finditer(text):
        tokens.append(match.group())
        token_offsets.append((match.start(), match.end()))

    # Build sorted, non-overlapping span index
    valid_spans = []
    for span in spans:
        if not isinstance(span, dict):
            continue
        start = span.get("start")
        end = span.get("end")
        label = span.get("label", "")
        if start is None or end is None:
            continue
        canonical = map_label(str(label), dataset_name)
        if canonical is not None:
            valid_spans.append((int(start), int(end), canonical))
    valid_spans.sort(key=lambda s: s[0])

    # Assign BIO tags to each token based on span overlap
    bio_tags = ["O"] * len(tokens)
    span_idx = 0
    for tok_i, (tok_start, tok_end) in enumerate(token_offsets):
        # Advance span pointer past tokens we've passed
        while span_idx < len(valid_spans) and valid_spans[span_idx][1] <= tok_start:
            span_idx += 1

        if span_idx >= len(valid_spans):
            break

        sp_start, sp_end, sp_label = valid_spans[span_idx]

        # Check if token overlaps with the current span
        if tok_start >= sp_start and tok_start < sp_end:
            # First token of the span gets B-, rest get I-
            if tok_start <= sp_start or (tok_i > 0 and bio_tags[tok_i - 1] == "O"):
                bio_tags[tok_i] = f"B-{sp_label}"
            else:
                bio_tags[tok_i] = f"I-{sp_label}"

    return tokens, bio_tags


def _detect_columns(dataset: Dataset) -> tuple[str, str]:
    """Auto-detect the tokens and NER tags columns in a dataset."""
    cols = dataset.column_names

    # Common token column names
    for name in ["tokens", "words", "token", "word", "mbert_text_tokens"]:
        if name in cols:
            token_col = name
            break
    else:
        raise ValueError(f"Cannot find token column in {cols}")

    # Common tag column names
    for name in ["ner_tags", "tags", "labels", "ner_labels", "entities", "mbert_bio_labels"]:
        if name in cols:
            tag_col = name
            break
    else:
        raise ValueError(f"Cannot find tag column in {cols}")

    return token_col, tag_col


def _normalize_tags(
    tags: list[str | int],
    dataset_name: str,
    tag_names: list[str] | None = None,
) -> list[str]:
    """Convert dataset tags to canonical BIO labels.

    Handles both string tags ("B-PERSON") and integer tags (with tag_names lookup).
    """
    result = []
    for tag in tags:
        # Convert integer tags to strings using the dataset's tag name list
        if isinstance(tag, int):
            if tag_names is not None and 0 <= tag < len(tag_names):
                tag = tag_names[tag]
            else:
                result.append("O")
                continue

        tag = str(tag).strip()
        if tag in ("O", "0", "o", ""):
            result.append("O")
            continue

        # Parse BIO prefix
        if tag.startswith(("B-", "I-")):
            prefix = tag[:2]
            entity_type = tag[2:]
        elif tag.startswith(("B_", "I_")):
            prefix = tag[0] + "-"
            entity_type = tag[2:]
        else:
            # No prefix — treat as B-
            prefix = "B-"
            entity_type = tag

        canonical = map_label(entity_type, dataset_name)
        if canonical is None:
            result.append("O")
        else:
            result.append(f"{prefix}{canonical}")

    return result


def _tokenize_and_align(
    tokens: list[str],
    bio_tags: list[str],
    tokenizer: PreTrainedTokenizerFast,
    max_seq_len: int = 256,
    max_char_len: int = 20,
) -> dict:
    """Tokenize word-level tokens and align BIO labels to subword tokens.

    For each word split into subwords:
    - First subword gets the original BIO tag
    - Continuation subwords get I- tag (if original was B- or I-) or O
    """
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=max_seq_len,
        padding=False,
        return_offsets_mapping=False,
    )

    word_ids = encoding.word_ids()

    aligned_labels = []
    aligned_char_ids = []
    prev_word_id = None

    for idx, word_id in enumerate(word_ids):
        if word_id is None:
            # Special tokens ([CLS], [SEP], padding)
            aligned_labels.append(-100)
            aligned_char_ids.append([0] * max_char_len)
        elif word_id != prev_word_id:
            # First subword of a new word
            if word_id < len(bio_tags):
                label_str = bio_tags[word_id]
                aligned_labels.append(LABEL_TO_ID.get(label_str, 0))
            else:
                aligned_labels.append(-100)
            # Character IDs from the original word
            if word_id < len(tokens):
                aligned_char_ids.append(text_to_char_ids(tokens[word_id], max_char_len))
            else:
                aligned_char_ids.append([0] * max_char_len)
        else:
            # Continuation subword — propagate I- tag
            if word_id < len(bio_tags):
                label_str = bio_tags[word_id]
                if label_str.startswith("B-"):
                    label_str = "I-" + label_str[2:]
                aligned_labels.append(LABEL_TO_ID.get(label_str, 0))
            else:
                aligned_labels.append(-100)
            if word_id < len(tokens):
                aligned_char_ids.append(text_to_char_ids(tokens[word_id], max_char_len))
            else:
                aligned_char_ids.append([0] * max_char_len)

        prev_word_id = word_id

    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "labels": aligned_labels,
        "char_ids": aligned_char_ids,
    }


def load_single_dataset(
    dataset_name: str,
    tokenizer: PreTrainedTokenizerFast,
    max_seq_len: int = 256,
    max_char_len: int = 20,
    max_examples: int | None = None,
    override_split: str | None = None,
) -> Dataset:
    """Load and preprocess a single PII dataset.

    Handles three formats:
    - Token-based (AI4Privacy, NCBI, MACCROBAT): pre-tokenized tokens + NER tags
    - Span-based (Nemotron, Gretel): raw text + character-offset span annotations
    - Entity-based (Gretel v1): raw text + entity values/types without offsets

    Args:
        override_split: If provided, load this specific split instead of config default.
    """
    config = DATASET_CONFIGS[dataset_name]
    data_format = config.get("format", "token")
    split = override_split or config.get("split", "train")
    logger.info(f"Loading {dataset_name} from {config['path']} (format={data_format}, split={split})...")

    if data_format == "synthetic":
        return _load_synthetic_dataset(
            dataset_name, tokenizer, max_seq_len, max_char_len, max_examples
        )

    raw = load_dataset(config["path"], split=split, trust_remote_code=False)

    # Filter to English using dataset-specific language column/value
    lang_col = config.get("language_col", "language")
    lang_val = config.get("language_value", "en")
    if lang_col in raw.column_names:
        raw = raw.filter(lambda x: x[lang_col] == lang_val, desc="Filtering to English")
        logger.info(f"  Filtered to {len(raw)} English examples")

    if max_examples is not None:
        raw = raw.select(range(min(max_examples, len(raw))))

    if data_format == "entity":
        text_col = config["text_col"]
        entities_col = config["entities_col"]

        def preprocess_entity(example):
            spans = _entities_to_spans(example[text_col], example[entities_col])
            tokens, bio_tags = _spans_to_bio(
                example[text_col], spans, dataset_name
            )
            if not tokens:
                tokens = ["[EMPTY]"]
                bio_tags = ["O"]
            return _tokenize_and_align(tokens, bio_tags, tokenizer, max_seq_len, max_char_len)

        processed = raw.map(
            preprocess_entity,
            remove_columns=raw.column_names,
            desc=f"Processing {dataset_name}",
        )
    elif data_format == "span":
        text_col = config["text_col"]
        spans_col = config["spans_col"]

        def preprocess_span(example):
            tokens, bio_tags = _spans_to_bio(
                example[text_col], example[spans_col], dataset_name
            )
            if not tokens:
                # Empty document — produce a minimal valid example
                tokens = ["[EMPTY]"]
                bio_tags = ["O"]
            return _tokenize_and_align(tokens, bio_tags, tokenizer, max_seq_len, max_char_len)

        processed = raw.map(
            preprocess_span,
            remove_columns=raw.column_names,
            desc=f"Processing {dataset_name}",
        )
    else:
        token_col, tag_col = _detect_columns(raw)

        # Check if tags are ClassLabel integers (need tag_names for conversion).
        # Use explicit config tag_names first, then try HF ClassLabel auto-detection.
        tag_names = config.get("tag_names")
        if tag_names is None:
            try:
                tag_names = raw.features[tag_col].feature.names
            except Exception:
                pass

        def preprocess_token(example):
            tokens = example[token_col]
            raw_tags = example[tag_col]
            bio_tags = _normalize_tags(raw_tags, dataset_name, tag_names)
            return _tokenize_and_align(tokens, bio_tags, tokenizer, max_seq_len, max_char_len)

        processed = raw.map(
            preprocess_token,
            remove_columns=raw.column_names,
            desc=f"Processing {dataset_name}",
        )

    return processed


def _load_synthetic_dataset(
    dataset_name: str,
    tokenizer: PreTrainedTokenizerFast,
    max_seq_len: int = 256,
    max_char_len: int = 20,
    max_examples: int | None = None,
) -> Dataset:
    """Load synthetic data from local JSONL files."""
    data_dir = _SYNTHETIC_DATA_DIR
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Synthetic data directory not found: {data_dir}. "
            "Run 'python scripts/generate_synthetic_data.py' first."
        )

    jsonl_files = sorted(data_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found in {data_dir}")

    raw = load_dataset("json", data_files=[str(f) for f in jsonl_files], split="train")
    logger.info(f"  Loaded {len(raw)} synthetic examples from {len(jsonl_files)} files")

    if max_examples is not None:
        raw = raw.select(range(min(max_examples, len(raw))))

    def preprocess_synthetic(example):
        text = example["text"]
        spans = example["spans"]
        if isinstance(spans, str):
            spans = ast.literal_eval(spans)
        tokens, bio_tags = _spans_to_bio(text, spans, "synthetic")
        if not tokens:
            tokens = ["[EMPTY]"]
            bio_tags = ["O"]
        return _tokenize_and_align(tokens, bio_tags, tokenizer, max_seq_len, max_char_len)

    processed = raw.map(
        preprocess_synthetic,
        remove_columns=raw.column_names,
        desc=f"Processing {dataset_name}",
    )
    return processed


def _build_oversample_label_ids(tiers: list[int]) -> set[int]:
    """Build set of label IDs that belong to the specified tiers."""
    tier_types = []
    if 1 in tiers:
        tier_types.extend(TIER_1)
    if 2 in tiers:
        tier_types.extend(TIER_2)

    label_ids = set()
    for etype in tier_types:
        b_key = f"B-{etype}"
        i_key = f"I-{etype}"
        if b_key in LABEL_TO_ID:
            label_ids.add(LABEL_TO_ID[b_key])
        if i_key in LABEL_TO_ID:
            label_ids.add(LABEL_TO_ID[i_key])
    return label_ids


def _oversample_rare_entities(
    dataset: Dataset,
    oversample_tiers: list[int],
    oversample_factor: int,
) -> Dataset:
    """Oversample examples containing rare entity types.

    Identifies examples that contain entities from the specified tiers,
    then duplicates them `oversample_factor` times.
    """
    target_label_ids = _build_oversample_label_ids(oversample_tiers)
    if not target_label_ids:
        return dataset

    # Find indices of examples containing target entities
    rare_indices = []
    for i in range(len(dataset)):
        labels = dataset[i]["labels"]
        if any(label_id in target_label_ids for label_id in labels):
            rare_indices.append(i)

    if not rare_indices:
        logger.warning("No examples found containing target tier entities for oversampling")
        return dataset

    logger.info(
        f"Oversampling: {len(rare_indices)} examples with tier {oversample_tiers} entities "
        f"x{oversample_factor} ({len(rare_indices) * oversample_factor} additional examples)"
    )

    # Create duplicated dataset
    oversampled_indices = rare_indices * oversample_factor
    oversampled = dataset.select(oversampled_indices)

    return concatenate_datasets([dataset, oversampled])


def load_pii_datasets(
    tokenizer: PreTrainedTokenizerFast,
    max_seq_len: int = 256,
    max_char_len: int = 20,
    max_examples_per_dataset: int | None = None,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    oversample_tiers: list[int] | None = None,
    oversample_factor: int = 2,
) -> DatasetDict:
    """Load all PII datasets, combine, and split into train/val/test.

    Datasets with a "splits" config key (e.g. NCBI) are loaded per-split and
    injected directly into the corresponding train/val/test partitions.
    Datasets without "splits" go through a combined random split.

    Args:
        oversample_tiers: List of tier numbers to oversample (e.g., [1, 2]).
            None disables oversampling.
        oversample_factor: Number of times to duplicate rare-entity examples.
    """
    # Datasets that go through random split
    unsplit_datasets = []
    # Pre-split datasets: {split_name: [dataset, ...]}
    presplit_datasets = {"train": [], "validation": [], "test": []}

    for name, config in DATASET_CONFIGS.items():
        try:
            if "splits" in config:
                # Load each split separately (e.g. NCBI train/val/test)
                for target_split, source_split in config["splits"].items():
                    ds = load_single_dataset(
                        name, tokenizer, max_seq_len, max_char_len,
                        max_examples_per_dataset, override_split=source_split,
                    )
                    presplit_datasets[target_split].append(ds)
                    logger.info(f"  {name}/{source_split}: {len(ds)} examples -> {target_split}")
            else:
                ds = load_single_dataset(
                    name, tokenizer, max_seq_len, max_char_len, max_examples_per_dataset
                )
                unsplit_datasets.append(ds)
                logger.info(f"  {name}: {len(ds)} examples")
        except Exception as e:
            logger.warning(f"  Failed to load {name}: {e}")
            continue

    if not unsplit_datasets and not any(presplit_datasets.values()):
        raise RuntimeError("No datasets loaded successfully")

    # Random split for unsplit datasets
    if unsplit_datasets:
        combined = concatenate_datasets(unsplit_datasets)
        logger.info(f"Combined (unsplit): {len(combined)} examples")

        test_size = test_ratio
        val_size = val_ratio / (1 - test_ratio)

        split1 = combined.train_test_split(test_size=test_size, seed=seed)
        split2 = split1["train"].train_test_split(test_size=val_size, seed=seed)

        presplit_datasets["train"].append(split2["train"])
        presplit_datasets["validation"].append(split2["test"])
        presplit_datasets["test"].append(split1["test"])

    # Merge all partitions
    train_ds = concatenate_datasets(presplit_datasets["train"]) if presplit_datasets["train"] else Dataset.from_dict({})
    val_ds = concatenate_datasets(presplit_datasets["validation"]) if presplit_datasets["validation"] else Dataset.from_dict({})
    test_ds = concatenate_datasets(presplit_datasets["test"]) if presplit_datasets["test"] else Dataset.from_dict({})

    logger.info(f"Final: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    # Oversample rare entities in training set only
    if oversample_tiers:
        train_ds = _oversample_rare_entities(train_ds, oversample_tiers, oversample_factor)
        logger.info(f"Train after oversampling: {len(train_ds)} examples")

    return DatasetDict(
        {
            "train": train_ds,
            "validation": val_ds,
            "test": test_ds,
        }
    )
