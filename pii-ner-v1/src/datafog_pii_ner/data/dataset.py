"""Dataset loading and preprocessing for PII NER training.

Loads AI4Privacy, Nemotron-PII, and Gretel datasets from HuggingFace,
maps labels to canonical schema, tokenizes with DeBERTa tokenizer,
and generates character IDs for the CharCNN.

Supports two data formats:
- Token-based: pre-tokenized tokens + NER tags (AI4Privacy)
- Span-based: raw text + character-offset spans (Nemotron, Gretel)
"""

import ast
import logging
import re

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from transformers import PreTrainedTokenizerFast

from .char_vocab import text_to_char_ids
from .label_schema import LABEL_TO_ID, map_label

logger = logging.getLogger(__name__)

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
}

# Regex for whitespace-aware word tokenization preserving character offsets
_WORD_RE = re.compile(r"\S+")


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
) -> Dataset:
    """Load and preprocess a single PII dataset.

    Handles two formats:
    - Token-based (AI4Privacy): pre-tokenized tokens + NER tag columns
    - Span-based (Nemotron, Gretel): raw text + character-offset span annotations
    """
    config = DATASET_CONFIGS[dataset_name]
    data_format = config.get("format", "token")
    logger.info(f"Loading {dataset_name} from {config['path']} (format={data_format})...")

    raw = load_dataset(config["path"], split=config["split"])

    # Filter to English using dataset-specific language column/value
    lang_col = config.get("language_col", "language")
    lang_val = config.get("language_value", "en")
    if lang_col in raw.column_names:
        raw = raw.filter(lambda x: x[lang_col] == lang_val, desc="Filtering to English")
        logger.info(f"  Filtered to {len(raw)} English examples")

    if max_examples is not None:
        raw = raw.select(range(min(max_examples, len(raw))))

    if data_format == "span":
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
        # HF datasets' List type has a broken __getattr__ so we use bare try/except.
        tag_names = None
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


def load_pii_datasets(
    tokenizer: PreTrainedTokenizerFast,
    max_seq_len: int = 256,
    max_char_len: int = 20,
    max_examples_per_dataset: int | None = None,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> DatasetDict:
    """Load all PII datasets, combine, and split into train/val/test."""
    all_datasets = []
    for name in DATASET_CONFIGS:
        try:
            ds = load_single_dataset(
                name, tokenizer, max_seq_len, max_char_len, max_examples_per_dataset
            )
            all_datasets.append(ds)
            logger.info(f"  {name}: {len(ds)} examples")
        except Exception as e:
            logger.warning(f"  Failed to load {name}: {e}")
            continue

    if not all_datasets:
        raise RuntimeError("No datasets loaded successfully")

    combined = concatenate_datasets(all_datasets)
    logger.info(f"Combined: {len(combined)} examples")

    # Split: train / val / test
    test_size = test_ratio
    val_size = val_ratio / (1 - test_ratio)  # Adjust for two-stage split

    split1 = combined.train_test_split(test_size=test_size, seed=seed)
    split2 = split1["train"].train_test_split(test_size=val_size, seed=seed)

    return DatasetDict(
        {
            "train": split2["train"],
            "validation": split2["test"],
            "test": split1["test"],
        }
    )
