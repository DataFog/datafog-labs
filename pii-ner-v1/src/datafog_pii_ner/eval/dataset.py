"""Dataset utilities for evaluation benchmarks."""

from __future__ import annotations

import logging

from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import PreTrainedTokenizerFast

from ..data.dataset import DATASET_CONFIGS, _detect_columns, _normalize_tags, _spans_to_bio
from .utils import tokens_to_text

logger = logging.getLogger(__name__)


def _truncate_tokens_and_tags(
    tokens: list[str],
    tags: list[str],
    tokenizer: PreTrainedTokenizerFast,
    max_seq_len: int,
) -> tuple[list[str], list[str]]:
    if max_seq_len is None:
        return tokens, tags

    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=max_seq_len,
        padding=False,
        return_offsets_mapping=False,
    )
    word_ids = encoding.word_ids()
    kept_word_ids = [wid for wid in word_ids if wid is not None]
    if not kept_word_ids:
        return tokens, tags
    cut = max(kept_word_ids) + 1
    return tokens[:cut], tags[:cut]


def _sanitize_example(tokens: list[str], tags: list[str]) -> tuple[list[str], list[str]]:
    if not tokens:
        return ["[EMPTY]"], ["O"]

    if len(tags) < len(tokens):
        tags = tags + ["O"] * (len(tokens) - len(tags))
    elif len(tags) > len(tokens):
        tags = tags[: len(tokens)]

    return tokens, tags


def _build_wordlevel_dataset(
    dataset_name: str,
    tokenizer: PreTrainedTokenizerFast,
    max_seq_len: int = 256,
    max_examples: int | None = None,
) -> Dataset:
    config = DATASET_CONFIGS[dataset_name]
    data_format = config.get("format", "token")
    logger.info(f"Loading {dataset_name} from {config['path']} (format={data_format})...")

    raw = load_dataset(config["path"], split=config["split"])

    # Filter to English if language column is present
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
            tokens, tags = _spans_to_bio(example[text_col], example[spans_col], dataset_name)
            tokens, tags = _sanitize_example(tokens, tags)
            tokens, tags = _truncate_tokens_and_tags(tokens, tags, tokenizer, max_seq_len)
            return {
                "text": tokens_to_text(tokens),
                "tokens": tokens,
                "bio_tags": tags,
                "source": dataset_name,
            }

        processed = raw.map(
            preprocess_span,
            remove_columns=raw.column_names,
            desc=f"Processing {dataset_name}",
        )
    else:
        token_col, tag_col = _detect_columns(raw)

        tag_names = None
        try:
            tag_names = raw.features[tag_col].feature.names
        except Exception:
            pass

        def preprocess_token(example):
            tokens = example[token_col]
            raw_tags = example[tag_col]
            tags = _normalize_tags(raw_tags, dataset_name, tag_names)
            tokens, tags = _sanitize_example(tokens, tags)
            tokens, tags = _truncate_tokens_and_tags(tokens, tags, tokenizer, max_seq_len)
            return {
                "text": tokens_to_text(tokens),
                "tokens": tokens,
                "bio_tags": tags,
                "source": dataset_name,
            }

        processed = raw.map(
            preprocess_token,
            remove_columns=raw.column_names,
            desc=f"Processing {dataset_name}",
        )

    def add_id(example, idx):
        return {"example_id": f"{dataset_name}-{idx}"}

    processed = processed.map(add_id, with_indices=True)
    return processed


def load_eval_dataset(
    dataset_name: str,
    tokenizer: PreTrainedTokenizerFast,
    split: str = "test",
    seed: int = 42,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    max_seq_len: int = 256,
    max_examples_per_dataset: int | None = None,
    max_examples: int | None = None,
) -> Dataset:
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unknown split: {split}")

    if dataset_name == "combined":
        datasets = []
        for name in DATASET_CONFIGS:
            ds = _build_wordlevel_dataset(
                name,
                tokenizer=tokenizer,
                max_seq_len=max_seq_len,
                max_examples=max_examples_per_dataset,
            )
            datasets.append(ds)
        combined = concatenate_datasets(datasets)

        test_size = test_ratio
        val_size = val_ratio / (1 - test_ratio)
        split1 = combined.train_test_split(test_size=test_size, seed=seed)
        split2 = split1["train"].train_test_split(test_size=val_size, seed=seed)
        splits = {
            "train": split2["train"],
            "validation": split2["test"],
            "test": split1["test"],
        }
    else:
        ds = _build_wordlevel_dataset(
            dataset_name,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            max_examples=max_examples_per_dataset,
        )
        test_size = test_ratio
        val_size = val_ratio / (1 - test_ratio)
        split1 = ds.train_test_split(test_size=test_size, seed=seed)
        split2 = split1["train"].train_test_split(test_size=val_size, seed=seed)
        splits = {
            "train": split2["train"],
            "validation": split2["test"],
            "test": split1["test"],
        }

    out = splits[split]
    if max_examples is not None:
        out = out.select(range(min(max_examples, len(out))))

    return out
