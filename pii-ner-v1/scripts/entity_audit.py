#!/usr/bin/env python3
"""Audit entity type frequencies across all training datasets.

Counts entity occurrences per type and per dataset to identify
class imbalance issues.
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from datasets import load_dataset
from datafog_pii_ner.data.label_schema import (
    ALL_ENTITY_TYPES,
    TIER_1,
    TIER_2,
    TIER_3,
    TIER_4,
    TIER_MAP,
    map_label,
)
from datafog_pii_ner.data.dataset import (
    DATASET_CONFIGS,
    _spans_to_bio,
    _detect_columns,
    _normalize_tags,
    _entities_to_spans,
)

import json
import re
_WORD_RE = re.compile(r"\S+")


def count_entities_token_format(raw, dataset_name):
    """Count entities in token-format dataset (AI4Privacy)."""
    token_col, tag_col = _detect_columns(raw)

    tag_names = None
    try:
        tag_names = raw.features[tag_col].feature.names
    except Exception:
        pass

    entity_counts = Counter()
    examples_with_entity = Counter()

    for i, example in enumerate(raw):
        tags = _normalize_tags(example[tag_col], dataset_name, tag_names)
        types_in_example = set()
        for tag in tags:
            if tag.startswith("B-"):
                entity_type = tag[2:]
                entity_counts[entity_type] += 1
                types_in_example.add(entity_type)
        for t in types_in_example:
            examples_with_entity[t] += 1

        if (i + 1) % 50000 == 0:
            print(f"  Processed {i+1} examples...")

    return entity_counts, examples_with_entity, len(raw)


def count_entities_span_format(raw, dataset_name, text_col, spans_col):
    """Count entities in span-format dataset (Nemotron, Gretel)."""
    entity_counts = Counter()
    examples_with_entity = Counter()

    for i, example in enumerate(raw):
        _, bio_tags = _spans_to_bio(example[text_col], example[spans_col], dataset_name)
        types_in_example = set()
        for tag in bio_tags:
            if tag.startswith("B-"):
                entity_type = tag[2:]
                entity_counts[entity_type] += 1
                types_in_example.add(entity_type)
        for t in types_in_example:
            examples_with_entity[t] += 1

        if (i + 1) % 50000 == 0:
            print(f"  Processed {i+1} examples...")

    return entity_counts, examples_with_entity, len(raw)


def count_entities_entity_format(raw, dataset_name, text_col, entities_col):
    """Count entities in entity-value format (Gretel v1)."""
    entity_counts = Counter()
    examples_with_entity = Counter()

    for i, example in enumerate(raw):
        spans = _entities_to_spans(example[text_col], example[entities_col])
        _, bio_tags = _spans_to_bio(example[text_col], spans, dataset_name)
        types_in_example = set()
        for tag in bio_tags:
            if tag.startswith("B-"):
                entity_type = tag[2:]
                entity_counts[entity_type] += 1
                types_in_example.add(entity_type)
        for t in types_in_example:
            examples_with_entity[t] += 1

        if (i + 1) % 50000 == 0:
            print(f"  Processed {i+1} examples...")

    return entity_counts, examples_with_entity, len(raw)


def count_entities_synthetic(data_dir):
    """Count entities in local synthetic JSONL files."""
    entity_counts = Counter()
    examples_with_entity = Counter()
    total = 0

    jsonl_files = sorted(Path(data_dir).glob("*.jsonl"))
    for filepath in jsonl_files:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                example = json.loads(line)
                total += 1
                types_in_example = set()
                for span in example.get("spans", []):
                    label = span.get("label", "")
                    if label:
                        entity_counts[label] += 1
                        types_in_example.add(label)
                for t in types_in_example:
                    examples_with_entity[t] += 1

    return entity_counts, examples_with_entity, total


def _load_and_count(name, config):
    """Load a single dataset and count its entities. Returns (counts, ex_counts, n)."""
    data_format = config.get("format", "token")

    if data_format == "synthetic":
        data_dir = config["path"]
        if not Path(data_dir).exists():
            print(f"  Synthetic data dir not found: {data_dir}, skipping")
            return Counter(), Counter(), 0
        return count_entities_synthetic(data_dir)

    # For pre-split datasets, load all splits and combine
    if "splits" in config:
        all_counts = Counter()
        all_ex_counts = Counter()
        total_n = 0
        for target_split, source_split in config["splits"].items():
            raw = load_dataset(config["path"], split=source_split, trust_remote_code=False)
            print(f"  Split {source_split}: {len(raw)} examples")
            counts, ex_counts, n = count_entities_token_format(raw, name)
            all_counts += counts
            all_ex_counts += ex_counts
            total_n += n
        return all_counts, all_ex_counts, total_n

    raw = load_dataset(config["path"], split=config.get("split", "train"), trust_remote_code=False)

    # Filter to English
    lang_col = config.get("language_col", "language")
    lang_val = config.get("language_value", "en")
    if lang_col in raw.column_names:
        raw = raw.filter(lambda x: x[lang_col] == lang_val, desc="Filtering English")
        print(f"  Filtered to {len(raw)} English examples")

    if data_format == "entity":
        return count_entities_entity_format(
            raw, name, config["text_col"], config["entities_col"]
        )
    elif data_format == "span":
        return count_entities_span_format(
            raw, name, config["text_col"], config["spans_col"]
        )
    else:
        return count_entities_token_format(raw, name)


def main():
    total_counts = Counter()
    total_examples = Counter()
    dataset_stats = {}

    for name, config in DATASET_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Loading {name} from {config['path']}...")
        print(f"{'='*60}")

        counts, ex_counts, n = _load_and_count(name, config)

        dataset_stats[name] = {
            "total_examples": n,
            "entity_counts": counts,
            "examples_with_entity": ex_counts,
        }

        total_counts += counts
        total_examples += ex_counts

        print(f"\n{name}: {n} examples")
        for etype in sorted(counts.keys(), key=lambda x: counts[x], reverse=True):
            tier = TIER_MAP.get(etype, "?")
            print(f"  T{tier} {etype:30s} {counts[etype]:>8,} entities  in {ex_counts[etype]:>8,} examples")

    # Summary
    print(f"\n{'='*60}")
    print("COMBINED ENTITY FREQUENCY AUDIT")
    print(f"{'='*60}")

    total_entities = sum(total_counts.values())
    print(f"\nTotal entities: {total_entities:,}")

    for tier_name, tier_list in [("TIER 1 (target >=0.98)", TIER_1),
                                  ("TIER 2 (target >=0.95)", TIER_2),
                                  ("TIER 3 (target >=0.90)", TIER_3),
                                  ("TIER 4 (target >=0.85)", TIER_4)]:
        print(f"\n--- {tier_name} ---")
        for etype in tier_list:
            count = total_counts.get(etype, 0)
            ex_count = total_examples.get(etype, 0)
            pct = 100.0 * count / total_entities if total_entities > 0 else 0
            print(f"  {etype:30s} {count:>8,} ({pct:5.2f}%)  in {ex_count:>8,} examples")

    # Imbalance ratio
    most_common = total_counts.most_common(1)[0] if total_counts else ("?", 0)
    least_common = total_counts.most_common()[-1] if total_counts else ("?", 0)
    if least_common[1] > 0:
        ratio = most_common[1] / least_common[1]
        print(f"\nImbalance ratio: {most_common[0]}({most_common[1]:,}) / "
              f"{least_common[0]}({least_common[1]:,}) = {ratio:.0f}x")

    # Types with zero occurrences
    zero_types = [t for t in ALL_ENTITY_TYPES if total_counts.get(t, 0) == 0]
    if zero_types:
        print(f"\nZERO OCCURRENCE TYPES: {zero_types}")


if __name__ == "__main__":
    main()
