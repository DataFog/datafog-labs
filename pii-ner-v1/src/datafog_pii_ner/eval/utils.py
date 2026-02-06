"""Utility helpers for evaluation and benchmarking."""

from __future__ import annotations

from typing import Iterable

from ..data.label_schema import ALL_ENTITY_TYPES, map_label_any


def tokens_to_text(tokens: list[str]) -> str:
    """Reconstruct text by joining tokens with single spaces."""
    return " ".join(tokens)


def tokens_to_offsets(tokens: list[str]) -> list[tuple[int, int]]:
    """Compute character offsets for tokens in the reconstructed text."""
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for token in tokens:
        start = cursor
        end = cursor + len(token)
        offsets.append((start, end))
        cursor = end + 1
    return offsets


def _span_fields(span) -> tuple[int | None, int | None, str | None]:
    if isinstance(span, dict):
        return span.get("start"), span.get("end"), span.get("label")
    return (
        getattr(span, "start", None),
        getattr(span, "end", None),
        getattr(span, "label", None),
    )


def normalize_span_label(label: str | None) -> str | None:
    if not label:
        return None
    canonical = map_label_any(str(label))
    if canonical in ALL_ENTITY_TYPES:
        return canonical
    return None


def spans_to_bio(tokens: list[str], spans: Iterable[object]) -> list[str]:
    """Convert predicted spans into BIO tags aligned to tokens.

    Tokens are assumed to correspond to a whitespace-joined text string.
    """
    if not tokens:
        return []

    offsets = tokens_to_offsets(tokens)
    tags = ["O"] * len(tokens)

    # Sort spans deterministically by start offset, longer first on ties.
    normalized_spans = []
    for span in spans:
        start, end, label = _span_fields(span)
        label = normalize_span_label(label)
        if start is None or end is None or label is None:
            continue
        normalized_spans.append((int(start), int(end), label))

    normalized_spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))

    for start, end, label in normalized_spans:
        if end <= start:
            continue
        overlap = [
            i for i, (tok_start, tok_end) in enumerate(offsets)
            if tok_end > start and tok_start < end
        ]
        if not overlap:
            continue
        if any(tags[i] != "O" for i in overlap):
            continue
        for j, idx in enumerate(overlap):
            prefix = "B-" if j == 0 else "I-"
            tags[idx] = f"{prefix}{label}"

    return tags
