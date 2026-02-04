"""Inference pipeline for PII detection.

Provides a simple API for detecting PII entities in text:

    pipeline = PiiPipeline.from_pretrained("DataFog/pii-ner-v1")
    entities = pipeline("My SSN is 123-45-6789")
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from ..data.char_vocab import text_to_char_ids
from ..data.label_schema import ID_TO_LABEL, TIER_MAP
from ..model.pii_model import PiiNerConfig, PiiNerModel

_WORD_RE = re.compile(r"\S+")


@dataclass
class PiiEntity:
    """A detected PII entity with character offsets."""

    text: str
    label: str
    start: int
    end: int
    tier: int

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "tier": self.tier,
        }


class PiiPipeline:
    """End-to-end PII detection pipeline.

    Tokenizes raw text, runs the model, and returns detected PII entities
    with character offsets.

    Usage:
        pipeline = PiiPipeline.from_pretrained("DataFog/pii-ner-v1")
        entities = pipeline("Contact john@example.com or call 555-0123")
        for e in entities:
            print(f"{e.label}: {e.text} [{e.start}:{e.end}]")
    """

    def __init__(
        self,
        model: PiiNerModel,
        tokenizer: PreTrainedTokenizerFast,
        device: str | torch.device | None = None,
        max_seq_len: int = 256,
        max_char_len: int = 20,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_char_len = max_char_len

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: str | torch.device | None = None,
        max_seq_len: int = 256,
        max_char_len: int = 20,
        **kwargs,
    ) -> PiiPipeline:
        """Load pipeline from a saved checkpoint or HuggingFace Hub.

        Args:
            model_path: Local path or HuggingFace Hub model ID
                (e.g. "DataFog/pii-ner-v1").
            device: Device to run on ("cpu", "cuda", "cuda:0", etc.).
                Auto-detects if None.
            max_seq_len: Maximum subword sequence length.
            max_char_len: Maximum characters per word for CharCNN.
            **kwargs: Passed to PiiNerModel.from_pretrained().
        """
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        config = PiiNerConfig.from_pretrained(model_path)
        model = PiiNerModel.from_pretrained(model_path, config=config, **kwargs)
        return cls(
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_seq_len=max_seq_len,
            max_char_len=max_char_len,
        )

    def __call__(
        self,
        text: str | list[str],
    ) -> list[PiiEntity] | list[list[PiiEntity]]:
        """Detect PII entities in text.

        Args:
            text: A string or list of strings to scan.

        Returns:
            List of PiiEntity for a single string, or list of lists for batch.
        """
        if isinstance(text, str):
            return self._predict_single(text)
        return [self._predict_single(t) for t in text]

    def _predict_single(self, text: str) -> list[PiiEntity]:
        """Detect PII entities in a single text string."""
        if not text or not text.strip():
            return []

        # Step 1: Whitespace-tokenize to get words + character offsets
        words = []
        word_offsets = []
        for match in _WORD_RE.finditer(text):
            words.append(match.group())
            word_offsets.append((match.start(), match.end()))

        if not words:
            return []

        # Step 2: Subword tokenize (matching training pipeline)
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_seq_len,
            padding=False,
            return_offsets_mapping=False,
        )

        word_ids = encoding.word_ids()

        # Step 3: Build char_ids aligned to subword tokens
        char_ids = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                char_ids.append([0] * self.max_char_len)
            elif word_id < len(words):
                char_ids.append(text_to_char_ids(words[word_id], self.max_char_len))
            else:
                char_ids.append([0] * self.max_char_len)
            prev_word_id = word_id

        # Step 4: Prepare tensors
        input_ids = torch.tensor([encoding["input_ids"]], dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(
            [encoding["attention_mask"]], dtype=torch.long, device=self.device
        )
        char_ids_tensor = torch.tensor([char_ids], dtype=torch.long, device=self.device)

        # Step 5: Run model
        with torch.no_grad():
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                char_ids=char_ids_tensor,
            )

        # Step 6: Extract predictions
        pred_ids = output.predictions[0].cpu().tolist()

        # Step 7: Convert BIO predictions to entity spans
        return self._decode_entities(pred_ids, word_ids, words, word_offsets, text)

    @staticmethod
    def _decode_entities(
        pred_ids: list[int],
        word_ids: list[int | None],
        words: list[str],
        word_offsets: list[tuple[int, int]],
        text: str,
    ) -> list[PiiEntity]:
        """Convert BIO tag predictions to PiiEntity spans.

        Merges subword predictions back to word level, then merges consecutive
        B-/I- tags of the same type into entity spans with character offsets.
        """
        # Map subword predictions back to word-level BIO tags.
        # For each word, take the prediction from its first subword token.
        word_labels: list[str] = ["O"] * len(words)
        seen_words: set[int] = set()

        for tok_idx, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            if word_id in seen_words:
                continue
            seen_words.add(word_id)
            if word_id < len(words) and tok_idx < len(pred_ids):
                label_str = ID_TO_LABEL.get(pred_ids[tok_idx], "O")
                word_labels[word_id] = label_str

        # Merge consecutive B-/I- tags into entity spans
        entities: list[PiiEntity] = []
        current_label: str | None = None
        current_start_word: int | None = None

        for word_idx, bio_tag in enumerate(word_labels):
            if bio_tag.startswith("B-"):
                # Close previous entity
                if current_label is not None and current_start_word is not None:
                    entities.append(
                        _make_entity(
                            current_label,
                            current_start_word,
                            word_idx - 1,
                            word_offsets,
                            text,
                        )
                    )
                current_label = bio_tag[2:]
                current_start_word = word_idx
            elif bio_tag.startswith("I-"):
                i_label = bio_tag[2:]
                if current_label == i_label:
                    # Continue current entity
                    pass
                else:
                    # I- tag without matching B- or different type — start new entity
                    if current_label is not None and current_start_word is not None:
                        entities.append(
                            _make_entity(
                                current_label,
                                current_start_word,
                                word_idx - 1,
                                word_offsets,
                                text,
                            )
                        )
                    current_label = i_label
                    current_start_word = word_idx
            else:
                # O tag — close any open entity
                if current_label is not None and current_start_word is not None:
                    entities.append(
                        _make_entity(
                            current_label,
                            current_start_word,
                            word_idx - 1,
                            word_offsets,
                            text,
                        )
                    )
                current_label = None
                current_start_word = None

        # Close final entity
        if current_label is not None and current_start_word is not None:
            entities.append(
                _make_entity(
                    current_label,
                    current_start_word,
                    len(words) - 1,
                    word_offsets,
                    text,
                )
            )

        return entities


def _make_entity(
    label: str,
    start_word: int,
    end_word: int,
    word_offsets: list[tuple[int, int]],
    text: str,
) -> PiiEntity:
    """Create a PiiEntity from word-level span indices."""
    char_start = word_offsets[start_word][0]
    char_end = word_offsets[end_word][1]
    return PiiEntity(
        text=text[char_start:char_end],
        label=label,
        start=char_start,
        end=char_end,
        tier=TIER_MAP.get(label, 0),
    )
