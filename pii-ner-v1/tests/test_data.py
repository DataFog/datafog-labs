"""Tests for the data pipeline components."""

import random

from datafog_pii_ner.data.char_vocab import VOCAB_SIZE, char_to_id, text_to_char_ids
from datafog_pii_ner.data.dataset import _entities_to_spans
from datafog_pii_ner.data.label_schema import (
    BIO_LABELS,
    LABEL_TO_ID,
    NUM_LABELS,
    map_label,
)
from datafog_pii_ner.data.synthetic import (
    SYNTHETIC_ENTITY_TYPES,
    generate_dataset,
    generate_example,
)


def test_bio_labels_start_with_o():
    """First label should always be O (outside)."""
    assert BIO_LABELS[0] == "O"


def test_bio_labels_count():
    """Should have O + 2 per entity type (B- and I-)."""
    from datafog_pii_ner.data.label_schema import ALL_ENTITY_TYPES

    expected = 1 + 2 * len(ALL_ENTITY_TYPES)
    assert NUM_LABELS == expected


def test_label_to_id_roundtrip():
    """Every label should map to a unique ID and back."""
    for label, idx in LABEL_TO_ID.items():
        assert BIO_LABELS[idx] == label


def test_map_label_ai4privacy():
    """AI4Privacy labels map correctly to canonical names."""
    assert map_label("FIRSTNAME", "ai4privacy") == "PERSON"
    assert map_label("SOCIALSECURITYNUMBER", "ai4privacy") == "SSN"
    assert map_label("EMAIL", "ai4privacy") == "EMAIL"
    assert map_label("nonexistent_type", "ai4privacy") is None


def test_map_label_nemotron():
    """Nemotron labels map correctly."""
    assert map_label("PER", "nemotron") == "PERSON"
    assert map_label("EMAIL_ADDRESS", "nemotron") == "EMAIL"
    assert map_label("SOCIAL_SECURITY_NUMBER", "nemotron") == "SSN"


def test_map_label_gretel():
    """Gretel labels (lowercase) map correctly."""
    assert map_label("person", "gretel") == "PERSON"
    assert map_label("ssn", "gretel") == "SSN"
    assert map_label("email", "gretel") == "EMAIL"


def test_char_to_id_ascii():
    """ASCII characters get consistent IDs."""
    assert char_to_id("a") == ord("a") + 2
    assert char_to_id("0") == ord("0") + 2
    assert char_to_id("@") == ord("@") + 2


def test_text_to_char_ids_padding():
    """Short text gets padded, long text gets truncated."""
    ids = text_to_char_ids("abc", max_len=10)
    assert len(ids) == 10
    assert ids[3:] == [0] * 7  # padded

    ids = text_to_char_ids("abcdefghijklmnop", max_len=5)
    assert len(ids) == 5


def test_vocab_size():
    """Vocab size accounts for PAD + UNK + 256 bytes."""
    assert VOCAB_SIZE == 258


# --- Entity-to-spans tests ---


def test_entities_to_spans_basic():
    """Correct offsets from entity-value format."""
    text = "My name is John Smith and my email is john@example.com"
    entities = [
        {"value": "John Smith", "type": "name"},
        {"value": "john@example.com", "type": "email"},
    ]
    spans = _entities_to_spans(text, entities)
    assert len(spans) == 2
    for span in spans:
        assert text[span["start"]:span["end"]] in ("John Smith", "john@example.com")
    # Check John Smith span
    john_span = [s for s in spans if s["label"] == "name"][0]
    assert text[john_span["start"]:john_span["end"]] == "John Smith"
    # Check email span
    email_span = [s for s in spans if s["label"] == "email"][0]
    assert text[email_span["start"]:email_span["end"]] == "john@example.com"


def test_entities_to_spans_not_found():
    """Graceful skip when entity not in text."""
    text = "Hello world"
    entities = [{"value": "nonexistent", "type": "name"}]
    spans = _entities_to_spans(text, entities)
    assert len(spans) == 0


def test_entities_to_spans_overlapping():
    """Longer match wins when entities overlap."""
    text = "Contact New York City office"
    entities = [
        {"value": "New York", "type": "location"},
        {"value": "New York City", "type": "location"},
    ]
    spans = _entities_to_spans(text, entities)
    # Should get one span for the longer match
    assert len(spans) == 1
    assert text[spans[0]["start"]:spans[0]["end"]] == "New York City"


def test_entities_to_spans_case_insensitive():
    """Case-insensitive fallback works."""
    text = "The patient is JOHN DOE"
    entities = [{"value": "john doe", "type": "name"}]
    spans = _entities_to_spans(text, entities)
    assert len(spans) == 1
    assert text[spans[0]["start"]:spans[0]["end"]] == "JOHN DOE"


def test_entities_to_spans_string_input():
    """Accepts string representation of entities list."""
    text = "SSN is 123-45-6789"
    entities = '[{"value": "123-45-6789", "type": "ssn"}]'
    spans = _entities_to_spans(text, entities)
    assert len(spans) == 1
    assert text[spans[0]["start"]:spans[0]["end"]] == "123-45-6789"


# --- New label map tests ---


def test_map_label_gretel_v1():
    """Gretel v1 labels map correctly to canonical names."""
    assert map_label("medical_record_number", "gretel_v1") == "MEDICAL_RECORD"
    assert map_label("device_identifier", "gretel_v1") == "DEVICE_ID"
    assert map_label("health_plan_beneficiary_number", "gretel_v1") == "INSURANCE_NUMBER"
    assert map_label("name", "gretel_v1") == "PERSON"
    assert map_label("email", "gretel_v1") == "EMAIL"


def test_map_label_ncbi():
    """NCBI Disease label maps to HEALTH_CONDITION."""
    assert map_label("Disease", "ncbi") == "HEALTH_CONDITION"
    assert map_label("unknown_type", "ncbi") is None


def test_map_label_maccrobat():
    """MACCROBAT disease labels map to HEALTH_CONDITION."""
    assert map_label("DISEASE_DISORDER", "maccrobat") == "HEALTH_CONDITION"
    assert map_label("SIGN_SYMPTOM", "maccrobat") == "HEALTH_CONDITION"
    assert map_label("MEDICATION", "maccrobat") is None


def test_map_label_synthetic():
    """Synthetic label map is identity for all canonical types."""
    from datafog_pii_ner.data.label_schema import ALL_ENTITY_TYPES

    for etype in ALL_ENTITY_TYPES:
        assert map_label(etype, "synthetic") == etype


# --- Synthetic data generation tests ---


def test_generate_example_valid_spans():
    """Synthetic examples have valid spans (text[start:end] matches value)."""
    rng = random.Random(42)
    for entity_type in SYNTHETIC_ENTITY_TYPES:
        example = generate_example(entity_type, rng)
        assert "text" in example
        assert "spans" in example
        assert len(example["spans"]) >= 1
        for span in example["spans"]:
            extracted = example["text"][span["start"]:span["end"]]
            assert len(extracted) > 0
            assert span["label"] == entity_type
            assert span["start"] >= 0
            assert span["end"] <= len(example["text"])


def test_generate_example_all_types():
    """Every synthetic entity type can generate examples."""
    rng = random.Random(123)
    for entity_type in SYNTHETIC_ENTITY_TYPES:
        example = generate_example(entity_type, rng)
        assert example["text"]
        assert example["spans"]


def test_generate_dataset_deterministic():
    """Same seed produces same output."""
    examples1 = generate_dataset("NATIONALITY", n=50, seed=42)
    examples2 = generate_dataset("NATIONALITY", n=50, seed=42)
    assert examples1 == examples2

    # Different seed produces different output
    examples3 = generate_dataset("NATIONALITY", n=50, seed=99)
    assert examples1 != examples3


def test_generate_dataset_correct_count():
    """generate_dataset returns exactly n examples."""
    examples = generate_dataset("RELIGION", n=100, seed=42)
    assert len(examples) == 100
