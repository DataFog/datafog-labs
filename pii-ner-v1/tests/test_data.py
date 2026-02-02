"""Tests for the data pipeline components."""

from datafog_pii_ner.data.char_vocab import VOCAB_SIZE, char_to_id, text_to_char_ids
from datafog_pii_ner.data.label_schema import (
    BIO_LABELS,
    LABEL_TO_ID,
    NUM_LABELS,
    map_label,
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
