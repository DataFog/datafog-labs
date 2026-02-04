"""Tests for the PII inference pipeline."""

import pytest
import torch
from transformers import AutoTokenizer

from datafog_pii_ner.data.label_schema import NUM_LABELS
from datafog_pii_ner.inference.pipeline import PiiEntity, PiiPipeline, _make_entity
from datafog_pii_ner.model.pii_model import PiiNerConfig, PiiNerModel


@pytest.fixture(scope="module")
def pipeline():
    """Create a pipeline with a freshly initialized (untrained) model."""
    config = PiiNerConfig(
        backbone="microsoft/deberta-v3-xsmall",
        num_labels=NUM_LABELS,
        char_vocab_size=258,
    )
    model = PiiNerModel(config)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-xsmall")
    return PiiPipeline(model=model, tokenizer=tokenizer, device="cpu")


def test_pipeline_returns_list(pipeline):
    """Pipeline returns a list of PiiEntity for single string input."""
    result = pipeline("Hello world")
    assert isinstance(result, list)
    assert all(isinstance(e, PiiEntity) for e in result)


def test_pipeline_empty_string(pipeline):
    """Pipeline handles empty string input."""
    assert pipeline("") == []
    assert pipeline("   ") == []


def test_pipeline_batch_input(pipeline):
    """Pipeline handles list of strings."""
    results = pipeline(["Hello", "World"])
    assert isinstance(results, list)
    assert len(results) == 2
    assert all(isinstance(r, list) for r in results)


def test_entity_offsets_valid(pipeline):
    """Detected entity offsets should be valid character positions."""
    text = "My name is John Smith and my email is test@example.com"
    entities = pipeline(text)
    for e in entities:
        assert 0 <= e.start < e.end <= len(text)
        assert text[e.start : e.end] == e.text


def test_entity_to_dict():
    """PiiEntity.to_dict() returns expected format."""
    entity = PiiEntity(text="123-45-6789", label="SSN", start=10, end=21, tier=1)
    d = entity.to_dict()
    assert d == {
        "text": "123-45-6789",
        "label": "SSN",
        "start": 10,
        "end": 21,
        "tier": 1,
    }


def test_make_entity_character_offsets():
    """_make_entity correctly maps word indices to character offsets."""
    text = "My SSN is 123-45-6789 ok"
    word_offsets = [(0, 2), (3, 6), (7, 9), (10, 21), (22, 24)]
    entity = _make_entity("SSN", 3, 3, word_offsets, text)
    assert entity.text == "123-45-6789"
    assert entity.start == 10
    assert entity.end == 21
    assert entity.label == "SSN"
    assert entity.tier == 1


def test_make_entity_multi_word():
    """_make_entity handles multi-word entities."""
    text = "I live at 123 Main Street today"
    word_offsets = [(0, 1), (2, 6), (7, 9), (10, 13), (14, 18), (19, 25), (26, 31)]
    entity = _make_entity("STREET_ADDRESS", 3, 5, word_offsets, text)
    assert entity.text == "123 Main Street"
    assert entity.start == 10
    assert entity.end == 25


def test_decode_entities_bio_sequence():
    """Test BIO tag decoding logic with a controlled prediction sequence."""
    from datafog_pii_ner.data.label_schema import LABEL_TO_ID

    text = "Call John Smith at john@acme.com please"
    words = ["Call", "John", "Smith", "at", "john@acme.com", "please"]
    word_offsets = [(0, 4), (5, 9), (10, 15), (16, 18), (19, 32), (33, 39)]

    # Simulate word_ids from tokenizer (simplified: 1 subword per word + CLS/SEP)
    word_ids = [None, 0, 1, 2, 3, 4, 5, None]

    # Build prediction IDs: O, B-PERSON, I-PERSON, O, B-EMAIL, O
    pred_ids = [
        0,  # [CLS] - ignored
        0,  # Call -> O
        LABEL_TO_ID["B-PERSON"],  # John -> B-PERSON
        LABEL_TO_ID["I-PERSON"],  # Smith -> I-PERSON
        0,  # at -> O
        LABEL_TO_ID["B-EMAIL"],  # john@acme.com -> B-EMAIL
        0,  # please -> O
        0,  # [SEP] - ignored
    ]

    entities = PiiPipeline._decode_entities(pred_ids, word_ids, words, word_offsets, text)

    assert len(entities) == 2

    assert entities[0].label == "PERSON"
    assert entities[0].text == "John Smith"
    assert entities[0].start == 5
    assert entities[0].end == 15

    assert entities[1].label == "EMAIL"
    assert entities[1].text == "john@acme.com"
    assert entities[1].start == 19
    assert entities[1].end == 32


def test_pipeline_long_text(pipeline):
    """Pipeline handles text longer than max_seq_len via truncation."""
    long_text = "word " * 500  # ~500 words, well over 256 subwords
    result = pipeline(long_text)
    assert isinstance(result, list)
