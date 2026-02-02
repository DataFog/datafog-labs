"""Tests for the Character CNN encoder."""

import torch
from datafog_pii_ner.model.char_cnn import CharCNNEncoder


def test_output_shape():
    """CharCNN produces correct output dimensions."""
    encoder = CharCNNEncoder(vocab_size=258, embed_dim=50, num_filters=(50, 50, 50))
    char_ids = torch.randint(0, 258, (2, 10, 20))  # batch=2, seq=10, chars=20
    output = encoder(char_ids)
    assert output.shape == (2, 10, 150)


def test_output_dim_property():
    """output_dim reflects total number of filters."""
    encoder = CharCNNEncoder(num_filters=(30, 40, 50))
    assert encoder.output_dim == 120


def test_padding_invariance():
    """Padding characters (ID=0) should not change output for non-padded tokens."""
    encoder = CharCNNEncoder()
    encoder.eval()

    # Same content, different padding
    char_ids_a = torch.tensor([[[3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]])
    char_ids_b = torch.tensor([[[3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]])

    with torch.no_grad():
        out_a = encoder(char_ids_a)
        out_b = encoder(char_ids_b)

    assert torch.allclose(out_a, out_b)


def test_different_inputs_different_outputs():
    """Different character patterns should produce different representations."""
    encoder = CharCNNEncoder()
    encoder.eval()

    # "123-45" pattern vs "abcdef" pattern
    char_ids_digits = torch.zeros(1, 1, 20, dtype=torch.long)
    char_ids_digits[0, 0, :6] = torch.tensor([51, 52, 53, 47, 54, 55])  # digits

    char_ids_alpha = torch.zeros(1, 1, 20, dtype=torch.long)
    char_ids_alpha[0, 0, :6] = torch.tensor([99, 100, 101, 102, 103, 104])  # letters

    with torch.no_grad():
        out_digits = encoder(char_ids_digits)
        out_alpha = encoder(char_ids_alpha)

    assert not torch.allclose(out_digits, out_alpha)
