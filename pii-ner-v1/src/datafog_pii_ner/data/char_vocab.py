"""Byte-level character vocabulary for the CharCNN encoder.

Simple fixed mapping: raw byte values (0-255) to integer IDs.
No learning required — this is a lookup table.
"""

PAD_ID = 0
UNK_ID = 1
OFFSET = 2  # Reserve 0 for padding, 1 for unknown


def char_to_id(char: str) -> int:
    """Convert a single character to its vocabulary ID."""
    byte_val = ord(char)
    if byte_val > 255:
        return UNK_ID
    return byte_val + OFFSET


def text_to_char_ids(text: str, max_len: int = 20) -> list[int]:
    """Convert a string to a list of character IDs, padded/truncated to max_len."""
    ids = [char_to_id(c) for c in text[:max_len]]
    # Pad with zeros
    ids.extend([PAD_ID] * (max_len - len(ids)))
    return ids


VOCAB_SIZE = 256 + OFFSET  # 258 total: PAD + UNK + 256 byte values
