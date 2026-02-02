"""Custom data collator that handles the extra char_ids tensor."""

import torch
from dataclasses import dataclass
from transformers import DataCollatorForTokenClassification


@dataclass
class PiiDataCollator(DataCollatorForTokenClassification):
    """Extends HF's token classification collator to also pad char_ids.

    char_ids is a 3D tensor (batch, seq_len, max_char_len) that needs
    custom padding logic since the default collator only handles 1D/2D tensors.
    """

    max_char_len: int = 20

    def __call__(self, features: list[dict]) -> dict:
        # Extract char_ids before passing to parent collator
        char_ids_list = [f.pop("char_ids") for f in features]

        # Let the parent handle input_ids, attention_mask, labels
        batch = super().__call__(features)

        # Pad char_ids to match the padded sequence length
        padded_seq_len = batch["input_ids"].shape[1]
        padded_char_ids = []
        for char_ids in char_ids_list:
            # char_ids is a list of lists: (seq_len, max_char_len)
            current_len = len(char_ids)
            padding_needed = padded_seq_len - current_len
            if padding_needed > 0:
                char_ids = char_ids + [[0] * self.max_char_len] * padding_needed
            elif padding_needed < 0:
                char_ids = char_ids[:padded_seq_len]
            padded_char_ids.append(char_ids)

        batch["char_ids"] = torch.tensor(padded_char_ids, dtype=torch.long)
        return batch
