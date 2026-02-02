import torch
import torch.nn as nn
from torchcrf import CRF as TorchCRF


class CRFHead(nn.Module):
    """Token classification head with CRF decoding.

    Linear projection to label space followed by a CRF layer that enforces
    valid BIO tag sequences during both training (loss) and inference (decoding).
    """

    def __init__(self, hidden_dim: int, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Linear(hidden_dim, num_labels)
        self.crf = TorchCRF(num_labels, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.num_labels = num_labels

    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_dim) fused features.
            labels: (batch_size, seq_len) BIO tag IDs. None at inference.
            attention_mask: (batch_size, seq_len) boolean mask.

        Returns:
            dict with 'loss' (training) and/or 'predictions' (inference).
        """
        emissions = self.classifier(self.dropout(hidden_states))

        # Convert attention mask to boolean for CRF
        mask = attention_mask.bool() if attention_mask is not None else None

        result = {}

        if labels is not None:
            # CRF expects labels without ignore_index padding.
            # Replace -100 (HF ignore index) with 0 — masked positions are excluded by mask.
            clamped_labels = labels.clamp(min=0)
            # CRF returns log-likelihood; negate for loss.
            log_likelihood = self.crf(emissions, clamped_labels, mask=mask, reduction="mean")
            result["loss"] = -log_likelihood

        # Always decode for predictions
        tag_sequences = self.crf.decode(emissions, mask=mask)
        # Pad sequences to seq_len (decode returns variable-length lists)
        batch_size, seq_len = emissions.shape[:2]
        predictions = torch.full(
            (batch_size, seq_len), 0, dtype=torch.long, device=emissions.device
        )
        for i, tags in enumerate(tag_sequences):
            predictions[i, : len(tags)] = torch.tensor(tags, device=emissions.device)
        result["predictions"] = predictions

        return result
