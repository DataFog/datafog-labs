import torch
import torch.nn as nn


class GatingFusion(nn.Module):
    """Sigmoid gating layer that dynamically fuses character and contextual features.

    For each token, learns a gate value g in [0, 1] that controls the mix:
        output = g * char_features + (1 - g) * context_features

    High g -> trust character patterns (structured PII like SSN, email).
    Low g  -> trust contextual meaning (soft PII like names, addresses).
    """

    def __init__(self, char_dim: int, context_dim: int, dropout: float = 0.1):
        super().__init__()
        self.char_proj = nn.Linear(char_dim, context_dim)
        self.gate = nn.Linear(context_dim * 2, context_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = context_dim

    def forward(
        self, char_features: torch.Tensor, context_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            char_features: (batch_size, seq_len, char_dim) from CharCNNEncoder.
            context_features: (batch_size, seq_len, context_dim) from DeBERTa.

        Returns:
            (batch_size, seq_len, context_dim) fused representation.
        """
        # Project char features to match context dimensionality
        char_proj = torch.relu(self.char_proj(char_features))
        char_proj = self.dropout(char_proj)

        # Compute gate from concatenated features
        combined = torch.cat([char_proj, context_features], dim=-1)
        g = torch.sigmoid(self.gate(combined))

        # Fuse: gate controls character vs context contribution
        return g * char_proj + (1 - g) * context_features
