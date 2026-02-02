import torch
import torch.nn as nn


class CharCNNEncoder(nn.Module):
    """Character-level CNN encoder for capturing orthographic patterns in tokens.

    Applies parallel Conv1D filters of different widths over character embeddings,
    then max-pools to produce a fixed-size representation per token.
    """

    def __init__(
        self,
        vocab_size: int = 256,
        embed_dim: int = 50,
        filter_widths: list[int] = (3, 4, 5),
        num_filters: list[int] = (50, 50, 50),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.char_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(embed_dim, nf, fw, padding=0)
                for fw, nf in zip(filter_widths, num_filters)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.output_dim = sum(num_filters)

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            char_ids: (batch_size, seq_len, max_char_len) integer tensor of character IDs.

        Returns:
            (batch_size, seq_len, output_dim) tensor of character-level features.
        """
        batch_size, seq_len, max_char_len = char_ids.shape

        # Flatten to (batch_size * seq_len, max_char_len) for embedding
        flat = char_ids.view(-1, max_char_len)

        # Embed: (batch * seq, max_char_len, embed_dim)
        embedded = self.char_embedding(flat)
        embedded = self.dropout(embedded)

        # Conv1D expects (batch, channels, length)
        embedded = embedded.transpose(1, 2)

        # Apply each filter and max-pool
        conv_outputs = []
        for conv in self.convs:
            # (batch * seq, num_filters, conv_out_len)
            c = torch.relu(conv(embedded))
            # Max-pool over the sequence dimension: (batch * seq, num_filters)
            c = c.max(dim=2).values
            conv_outputs.append(c)

        # Concatenate: (batch * seq, total_filters)
        output = torch.cat(conv_outputs, dim=1)
        output = self.dropout(output)

        # Reshape back: (batch_size, seq_len, output_dim)
        return output.view(batch_size, seq_len, self.output_dim)
