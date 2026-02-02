from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import DebertaV2Model, DebertaV2PreTrainedModel, PretrainedConfig
from transformers.utils import ModelOutput

from .char_cnn import CharCNNEncoder
from .crf import CRFHead
from .gating_fusion import GatingFusion


@dataclass
class PiiNerOutput(ModelOutput):
    loss: torch.Tensor | None = None
    predictions: torch.Tensor | None = None
    hidden_states: torch.Tensor | None = None


class PiiNerConfig(PretrainedConfig):
    model_type = "pii_ner"

    def __init__(
        self,
        backbone: str = "microsoft/deberta-v3-xsmall",
        num_labels: int = 3,
        char_vocab_size: int = 256,
        char_embed_dim: int = 50,
        char_cnn_filters: list[int] = (50, 50, 50),
        char_cnn_widths: list[int] = (3, 4, 5),
        max_char_len: int = 20,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backbone = backbone
        self.num_labels = num_labels
        self.char_vocab_size = char_vocab_size
        self.char_embed_dim = char_embed_dim
        self.char_cnn_filters = list(char_cnn_filters)
        self.char_cnn_widths = list(char_cnn_widths)
        self.max_char_len = max_char_len
        self.dropout = dropout


class PiiNerModel(DebertaV2PreTrainedModel):
    """Hybrid character-contextual PII detection model.

    Combines DeBERTa-v3-xsmall (contextual encoder) with a CharCNN (pattern encoder)
    via a learned gating fusion, followed by a CRF output head for BIO tagging.
    """

    config_class = PiiNerConfig

    def __init__(self, config: PiiNerConfig):
        # Initialize the backbone config from pretrained
        super().__init__(config)

        self.deberta = DebertaV2Model.from_pretrained(config.backbone)
        context_dim = self.deberta.config.hidden_size

        self.char_cnn = CharCNNEncoder(
            vocab_size=config.char_vocab_size,
            embed_dim=config.char_embed_dim,
            filter_widths=config.char_cnn_widths,
            num_filters=config.char_cnn_filters,
            dropout=config.dropout,
        )

        self.fusion = GatingFusion(
            char_dim=self.char_cnn.output_dim,
            context_dim=context_dim,
            dropout=config.dropout,
        )

        self.crf_head = CRFHead(
            hidden_dim=context_dim,
            num_labels=config.num_labels,
            dropout=config.dropout,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        char_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> PiiNerOutput:
        """
        Args:
            input_ids: (batch, seq_len) subword token IDs.
            attention_mask: (batch, seq_len) padding mask.
            char_ids: (batch, seq_len, max_char_len) character IDs per token.
            labels: (batch, seq_len) BIO tag IDs for training.

        Returns:
            PiiNerOutput with loss (training) and predictions (always).
        """
        # Contextual encoder
        deberta_output = self.deberta(
            input_ids=input_ids, attention_mask=attention_mask
        )
        context_features = deberta_output.last_hidden_state

        # Character encoder + fusion
        if char_ids is not None:
            char_features = self.char_cnn(char_ids)
            fused = self.fusion(char_features, context_features)
        else:
            # Fallback: context only (e.g., if char_ids not available)
            fused = context_features

        # Ensure consistent dtype (DeBERTa may output FP16 while head is FP32)
        fused = fused.float()

        # CRF classification
        crf_output = self.crf_head(
            hidden_states=fused,
            labels=labels,
            attention_mask=attention_mask,
        )

        return PiiNerOutput(
            loss=crf_output.get("loss"),
            predictions=crf_output["predictions"],
            hidden_states=fused,
        )
