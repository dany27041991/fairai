"""VAN SingleSentence classifier matching AS-IS Thesis architecture."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoModel, BertConfig, BertModel

from fairai.infrastructure.torch_van.layers import (
    AttentionPooling,
    BiasProjectionLayer,
    DynamicDropout,
    MultiHeadAttentionPooling,
    ResidualBlock,
)

# all-MiniLM-L6-v2 compatible (matches AS-IS checkpoint tensor shapes)
_MINILM_L6_CONFIG = BertConfig(
    vocab_size=30522,
    hidden_size=384,
    num_hidden_layers=6,
    num_attention_heads=12,
    intermediate_size=1536,
    max_position_embeddings=512,
    type_vocab_size=2,
)


@dataclass(frozen=True, slots=True)
class VanArchConfig:
    backbone: str = "sentence-transformers/all-MiniLM-L6-v2"
    num_labels: int = 3
    dropout: float = 0.3
    dropout_reg: float = 0.5
    num_layers: int = 2
    hidden_size: int = 512
    att_dropout: float = 0.1
    multihead_num_heads: int = 1
    bias_dim: int = 256
    bias_num_layers: int = 1
    bias_dropout: float = 0.0
    bias_lambda: float = 1.0
    # Ablation: False → nn.Identity (no BiasProjectionLayer params)
    use_bias_projection: bool = True


class VanClassifier(nn.Module):
    """Encoder + attention pool + bias projection + residual MLP head."""

    def __init__(self, cfg: VanArchConfig, *, init_backbone_weights: bool = False) -> None:
        super().__init__()
        self.cfg = cfg
        if init_backbone_weights:
            self.base_model = AutoModel.from_pretrained(cfg.backbone)
        else:
            # Offline-friendly: architecture from known MiniLM config;
            # weights come entirely from the FAIRAI .pt state_dict.
            if "MiniLM-L6" in cfg.backbone or "all-MiniLM-L6" in cfg.backbone:
                self.base_model = BertModel(_MINILM_L6_CONFIG)
            else:
                self.base_model = AutoModel.from_pretrained(cfg.backbone)

        hidden = self.base_model.config.hidden_size
        if cfg.multihead_num_heads > 1:
            self.attention_pooling: nn.Module = MultiHeadAttentionPooling(
                hidden_size=hidden,
                num_heads=cfg.multihead_num_heads,
                att_dropout=cfg.att_dropout,
            )
        else:
            self.attention_pooling = AttentionPooling(
                hidden_size=hidden, att_dropout=cfg.att_dropout
            )

        self.layer_norm = nn.LayerNorm(hidden)
        self.dropout = DynamicDropout(cfg.dropout)
        self.dropout_reg = DynamicDropout(cfg.dropout_reg)
        if cfg.use_bias_projection:
            self.bias_layer: nn.Module = BiasProjectionLayer(
                in_dim=hidden,
                hidden_dim=cfg.bias_dim,
                num_layers=cfg.bias_num_layers,
                dropout=cfg.bias_dropout,
                lambda_factor=cfg.bias_lambda,
            )
        else:
            self.bias_layer = nn.Identity()

        blocks: list[nn.Module] = []
        in_size = hidden
        for _ in range(cfg.num_layers):
            blocks.append(ResidualBlock(in_size, cfg.hidden_size, cfg.dropout))
            in_size = cfg.hidden_size
        blocks.append(nn.Linear(cfg.hidden_size, cfg.num_labels))
        self.fc = nn.Sequential(*blocks)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.attention_pooling(outputs.last_hidden_state, mask=attention_mask)
        pooled = self.layer_norm(pooled)
        pooled = self.dropout(pooled)
        pooled = self.dropout_reg(pooled)
        pooled = self.bias_layer(pooled)
        logits = self.fc(pooled)
        return logits, pooled
