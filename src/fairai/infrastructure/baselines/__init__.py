"""Classical baselines (frozen encoder + linear heads)."""

from fairai.infrastructure.baselines.frozen_encoder import (
    FrozenEncoder,
    encode_texts,
    mean_pool,
)

__all__ = ["FrozenEncoder", "encode_texts", "mean_pool"]
