"""
models/heads/binary.py
----------------------
Binary deepfake detection head for DRIFT.

Responsibility:
    Map a feature vector produced by a FeatureExtractor to a single real-valued
    logit indicating whether the input image is real (0) or fake (1).
    Uses BCEWithLogitsLoss-compatible output (raw logit, no sigmoid).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional


class BinaryDetectionHead(nn.Module):
    """Fully-connected binary classification head for deepfake detection.

    Transforms a feature vector ``[B, feature_dim]`` into a logit
    ``[B, 1]``.  The head is intentionally lightweight — the heavy lifting
    is done by the FeatureExtractor + ADMBackbone pipeline.

    Architecture (default)::

        Linear(feature_dim → hidden_dim)
        LayerNorm(hidden_dim)
        GELU
        Dropout(dropout)
        Linear(hidden_dim → 1)

    Args:
        feature_dim: Input feature vector dimensionality.  Must match
            ``FeatureExtractor.feature_dim``.
        hidden_dim: Width of the intermediate projection layer.
        dropout: Dropout probability applied before the final linear layer.

    Example::

        head = BinaryDetectionHead(feature_dim=768, hidden_dim=256)
        logits = head(features)           # [B, 1]
        probs  = torch.sigmoid(logits)    # [B, 1], probability of being fake
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Apply standard weight initialisation."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            features: Feature tensor ``[B, feature_dim]`` from a
                ``FeatureExtractor``.

        Returns:
            Raw logit tensor ``[B, 1]``.  Pass through ``torch.sigmoid()``
            for probability, or use directly with ``BCEWithLogitsLoss``.

        Raises:
            ValueError: If the feature dimensionality does not match
                ``self.feature_dim``.
        """
        if features.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected feature_dim={self.feature_dim}, "
                f"got {features.shape[1]}."
            )
        return self.net(features)

    def predict_proba(self, features: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper that returns sigmoid probabilities.

        Args:
            features: Feature tensor ``[B, feature_dim]``.

        Returns:
            Probability tensor ``[B, 1]`` in range ``[0, 1]`` where values
            close to 1 indicate a fake image.
        """
        with torch.no_grad():
            logits = self.forward(features)
        return torch.sigmoid(logits)

    def __repr__(self) -> str:
        return (
            f"BinaryDetectionHead("
            f"feature_dim={self.feature_dim}, "
            f"hidden_dim={self.hidden_dim}"
            f")"
        )
