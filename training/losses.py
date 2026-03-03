"""
training/losses.py
------------------
Loss functions used across DRIFT training phases.

Responsibility:
    Centralise all loss computation so that the Trainer can swap between
    different objectives without modifying training loop logic.

Available losses:

    ``BinaryDetectionLoss`` — BCE-based loss for deepfake binary classification.
    ``AttributionLoss``     — Cross-entropy loss for generator attribution.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class BinaryDetectionLoss(nn.Module):
    """Binary cross-entropy loss for deepfake detection.

    Wraps ``nn.BCEWithLogitsLoss`` for use with the ``BinaryDetectionHead``
    which outputs raw logits (no sigmoid).  Supports optional per-sample
    weighting to handle class imbalance.

    Args:
        pos_weight: Scalar weight for the positive (fake) class.  Pass a
            value > 1 to up-weight fake samples when the dataset is
            real-heavy.  ``None`` applies equal weights (default).
        reduction: Reduction method: ``"mean"`` (default) | ``"sum"`` |
            ``"none"``.

    Example::

        loss_fn = BinaryDetectionLoss(pos_weight=2.0)
        loss = loss_fn(logits, labels)   # logits: [B,1], labels: [B]
    """

    def __init__(
        self,
        pos_weight: Optional[float] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        pw = torch.tensor([pos_weight]) if pos_weight is not None else None
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pw, reduction=reduction)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute BCE loss.

        Args:
            logits: Raw logit tensor ``[B, 1]`` or ``[B]`` from
                ``BinaryDetectionHead``.
            labels: Binary label tensor ``[B]`` of type ``float``.
                0 = real, 1 = fake.

        Returns:
            Scalar loss tensor.
        """
        logits = logits.squeeze(1) if logits.ndim == 2 else logits
        labels = labels.float()
        return self.bce(logits, labels)


class AttributionLoss(nn.Module):
    """Cross-entropy loss for generator attribution.

    Used during optional end-to-end fine-tuning of the attribution head.

    Args:
        num_generators: Number of generator classes (output dim).
        label_smoothing: Label smoothing factor in ``[0, 1)``.
        reduction: Reduction method: ``"mean"`` (default) | ``"sum"``.

    Example::

        loss_fn = AttributionLoss(num_generators=14)
        loss = loss_fn(logits, generator_ids)
    """

    def __init__(
        self,
        num_generators: int,
        label_smoothing: float = 0.1,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.num_generators = num_generators
        self.ce = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            reduction=reduction,
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cross-entropy attribution loss.

        Args:
            logits: Logit tensor ``[B, num_generators]`` from the attribution
                head.
            targets: Integer class-index tensor ``[B]``.

        Returns:
            Scalar loss tensor.
        """
        return self.ce(logits, targets)


class CombinedLoss(nn.Module):
    """Weighted sum of binary detection loss and attribution loss.

    Useful when training jointly on both objectives.

    Args:
        detection_weight: Weight for the binary detection component.
        attribution_weight: Weight for the attribution component.
        pos_weight: Positive-class weight passed to ``BinaryDetectionLoss``.
        num_generators: Number of generator classes.
        label_smoothing: Label smoothing for ``AttributionLoss``.

    Example::

        loss_fn = CombinedLoss(num_generators=14)
        loss = loss_fn(det_logits, labels, attr_logits, generator_ids)
    """

    def __init__(
        self,
        detection_weight: float = 1.0,
        attribution_weight: float = 1.0,
        pos_weight: Optional[float] = None,
        num_generators: int = 14,
        label_smoothing: float = 0.1,
    ) -> None:
        super().__init__()
        self.detection_weight = detection_weight
        self.attribution_weight = attribution_weight
        self.detection_loss = BinaryDetectionLoss(pos_weight=pos_weight)
        self.attribution_loss = AttributionLoss(
            num_generators=num_generators,
            label_smoothing=label_smoothing,
        )

    def forward(
        self,
        det_logits: torch.Tensor,
        labels: torch.Tensor,
        attr_logits: torch.Tensor,
        generator_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined loss.

        Args:
            det_logits: Binary detection logits ``[B, 1]``.
            labels: Binary labels ``[B]``.
            attr_logits: Attribution logits ``[B, num_generators]``.
            generator_ids: Generator class indices ``[B]``.

        Returns:
            Scalar combined loss tensor.
        """
        det_loss = self.detection_loss(det_logits, labels)
        attr_loss = self.attribution_loss(attr_logits, generator_ids)
        return (
            self.detection_weight * det_loss
            + self.attribution_weight * attr_loss
        )
