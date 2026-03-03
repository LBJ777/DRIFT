"""
training/trainer.py
-------------------
Generic training loop for DRIFT experiments.

Responsibility:
    Orchestrate the full training pipeline:

    1. For each batch, run the ADMBackbone inversion to obtain (x_T, intermediates).
    2. Feed the inversion output through a FeatureExtractor.
    3. Pass the feature vector through a detection/attribution head.
    4. Compute loss, back-propagate, and update weights.
    5. Track metrics and emit logs.

The Trainer is deliberately head-agnostic: it works with any
``nn.Module`` head and any ``FeatureExtractor`` sub-class.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from ..models.backbone.adm_wrapper import ADMBackbone
from ..models.features.base import FeatureExtractor
from .losses import BinaryDetectionLoss

logger = logging.getLogger(__name__)


class DRIFTTrainer:
    """Coordinates the DRIFT training loop across backbone, feature extractor,
    and classification head.

    The trainer handles:

    * ADM DDIM inversion of each batch of images.
    * Feature extraction from the inversion outputs.
    * Loss computation and parameter optimisation.
    * Epoch-level metrics aggregation and validation.
    * Checkpoint saving.

    Args:
        backbone: A configured ``ADMBackbone`` instance. In mock mode the
            inversion returns random Gaussian noise (useful for dry runs).
        feature_extractor: A ``FeatureExtractor`` sub-class instance that
            converts (x_T, intermediates) → feature vectors.
        head: An ``nn.Module`` that maps feature vectors to logits.
        config: Configuration dictionary.  Expected keys (all optional with
            sensible defaults):

            * ``lr``            (float)  — learning rate (default 1e-4)
            * ``optim``         (str)    — ``"adam"`` or ``"sgd"``
            * ``weight_decay``  (float)  — L2 regularisation (default 1e-4)
            * ``grad_clip``     (float)  — gradient clipping norm (default 1.0)
            * ``checkpoint_dir``(str)    — where to save checkpoints
            * ``device``        (str)    — ``"cuda"`` or ``"cpu"``
            * ``amp``           (bool)   — use automatic mixed precision

    Example::

        trainer = DRIFTTrainer(
            backbone=ADMBackbone("mock", device="cpu"),
            feature_extractor=MyF1Extractor(),
            head=BinaryDetectionHead(feature_dim=128),
            config={"lr": 1e-4, "device": "cpu"},
        )
        history = trainer.train(train_loader, val_loader, num_epochs=30)
    """

    def __init__(
        self,
        backbone: ADMBackbone,
        feature_extractor: FeatureExtractor,
        head: nn.Module,
        config: Dict[str, Any],
    ) -> None:
        self.backbone = backbone
        self.feature_extractor = feature_extractor
        self.head = head
        self.config = config

        self.device = torch.device(config.get("device", "cuda"))
        self.head = self.head.to(self.device)

        self.loss_fn = BinaryDetectionLoss(
            pos_weight=config.get("pos_weight", None)
        )
        self.optimizer: Optimizer = self._build_optimizer()
        self.scheduler: Optional[_LRScheduler] = self._build_scheduler()

        self.use_amp: bool = config.get("amp", False)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.grad_clip: float = float(config.get("grad_clip", 1.0))

        self.checkpoint_dir: str = config.get("checkpoint_dir", "./checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # History tracking
        self._history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_auc": [],
        }

    # ------------------------------------------------------------------
    # Builder helpers
    # ------------------------------------------------------------------

    def _build_optimizer(self) -> Optimizer:
        """Construct the optimiser specified in *self.config*.

        Returns:
            A configured ``torch.optim.Optimizer``.

        Raises:
            ValueError: If ``config["optim"]`` is not ``"adam"`` or ``"sgd"``.
        """
        lr = float(self.config.get("lr", 1e-4))
        wd = float(self.config.get("weight_decay", 1e-4))
        beta1 = float(self.config.get("beta1", 0.9))
        optim_name = self.config.get("optim", "adam").lower()

        params = self.head.parameters()

        if optim_name == "adam":
            return torch.optim.AdamW(
                params, lr=lr, betas=(beta1, 0.999), weight_decay=wd
            )
        elif optim_name == "sgd":
            return torch.optim.SGD(
                params, lr=lr, momentum=0.9, weight_decay=wd
            )
        else:
            raise ValueError(
                f"Unsupported optimiser '{optim_name}'. Choose 'adam' or 'sgd'."
            )

    def _build_scheduler(self) -> Optional[_LRScheduler]:
        """Construct an LR scheduler if specified in *self.config*.

        Returns:
            A scheduler instance, or ``None`` if ``config["lr_scheduler"]``
            is ``"none"`` or absent.
        """
        scheduler_name = self.config.get("lr_scheduler", "cosine").lower()
        num_epochs = int(self.config.get("num_epochs", 30))

        if scheduler_name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=num_epochs
            )
        elif scheduler_name == "step":
            step_size = int(self.config.get("lr_step_size", 5))
            gamma = float(self.config.get("lr_gamma", 0.1))
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size, gamma=gamma
            )
        elif scheduler_name in ("none", ""):
            return None
        else:
            logger.warning(
                "Unknown lr_scheduler '%s'. Disabling scheduler.", scheduler_name
            )
            return None

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    def _forward_batch(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """Run backbone inversion + feature extraction for a single batch.

        Args:
            images: Image batch ``[B, 3, H, W]`` on ``self.device``.

        Returns:
            Tuple ``(features, intermediates)``:

            * ``features`` — ``[B, feature_dim]`` tensor.
            * ``intermediates`` — list of intermediate tensors or ``None``.
        """
        need_intermediates = self.config.get("return_intermediates", False)
        x_T, intermediates = self.backbone.invert(
            images, return_intermediates=need_intermediates
        )
        features = self.feature_extractor.extract(x_T, intermediates)
        return features, intermediates

    # ------------------------------------------------------------------
    # Train / validate
    # ------------------------------------------------------------------

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Run a single training epoch.

        Iterates over the DataLoader, performs forward passes through the
        full pipeline (backbone → feature extractor → head), computes the
        loss, and updates the head parameters.

        Args:
            dataloader: A DataLoader returning ``(image, label)`` tuples
                where ``label`` is a float tensor (0 real, 1 fake).

        Returns:
            Dictionary with key ``"train_loss"`` containing the mean loss
            for this epoch.
        """
        self.head.train()
        total_loss = 0.0
        n_batches = 0

        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(self.device)
            labels = labels.to(self.device).float()

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                features, _ = self._forward_batch(images)
                logits = self.head(features)
                loss = self.loss_fn(logits, labels)

            self.scaler.scale(loss).backward()

            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.head.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            n_batches += 1

            log_every = int(self.config.get("log_every", 50))
            if (batch_idx + 1) % log_every == 0:
                logger.info(
                    "  [batch %d/%d] loss=%.4f",
                    batch_idx + 1,
                    len(dataloader),
                    loss.item(),
                )

        return {"train_loss": total_loss / max(n_batches, 1)}

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Run validation and compute loss + AUC.

        Args:
            dataloader: A DataLoader returning ``(image, label)`` tuples.

        Returns:
            Dictionary with keys ``"val_loss"`` and ``"val_auc"``.
        """
        from ..evaluation.metrics import compute_auc

        self.head.eval()
        total_loss = 0.0
        all_labels: List[int] = []
        all_scores: List[float] = []

        for images, labels in dataloader:
            images = images.to(self.device)
            labels_dev = labels.to(self.device).float()

            features, _ = self._forward_batch(images)
            logits = self.head(features)
            loss = self.loss_fn(logits, labels_dev)

            total_loss += loss.item()
            scores = torch.sigmoid(logits.squeeze(1)).cpu().tolist()
            all_scores.extend(scores if isinstance(scores, list) else [scores])
            lbls = labels.cpu().tolist()
            all_labels.extend(lbls if isinstance(lbls, list) else [lbls])

        val_auc = compute_auc(all_labels, all_scores)
        return {
            "val_loss": total_loss / max(len(dataloader), 1),
            "val_auc": val_auc,
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        num_epochs: Optional[int] = None,
    ) -> Dict[str, List[float]]:
        """Run the full training loop.

        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.  If ``None``,
                validation is skipped.
            num_epochs: Number of training epochs.  Falls back to
                ``self.config["num_epochs"]`` (default 30).

        Returns:
            Training history dictionary with keys ``"train_loss"``,
            ``"val_loss"``, and ``"val_auc"``.
        """
        if num_epochs is None:
            num_epochs = int(self.config.get("num_epochs", 30))

        save_every = int(self.config.get("save_every", 5))
        val_every = int(self.config.get("val_every", 1))

        logger.info(
            "Starting training: epochs=%d, device=%s", num_epochs, self.device
        )

        for epoch in range(1, num_epochs + 1):
            t0 = time.time()
            train_metrics = self.train_epoch(train_loader)
            elapsed = time.time() - t0

            self._history["train_loss"].append(train_metrics["train_loss"])

            log_msg = (
                f"Epoch {epoch}/{num_epochs} | "
                f"train_loss={train_metrics['train_loss']:.4f} | "
                f"elapsed={elapsed:.1f}s"
            )

            if val_loader is not None and epoch % val_every == 0:
                val_metrics = self.validate(val_loader)
                self._history["val_loss"].append(val_metrics["val_loss"])
                self._history["val_auc"].append(val_metrics["val_auc"])
                log_msg += (
                    f" | val_loss={val_metrics['val_loss']:.4f} "
                    f"| val_auc={val_metrics['val_auc']:.4f}"
                )

            logger.info(log_msg)

            if self.scheduler is not None:
                self.scheduler.step()

            if epoch % save_every == 0:
                self.save_checkpoint(epoch)

        logger.info("Training complete.")
        return dict(self._history)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, epoch: int, tag: str = "") -> str:
        """Save the head's state dict and optimiser state to disk.

        Args:
            epoch: Current epoch number (embedded in the filename).
            tag: Optional extra string appended to the checkpoint filename.

        Returns:
            Absolute path to the saved checkpoint file.
        """
        fname = f"drift_epoch{epoch:04d}{('_' + tag) if tag else ''}.pth"
        path = os.path.join(self.checkpoint_dir, fname)
        torch.save(
            {
                "epoch": epoch,
                "head_state_dict": self.head.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "history": self._history,
            },
            path,
        )
        logger.info("Checkpoint saved to '%s'.", path)
        return path

    def load_checkpoint(self, path: str) -> int:
        """Restore head and optimiser state from a checkpoint file.

        Args:
            path: Path to a ``.pth`` file previously saved by
                ``save_checkpoint()``.

        Returns:
            The epoch number stored in the checkpoint.
        """
        state = torch.load(path, map_location=self.device)
        self.head.load_state_dict(state["head_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self._history = state.get("history", self._history)
        epoch = state.get("epoch", 0)
        logger.info("Checkpoint loaded from '%s' (epoch %d).", path, epoch)
        return epoch

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def history(self) -> Dict[str, List[float]]:
        """Training history dictionary (updated in-place during training).

        Returns:
            Dict with keys ``"train_loss"``, ``"val_loss"``, ``"val_auc"``.
        """
        return dict(self._history)
