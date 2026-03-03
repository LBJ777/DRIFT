"""
utils/checkpointing.py
----------------------
Model checkpoint save / load utilities for DRIFT.

Responsibility:
    Provide lightweight, format-agnostic helpers for persisting and restoring
    PyTorch model state dicts, optimiser states, and arbitrary metadata.
    Also exposes a ``CheckpointManager`` class that tracks the best model
    by a scalar metric and enforces a rolling keep-last-N policy.
"""

from __future__ import annotations

import glob
import logging
import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: str,
    model: nn.Module,
    epoch: int,
    optimizer: Optional[Any] = None,
    scheduler: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Save a model checkpoint to disk.

    Args:
        path: Destination file path (should end in ``.pth``).
        model: The ``nn.Module`` whose ``state_dict`` is saved.
        epoch: Current training epoch (stored in the checkpoint for reference).
        optimizer: Optional optimiser; its ``state_dict`` is included if
            provided.
        scheduler: Optional LR scheduler; its ``state_dict`` is included if
            provided.
        metadata: Optional dict of extra key-value pairs (e.g. val AUC,
            config hash) that are stored alongside the model weights.

    Returns:
        The absolute path to the saved checkpoint file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    state: Dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
    }

    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    if metadata is not None:
        state["metadata"] = metadata

    torch.save(state, path)
    logger.info("Checkpoint saved: '%s' (epoch %d).", path, epoch)
    return os.path.abspath(path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[Any] = None,
    scheduler: Optional[Any] = None,
    device: Optional[str] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """Load a checkpoint from disk and restore model (and optionally optimiser)
    state.

    Args:
        path: Path to the ``.pth`` checkpoint file.
        model: The ``nn.Module`` into which weights are loaded in-place.
        optimizer: If provided, the optimiser state is also restored.
        scheduler: If provided, the scheduler state is also restored.
        device: Device string for ``map_location`` (e.g. ``"cpu"``).
            ``None`` auto-maps to the model's current device.
        strict: Passed to ``model.load_state_dict()``.  ``False`` allows
            partial loading (useful for transfer learning).

    Returns:
        The full checkpoint dictionary (includes ``epoch``, ``metadata``, etc.).

    Raises:
        FileNotFoundError: If *path* does not exist on disk.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: '{path}'.")

    map_location = device if device else "cpu"
    state = torch.load(path, map_location=map_location)

    model.load_state_dict(state["model_state_dict"], strict=strict)

    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in state:
        scheduler.load_state_dict(state["scheduler_state_dict"])

    epoch = state.get("epoch", 0)
    logger.info("Checkpoint loaded from '%s' (epoch %d).", path, epoch)
    return state


# ---------------------------------------------------------------------------
# CheckpointManager
# ---------------------------------------------------------------------------

class CheckpointManager:
    """Manages checkpoint saving with best-model tracking and rolling cleanup.

    Saves a checkpoint every call to ``save()`` and additionally saves a
    dedicated ``best.pth`` whenever the monitored metric improves.  Keeps
    at most *keep_last* regular checkpoints on disk.

    Args:
        checkpoint_dir: Directory where checkpoints are stored.
        metric_name: Name of the scalar metric used to determine the
            "best" model (e.g. ``"val_auc"``).
        mode: ``"max"`` if higher metric is better (default, e.g. AUC),
            ``"min"`` if lower is better (e.g. loss).
        keep_last: Number of most-recent regular checkpoints to retain.
            Older ones are deleted automatically.

    Example::

        manager = CheckpointManager("./checkpoints", metric_name="val_auc")
        for epoch in range(30):
            ...
            is_best = manager.save(
                model, epoch, optimizer, metrics={"val_auc": 0.95}
            )
    """

    def __init__(
        self,
        checkpoint_dir: str,
        metric_name: str = "val_auc",
        mode: str = "max",
        keep_last: int = 5,
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.metric_name = metric_name
        self.mode = mode
        self.keep_last = keep_last

        os.makedirs(checkpoint_dir, exist_ok=True)

        self._best_metric: Optional[float] = None
        self._saved_paths: List[str] = []

    @property
    def best_metric(self) -> Optional[float]:
        """The best metric value observed so far, or ``None`` if never saved."""
        return self._best_metric

    def save(
        self,
        model: nn.Module,
        epoch: int,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> bool:
        """Save a checkpoint and update the best model if the metric improved.

        Args:
            model: Model to checkpoint.
            epoch: Current epoch number.
            optimizer: Optimiser (optional).
            scheduler: LR scheduler (optional).
            metrics: Dict of metric values.  Must contain ``self.metric_name``
                to enable best-model tracking.

        Returns:
            ``True`` if this checkpoint is the new best; ``False`` otherwise.
        """
        # Regular epoch checkpoint
        fname = f"epoch_{epoch:04d}.pth"
        regular_path = os.path.join(self.checkpoint_dir, fname)
        save_checkpoint(
            path=regular_path,
            model=model,
            epoch=epoch,
            optimizer=optimizer,
            scheduler=scheduler,
            metadata=metrics,
        )
        self._saved_paths.append(regular_path)

        # Enforce keep_last policy
        self._cleanup()

        # Best-model tracking
        is_best = False
        if metrics is not None and self.metric_name in metrics:
            current_metric = float(metrics[self.metric_name])
            if self._is_better(current_metric):
                self._best_metric = current_metric
                best_path = os.path.join(self.checkpoint_dir, "best.pth")
                save_checkpoint(
                    path=best_path,
                    model=model,
                    epoch=epoch,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    metadata=metrics,
                )
                logger.info(
                    "New best model: %s=%.4f (epoch %d) → saved to '%s'.",
                    self.metric_name,
                    current_metric,
                    epoch,
                    best_path,
                )
                is_best = True

        return is_best

    def load_best(
        self,
        model: nn.Module,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        device: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load the best checkpoint saved so far.

        Args:
            model: Model to restore.
            optimizer: Optional optimiser to restore.
            scheduler: Optional scheduler to restore.
            device: Device for ``map_location``.

        Returns:
            The raw checkpoint dictionary.

        Raises:
            FileNotFoundError: If no ``best.pth`` exists in the checkpoint dir.
        """
        best_path = os.path.join(self.checkpoint_dir, "best.pth")
        return load_checkpoint(
            path=best_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _is_better(self, value: float) -> bool:
        if self._best_metric is None:
            return True
        if self.mode == "max":
            return value > self._best_metric
        return value < self._best_metric

    def _cleanup(self) -> None:
        """Delete oldest regular checkpoints beyond *keep_last*."""
        while len(self._saved_paths) > self.keep_last:
            oldest = self._saved_paths.pop(0)
            if os.path.isfile(oldest):
                os.remove(oldest)
                logger.debug("Deleted old checkpoint: '%s'.", oldest)
