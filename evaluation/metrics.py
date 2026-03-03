"""
evaluation/metrics.py
---------------------
All quantitative evaluation metrics used in DRIFT.

Responsibility:
    Provide pure-function metric utilities that can be imported and called
    from any evaluation script, Jupyter notebook, or the DRIFTEvaluator.
    All functions operate on plain Python lists or NumPy arrays so they
    remain independent of the model implementation.

Metrics implemented:
    - ``compute_auc``                — ROC-AUC (binary detection)
    - ``compute_ap``                 — Average Precision (binary detection)
    - ``compute_attribution_accuracy``  — top-1 accuracy (generator attribution)
    - ``measure_inference_time``     — latency in ms/image
    - ``compute_cross_generator_auc``   — per-generator AUC + mean
"""

from __future__ import annotations

import time
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    accuracy_score,
)
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Binary detection metrics
# ---------------------------------------------------------------------------

def compute_auc(
    y_true: Sequence[int],
    y_scores: Sequence[float],
) -> float:
    """Compute the ROC-AUC score for binary deepfake detection.

    Args:
        y_true: Ground-truth binary labels (0 = real, 1 = fake).
            Any sequence of integers or floats is accepted.
        y_scores: Predicted scores (e.g. sigmoid probabilities or raw logits).
            Higher scores should correspond to the positive (fake) class.

    Returns:
        AUC as a float in ``[0, 1]``.  Returns 0.5 if only one class is
        present in *y_true* (degenerate case).
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_scores_arr = np.asarray(y_scores, dtype=float)

    if len(np.unique(y_true_arr)) < 2:
        logger.warning(
            "compute_auc: only one class present in y_true — returning 0.5."
        )
        return 0.5

    return float(roc_auc_score(y_true_arr, y_scores_arr))


def compute_ap(
    y_true: Sequence[int],
    y_scores: Sequence[float],
) -> float:
    """Compute Average Precision (area under the PR curve) for binary detection.

    Args:
        y_true: Ground-truth binary labels (0 = real, 1 = fake).
        y_scores: Predicted scores for the positive (fake) class.

    Returns:
        Average Precision as a float in ``[0, 1]``.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_scores_arr = np.asarray(y_scores, dtype=float)
    return float(average_precision_score(y_true_arr, y_scores_arr))


def compute_accuracy(
    y_true: Sequence[int],
    y_scores: Sequence[float],
    threshold: float = 0.5,
) -> float:
    """Compute binary classification accuracy at a fixed decision threshold.

    Args:
        y_true: Ground-truth binary labels.
        y_scores: Predicted scores for the positive class.
        threshold: Decision boundary (default 0.5).

    Returns:
        Accuracy in ``[0, 1]``.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred = (np.asarray(y_scores, dtype=float) >= threshold).astype(int)
    return float(accuracy_score(y_true_arr, y_pred))


# ---------------------------------------------------------------------------
# Attribution metrics
# ---------------------------------------------------------------------------

def compute_attribution_accuracy(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
) -> float:
    """Compute top-1 accuracy for generator attribution.

    Args:
        y_true: Ground-truth generator names or integer class indices.
            Any hashable sequence element is accepted.
        y_pred: Predicted generator names or class indices (same type as
            *y_true*).

    Returns:
        Fraction of correctly attributed samples in ``[0, 1]``.

    Example::

        acc = compute_attribution_accuracy(
            ["ProGAN", "StyleGAN", "ProGAN"],
            ["ProGAN", "ProGAN",   "ProGAN"],
        )
        # acc == 0.6667
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    if len(y_true_arr) == 0:
        return 0.0
    return float(np.mean(y_true_arr == y_pred_arr))


# ---------------------------------------------------------------------------
# Inference speed
# ---------------------------------------------------------------------------

def measure_inference_time(
    model: Any,
    dataloader: DataLoader,
    device: Union[str, torch.device] = "cuda",
    n_warmup_batches: int = 5,
    max_batches: Optional[int] = None,
) -> float:
    """Measure model inference latency in milliseconds per image.

    Runs the model on batches from *dataloader* without computing gradients
    and records wall-clock time.  A warmup phase is run first to allow GPU
    JIT compilation and cache warm-up.

    The ``model`` must accept a single ``torch.Tensor`` argument (images)
    and can be any callable (``nn.Module``, lambda, etc.).

    Args:
        model: Callable that accepts ``images: torch.Tensor`` on *device*.
        dataloader: DataLoader returning ``(images, labels)`` tuples.
        device: Device to run inference on.
        n_warmup_batches: Number of initial batches excluded from timing.
        max_batches: If set, stop after this many batches (post warm-up).
            ``None`` uses all available batches.

    Returns:
        Average inference time in **milliseconds per image**.

    Example::

        ms_per_img = measure_inference_time(model, test_loader, device="cuda")
        print(f"Throughput: {1000 / ms_per_img:.1f} images/sec")
    """
    device = torch.device(device)
    total_time_s = 0.0
    total_images = 0
    batch_count = 0

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.to(device)
            batch_size = images.shape[0]

            if batch_idx < n_warmup_batches:
                _ = model(images)
                continue

            if max_batches is not None and batch_count >= max_batches:
                break

            # Synchronise GPU before timing
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_start = time.perf_counter()

            _ = model(images)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t_end = time.perf_counter()

            total_time_s += t_end - t_start
            total_images += batch_size
            batch_count += 1

    if total_images == 0:
        logger.warning(
            "measure_inference_time: no batches were timed (dataloader may be empty "
            "or n_warmup_batches exceeds dataset size). Returning 0.0."
        )
        return 0.0

    ms_per_image = (total_time_s / total_images) * 1000.0
    logger.info(
        "Inference time: %.3f ms/image (total_images=%d, total_time=%.3fs)",
        ms_per_image,
        total_images,
        total_time_s,
    )
    return ms_per_image


# ---------------------------------------------------------------------------
# Cross-generator evaluation
# ---------------------------------------------------------------------------

def compute_cross_generator_auc(
    model: Any,
    test_loaders: Dict[str, DataLoader],
    device: Union[str, torch.device] = "cuda",
    score_fn: Optional[Any] = None,
) -> Dict[str, float]:
    """Compute ROC-AUC separately for each generator and report the mean.

    For each generator's test loader, the function runs *model* inference,
    collects logit/probability scores, and computes AUC vs. the real images
    pooled together.

    The *model* is expected to be a callable that:

    1. Accepts a batch of images ``[B, 3, H, W]`` on *device*.
    2. Returns a tensor of shape ``[B]`` or ``[B, 1]`` containing scores
       (higher = more likely fake).

    If *score_fn* is provided, it is used instead of calling *model* directly.
    This is useful when the full pipeline (backbone + extractor + head) must
    be called as separate steps.

    Args:
        model: Inference callable (see above).
        test_loaders: Dict mapping generator name → DataLoader.  Each
            DataLoader should return ``(image, label)`` tuples.
        device: Device for tensor operations.
        score_fn: Optional replacement for *model* when the inference
            pipeline cannot be reduced to a single callable.

    Returns:
        Dictionary with one entry per generator (AUC value) plus a
        ``"mean"`` key containing the macro-average AUC.

    Example::

        results = compute_cross_generator_auc(
            model=pipeline,
            test_loaders={"ProGAN": loader_progan, "ADM": loader_adm},
        )
        # results == {"ProGAN": 0.92, "ADM": 0.87, "mean": 0.895}
    """
    device = torch.device(device)
    call_fn = score_fn if score_fn is not None else model

    per_generator_auc: Dict[str, float] = {}

    with torch.no_grad():
        for gen_name, loader in test_loaders.items():
            y_true: List[int] = []
            y_scores: List[float] = []

            for images, labels in loader:
                images = images.to(device)

                raw_scores = call_fn(images)

                if isinstance(raw_scores, torch.Tensor):
                    raw_scores = raw_scores.squeeze().cpu()
                    if raw_scores.ndim == 0:
                        raw_scores = raw_scores.unsqueeze(0)
                    scores_list = raw_scores.tolist()
                else:
                    scores_list = list(raw_scores)

                labels_list = (
                    labels.cpu().tolist()
                    if isinstance(labels, torch.Tensor)
                    else list(labels)
                )

                y_true.extend(labels_list)
                y_scores.extend(scores_list)

            auc = compute_auc(y_true, y_scores)
            per_generator_auc[gen_name] = auc
            logger.info("  [%s] AUC=%.4f (n=%d)", gen_name, auc, len(y_true))

    if per_generator_auc:
        per_generator_auc["mean"] = float(
            np.mean([v for k, v in per_generator_auc.items() if k != "mean"])
        )
    else:
        per_generator_auc["mean"] = 0.0

    logger.info(
        "Cross-generator AUC: mean=%.4f", per_generator_auc["mean"]
    )
    return per_generator_auc


# ---------------------------------------------------------------------------
# Composite summary
# ---------------------------------------------------------------------------

def compute_all_metrics(
    y_true: Sequence[int],
    y_scores: Sequence[float],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute AUC, AP, and accuracy in one call.

    Args:
        y_true: Ground-truth binary labels.
        y_scores: Predicted scores for the positive class.
        threshold: Decision threshold for accuracy computation.

    Returns:
        Dict with keys ``"auc"``, ``"ap"``, ``"acc"``.
    """
    return {
        "auc": compute_auc(y_true, y_scores),
        "ap": compute_ap(y_true, y_scores),
        "acc": compute_accuracy(y_true, y_scores, threshold=threshold),
    }
