"""
experiments/phase1_binary.py
-----------------------------
Phase 1 — Binary Deepfake Detection with the F1 Endpoint Feature Extractor.

This script implements the complete training + evaluation pipeline for DRIFT
Phase 1.  It can run in two modes:

  Real mode  — Requires a real ADM checkpoint and a dataset directory laid
               out in the AIGCDetectBenchmark format.
  Mock mode  — Generates synthetic data in-memory; useful for development
               and CI/CD verification without a GPU or real data.

Usage
-----
# Mock end-to-end test (no GPU or data needed)
python experiments/phase1_binary.py \\
    --mock --num_samples 40 --epochs 2 \\
    --output_dir /tmp/phase1_test

# Real training
python experiments/phase1_binary.py \\
    --data_dir /path/to/datasets \\
    --train_generators ProGAN \\
    --test_generators ProGAN,StyleGAN2,SD_v1.4,ADM \\
    --model_path /path/to/256x256_diffusion_uncond.pt \\
    --output_dir ./results/phase1 \\
    --ddim_steps 20 \\
    --epochs 10 --lr 1e-4 --batch_size 32 \\
    --precompute
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Path bootstrap — allow running this file directly from the project root
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from DRIFT.models.backbone.adm_wrapper import ADMBackbone
from DRIFT.models.features.endpoint import EndpointFeatureExtractor
from DRIFT.models.heads.binary import BinaryDetectionHead
from DRIFT.training.trainer import DRIFTTrainer
from DRIFT.training.losses import BinaryDetectionLoss
from DRIFT.evaluation.metrics import compute_auc, compute_cross_generator_auc
from DRIFT.evaluation.evaluator import DRIFTEvaluator
from DRIFT.utils.logger import get_logger, setup_logger
from DRIFT.utils.checkpointing import CheckpointManager

# ---------------------------------------------------------------------------
# DIRE baseline AUC values (from DIRE paper Table 1, used for comparison)
# ---------------------------------------------------------------------------
DIRE_REPORTED: Dict[str, float] = {
    "ProGAN":    0.999,
    "StyleGAN":  0.939,
    "StyleGAN2": 0.877,
    "BigGAN":    0.694,
    "CycleGAN":  0.745,
    "GauGAN":    0.666,
    "ADM":       0.657,
    "SD_v1.4":   0.711,
}

logger = logging.getLogger("DRIFT.phase1")


# ===========================================================================
# Mock data utilities
# ===========================================================================

class _MockDataset(torch.utils.data.Dataset):
    """Synthetic dataset for smoke-testing without real images.

    Generates random noise tensors labelled as real (0) or fake (1) and
    injects a weak discriminative signal into fake samples so that a
    trained classifier can achieve AUC > 0.5 on validation data.

    Args:
        n_samples: Total number of samples in this split.
        image_size: Spatial resolution (height == width).
        fake_signal_strength: Magnitude of the bias added to fake samples
            (higher → easier classification problem).
    """

    def __init__(
        self,
        n_samples: int,
        image_size: int = 64,
        fake_signal_strength: float = 0.3,
    ) -> None:
        self.n_samples = n_samples
        self.image_size = image_size
        self.fake_signal_strength = fake_signal_strength

        torch.manual_seed(42)
        self.labels = torch.randint(0, 2, (n_samples,))  # 0=real, 1=fake

        self.images = torch.randn(n_samples, 3, image_size, image_size)

        # Inject a weak structured signal into fake samples
        fake_mask = self.labels == 1
        self.images[fake_mask, 0] += fake_signal_strength   # bias channel 0
        self.images[fake_mask] *= 1.1                        # slight amplitude boost

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.images[idx], int(self.labels[idx])


def _build_mock_loaders(
    n_train: int,
    n_val: int,
    n_test: int,
    image_size: int,
    batch_size: int,
    generator_names: List[str],
) -> Tuple[DataLoader, DataLoader, Dict[str, DataLoader]]:
    """Build train, validation, and per-generator test DataLoaders.

    Args:
        n_train: Number of training samples.
        n_val: Number of validation samples.
        n_test: Number of test samples *per generator*.
        image_size: Spatial resolution for mock images.
        batch_size: DataLoader batch size.
        generator_names: List of generator names for the test loaders.

    Returns:
        Tuple ``(train_loader, val_loader, test_loaders_dict)``.
    """
    train_ds = _MockDataset(n_train, image_size, fake_signal_strength=0.5)
    val_ds   = _MockDataset(n_val,   image_size, fake_signal_strength=0.5)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    test_loaders: Dict[str, DataLoader] = {}
    for gen_name in generator_names:
        # Each generator gets a slightly different signal strength to
        # simulate variation in detection difficulty
        strength = 0.2 + 0.1 * (hash(gen_name) % 5)
        ds = _MockDataset(n_test, image_size, fake_signal_strength=strength)
        test_loaders[gen_name] = DataLoader(
            ds, batch_size=batch_size, shuffle=False
        )

    return train_loader, val_loader, test_loaders


# ===========================================================================
# Inference pipeline (backbone + extractor + head combined as callable)
# ===========================================================================

class _InferencePipeline:
    """Wraps backbone + feature extractor + head into a single callable.

    The resulting object satisfies the interface expected by
    :class:`DRIFTEvaluator` and :func:`compute_cross_generator_auc`:
    it accepts a batch of images and returns a score tensor.

    Args:
        backbone: Configured :class:`ADMBackbone`.
        extractor: A :class:`FeatureExtractor` sub-class instance.
        head: The trained :class:`BinaryDetectionHead`.
        device: Torch device.
    """

    def __init__(
        self,
        backbone: ADMBackbone,
        extractor: EndpointFeatureExtractor,
        head: BinaryDetectionHead,
        device: torch.device,
    ) -> None:
        self.backbone = backbone
        self.extractor = extractor
        self.head = head
        self.device = device
        self.head.eval()

    @torch.no_grad()
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """Run the full pipeline and return fake probabilities.

        Args:
            images: ``[B, 3, H, W]`` batch on any device.

        Returns:
            ``[B]`` tensor of fake probabilities in ``[0, 1]``.
        """
        images = images.to(self.device)
        x_T, intermediates = self.backbone.invert(images, return_intermediates=False)
        features = self.extractor.extract(x_T, intermediates)
        logits = self.head(features)
        return torch.sigmoid(logits.squeeze(1))


# ===========================================================================
# Plotting utilities
# ===========================================================================

def _plot_training_curves(history: Dict, output_dir: str) -> None:
    """Save loss and AUC learning curves to ``training_curves.png``.

    Args:
        history: Dict returned by :meth:`DRIFTTrainer.train`.
        output_dir: Directory where the PNG is saved.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping training_curves.png.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curve
    ax = axes[0]
    epochs = list(range(1, len(history["train_loss"]) + 1))
    ax.plot(epochs, history["train_loss"], label="Train Loss", marker="o")
    if history["val_loss"]:
        ax.plot(epochs, history["val_loss"], label="Val Loss", marker="s")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # AUC curve
    ax = axes[1]
    if history["val_auc"]:
        ax.plot(epochs, history["val_auc"], label="Val AUC", marker="s", color="green")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC")
    ax.set_title("Validation AUC over Epochs")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Training curves saved to '%s'.", path)


def _plot_roc_curves(
    pipeline: _InferencePipeline,
    test_loaders: Dict[str, DataLoader],
    output_dir: str,
    device: torch.device,
) -> None:
    """Save per-generator ROC curves to ``roc_curves.png``.

    Args:
        pipeline: Inference callable.
        test_loaders: Dict mapping generator name → DataLoader.
        output_dir: Output directory.
        device: Torch device.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve
    except ImportError:
        logger.warning("matplotlib / sklearn not available — skipping roc_curves.png.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    for gen_name, loader in test_loaders.items():
        y_true: List[int] = []
        y_scores: List[float] = []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                scores = pipeline(images).cpu().tolist()
                if isinstance(scores, float):
                    scores = [scores]
                y_scores.extend(scores)
                lbls = labels.tolist() if isinstance(labels, torch.Tensor) else list(labels)
                y_true.extend(lbls)

        if len(set(y_true)) < 2:
            continue

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc_val = compute_auc(y_true, y_scores)
        ax.plot(fpr, tpr, label=f"{gen_name} (AUC={auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Cross-Generator Evaluation")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "roc_curves.png")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("ROC curves saved to '%s'.", path)


def _plot_confusion_matrix(
    pipeline: _InferencePipeline,
    test_loaders: Dict[str, DataLoader],
    output_dir: str,
    device: torch.device,
) -> None:
    """Save an aggregated confusion matrix to ``confusion_matrix.png``.

    Args:
        pipeline: Inference callable.
        test_loaders: Dict of test DataLoaders.
        output_dir: Output directory.
        device: Torch device.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    except ImportError:
        logger.warning("matplotlib / sklearn not available — skipping confusion_matrix.png.")
        return

    all_true: List[int] = []
    all_pred: List[int] = []

    with torch.no_grad():
        for loader in test_loaders.values():
            for images, labels in loader:
                images = images.to(device)
                scores = pipeline(images).cpu()
                preds = (scores >= 0.5).long().tolist()
                if isinstance(preds, int):
                    preds = [preds]
                lbls = labels.tolist() if isinstance(labels, torch.Tensor) else list(labels)
                all_pred.extend(preds)
                all_true.extend(lbls)

    cm = confusion_matrix(all_true, all_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Fake"])

    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix (all generators pooled)")

    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Confusion matrix saved to '%s'.", path)


def _save_auc_table(
    auc_results: Dict[str, float],
    output_dir: str,
    model_name: str = "DRIFT-F1",
) -> None:
    """Save cross-generator AUC as JSON and Markdown table.

    Args:
        auc_results: Dict mapping generator name → AUC value.
        output_dir: Directory where files are written.
        model_name: Method name shown in the Markdown table header.
    """
    # JSON
    json_path = os.path.join(output_dir, "cross_generator_auc.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(auc_results, f, indent=2)
    logger.info("Cross-generator AUC JSON saved to '%s'.", json_path)

    # Markdown table with DIRE comparison
    md_path = os.path.join(output_dir, "cross_generator_auc_table.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Phase 1 — Cross-Generator AUC Comparison\n\n")
        f.write(f"| Generator | {model_name} | DIRE (reported) |\n")
        f.write("|-----------|-------------|------------------|\n")

        for gen, auc in auc_results.items():
            if gen == "mean":
                continue
            dire_auc = DIRE_REPORTED.get(gen, "—")
            dire_str = f"{dire_auc:.3f}" if isinstance(dire_auc, float) else dire_auc
            f.write(f"| {gen} | {auc:.3f} | {dire_str} |\n")

        mean_auc = auc_results.get("mean", "—")
        mean_str = f"{mean_auc:.3f}" if isinstance(mean_auc, float) else mean_auc
        f.write(f"| **Mean** | **{mean_str}** | — |\n")

    logger.info("Cross-generator AUC table saved to '%s'.", md_path)


# ===========================================================================
# Real-data DataLoader builder
# ===========================================================================

def _build_real_loaders(
    args: argparse.Namespace,
) -> Tuple[DataLoader, DataLoader, Dict[str, DataLoader]]:
    """Build DataLoaders from the AIGCDetectBenchmark directory layout.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Tuple ``(train_loader, val_loader, test_loaders_dict)``.
    """
    from DRIFT.data.dataloader import DRIFTDataLoader

    train_generators = [g.strip() for g in args.train_generators.split(",")]
    test_generators  = [g.strip() for g in args.test_generators.split(",")]

    # Build a combined train/val loader from the training generators
    # DRIFTDataLoader handles splitting internally
    train_dl_obj = DRIFTDataLoader(
        root=args.data_dir,
        mode="binary_mode",
        split="train",
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    val_dl_obj = DRIFTDataLoader(
        root=args.data_dir,
        mode="binary_mode",
        split="val",
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    train_loader = train_dl_obj.get_dataloader()
    val_loader   = val_dl_obj.get_dataloader()

    # Per-generator test loaders
    test_loaders: Dict[str, DataLoader] = {}
    for gen in test_generators:
        gen_root = os.path.join(args.data_dir, gen)
        if not os.path.isdir(gen_root):
            logger.warning(
                "Test generator directory not found: '%s'. Skipping.", gen_root
            )
            continue
        dl_obj = DRIFTDataLoader(
            root=args.data_dir,
            mode="binary_mode",
            split="test",
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        test_loaders[gen] = dl_obj.get_dataloader()

    return train_loader, val_loader, test_loaders


# ===========================================================================
# Optional x_T pre-computation step
# ===========================================================================

def _precompute_x_T(
    backbone: ADMBackbone,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loaders: Dict[str, DataLoader],
    cache_dir: str,
    batch_size: int,
) -> None:
    """Pre-compute x_T for all splits and cache to disk.

    This is a best-effort step: if images cannot be loaded or backbone
    fails, a warning is emitted but the script continues.

    Args:
        backbone: Configured ADMBackbone.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        test_loaders: Test DataLoaders.
        cache_dir: Cache directory.
        batch_size: Backbone batch size for pre-computation.
    """
    from DRIFT.models.preprocessing.feature_cache import FeatureCache

    cache = FeatureCache(cache_dir=cache_dir, backbone=backbone)

    def _collect_paths(loader: DataLoader) -> List[str]:
        # Attempt to extract file paths from the dataset
        ds = loader.dataset
        if hasattr(ds, "samples"):
            return [s[0] for s in ds.samples]
        logger.warning("Cannot extract image paths from DataLoader dataset — skipping pre-computation for this split.")
        return []

    all_paths: List[str] = []
    for loader in [train_loader, val_loader] + list(test_loaders.values()):
        all_paths.extend(_collect_paths(loader))

    if all_paths:
        logger.info("Pre-computing x_T for %d images ...", len(all_paths))
        cache.precompute_dataset(all_paths, batch_size=batch_size)
        stats = cache.cache_stats()
        logger.info("Cache stats: %s", stats)
    else:
        logger.warning("No image paths found — skipping x_T pre-computation.")


# ===========================================================================
# Main entry point
# ===========================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="DRIFT Phase 1 — Binary Deepfake Detection (F1 Endpoint Features)"
    )

    # Mode
    parser.add_argument(
        "--mock", action="store_true",
        help="Run in mock mode (no real data or model needed)."
    )
    parser.add_argument(
        "--num_samples", type=int, default=100,
        help="Number of synthetic samples per split in mock mode."
    )

    # Paths (real mode)
    parser.add_argument("--data_dir",         type=str, default="",
                        help="Root directory of the AIGCDetectBenchmark dataset.")
    parser.add_argument("--train_generators", type=str, default="ProGAN",
                        help="Comma-separated list of training generator names.")
    parser.add_argument("--test_generators",  type=str,
                        default="ProGAN,StyleGAN2,SD_v1.4,ADM",
                        help="Comma-separated list of test generator names.")
    parser.add_argument("--model_path",       type=str, default="mock",
                        help="Path to the ADM checkpoint (or 'mock').")
    parser.add_argument("--output_dir",       type=str, default="./results/phase1",
                        help="Directory for output files.")
    parser.add_argument("--precompute",       action="store_true",
                        help="Pre-compute and cache all x_T before training.")

    # Model / training hyper-parameters
    parser.add_argument("--ddim_steps", type=int,   default=20)
    parser.add_argument("--image_size", type=int,   default=64,
                        help="Spatial resolution (default 64 for speed; 256 for real runs).")
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int,   default=16)
    parser.add_argument("--num_workers",type=int,   default=0)
    parser.add_argument("--device",     type=str,   default="auto",
                        help="Device: 'auto', 'cuda', 'mps', or 'cpu'.")
    parser.add_argument("--feature_dim",type=int,   default=128,
                        help="Output dimensionality of the F1 extractor.")
    parser.add_argument("--hidden_dim", type=int,   default=256,
                        help="Hidden dim of BinaryDetectionHead.")
    parser.add_argument("--log_every",  type=int,   default=10,
                        help="Log loss every N batches.")

    return parser.parse_args()


def _resolve_device(device_str: str) -> str:
    """Resolve device string, falling back to CPU if requested is unavailable.

    Args:
        device_str: One of ``"auto"``, ``"cuda"``, ``"mps"``, ``"cpu"``.

    Returns:
        Device string suitable for ``torch.device()``.
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device_str


def main() -> None:
    """Entry point for Phase 1 training and evaluation."""
    args = parse_args()

    # ------------------------------------------------------------------
    # Step 1: Setup logging and output directory
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logger(
        name="DRIFT",
        log_level="INFO",
        log_dir=args.output_dir,
        log_filename="phase1.log",
    )
    logger.info("=" * 70)
    logger.info("DRIFT Phase 1 — Binary Detection (F1 Endpoint Features)")
    logger.info("=" * 70)
    logger.info("Arguments: %s", vars(args))

    device_str = _resolve_device(args.device)
    device = torch.device(device_str)
    logger.info("Device: %s", device)

    # ------------------------------------------------------------------
    # Step 2: Initialise ADMBackbone
    # ------------------------------------------------------------------
    logger.info("Step 2: Initialising ADMBackbone ...")
    model_path = "mock" if args.mock else args.model_path
    backbone = ADMBackbone(
        model_path=model_path,
        device=device_str,
        ddim_steps=args.ddim_steps,
        image_size=args.image_size,
    )
    logger.info("Backbone: %s", backbone)

    # ------------------------------------------------------------------
    # Step 3: Build DataLoaders
    # ------------------------------------------------------------------
    logger.info("Step 3: Building DataLoaders ...")

    if args.mock:
        n_train = args.num_samples
        n_val   = max(10, args.num_samples // 5)
        n_test  = max(10, args.num_samples // 5)
        test_generator_names = [g.strip() for g in args.test_generators.split(",")]

        train_loader, val_loader, test_loaders = _build_mock_loaders(
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            image_size=args.image_size,
            batch_size=args.batch_size,
            generator_names=test_generator_names,
        )
        logger.info(
            "Mock mode: train=%d, val=%d, test=%d/generator, generators=%s",
            n_train, n_val, n_test, test_generator_names,
        )
    else:
        if not args.data_dir:
            logger.error("--data_dir is required in real mode.")
            sys.exit(1)
        train_loader, val_loader, test_loaders = _build_real_loaders(args)

    # ------------------------------------------------------------------
    # Step 3b: Optional x_T pre-computation
    # ------------------------------------------------------------------
    if args.precompute and not args.mock:
        logger.info("Step 3b: Pre-computing x_T cache ...")
        cache_dir = os.path.join(args.output_dir, "x_T_cache")
        _precompute_x_T(
            backbone=backbone,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loaders=test_loaders,
            cache_dir=cache_dir,
            batch_size=args.batch_size,
        )

    # ------------------------------------------------------------------
    # Step 4: Initialise feature extractor and detection head
    # ------------------------------------------------------------------
    logger.info("Step 4: Initialising EndpointFeatureExtractor and BinaryDetectionHead ...")
    extractor = EndpointFeatureExtractor(feature_dim=args.feature_dim)
    head = BinaryDetectionHead(
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
    )
    logger.info("Extractor: %s", extractor)
    logger.info("Head: %s", head)

    # ------------------------------------------------------------------
    # Step 5: Training loop
    # ------------------------------------------------------------------
    logger.info("Step 5: Starting training for %d epoch(s) ...", args.epochs)
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")

    trainer_config = {
        "lr":               args.lr,
        "device":           device_str,
        "num_epochs":       args.epochs,
        "checkpoint_dir":   checkpoint_dir,
        "log_every":        args.log_every,
        "val_every":        1,
        "save_every":       max(1, args.epochs),   # save at least once
        "amp":              False,
        "return_intermediates": False,
    }

    trainer = DRIFTTrainer(
        backbone=backbone,
        feature_extractor=extractor,
        head=head,
        config=trainer_config,
    )

    t_start = time.time()
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
    )
    train_elapsed = time.time() - t_start
    logger.info("Training complete in %.1f seconds.", train_elapsed)

    # ------------------------------------------------------------------
    # Step 5b: Save best model checkpoint
    # ------------------------------------------------------------------
    best_model_path = os.path.join(args.output_dir, "best_model.pt")
    # Use the CheckpointManager to save the final head
    ckpt_manager = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        metric_name="val_auc",
        mode="max",
    )
    val_auc_list = history.get("val_auc", [])
    final_val_auc = val_auc_list[-1] if val_auc_list else 0.0
    ckpt_manager.save(
        model=head,
        epoch=args.epochs,
        metrics={"val_auc": final_val_auc},
    )

    # Also save a standalone "best_model.pt" at the output root
    torch.save(
        {
            "head_state_dict":      head.state_dict(),
            "feature_dim":          args.feature_dim,
            "hidden_dim":           args.hidden_dim,
            "val_auc":              final_val_auc,
            "epochs_trained":       args.epochs,
            "history":              history,
        },
        best_model_path,
    )
    logger.info("Best model saved to '%s'.", best_model_path)

    # ------------------------------------------------------------------
    # Step 5c: Plot training curves
    # ------------------------------------------------------------------
    _plot_training_curves(history, args.output_dir)

    # ------------------------------------------------------------------
    # Step 6: Cross-generator evaluation
    # ------------------------------------------------------------------
    logger.info("Step 6: Cross-generator evaluation ...")
    pipeline = _InferencePipeline(
        backbone=backbone,
        extractor=extractor,
        head=head,
        device=device,
    )

    # Per-generator AUC
    auc_results: Dict[str, float] = {}
    for gen_name, loader in test_loaders.items():
        y_true: List[int] = []
        y_scores: List[float] = []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                scores = pipeline(images).cpu().tolist()
                if isinstance(scores, float):
                    scores = [scores]
                y_scores.extend(scores)
                lbls = (
                    labels.tolist()
                    if isinstance(labels, torch.Tensor)
                    else list(labels)
                )
                y_true.extend(lbls)

        auc = compute_auc(y_true, y_scores)
        auc_results[gen_name] = auc
        logger.info("  [%s] AUC = %.4f (n=%d)", gen_name, auc, len(y_true))

    if auc_results:
        auc_results["mean"] = float(np.mean(list(auc_results.values())))
        logger.info("Mean AUC across generators: %.4f", auc_results["mean"])

    # ------------------------------------------------------------------
    # Step 7: Save results
    # ------------------------------------------------------------------
    logger.info("Step 7: Saving results to '%s' ...", args.output_dir)

    _save_auc_table(auc_results, args.output_dir)
    _plot_roc_curves(pipeline, test_loaders, args.output_dir, device)

    # Confusion matrix (especially useful in mock mode)
    _plot_confusion_matrix(pipeline, test_loaders, args.output_dir, device)

    # ------------------------------------------------------------------
    # Final summary printout
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("DRIFT Phase 1 — Final Results Summary")
    print("=" * 70)
    print(f"{'Generator':<20} {'DRIFT-F1 AUC':>14} {'DIRE (reported)':>18}")
    print("-" * 54)
    for gen, auc in auc_results.items():
        if gen == "mean":
            continue
        dire = DIRE_REPORTED.get(gen, "—")
        dire_str = f"{dire:.3f}" if isinstance(dire, float) else dire
        print(f"  {gen:<18} {auc:>14.4f} {dire_str:>18}")
    mean_auc = auc_results.get("mean", 0.0)
    print("-" * 54)
    print(f"  {'Mean':<18} {mean_auc:>14.4f}")
    print("=" * 70)
    print(f"Output directory: {os.path.abspath(args.output_dir)}")
    print(f"Training time:    {train_elapsed:.1f}s")
    print(f"Final val AUC:    {final_val_auc:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
