"""
data/transforms.py
------------------
Image pre-processing and augmentation pipelines for DRIFT.

Responsibility:
    Provide standard train/val/test transforms compatible with the
    AIGCDetectBenchmark augmentation scheme, including optional JPEG
    compression and Gaussian blur augmentations used during training.

All transforms use the ImageNet mean/std normalisation that is standard
across the AIGCDetectBenchmark codebase.
"""

from __future__ import annotations

import random
from io import BytesIO
from typing import Optional, Tuple, List

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# ---------------------------------------------------------------------------
# ImageNet normalisation constants (matches AIGCDetectBenchmark)
# ---------------------------------------------------------------------------
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Augmentation helpers (mirroring AIGCDetectBenchmark/data/datasets.py)
# ---------------------------------------------------------------------------

class JPEGCompression:
    """Apply random JPEG compression to a PIL image.

    Args:
        quality_range: ``(low, high)`` inclusive range for JPEG quality.
            Lower quality = more compression artefacts.
    """

    def __init__(self, quality_range: Tuple[int, int] = (75, 95)) -> None:
        self.quality_range = quality_range

    def __call__(self, img: Image.Image) -> Image.Image:
        quality = random.randint(*self.quality_range)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(quality_range={self.quality_range})"


class GaussianBlur:
    """Apply Gaussian blur with a randomly sampled radius.

    Args:
        sigma_range: ``(low, high)`` range for the Gaussian sigma value.
    """

    def __init__(self, sigma_range: Tuple[float, float] = (0.1, 2.0)) -> None:
        self.sigma_range = sigma_range

    def __call__(self, img: Image.Image) -> Image.Image:
        import PIL.ImageFilter as IF

        sigma = random.uniform(*self.sigma_range)
        return img.filter(IF.GaussianBlur(radius=sigma))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(sigma_range={self.sigma_range})"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_transforms(
    split: str,
    image_size: int = 256,
    noise_type: Optional[str] = None,
    jpeg_quality_range: Tuple[int, int] = (75, 95),
    blur_sigma_range: Tuple[float, float] = (0.1, 2.0),
    no_flip: bool = False,
    no_crop: bool = False,
    no_resize: bool = False,
) -> T.Compose:
    """Return a ``torchvision.transforms.Compose`` pipeline for the given split.

    The pipeline mirrors the AIGCDetectBenchmark approach:

    * **Training** — random resize → optional noise augmentation →
      random crop → random horizontal flip → ToTensor → Normalize
    * **Validation / Test** — resize → center crop → ToTensor → Normalize

    Args:
        split: One of ``"train"``, ``"val"``, ``"test"``.
        image_size: Target spatial resolution (height == width).
        noise_type: Optional noise augmentation applied *during training only*.
            ``"jpg"`` applies JPEG compression, ``"blur"`` applies Gaussian
            blur, ``None`` / ``"none"`` disables augmentation.
        jpeg_quality_range: ``(low, high)`` JPEG quality when
            ``noise_type="jpg"``.
        blur_sigma_range: ``(low, high)`` sigma when ``noise_type="blur"``.
        no_flip: Disable random horizontal flip even for training.
        no_crop: Disable cropping (return full resized image).
        no_resize: Disable resizing step (use raw image size).

    Returns:
        A ``torchvision.transforms.Compose`` object ready for use in a
        PyTorch Dataset.

    Raises:
        ValueError: If *split* is not one of ``"train"``, ``"val"``, ``"test"``.
    """
    if split not in ("train", "val", "test"):
        raise ValueError(
            f"split must be 'train', 'val', or 'test', got '{split}'."
        )

    is_train = split == "train"
    steps: List = []

    # ----- Resize ----------------------------------------------------------
    if not no_resize:
        steps.append(T.Resize((image_size, image_size)))

    # ----- Training-only augmentations ------------------------------------
    if is_train:
        # Optional noise augmentation (JPEG / Gaussian blur)
        if noise_type == "jpg":
            steps.append(JPEGCompression(quality_range=jpeg_quality_range))
        elif noise_type == "blur":
            steps.append(GaussianBlur(sigma_range=blur_sigma_range))

        # Crop
        if not no_crop:
            steps.append(T.RandomCrop(image_size))

        # Flip
        if not no_flip:
            steps.append(T.RandomHorizontalFlip())

    else:
        # Validation / test crop
        if not no_crop:
            steps.append(T.CenterCrop(image_size))

    # ----- Tensor conversion + normalisation --------------------------------
    steps.append(T.ToTensor())
    steps.append(T.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)))

    return T.Compose(steps)


def get_drift_transforms(
    split: str,
    image_size: int = 256,
) -> T.Compose:
    """Convenience wrapper that returns the default DRIFT transform without
    any noise augmentation.

    Equivalent to ``get_transforms(split, image_size, noise_type=None)``.

    Args:
        split: One of ``"train"``, ``"val"``, ``"test"``.
        image_size: Target spatial resolution.

    Returns:
        A ``torchvision.transforms.Compose`` pipeline.
    """
    return get_transforms(split=split, image_size=image_size, noise_type=None)


def denormalize(
    tensor: torch.Tensor,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
) -> torch.Tensor:
    """Invert ImageNet normalisation for visualisation.

    Args:
        tensor: Normalised image tensor ``[C, H, W]`` or ``[B, C, H, W]``.
        mean: Per-channel mean used during normalisation.
        std: Per-channel std used during normalisation.

    Returns:
        Denormalised tensor clipped to ``[0, 1]``.
    """
    mean_t = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std_t = torch.tensor(std, dtype=tensor.dtype, device=tensor.device)

    # Handle both [C, H, W] and [B, C, H, W]
    if tensor.ndim == 3:
        mean_t = mean_t.view(-1, 1, 1)
        std_t = std_t.view(-1, 1, 1)
    elif tensor.ndim == 4:
        mean_t = mean_t.view(1, -1, 1, 1)
        std_t = std_t.view(1, -1, 1, 1)

    return (tensor * std_t + mean_t).clamp(0.0, 1.0)
