"""
data/dataloader.py
------------------
Main data-loading module for DRIFT.

Responsibility:
    Provide a unified PyTorch Dataset + DataLoader wrapper that is compatible
    with the AIGCDetectBenchmark directory layout (sub-directories named
    ``0_real`` and ``1_fake``) and supports both *binary detection* and
    *generator attribution* modes.

Directory layout expected on disk::

    <root>/
    ├── <generator_A>/
    │   ├── 0_real/
    │   │   └── *.png / *.jpg / ...
    │   └── 1_fake/
    │       └── *.png / *.jpg / ...
    ├── <generator_B>/
    │   ├── 0_real/
    │   └── 1_fake/
    └── ...

Alternatively, a flat layout is also accepted::

    <root>/
    ├── 0_real/
    │   └── *.png / *.jpg / ...
    └── 1_fake/
        └── *.png / *.jpg / ...
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from PIL import Image, ImageFile
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from .transforms import get_transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------
_REAL_DIR = "0_real"
_FAKE_DIR = "1_fake"
_SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _collect_images(directory: Union[str, Path]) -> List[str]:
    """Recursively collect all image file paths under *directory*.

    Args:
        directory: Root directory to scan.

    Returns:
        Sorted list of absolute image file paths.
    """
    directory = Path(directory)
    paths: List[str] = []
    for root, _, files in os.walk(directory):
        for fname in files:
            if Path(fname).suffix.lower() in _SUPPORTED_EXTENSIONS:
                paths.append(str(Path(root) / fname))
    return sorted(paths)


def _discover_generators(root: Union[str, Path]) -> List[str]:
    """Discover generator sub-directories inside *root*.

    A generator sub-directory is any directory that contains either a
    ``0_real`` or ``1_fake`` child folder.

    If no such structure is found the root itself is treated as the single
    (anonymous) source.

    Args:
        root: Dataset root path.

    Returns:
        List of generator directory names (not full paths).
    """
    root = Path(root)
    generators: List[str] = []

    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        if (entry / _REAL_DIR).is_dir() or (entry / _FAKE_DIR).is_dir():
            generators.append(entry.name)

    return generators


# ---------------------------------------------------------------------------
# Core Dataset
# ---------------------------------------------------------------------------

class _DRIFTDataset(Dataset):
    """Internal PyTorch Dataset backing DRIFTDataLoader.

    Args:
        samples: List of (image_path, binary_label, generator_name) tuples.
            binary_label is 0 for real, 1 for fake.
        transform: torchvision transform applied to each PIL image.
        mode: ``"binary_mode"`` returns ``(image, int_label)``;
              ``"attribution_mode"`` returns ``(image, generator_name_str)``.
    """

    def __init__(
        self,
        samples: List[Tuple[str, int, str]],
        transform,
        mode: str,
    ) -> None:
        self.samples = samples
        self.transform = transform
        self.mode = mode

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        img_path, binary_label, generator_name = self.samples[index]

        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        if self.mode == "binary_mode":
            return img, binary_label
        elif self.mode == "attribution_mode":
            return img, generator_name
        else:
            raise ValueError(
                f"Unknown mode '{self.mode}'. "
                "Choose 'binary_mode' or 'attribution_mode'."
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class DRIFTDataLoader:
    """Unified data-loading interface for DRIFT experiments.

    Supports two operational modes:

    * ``binary_mode`` — returns ``(image_tensor, label)`` where label is
      ``0`` (real) or ``1`` (fake). Used for binary deepfake detection.
    * ``attribution_mode`` — returns ``(image_tensor, generator_name)``
      where ``generator_name`` is a string identifying the source generator.
      Used for Phase 4 generator attribution.

    Example::

        loader = DRIFTDataLoader(
            root="/data/AIGCBenchmark",
            mode="binary_mode",
            split="train",
            batch_size=32,
            num_samples=500,
        )
        dl = loader.get_dataloader()
        for images, labels in dl:
            ...

    Args:
        root: Path to the dataset root (AIGCDetectBenchmark layout).
        mode: ``"binary_mode"`` or ``"attribution_mode"``.
        split: One of ``"train"``, ``"val"``, ``"test"``.
        image_size: Spatial resolution for cropping/resizing.
        batch_size: DataLoader batch size.
        num_workers: Number of DataLoader worker processes.
        num_samples: If set, cap the number of samples per class per generator.
            ``None`` means use all available samples.
        split_ratios: Dict with keys ``"train"``, ``"val"``, ``"test"`` whose
            values are non-negative floats that sum to 1.0.
        seed: Random seed for reproducible dataset splitting.
        pin_memory: Whether to use pinned memory in the DataLoader.
        shuffle: Override shuffle behaviour. ``None`` auto-sets shuffle=True
            for ``"train"`` and False otherwise.
    """

    def __init__(
        self,
        root: Union[str, Path],
        mode: str = "binary_mode",
        split: str = "train",
        image_size: int = 256,
        batch_size: int = 32,
        num_workers: int = 4,
        num_samples: Optional[int] = None,
        split_ratios: Optional[Dict[str, float]] = None,
        seed: int = 42,
        pin_memory: bool = True,
        shuffle: Optional[bool] = None,
    ) -> None:
        if mode not in ("binary_mode", "attribution_mode"):
            raise ValueError(
                f"mode must be 'binary_mode' or 'attribution_mode', got '{mode}'."
            )
        if split not in ("train", "val", "test"):
            raise ValueError(
                f"split must be 'train', 'val', or 'test', got '{split}'."
            )

        self.root = Path(root)
        self.mode = mode
        self.split = split
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_samples = num_samples
        self.seed = seed
        self.pin_memory = pin_memory

        if split_ratios is None:
            split_ratios = {"train": 0.7, "val": 0.15, "test": 0.15}
        _total = sum(split_ratios.values())
        if abs(_total - 1.0) > 1e-6:
            raise ValueError(
                f"split_ratios must sum to 1.0, got {_total:.4f}."
            )
        self.split_ratios = split_ratios

        self.shuffle = (split == "train") if shuffle is None else shuffle

        # Build samples list
        self._samples: List[Tuple[str, int, str]] = self._build_samples()
        self._print_stats()

        # Transforms
        self._transform = get_transforms(split=split, image_size=image_size)

    # ------------------------------------------------------------------
    # Sample construction
    # ------------------------------------------------------------------

    def _build_samples(self) -> List[Tuple[str, int, str]]:
        """Scan the dataset root and build the (path, label, generator) list.

        Returns:
            List of tuples ``(image_path, binary_label, generator_name)``.
        """
        generators = _discover_generators(self.root)
        all_samples: List[Tuple[str, int, str]] = []

        if generators:
            # Multi-generator layout
            for gen_name in generators:
                gen_dir = self.root / gen_name
                all_samples.extend(
                    self._collect_from_generator(gen_dir, gen_name)
                )
        else:
            # Flat layout (root contains 0_real / 1_fake directly)
            all_samples.extend(
                self._collect_from_generator(self.root, generator_name="unknown")
            )

        if not all_samples:
            raise RuntimeError(
                f"No images found under '{self.root}'. "
                "Make sure the directory contains '0_real' and/or '1_fake' sub-folders "
                "with supported image files (.jpg, .jpeg, .png, .bmp, .tiff, .webp)."
            )

        # Reproducible deterministic split
        rng = random.Random(self.seed)
        rng.shuffle(all_samples)
        all_samples = self._apply_split(all_samples)

        return all_samples

    def _collect_from_generator(
        self,
        gen_dir: Path,
        generator_name: str,
    ) -> List[Tuple[str, int, str]]:
        """Collect real and fake images from a single generator directory.

        Args:
            gen_dir: Path to the generator sub-directory.
            generator_name: Human-readable generator identifier.

        Returns:
            List of ``(path, binary_label, generator_name)`` tuples.
        """
        samples: List[Tuple[str, int, str]] = []

        real_dir = gen_dir / _REAL_DIR
        if real_dir.is_dir():
            real_paths = _collect_images(real_dir)
            if self.num_samples is not None:
                real_paths = real_paths[: self.num_samples]
            samples.extend((p, 0, generator_name) for p in real_paths)

        fake_dir = gen_dir / _FAKE_DIR
        if fake_dir.is_dir():
            fake_paths = _collect_images(fake_dir)
            if self.num_samples is not None:
                fake_paths = fake_paths[: self.num_samples]
            samples.extend((p, 1, generator_name) for p in fake_paths)

        return samples

    def _apply_split(
        self,
        samples: List[Tuple[str, int, str]],
    ) -> List[Tuple[str, int, str]]:
        """Return the slice of *samples* corresponding to ``self.split``.

        Args:
            samples: Shuffled list of all samples.

        Returns:
            Sub-list for the requested split.
        """
        n = len(samples)
        train_end = int(n * self.split_ratios["train"])
        val_end = train_end + int(n * self.split_ratios["val"])

        if self.split == "train":
            return samples[:train_end]
        elif self.split == "val":
            return samples[train_end:val_end]
        else:  # test
            return samples[val_end:]

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def _print_stats(self) -> None:
        """Print a concise dataset summary to stdout."""
        n_real = sum(1 for _, lbl, _ in self._samples if lbl == 0)
        n_fake = sum(1 for _, lbl, _ in self._samples if lbl == 1)

        generators: Dict[str, int] = {}
        for _, _, gen in self._samples:
            generators[gen] = generators.get(gen, 0) + 1

        print(
            f"[DRIFTDataLoader] split={self.split!r} | mode={self.mode!r} | "
            f"total={len(self._samples)} (real={n_real}, fake={n_fake})"
        )
        for gen, cnt in sorted(generators.items()):
            print(f"  generator={gen!r}: {cnt} samples")

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def get_dataset(self) -> _DRIFTDataset:
        """Return the underlying PyTorch Dataset object.

        Returns:
            A ``_DRIFTDataset`` instance ready for indexing.
        """
        return _DRIFTDataset(
            samples=self._samples,
            transform=self._transform,
            mode=self.mode,
        )

    def get_dataloader(self, **kwargs) -> DataLoader:
        """Return a configured PyTorch DataLoader.

        Args:
            **kwargs: Additional keyword arguments forwarded to
                ``torch.utils.data.DataLoader`` (e.g. ``drop_last``).

        Returns:
            A ``DataLoader`` instance.
        """
        dataset = self.get_dataset()
        loader_kwargs = dict(
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        loader_kwargs.update(kwargs)
        return DataLoader(dataset, **loader_kwargs)

    def __len__(self) -> int:
        """Return the number of samples in the current split."""
        return len(self._samples)

    @property
    def generator_names(self) -> List[str]:
        """Sorted list of unique generator names present in this split.

        Returns:
            List of generator name strings.
        """
        return sorted({gen for _, _, gen in self._samples})

    @property
    def samples(self) -> List[Tuple[str, int, str]]:
        """Read-only view of the ``(path, label, generator)`` sample list.

        Returns:
            The internal sample list.
        """
        return self._samples
