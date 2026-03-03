"""
models/preprocessing/feature_cache.py
--------------------------------------
Disk-level cache for pre-computed DDIM inversion endpoints (x_T).

Responsibility:
    Avoid re-running the expensive ADMBackbone DDIM inversion on every
    training epoch by persisting the result for each image as a ``.pt``
    file.  Supports:

    * Deterministic file naming: ``md5(image_path).pt``
    * Resumable batch pre-computation (skips already-cached files)
    * Progress display via ``tqdm``
    * On-demand single-file get (compute-and-cache on first access)
    * Cache statistics (file count, total size, coverage fraction)

Example::

    backbone = ADMBackbone("mock", device="cpu")
    cache = FeatureCache("/tmp/drift_cache", backbone)
    cache.precompute_dataset(image_paths, batch_size=8)
    entry = cache.get(image_paths[0])
    x_T = entry["x_T"]
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Dict, List, Optional, Union

import torch
from PIL import Image

logger = logging.getLogger(__name__)

try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False
    logger.warning(
        "tqdm not installed — progress bars will be disabled. "
        "Install with: pip install tqdm"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _md5_key(image_path: str) -> str:
    """Return a hex MD5 digest of *image_path* (used as cache filename stem).

    Args:
        image_path: Absolute or relative path to an image file.

    Returns:
        32-character lowercase hex string.
    """
    return hashlib.md5(image_path.encode("utf-8")).hexdigest()


def _load_image_tensor(
    image_path: str,
    image_size: int = 256,
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """Load a PIL image, resize, and convert to a ``[-1, 1]`` float tensor.

    Args:
        image_path: Path to the image file.
        image_size: Spatial resolution to resize to (square crop).
        device: Torch device for the output tensor.

    Returns:
        Tensor of shape ``[1, 3, image_size, image_size]``.
    """
    import torchvision.transforms.functional as TF

    img = Image.open(image_path).convert("RGB")
    img = img.resize((image_size, image_size), Image.BILINEAR)
    t = TF.to_tensor(img)                 # [3, H, W], range [0, 1]
    t = t * 2.0 - 1.0                     # map to [-1, 1]
    return t.unsqueeze(0).to(device)      # [1, 3, H, W]


# ---------------------------------------------------------------------------
# FeatureCache
# ---------------------------------------------------------------------------

class FeatureCache:
    """Pre-compute and cache the DDIM inversion result (x_T) for each image.

    Cache layout on disk::

        <cache_dir>/
        ├── <md5_of_path_1>.pt
        ├── <md5_of_path_2>.pt
        └── ...

    Each ``.pt`` file contains a Python dict::

        {
            "x_T":           torch.Tensor [1, 3, H, W],
            "intermediates": list[Tensor] | None,
            "source":        str  # absolute image path
        }

    Args:
        cache_dir: Directory where ``.pt`` cache files are stored.
        backbone: A configured :class:`ADMBackbone` instance used to run
            DDIM inversion when a cache file does not yet exist.
        image_size: Spatial resolution passed to the image loader (default
            256, matching ADM training resolution).
    """

    def __init__(
        self,
        cache_dir: str,
        backbone,  # ADMBackbone — avoid circular import with type annotation
        image_size: int = 256,
    ) -> None:
        self.cache_dir = os.path.abspath(cache_dir)
        self.backbone = backbone
        self.image_size = image_size
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info("FeatureCache initialised: cache_dir='%s'", self.cache_dir)

    # ------------------------------------------------------------------
    # File naming
    # ------------------------------------------------------------------

    def _cache_path(self, image_path: str) -> str:
        """Return the full path to the cache file for *image_path*.

        Args:
            image_path: Source image path (used as hash input).

        Returns:
            Absolute path ending in ``<md5>.pt``.
        """
        key = _md5_key(os.path.abspath(image_path))
        return os.path.join(self.cache_dir, key + ".pt")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_cached(self, image_path: str) -> bool:
        """Check whether *image_path* already has a cache entry on disk.

        Args:
            image_path: Source image path.

        Returns:
            ``True`` if the corresponding ``.pt`` file exists.
        """
        return os.path.isfile(self._cache_path(image_path))

    def get(
        self,
        image_path: str,
        return_intermediates: bool = False,
    ) -> Dict:
        """Retrieve the cached inversion result for *image_path*.

        If no cache file exists, runs ADM inversion, caches the result, and
        returns it.

        Args:
            image_path: Absolute or relative path to the source image.
            return_intermediates: Whether to compute and cache intermediate
                tensors if not yet cached.  Ignored when the cache file
                already exists (the stored ``intermediates`` value is
                returned regardless).

        Returns:
            Dict with keys ``"x_T"``, ``"intermediates"``, ``"source"``.
        """
        cache_file = self._cache_path(image_path)

        if os.path.isfile(cache_file):
            try:
                entry = torch.load(cache_file, map_location="cpu")
                return entry
            except Exception as exc:
                logger.warning(
                    "Corrupt cache file '%s' (error: %s). Recomputing.",
                    cache_file,
                    exc,
                )
                os.remove(cache_file)

        # Compute and cache
        entry = self._compute_and_save(image_path, return_intermediates)
        return entry

    def precompute_dataset(
        self,
        image_paths: List[str],
        return_intermediates: bool = False,
        num_workers: int = 1,
        batch_size: int = 4,
    ) -> None:
        """Batch pre-compute x_T for a list of image paths.

        Skips images that are already cached.  Processes images in batches
        of *batch_size* and shows a ``tqdm`` progress bar.

        Args:
            image_paths: List of absolute paths to images.
            return_intermediates: Whether to also compute and store the
                intermediate DDIM latents.
            num_workers: Currently unused (reserved for future multi-process
                support).  Inversion is run on the backbone's device.
            batch_size: Number of images to invert in a single backbone call.
        """
        pending = [p for p in image_paths if not self.is_cached(p)]
        already_done = len(image_paths) - len(pending)

        logger.info(
            "precompute_dataset: %d total, %d already cached, %d to compute.",
            len(image_paths),
            already_done,
            len(pending),
        )

        if not pending:
            logger.info("All images already cached — nothing to do.")
            return

        # Wrap in progress bar if available
        def _make_progress(iterable, **kwargs):
            if _HAS_TQDM:
                return _tqdm(iterable, **kwargs)
            return iterable

        # Process in batches
        for start_idx in _make_progress(
            range(0, len(pending), batch_size),
            desc="Precomputing x_T",
            unit="batch",
        ):
            batch_paths = pending[start_idx : start_idx + batch_size]
            self._compute_batch(batch_paths, return_intermediates)

        logger.info("precompute_dataset complete. Computed %d entries.", len(pending))

    def cache_stats(self) -> Dict:
        """Return statistics about the current cache state.

        Returns:
            Dict with keys:
            * ``"num_files"``   — number of ``.pt`` files in the cache dir.
            * ``"total_size_mb"`` — total size of cached files in MB.
            * ``"cache_dir"``   — absolute path to the cache directory.
        """
        pt_files = [
            f for f in os.listdir(self.cache_dir) if f.endswith(".pt")
        ]
        total_bytes = sum(
            os.path.getsize(os.path.join(self.cache_dir, f))
            for f in pt_files
        )
        return {
            "num_files": len(pt_files),
            "total_size_mb": round(total_bytes / (1024 ** 2), 2),
            "cache_dir": self.cache_dir,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_and_save(
        self,
        image_path: str,
        return_intermediates: bool,
    ) -> Dict:
        """Run backbone inversion on a single image and persist the result.

        Args:
            image_path: Path to the source image.
            return_intermediates: Whether to compute intermediate tensors.

        Returns:
            The cache entry dict.
        """
        device = self.backbone.device

        try:
            x0 = _load_image_tensor(image_path, self.image_size, device)
        except Exception as exc:
            logger.error(
                "Failed to load image '%s': %s. Returning zero x_T.", image_path, exc
            )
            shape = (1, 3, self.image_size, self.image_size)
            x_T = torch.zeros(shape)
            entry = {"x_T": x_T.cpu(), "intermediates": None, "source": image_path}
            torch.save(entry, self._cache_path(image_path))
            return entry

        with torch.no_grad():
            x_T, intermediates = self.backbone.invert(
                x0, return_intermediates=return_intermediates
            )

        # Move to CPU before saving to keep cache device-agnostic
        entry: Dict = {
            "x_T": x_T.cpu(),
            "intermediates": (
                [t.cpu() for t in intermediates]
                if intermediates is not None
                else None
            ),
            "source": os.path.abspath(image_path),
        }

        torch.save(entry, self._cache_path(image_path))
        return entry

    def _compute_batch(
        self,
        image_paths: List[str],
        return_intermediates: bool,
    ) -> None:
        """Invert a batch of images and save each result to the cache.

        Loads all images in *image_paths*, stacks them into a single tensor,
        calls ``backbone.invert()`` once, then splits and saves individually.

        Args:
            image_paths: Batch of image file paths.
            return_intermediates: Whether to compute intermediates.
        """
        device = self.backbone.device
        images = []

        for path in image_paths:
            try:
                t = _load_image_tensor(path, self.image_size, device)
                images.append(t)
            except Exception as exc:
                logger.warning("Skipping '%s' (load error: %s).", path, exc)
                # Insert a placeholder zero tensor to keep batch index alignment
                shape = (1, 3, self.image_size, self.image_size)
                images.append(torch.zeros(shape, device=device))

        if not images:
            return

        # Stack into [B, 3, H, W]
        x0_batch = torch.cat(images, dim=0)  # [B, 3, H, W]

        with torch.no_grad():
            x_T_batch, intermediates_batch = self.backbone.invert(
                x0_batch, return_intermediates=return_intermediates
            )

        # Save each sample individually
        B = x_T_batch.shape[0]
        for i, path in enumerate(image_paths):
            if i >= B:
                break

            x_T_i = x_T_batch[i : i + 1].cpu()  # [1, 3, H, W]

            if intermediates_batch is not None:
                # intermediates is a list of [B, 3, H, W] tensors
                ints_i = [t[i : i + 1].cpu() for t in intermediates_batch]
            else:
                ints_i = None

            entry: Dict = {
                "x_T": x_T_i,
                "intermediates": ints_i,
                "source": os.path.abspath(path),
            }
            torch.save(entry, self._cache_path(path))
