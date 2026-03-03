"""
models/features/base.py
-----------------------
Abstract base class for all DRIFT feature extraction schemes.

Responsibility:
    Define the contract that Phase 1 and Phase 2 Agents must fulfill when
    implementing feature schemes F1, F2, F3, and F4.  All concrete
    implementations must inherit from ``FeatureExtractor`` and implement:

    * ``extract()`` — maps DDIM inversion outputs to a fixed-length feature
      vector.
    * ``feature_dim`` — property returning the output dimensionality.

Feature scheme overview (implementations live in Phase 1–2):

    F1 — Per-channel statistics of the terminal noise x_T
         (mean, std, skewness, kurtosis → compact fixed-size vector).

    F2 — Trajectory-level statistics computed from the sequence of
         intermediate latent states [x_{t1}, ..., x_T].

    F3 — Power-spectral-density (PSD) profile of x_T mapped to a
         frequency-domain descriptor.

    F4 — Wasserstein distances between consecutive intermediate
         distributions along the DDIM trajectory.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import torch


class FeatureExtractor(ABC):
    """Abstract base class for all DRIFT feature extraction schemes (F1–F4).

    Phase 1 and Phase 2 Agents must subclass this class and implement
    both ``extract()`` and the ``feature_dim`` property.

    Concrete sub-classes may add their own ``__init__`` parameters (e.g.
    number of frequency bins, which statistical moments to compute) but must
    call ``super().__init__()`` to ensure correct ABC registration.

    Example (minimal concrete implementation)::

        class MyFeatureExtractor(FeatureExtractor):
            @property
            def feature_dim(self) -> int:
                return 12

            def extract(
                self,
                x_T: torch.Tensor,
                intermediates: list | None = None,
            ) -> torch.Tensor:
                B = x_T.shape[0]
                mean = x_T.view(B, -1).mean(dim=1, keepdim=True)
                std  = x_T.view(B, -1).std(dim=1, keepdim=True)
                return torch.cat([mean, std], dim=1)
    """

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def extract(
        self,
        x_T: torch.Tensor,
        intermediates: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Extract a fixed-length feature vector from DDIM inversion outputs.

        Args:
            x_T: Terminal noise tensor output by
                ``ADMBackbone.invert()``.  Shape ``[B, 3, H, W]``.
            intermediates: Ordered list of intermediate latent tensors
                ``[x_{t_1}, ..., x_{t_{K-1}}, x_T]`` returned when
                ``ADMBackbone.invert(return_intermediates=True)`` is called.
                ``None`` is acceptable for schemes that only need ``x_T``
                (e.g. F1, F3).

        Returns:
            Feature tensor of shape ``[B, self.feature_dim]``.  Each row
            is the descriptor for one image in the batch.

        Raises:
            NotImplementedError: Always — must be overridden in a sub-class.
        """
        raise NotImplementedError("To be implemented in Phase 1")

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Dimensionality of the feature vector returned by ``extract()``.

        Returns:
            A positive integer representing the length of the output
            descriptor for a single image.

        Raises:
            NotImplementedError: Always — must be overridden in a sub-class.
        """
        raise NotImplementedError("To be implemented in Phase 1")

    # ------------------------------------------------------------------
    # Convenience helpers (non-abstract, available to all sub-classes)
    # ------------------------------------------------------------------

    def __call__(
        self,
        x_T: torch.Tensor,
        intermediates: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Shorthand for ``self.extract(x_T, intermediates)``.

        Allows the extractor to be used as a callable, e.g.::

            features = extractor(x_T, intermediates)

        Args:
            x_T: Terminal noise tensor ``[B, 3, H, W]``.
            intermediates: Optional list of intermediate tensors.

        Returns:
            Feature tensor ``[B, feature_dim]``.
        """
        return self.extract(x_T, intermediates)

    def validate_output(self, features: torch.Tensor) -> None:
        """Assert that *features* has the expected shape ``[B, feature_dim]``.

        Call this inside ``extract()`` before returning to catch shape bugs
        early during development.

        Args:
            features: Candidate output of ``extract()``.

        Raises:
            ValueError: If *features* does not have exactly 2 dimensions, or
                if the last dimension does not match ``self.feature_dim``.
        """
        if features.ndim != 2:
            raise ValueError(
                f"FeatureExtractor.extract() must return a 2-D tensor "
                f"[B, feature_dim], got shape {tuple(features.shape)}."
            )
        if features.shape[1] != self.feature_dim:
            raise ValueError(
                f"Output feature dim mismatch: expected {self.feature_dim}, "
                f"got {features.shape[1]}."
            )

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        try:
            dim = self.feature_dim
        except NotImplementedError:
            dim = "?"
        return f"{cls}(feature_dim={dim})"
