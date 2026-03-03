"""
models/heads/attribution.py
---------------------------
Generator attribution head for DRIFT Phase 4.

Responsibility:
    Implement the two-stage S3 attribution strategy:

    Stage 1  — Unsupervised GMM clustering of feature vectors to discover
               latent generator clusters without label supervision.
               (Mirrors the GMM approach in SDAIE-main/oc_funs.py.)

    Stage 2  — Lightweight linear classifier trained on a small fraction
               of labelled data to align cluster indices to generator names.

The GMM fitting code is deliberately compatible with the scikit-learn
``GaussianMixture`` API used in SDAIE's ``train_GMM_sklearn`` function.
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GeneratorAttributionHead
# ---------------------------------------------------------------------------

class GeneratorAttributionHead(nn.Module):
    """Two-stage generator attribution head.

    Stage 1 (GMM) is unsupervised and does not require labels.
    Stage 2 (linear alignment) requires a small amount of labelled data to
    map the GMM cluster indices to interpretable generator names.

    Args:
        feature_dim: Dimensionality of the input feature vector.  Must match
            ``FeatureExtractor.feature_dim``.
        num_generators: Number of known generator classes (used only as
            a hint; the GMM can discover more latent components).

    Example::

        head = GeneratorAttributionHead(feature_dim=768, num_generators=14)

        # Stage 1: fit GMM on unlabelled features
        head.fit_gmm(all_features_np, n_components=15)

        # Stage 2: align clusters using a labelled subset
        head.align_labels(labelled_features_np, labelled_generator_names)

        # Predict generator for new features
        preds = head.predict(new_features_np)
    """

    def __init__(
        self,
        feature_dim: int,
        num_generators: int,
    ) -> None:
        super().__init__()

        self.feature_dim = feature_dim
        self.num_generators = num_generators

        # GMM model (scikit-learn) — fitted in stage 1
        self._gmm: Optional[GaussianMixture] = None

        # Linear alignment head (scikit-learn) — fitted in stage 2
        self._linear_clf: Optional[LogisticRegression] = None

        # Label encoder to map generator names ↔ integer indices
        self._label_encoder: LabelEncoder = LabelEncoder()

        # NN linear layer for gradient-based fine-tuning (optional, stage 2)
        self.nn_linear = nn.Linear(num_generators, num_generators, bias=True)

    # ------------------------------------------------------------------
    # Stage 1: GMM
    # ------------------------------------------------------------------

    def fit_gmm(
        self,
        features: np.ndarray,
        n_components: int = 15,
        init_params: str = "k-means++",
        random_state: int = 42,
        max_iter: int = 200,
    ) -> "GeneratorAttributionHead":
        """Fit a Gaussian Mixture Model on the provided feature vectors.

        This mirrors ``train_GMM_sklearn`` in ``SDAIE-main/oc_funs.py`` but
        exposes all relevant hyper-parameters and stores the trained model
        on ``self`` for downstream use.

        Args:
            features: Feature matrix ``[N, feature_dim]`` as a numpy array.
                No labels required.
            n_components: Number of GMM mixture components.  Should be >=
                ``self.num_generators`` to allow for intra-generator variation.
            init_params: GMM initialisation strategy passed to scikit-learn's
                ``GaussianMixture``.  ``"k-means++"`` is recommended for
                stable convergence.
            random_state: Random seed for reproducibility.
            max_iter: Maximum EM iterations.

        Returns:
            ``self`` (for method chaining).

        Raises:
            ValueError: If *features* has fewer dimensions than
                ``self.feature_dim``.
        """
        if features.ndim != 2:
            raise ValueError(
                f"features must be 2-D [N, feature_dim], got shape {features.shape}."
            )
        if features.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected feature_dim={self.feature_dim}, "
                f"got {features.shape[1]}."
            )

        logger.info(
            "Fitting GMM: n_components=%d, n_samples=%d, feature_dim=%d ...",
            n_components,
            len(features),
            self.feature_dim,
        )

        self._gmm = GaussianMixture(
            n_components=n_components,
            init_params=init_params,
            random_state=random_state,
            max_iter=max_iter,
        )
        self._gmm.fit(features)

        logger.info(
            "GMM fitted. Converged=%s, lower_bound=%.4f",
            self._gmm.converged_,
            self._gmm.lower_bound_,
        )
        return self

    def gmm_score_samples(self, features: np.ndarray) -> np.ndarray:
        """Compute the per-sample log-likelihood under the fitted GMM.

        Args:
            features: Feature matrix ``[N, feature_dim]``.

        Returns:
            Log-likelihood array of shape ``[N]``.

        Raises:
            RuntimeError: If ``fit_gmm()`` has not been called.
        """
        self._require_gmm()
        return self._gmm.score_samples(features)  # type: ignore[union-attr]

    def gmm_predict_cluster(self, features: np.ndarray) -> np.ndarray:
        """Assign each feature vector to the most likely GMM component.

        Args:
            features: Feature matrix ``[N, feature_dim]``.

        Returns:
            Integer array of cluster indices, shape ``[N]``.

        Raises:
            RuntimeError: If ``fit_gmm()`` has not been called.
        """
        self._require_gmm()
        return self._gmm.predict(features)  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Stage 2: Linear alignment
    # ------------------------------------------------------------------

    def align_labels(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        C: float = 1.0,
        max_iter: int = 1000,
    ) -> "GeneratorAttributionHead":
        """Train a logistic regression to align GMM clusters → generator labels.

        Uses the GMM cluster assignments as an intermediate representation
        (``[N, n_components]`` soft probabilities) and trains a linear
        classifier on top to map them to human-readable generator names.

        This two-stage approach mirrors SDAIE's strategy: unsupervised
        discovery first, then supervised alignment with minimal labels.

        Args:
            features: Feature matrix ``[N, feature_dim]`` as a numpy array.
            labels: String or integer array of generator names/IDs, shape
                ``[N]``.  Must be alignable via scikit-learn's
                ``LabelEncoder``.
            C: Regularisation inverse strength for logistic regression.
            max_iter: Max solver iterations.

        Returns:
            ``self`` (for method chaining).

        Raises:
            RuntimeError: If ``fit_gmm()`` has not been called first.
            ValueError: If *features* and *labels* have different lengths.
        """
        self._require_gmm()

        if len(features) != len(labels):
            raise ValueError(
                f"features and labels must have the same length, "
                f"got {len(features)} vs {len(labels)}."
            )

        logger.info(
            "Aligning GMM clusters to generator labels. "
            "n_samples=%d, unique_labels=%d ...",
            len(features),
            len(np.unique(labels)),
        )

        # Encode string labels to integers
        int_labels = self._label_encoder.fit_transform(labels)

        # Convert features to GMM soft probabilities
        gmm_probs = self._gmm.predict_proba(features)  # type: ignore[union-attr]

        # Fit logistic regression on soft probabilities
        self._linear_clf = LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=42,
            multi_class="multinomial",
            solver="lbfgs",
        )
        self._linear_clf.fit(gmm_probs, int_labels)

        logger.info(
            "Linear alignment fitted. Classes: %s",
            list(self._label_encoder.classes_),
        )
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict generator names for a batch of feature vectors.

        Args:
            features: Feature matrix ``[N, feature_dim]``.

        Returns:
            Array of predicted generator name strings, shape ``[N]``.

        Raises:
            RuntimeError: If either ``fit_gmm()`` or ``align_labels()`` has
                not been called.
        """
        self._require_gmm()
        self._require_linear_clf()

        gmm_probs = self._gmm.predict_proba(features)  # type: ignore[union-attr]
        int_preds = self._linear_clf.predict(gmm_probs)  # type: ignore[union-attr]
        return self._label_encoder.inverse_transform(int_preds)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return class probability matrix for a batch of feature vectors.

        Args:
            features: Feature matrix ``[N, feature_dim]``.

        Returns:
            Probability matrix ``[N, num_generators]``.

        Raises:
            RuntimeError: If either ``fit_gmm()`` or ``align_labels()`` has
                not been called.
        """
        self._require_gmm()
        self._require_linear_clf()

        gmm_probs = self._gmm.predict_proba(features)  # type: ignore[union-attr]
        return self._linear_clf.predict_proba(gmm_probs)  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist the GMM and linear classifier to disk using pickle.

        Args:
            path: File path (e.g. ``"./checkpoints/attribution_head.pkl"``).
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        state = {
            "feature_dim": self.feature_dim,
            "num_generators": self.num_generators,
            "gmm": self._gmm,
            "linear_clf": self._linear_clf,
            "label_encoder": self._label_encoder,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info("Attribution head saved to '%s'.", path)

    @classmethod
    def load(cls, path: str) -> "GeneratorAttributionHead":
        """Load a previously saved attribution head from disk.

        Args:
            path: Path to a ``.pkl`` file saved by ``save()``.

        Returns:
            A restored ``GeneratorAttributionHead`` instance.
        """
        with open(path, "rb") as f:
            state = pickle.load(f)

        head = cls(
            feature_dim=state["feature_dim"],
            num_generators=state["num_generators"],
        )
        head._gmm = state["gmm"]
        head._linear_clf = state["linear_clf"]
        head._label_encoder = state["label_encoder"]
        logger.info("Attribution head loaded from '%s'.", path)
        return head

    # ------------------------------------------------------------------
    # nn.Module forward (optional gradient path for fine-tuning)
    # ------------------------------------------------------------------

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Optional differentiable forward pass for gradient-based fine-tuning.

        In the default two-stage pipeline this is not called — use
        ``predict()`` / ``predict_proba()`` instead.  This method is
        provided so that the head can optionally be plugged into a
        ``nn.Module`` training loop for end-to-end fine-tuning.

        Args:
            features: Feature tensor ``[B, feature_dim]``.

        Returns:
            Logit tensor ``[B, num_generators]``.

        Raises:
            NotImplementedError: Always — implement in Phase 4 if needed.
        """
        raise NotImplementedError(
            "Differentiable forward pass to be implemented in Phase 4. "
            "Use predict() / predict_proba() for the GMM+linear pipeline."
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _require_gmm(self) -> None:
        if self._gmm is None:
            raise RuntimeError(
                "GMM has not been fitted. Call fit_gmm() before using this method."
            )

    def _require_linear_clf(self) -> None:
        if self._linear_clf is None:
            raise RuntimeError(
                "Linear classifier has not been fitted. "
                "Call align_labels() before using this method."
            )

    @property
    def is_fitted(self) -> bool:
        """``True`` if both GMM and linear classifier have been fitted."""
        return self._gmm is not None and self._linear_clf is not None

    @property
    def classes_(self) -> Optional[np.ndarray]:
        """Array of known generator class names (``None`` until aligned)."""
        if hasattr(self._label_encoder, "classes_"):
            return self._label_encoder.classes_
        return None

    def __repr__(self) -> str:
        return (
            f"GeneratorAttributionHead("
            f"feature_dim={self.feature_dim}, "
            f"num_generators={self.num_generators}, "
            f"gmm_fitted={self._gmm is not None}, "
            f"linear_fitted={self._linear_clf is not None}"
            f")"
        )
