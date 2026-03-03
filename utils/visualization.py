"""
utils/visualization.py
----------------------
Visualisation utilities for DRIFT experiment analysis.

Responsibility:
    Provide plotting functions for key diagnostic figures mentioned in the
    paper, including:

    - ``plot_tsne``               — 2D t-SNE scatter of feature vectors
    - ``plot_psd_comparison``     — PSD curves per source generator
    - ``plot_wasserstein_heatmap``— pairwise Wasserstein distance matrix
    - ``plot_trajectory``         — DDIM inversion trajectory visualisation

All functions save figures to disk (PNG) and optionally return the
``matplotlib.figure.Figure`` object for use in notebooks.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

logger = logging.getLogger(__name__)

# Matplotlib style — use a clean theme if available
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    pass  # Older matplotlib — default style is fine


# ---------------------------------------------------------------------------
# t-SNE
# ---------------------------------------------------------------------------

def plot_tsne(
    features: np.ndarray,
    labels: Sequence,
    output_path: str,
    perplexity: float = 30.0,
    n_iter: int = 1000,
    random_state: int = 42,
    title: str = "Feature Space (t-SNE)",
    figsize: Tuple[int, int] = (10, 8),
    point_size: float = 8.0,
    alpha: float = 0.7,
    dpi: int = 150,
) -> plt.Figure:
    """Compute and plot a 2-D t-SNE embedding of *features*.

    Uses ``sklearn.manifold.TSNE`` for dimensionality reduction.  Each unique
    label value is assigned a distinct colour.

    Args:
        features: Feature matrix ``[N, D]``.
        labels: Sequence of N label values (strings or integers).  Used to
            colour the scatter plot.
        output_path: Absolute path where the PNG figure is saved.
        perplexity: t-SNE perplexity hyper-parameter.
        n_iter: Number of t-SNE optimisation iterations.
        random_state: Random seed for reproducibility.
        title: Plot title string.
        figsize: Figure size in inches ``(width, height)``.
        point_size: Scatter plot marker size.
        alpha: Marker transparency.
        dpi: Output resolution in dots per inch.

    Returns:
        The ``matplotlib.figure.Figure`` object.

    Example::

        plot_tsne(features_np, label_array, output_path="./results/tsne.png")
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for plot_tsne. "
            "Install with: pip install scikit-learn"
        ) from exc

    logger.info(
        "Computing t-SNE: n_samples=%d, feature_dim=%d ...",
        len(features),
        features.shape[1],
    )

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    )
    embedding = tsne.fit_transform(features)

    labels_arr = np.asarray(labels)
    unique_labels = sorted(set(labels_arr.tolist()))
    cmap = plt.get_cmap("tab20")
    color_map = {lbl: cmap(i / max(len(unique_labels) - 1, 1)) for i, lbl in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=figsize)
    for lbl in unique_labels:
        mask = labels_arr == lbl
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=[color_map[lbl]],
            label=str(lbl),
            s=point_size,
            alpha=alpha,
        )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("t-SNE 1", fontsize=12)
    ax.set_ylabel("t-SNE 2", fontsize=12)
    ax.legend(
        markerscale=3,
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        borderaxespad=0,
        fontsize=8,
    )
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    logger.info("t-SNE plot saved to '%s'.", output_path)
    return fig


# ---------------------------------------------------------------------------
# PSD comparison
# ---------------------------------------------------------------------------

def plot_psd_comparison(
    xT_by_source: Dict[str, np.ndarray],
    output_path: str,
    figsize: Tuple[int, int] = (10, 5),
    dpi: int = 150,
    log_scale: bool = True,
    title: str = "PSD Comparison of x_T across Sources",
) -> plt.Figure:
    """Plot the azimuthally-averaged 2-D Power Spectral Density of x_T images.

    For each source key in *xT_by_source*, the function computes the average
    radial PSD across all provided samples and overlays the curves on a single
    frequency-domain plot.

    Args:
        xT_by_source: Dict mapping source name → numpy array of shape
            ``[N, C, H, W]`` or ``[N, H, W]`` containing x_T images.
        output_path: Absolute path for the saved PNG file.
        figsize: Figure size in inches.
        dpi: Output resolution.
        log_scale: If ``True``, use log scale on the y-axis.
        title: Plot title string.

    Returns:
        The ``matplotlib.figure.Figure`` object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.get_cmap("tab10")

    for i, (source_name, images) in enumerate(xT_by_source.items()):
        images = np.asarray(images)

        # Flatten channel dim if present
        if images.ndim == 4:
            # Average over channels to get grayscale PSD
            images = images.mean(axis=1)  # [N, H, W]

        radial_psds: List[np.ndarray] = []
        for img in images:
            psd = _compute_radial_psd(img)
            radial_psds.append(psd)

        mean_psd = np.mean(np.stack(radial_psds, axis=0), axis=0)
        freqs = np.arange(len(mean_psd))

        ax.plot(
            freqs,
            mean_psd,
            label=source_name,
            color=cmap(i % 10),
            linewidth=1.5,
            alpha=0.85,
        )

    if log_scale:
        ax.set_yscale("log")

    ax.set_xlabel("Radial Frequency (pixels)", fontsize=12)
    ax.set_ylabel("Power Spectral Density", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=9, bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    logger.info("PSD comparison plot saved to '%s'.", output_path)
    return fig


def _compute_radial_psd(img: np.ndarray) -> np.ndarray:
    """Compute the 1-D radially-averaged PSD of a 2-D image.

    Args:
        img: 2-D float array ``[H, W]``.

    Returns:
        1-D array of length ``min(H, W) // 2`` containing mean power per
        radial frequency bin.
    """
    H, W = img.shape[-2], img.shape[-1]
    if img.ndim > 2:
        img = img[0]  # Take first channel/slice

    # 2-D FFT and power spectrum
    f = np.fft.fft2(img.astype(np.float64))
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift) ** 2

    # Build radial frequency grid
    cy, cx = H // 2, W // 2
    y_idx, x_idx = np.ogrid[:H, :W]
    r = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2).astype(int)

    max_r = min(cy, cx)
    radial_psd = np.zeros(max_r)
    for radius in range(max_r):
        mask = r == radius
        if mask.any():
            radial_psd[radius] = magnitude_spectrum[mask].mean()

    return radial_psd


# ---------------------------------------------------------------------------
# Wasserstein heatmap
# ---------------------------------------------------------------------------

def plot_wasserstein_heatmap(
    distance_matrix: np.ndarray,
    labels: Sequence[str],
    output_path: str,
    figsize: Optional[Tuple[int, int]] = None,
    dpi: int = 150,
    cmap: str = "viridis",
    title: str = "Pairwise Wasserstein Distance Matrix",
) -> plt.Figure:
    """Plot a heatmap of pairwise Wasserstein distances between generators.

    Args:
        distance_matrix: Square matrix ``[N, N]`` of non-negative pairwise
            distances.  Typically computed over x_T distributions.
        labels: Ordered list of N generator labels for axis ticks.
        output_path: Absolute path for the saved PNG file.
        figsize: Figure size in inches.  Defaults to ``(N, N)`` scaled.
        dpi: Output resolution.
        cmap: Matplotlib colormap name.
        title: Plot title.

    Returns:
        The ``matplotlib.figure.Figure`` object.
    """
    n = len(labels)
    if figsize is None:
        size = max(6, n)
        figsize = (size, int(size * 0.85))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(distance_matrix, cmap=cmap, aspect="auto")
    plt.colorbar(im, ax=ax, label="Wasserstein Distance")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(title, fontsize=13)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = distance_matrix[i, j]
            text_color = "white" if val > (distance_matrix.max() * 0.6) else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=6, color=text_color)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    logger.info("Wasserstein heatmap saved to '%s'.", output_path)
    return fig


# ---------------------------------------------------------------------------
# Trajectory plot
# ---------------------------------------------------------------------------

def plot_trajectory(
    intermediates: List[np.ndarray],
    output_path: str,
    max_frames: int = 10,
    figsize: Optional[Tuple[int, int]] = None,
    dpi: int = 150,
    title: str = "DDIM Inversion Trajectory",
) -> plt.Figure:
    """Visualise the DDIM inversion trajectory as a grid of intermediate images.

    Each frame in *intermediates* is rescaled to ``[0, 1]`` and displayed in
    a single row.  Useful for qualitative inspection of the inversion process.

    Args:
        intermediates: Ordered list of intermediate latent tensors.  Each
            element should be a numpy array of shape ``[3, H, W]`` or
            ``[H, W]`` representing a single image.
        output_path: Absolute path for the saved PNG file.
        max_frames: Maximum number of frames to display.  If there are more
            intermediates, evenly-spaced frames are selected.
        figsize: Figure size in inches.  Auto-computed if ``None``.
        dpi: Output resolution.
        title: Plot title.

    Returns:
        The ``matplotlib.figure.Figure`` object.
    """
    import math

    n_total = len(intermediates)
    if n_total == 0:
        logger.warning("plot_trajectory: empty intermediates list.")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No intermediates", ha="center", va="center")
        fig.savefig(output_path, dpi=dpi)
        return fig

    # Sub-sample evenly if needed
    if n_total > max_frames:
        indices = np.linspace(0, n_total - 1, max_frames, dtype=int).tolist()
    else:
        indices = list(range(n_total))

    n_cols = len(indices)
    if figsize is None:
        figsize = (n_cols * 2, 2.5)

    fig, axes = plt.subplots(1, n_cols, figsize=figsize)
    if n_cols == 1:
        axes = [axes]

    for plot_idx, frame_idx in enumerate(indices):
        frame = np.asarray(intermediates[frame_idx], dtype=np.float32)

        # Handle [C, H, W] → [H, W, C]
        if frame.ndim == 3 and frame.shape[0] in (1, 3):
            frame = np.transpose(frame, (1, 2, 0))

        # Normalise to [0, 1] for display
        frame_min, frame_max = frame.min(), frame.max()
        if frame_max > frame_min:
            frame = (frame - frame_min) / (frame_max - frame_min)
        else:
            frame = np.zeros_like(frame)

        ax = axes[plot_idx]
        if frame.shape[-1] == 1:
            ax.imshow(frame[..., 0], cmap="gray")
        else:
            ax.imshow(frame.clip(0, 1))

        # Label by original frame index (timestep)
        t_fraction = frame_idx / max(n_total - 1, 1)
        ax.set_title(f"t={t_fraction:.2f}", fontsize=8)
        ax.axis("off")

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    logger.info("Trajectory plot saved to '%s'.", output_path)
    return fig
