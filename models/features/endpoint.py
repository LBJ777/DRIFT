"""
models/features/endpoint.py
----------------------------
F1: Endpoint feature extractor for DRIFT Phase 1.

Extracts a 60-dimensional (padded to 128) descriptor from the DDIM inversion
terminal noise x_T.  All operations run on GPU tensors without numpy
conversion.

Feature groups
--------------
G1  Statistical moments        (12 dim): per-channel mean, std, skewness, kurtosis
G2  Gaussianity deviation       (3 dim): per-channel KS-like statistic vs N(0,1)
G3  Spatial auto-correlation    (9 dim): per-channel lag-[1,2,4] ACF in H and W
G4  Radial PSD (8 bands × 3ch)(24 dim): rfft2 power spectrum in 8 frequency rings
G5  VAE periodicity indicator  (12 dim): energy at H/8 ± 1 and W/8 ± 1 in rfft2

Total raw:  12 + 3 + 9 + 24 + 12 = 60 dim, zero-padded to feature_dim=128.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import FeatureExtractor


class EndpointFeatureExtractor(FeatureExtractor):
    """F1 endpoint descriptor extracted from the DDIM inversion terminal noise.

    Inherits from :class:`FeatureExtractor` and implements all five feature
    groups described in the module docstring.  The output vector is L2-
    normalised before returning so that downstream linear classifiers work
    without manual scaling.

    Args:
        feature_dim: Output dimensionality.  Raw features are 60-D; this
            value must be >= 60 so zero-padding can reach it.  Default 128.
        n_freq_bands: Number of radial frequency bands in G4 (default 8).
        normalize: If ``True`` (default), apply L2 normalisation to the
            output vector.

    Example::

        extractor = EndpointFeatureExtractor()
        features = extractor.extract(x_T)  # [B, 128]
    """

    # Raw feature count before padding
    _RAW_DIM: int = 60  # 12 + 3 + 9 + 24 + 12

    def __init__(
        self,
        feature_dim: int = 128,
        n_freq_bands: int = 8,
        normalize: bool = True,
    ) -> None:
        super().__init__()

        if feature_dim < self._RAW_DIM:
            raise ValueError(
                f"feature_dim must be >= {self._RAW_DIM} (raw feature size), "
                f"got {feature_dim}."
            )

        self._feature_dim = feature_dim
        self.n_freq_bands = n_freq_bands
        self.normalize = normalize

        # LayerNorm applied to raw features before zero-padding
        self._layer_norm = nn.LayerNorm(self._RAW_DIM)

    # ------------------------------------------------------------------
    # FeatureExtractor interface
    # ------------------------------------------------------------------

    @property
    def feature_dim(self) -> int:
        """Output feature dimensionality (default 128)."""
        return self._feature_dim

    def extract(
        self,
        x_T: torch.Tensor,
        intermediates: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Extract a 128-D descriptor from the DDIM inversion terminal noise.

        Args:
            x_T: Terminal noise tensor ``[B, 3, H, W]``.  Values are expected
                to lie roughly in ``[-3, 3]`` (standard Gaussian noise).
            intermediates: Ignored by F1 (passed for interface compatibility).

        Returns:
            Feature tensor ``[B, feature_dim]``.
        """
        B = x_T.shape[0]

        g1 = self._group1_moments(x_T)          # [B, 12]
        g2 = self._group2_gaussianity(x_T)      # [B,  3]
        g3 = self._group3_spatial_acf(x_T)      # [B,  9]
        g4 = self._group4_radial_psd(x_T)       # [B, 24]
        g5 = self._group5_vae_periodicity(x_T)  # [B, 12]

        raw = torch.cat([g1, g2, g3, g4, g5], dim=1)  # [B, 60]

        # Normalise scale with LayerNorm (per-sample, across feature dim)
        # Move LayerNorm to same device as input to handle MPS/CUDA
        raw = self._layer_norm.to(raw.device)(raw)

        # Zero-pad to feature_dim
        pad_size = self._feature_dim - self._RAW_DIM
        if pad_size > 0:
            padding = torch.zeros(B, pad_size, device=x_T.device, dtype=raw.dtype)
            features = torch.cat([raw, padding], dim=1)  # [B, feature_dim]
        else:
            features = raw

        # Optional L2 normalisation
        if self.normalize:
            features = F.normalize(features, p=2, dim=1)

        self.validate_output(features)
        return features

    # ------------------------------------------------------------------
    # G1 — Per-channel statistical moments (12 dim)
    # ------------------------------------------------------------------

    def _group1_moments(self, x_T: torch.Tensor) -> torch.Tensor:
        """Compute mean, std, skewness, kurtosis for each of the 3 channels.

        Args:
            x_T: ``[B, 3, H, W]``

        Returns:
            ``[B, 12]``
        """
        B, C, H, W = x_T.shape
        # Flatten spatial dims: [B, C, N]
        x = x_T.view(B, C, -1)

        mu = x.mean(dim=2)               # [B, C]
        std = x.std(dim=2, unbiased=False) + 1e-8  # [B, C]

        # Centre
        x_c = x - mu.unsqueeze(2)        # [B, C, N]

        skew = (x_c ** 3).mean(dim=2) / (std ** 3)       # [B, C]
        kurt = (x_c ** 4).mean(dim=2) / (std ** 4) - 3.0  # [B, C]

        # Concatenate: [B, C, 4] → [B, 12]
        moments = torch.stack([mu, std, skew, kurt], dim=2)  # [B, C, 4]
        return moments.view(B, -1)  # [B, 12]

    # ------------------------------------------------------------------
    # G2 — Gaussianity deviation (3 dim)
    # ------------------------------------------------------------------

    def _group2_gaussianity(self, x_T: torch.Tensor) -> torch.Tensor:
        """Approximate KS statistic vs N(0,1) for each channel.

        The exact KS statistic requires sorting (O(N log N)) which is
        expensive on GPU.  We use a quantile-based approximation:
        sample 256 order statistics and compare them to the theoretical
        N(0,1) quantiles, taking the maximum absolute deviation.

        Args:
            x_T: ``[B, 3, H, W]``

        Returns:
            ``[B, 3]``
        """
        B, C, H, W = x_T.shape
        x = x_T.view(B, C, -1)  # [B, C, N]
        N = x.shape[2]

        # Sort along spatial dimension to get empirical CDF values
        x_sorted = x.sort(dim=2).values  # [B, C, N]

        # Subsample to 256 equally-spaced order statistics for efficiency
        n_pts = min(256, N)
        indices = torch.linspace(0, N - 1, n_pts, device=x_T.device).long()
        x_sub = x_sorted[:, :, indices]  # [B, C, n_pts]

        # Empirical CDF values at the sampled order statistics
        ecdf = (indices.float() + 0.5) / N  # [n_pts]

        # Theoretical N(0,1) quantiles via error function approximation
        # Use the Beasley-Springer-Moro approximation to inverse normal CDF
        theoretical = self._approx_norm_ppf(ecdf)  # [n_pts]
        theoretical = theoretical.to(x_T.device).view(1, 1, n_pts)

        # KS statistic: max |F_empirical(x) - F_N(0,1)(x)|
        # We evaluate the normal CDF at sampled x values, then diff from ecdf
        norm_cdf_at_x = 0.5 * (1.0 + torch.erf(x_sub / (2.0 ** 0.5)))  # [B, C, n_pts]
        ecdf_bc = ecdf.view(1, 1, n_pts).to(x_T.device)

        ks = (norm_cdf_at_x - ecdf_bc).abs().max(dim=2).values  # [B, C]
        return ks  # [B, 3]

    @staticmethod
    def _approx_norm_ppf(p: torch.Tensor) -> torch.Tensor:
        """Rational approximation to the standard normal inverse CDF (ppf).

        Uses the Peter Acklam rational approximation which achieves
        maximum error < 1.15e-9 over the entire (0, 1) range.

        Args:
            p: Probability tensor with values in (0, 1).

        Returns:
            Tensor of the same shape containing N(0,1) quantiles.
        """
        # Coefficients for the rational approximation
        a = [-3.969683028665376e+01,  2.209460984245205e+02,
             -2.759285104469687e+02,  1.383577518672690e+02,
             -3.066479806614716e+01,  2.506628277459239e+00]
        b = [-5.447609879822406e+01,  1.615858368580409e+02,
             -1.556989798598866e+02,  6.680131188771972e+01,
             -1.328068155288572e+01]
        c = [-7.784894002430293e-03, -3.223964580411365e-01,
             -2.400758277161838e+00, -2.549732539343734e+00,
              4.374664141464968e+00,  2.938163982698783e+00]
        d = [7.784695709041462e-03,  3.224671290700398e-01,
             2.445134137142996e+00,  3.754408661907416e+00]

        p_low  = 0.02425
        p_high = 1 - p_low

        q = torch.where(p < p_high, p, 1 - p)

        # Region 1: lower tail (and by symmetry, upper tail)
        lo = (q < p_low)
        t_lo = (-2 * q.clamp(min=1e-10).log()).sqrt()
        num_lo = (((((c[0]*t_lo + c[1])*t_lo + c[2])*t_lo + c[3])*t_lo + c[4])*t_lo + c[5])
        den_lo = ((((d[0]*t_lo + d[1])*t_lo + d[2])*t_lo + d[3])*t_lo + 1)
        x_lo = num_lo / den_lo

        # Region 2: central region
        q2 = q - 0.5
        r = q2 * q2
        num_c = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q2
        den_c = ((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1
        x_c = num_c / den_c

        x_ppf = torch.where(lo, x_lo, x_c)

        # Reflect for upper tail
        x_ppf = torch.where(p < p_high, x_ppf, -x_ppf)
        return x_ppf

    # ------------------------------------------------------------------
    # G3 — Spatial auto-correlation (9 dim)
    # ------------------------------------------------------------------

    def _group3_spatial_acf(self, x_T: torch.Tensor) -> torch.Tensor:
        """Normalised spatial auto-correlation at lags [1, 2, 4].

        Uses depthwise ``conv2d`` to compute the spatial cross-correlation
        between x_T and a shifted version of itself, one channel at a time.
        Both horizontal and vertical shifts are computed; the two values are
        averaged to produce one scalar per (channel, lag) pair.

        Args:
            x_T: ``[B, 3, H, W]``

        Returns:
            ``[B, 9]`` (3 channels × 3 lags)
        """
        B, C, H, W = x_T.shape
        lags = [1, 2, 4]
        acf_values = []

        for lag in lags:
            lag_acfs_ch = []
            for ch in range(C):
                xc = x_T[:, ch:ch+1, :, :]  # [B, 1, H, W]

                # Normalise per sample to zero mean, unit variance
                mu_c = xc.mean(dim=(2, 3), keepdim=True)
                std_c = xc.std(dim=(2, 3), keepdim=True) + 1e-8
                xn = (xc - mu_c) / std_c  # [B, 1, H, W]

                N = H * W  # number of spatial locations

                # Horizontal lag: correlate xn[..., :-lag] with xn[..., lag:]
                x_left  = xn[:, :, :, :-lag]  # [B, 1, H, W-lag]
                x_right = xn[:, :, :, lag:]   # [B, 1, H, W-lag]
                acf_h = (x_left * x_right).mean(dim=(2, 3))  # [B, 1]

                # Vertical lag: correlate xn[..., :-lag, :] with xn[..., lag:, :]
                x_top = xn[:, :, :-lag, :]   # [B, 1, H-lag, W]
                x_bot = xn[:, :, lag:, :]    # [B, 1, H-lag, W]
                acf_v = (x_top * x_bot).mean(dim=(2, 3))  # [B, 1]

                # Average H and V directions
                acf_ch = (acf_h + acf_v) * 0.5  # [B, 1]
                lag_acfs_ch.append(acf_ch)

            # Stack across channels: [B, C]
            acf_lag = torch.cat(lag_acfs_ch, dim=1)  # [B, 3]
            acf_values.append(acf_lag)

        # [B, 3 lags × 3 channels] = [B, 9]
        return torch.cat(acf_values, dim=1)

    # ------------------------------------------------------------------
    # G4 — Global frequency-domain statistics (24 dim)
    # ------------------------------------------------------------------

    def _group4_radial_psd(self, x_T: torch.Tensor) -> torch.Tensor:
        """Compute power spectral density in radial frequency bands.

        Each channel is transformed with rfft2; the power spectrum is then
        binned into ``n_freq_bands`` annular rings.  Mean and std are computed
        per band per channel → 2 × n_freq_bands × C = 2 × 8 × 3 = 48 values.

        Wait — per the specification: "8频带 × 3通道 = 24维 (mean only per band)"
        The spec says "每带取均值和标准差" for 24 total, so 8 bands × 3 ch = 24
        means only mean is kept.  We keep the mean to match the target 24-D.

        Args:
            x_T: ``[B, 3, H, W]``

        Returns:
            ``[B, 24]``
        """
        B, C, H, W = x_T.shape

        # rfft2 output shape: [B, C, H, W//2+1]
        freq = torch.fft.rfft2(x_T, norm="ortho")
        power = freq.real ** 2 + freq.imag ** 2  # [B, C, H, W//2+1]

        Hf, Wf = power.shape[2], power.shape[3]

        # Build radial frequency grid
        fy = torch.fft.fftfreq(H, device=x_T.device).view(-1, 1).expand(Hf, Wf)[:Hf, :]
        fx = torch.fft.rfftfreq(W, device=x_T.device).view(1, -1).expand(Hf, Wf)
        r = (fx ** 2 + fy ** 2).sqrt()  # [Hf, Wf], range [0, ~0.7]

        r_max = r.max().clamp(min=1e-8)
        r_norm = r / r_max  # normalise to [0, 1]

        # Bin edges
        bins = torch.linspace(0.0, 1.0, self.n_freq_bands + 1, device=x_T.device)

        band_means = []
        for b in range(self.n_freq_bands):
            lo, hi = bins[b], bins[b + 1]
            if b == self.n_freq_bands - 1:
                # Include the upper edge in the last bin
                mask = (r_norm >= lo) & (r_norm <= hi)
            else:
                mask = (r_norm >= lo) & (r_norm < hi)

            mask_bc = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, Hf, Wf]
            count = mask.float().sum().clamp(min=1.0)

            # Mean power per channel in this band
            # power: [B, C, Hf, Wf], mask broadcasts over B and C
            band_power = (power * mask_bc.float()).sum(dim=(2, 3)) / count  # [B, C]
            band_means.append(band_power)

        # [B, n_freq_bands, C] → [B, n_freq_bands × C]
        stacked = torch.stack(band_means, dim=1)  # [B, n_bands, C]
        g4 = stacked.view(B, self.n_freq_bands * C)  # [B, 24]

        # Log-scale to compress dynamic range
        g4 = torch.log1p(g4)
        return g4

    # ------------------------------------------------------------------
    # G5 — VAE periodicity indicator (12 dim)
    # ------------------------------------------------------------------

    def _group5_vae_periodicity(self, x_T: torch.Tensor) -> torch.Tensor:
        """Extract spectral energy at the SD-VAE 8px periodicity frequency.

        SD (Stable Diffusion) and DALL-E generate images with latent codes
        compressed by a factor of 8, introducing a characteristic spectral
        peak at spatial frequency f = 1/8 pixels.  We extract the energy in
        a narrow window around H//8 and W//8 in the rfft2 output.

        Per channel we extract 4 values:
          (H/8 - 1, W/8), (H/8 + 1, W/8), (H/8, W/8 - 1), (H/8, W/8 + 1)
        → 4 × 3 channels = 12 dim.

        Args:
            x_T: ``[B, 3, H, W]``

        Returns:
            ``[B, 12]``
        """
        B, C, H, W = x_T.shape

        freq = torch.fft.rfft2(x_T, norm="ortho")       # [B, C, H, W//2+1]
        power = freq.real ** 2 + freq.imag ** 2          # [B, C, H, W//2+1]

        Hf = power.shape[2]
        Wf = power.shape[3]

        h_peak = H // 8  # row index corresponding to 1/8 frequency
        w_peak = W // 8  # col index

        # Clamp offsets to stay within valid bounds
        rows = [
            max(0, h_peak - 1),
            min(Hf - 1, h_peak + 1),
        ]
        cols = [
            max(0, w_peak - 1),
            min(Wf - 1, w_peak + 1),
        ]

        # Collect 4 index pairs: row ± 1 at w_peak, w ± 1 at h_peak
        idx_pairs = [
            (rows[0], w_peak),
            (rows[1], w_peak),
            (h_peak, cols[0]),
            (h_peak, cols[1]),
        ]

        # Clamp all indices to valid range
        idx_pairs = [
            (min(r, Hf - 1), min(c, Wf - 1))
            for r, c in idx_pairs
        ]

        vals = []
        for (ri, ci) in idx_pairs:
            v = power[:, :, ri, ci]  # [B, C]
            vals.append(v)

        # [B, 4, C] → [B, 4 × C] = [B, 12]
        g5 = torch.stack(vals, dim=1).view(B, -1)  # [B, 12]
        g5 = torch.log1p(g5)
        return g5
