"""
models/backbone/adm_wrapper.py
------------------------------
Wraps the ADM (Ablated Diffusion Model) guided-diffusion codebase to provide a
clean, DRIFT-specific API for DDIM inversion.

Responsibility:
    Encapsulate model loading, DDIM timestep scheduling, and the forward
    inversion pass (x₀ → x_T) so that all downstream feature extractors
    receive a uniform interface regardless of ADM implementation details.

The underlying guided-diffusion code lives at:
    ``AIGCDetectBenchmark-main/preprocessing_model/``

This wrapper adds the preprocessing_model directory to ``sys.path`` at import
time so the guided_diffusion package can be found without any install step.
"""

from __future__ import annotations

import sys
import os
import logging
from typing import List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path bootstrapping — add guided-diffusion to sys.path
# ---------------------------------------------------------------------------
_PREPROCESSING_MODEL_DIR = os.path.join(
    os.path.dirname(__file__),  # models/backbone/
    "..", "..", "..",           # up to Deepfake_Project/
    "AIGCDetectBenchmark-main",
    "preprocessing_model",
)
_PREPROCESSING_MODEL_DIR = os.path.abspath(_PREPROCESSING_MODEL_DIR)

if _PREPROCESSING_MODEL_DIR not in sys.path:
    sys.path.insert(0, _PREPROCESSING_MODEL_DIR)


# ---------------------------------------------------------------------------
# ADMBackbone
# ---------------------------------------------------------------------------

class ADMBackbone:
    """Wrapper around the ADM UNet + SpacedDiffusion for DDIM inversion.

    Loads an unconditional ADM checkpoint and exposes a simple ``invert()``
    method that maps a batch of images x₀ to their noise-space endpoint x_T
    via the DDIM reverse ODE.  Intermediate latent states can optionally be
    returned for trajectory-based feature extractors (F2/F3/F4).

    A ``mock_invert()`` method is provided for unit-testing without a real
    checkpoint; it simply returns Gaussian noise with the correct shape.

    Args:
        model_path: Absolute path to the ADM ``.pt`` checkpoint file.
            Pass ``""`` or ``"mock"`` to operate in mock mode (no file loaded).
        device: PyTorch device string, e.g. ``"cuda"`` or ``"cpu"``.
        ddim_steps: Number of DDIM steps for the inversion.  The default of
            20 matches the DIRE paper.
        image_size: Spatial resolution expected by the model (default 256).
        sampling_strategy: ``"uniform"`` (evenly spaced timesteps) or
            ``"front_dense"`` (more steps concentrated near t=0).

    Example::

        backbone = ADMBackbone(
            model_path="/checkpoints/256x256_diffusion_uncond.pt",
            device="cuda",
            ddim_steps=20,
        )
        x_T, intermediates = backbone.invert(x0, return_intermediates=True)
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        ddim_steps: int = 20,
        image_size: int = 256,
        sampling_strategy: str = "uniform",
    ) -> None:
        self.model_path = model_path
        self.device = torch.device(device)
        self.ddim_steps = ddim_steps
        self.image_size = image_size
        self.sampling_strategy = sampling_strategy

        self._mock_mode: bool = model_path in ("", "mock")
        self._model = None
        self._diffusion = None

        if not self._mock_mode:
            self._load_model()
        else:
            logger.info(
                "ADMBackbone initialised in MOCK mode — "
                "invert() will return Gaussian noise."
            )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load the ADM UNet and SpacedDiffusion from *self.model_path*.

        Uses the ``script_util.create_model_and_diffusion`` factory from the
        guided-diffusion library.  The diffusion is configured for DDIM
        with ``self.ddim_steps`` steps.

        Raises:
            ImportError: If the guided-diffusion package cannot be found on
                ``sys.path``.
            FileNotFoundError: If *model_path* does not exist on disk.
        """
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(
                f"ADM checkpoint not found at '{self.model_path}'. "
                "Download it from: "
                "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/"
                "256x256_diffusion_uncond.pt"
            )

        try:
            from guided_diffusion import script_util, gaussian_diffusion
            from guided_diffusion.respace import SpacedDiffusion, space_timesteps
        except ImportError as exc:
            raise ImportError(
                "Cannot import guided_diffusion. "
                f"Expected package at: {_PREPROCESSING_MODEL_DIR}\n"
                f"Original error: {exc}"
            ) from exc

        logger.info("Loading ADM model from '%s' ...", self.model_path)

        # Build model + diffusion using the same defaults as DIRE
        model_kwargs = script_util.model_and_diffusion_defaults()
        model_kwargs.update(
            {
                "image_size": self.image_size,
                "timestep_respacing": f"ddim{self.ddim_steps}",
            }
        )

        self._model, self._diffusion = script_util.create_model_and_diffusion(
            **model_kwargs
        )

        state_dict = torch.load(self.model_path, map_location="cpu")
        self._model.load_state_dict(state_dict)
        self._model.to(self.device)
        self._model.eval()

        logger.info(
            "ADM model loaded successfully. "
            "DDIM steps=%d, image_size=%d, strategy=%s",
            self.ddim_steps,
            self.image_size,
            self.sampling_strategy,
        )

    # ------------------------------------------------------------------
    # Timestep scheduling
    # ------------------------------------------------------------------

    def _get_timestep_sequence(self) -> List[int]:
        """Return the ordered list of diffusion timesteps for inversion.

        The sequence goes from t=0 (clean) to t=T (noise), i.e. the
        *reverse* direction used in DDIM inversion.

        Returns:
            List of integer timestep indices in ascending order (0 → T).

        Raises:
            NotImplementedError: If ``sampling_strategy`` is unknown.
        """
        total_steps = 1000  # ADM base diffusion steps

        if self.sampling_strategy == "uniform":
            # Evenly spaced from 0 to total_steps
            stride = total_steps // self.ddim_steps
            timesteps = list(range(0, total_steps, stride))[: self.ddim_steps]

        elif self.sampling_strategy == "front_dense":
            # More timesteps concentrated near t=0 (early trajectory detail)
            # Use a quadratic schedule: more granularity at small t
            raw = np.linspace(0, 1, self.ddim_steps) ** 2
            timesteps = (raw * (total_steps - 1)).astype(int).tolist()
            # Deduplicate while preserving order
            seen = set()
            unique = []
            for t in timesteps:
                if t not in seen:
                    seen.add(t)
                    unique.append(t)
            timesteps = unique

        else:
            raise NotImplementedError(
                f"Unknown sampling_strategy '{self.sampling_strategy}'. "
                "Supported: 'uniform', 'front_dense'."
            )

        return sorted(timesteps)

    # ------------------------------------------------------------------
    # Core inversion
    # ------------------------------------------------------------------

    @torch.no_grad()
    def invert(
        self,
        x0: torch.Tensor,
        return_intermediates: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """Perform DDIM inversion: x₀ → x_T.

        Maps a batch of clean images to their noise-space endpoint by
        iterating the DDIM reverse ODE from t=0 to t=T.

        Args:
            x0: Input image batch ``[B, 3, H, W]`` with values in ``[-1, 1]``.
                The images must already be normalised to this range; use
                ``data.transforms.denormalize`` if coming from an ImageNet-
                normalised DataLoader.
            return_intermediates: If ``True``, also return all intermediate
                latent states ``[x_{t_1}, x_{t_2}, ..., x_T]`` as a list of
                tensors of the same shape as *x0*.  Required by F2/F3/F4
                feature extractors.

        Returns:
            A tuple ``(x_T, intermediates)`` where:

            * ``x_T`` — final noise tensor ``[B, 3, H, W]``.
            * ``intermediates`` — list of ``[B, 3, H, W]`` tensors if
              ``return_intermediates=True``, else ``None``.

        Raises:
            RuntimeError: If called in non-mock mode without a loaded model.
        """
        if self._mock_mode:
            return self.mock_invert(x0, return_intermediates=return_intermediates)

        if self._model is None or self._diffusion is None:
            raise RuntimeError(
                "Model not loaded. Call ADMBackbone.__init__ with a valid model_path."
            )

        x0 = x0.to(self.device)
        B = x0.shape[0]

        timesteps = self._get_timestep_sequence()
        intermediates: Optional[List[torch.Tensor]] = [] if return_intermediates else None

        # DDIM inversion: iterate forward from x_0 towards x_T
        # At each step, we use the DDIM deterministic forward map:
        #   x_{t+1} = sqrt(alpha_{t+1}) * x0_pred + sqrt(1 - alpha_{t+1}) * eps_pred
        xt = x0.clone()

        alphas_cumprod = torch.from_numpy(
            self._diffusion.alphas_cumprod.copy()
        ).float().to(self.device)

        for i, t_idx in enumerate(timesteps):
            t_tensor = torch.full(
                (B,), t_idx, device=self.device, dtype=torch.long
            )

            # Predict noise using ADM UNet
            model_output = self._model(xt, t_tensor)

            # If model predicts (eps, variance) concatenated, take first half
            if model_output.shape[1] == 6:  # learn_sigma=True
                eps_pred, _ = model_output.chunk(2, dim=1)
            else:
                eps_pred = model_output

            # DDIM deterministic inversion step
            alpha_t = alphas_cumprod[t_idx]
            if i + 1 < len(timesteps):
                alpha_next = alphas_cumprod[timesteps[i + 1]]
            else:
                alpha_next = torch.tensor(0.0, device=self.device)

            # Predicted x_0 from current xt and eps_pred
            x0_pred = (
                xt - (1 - alpha_t).sqrt() * eps_pred
            ) / alpha_t.sqrt()
            x0_pred = x0_pred.clamp(-1.0, 1.0)

            # DDIM forward step
            xt = alpha_next.sqrt() * x0_pred + (1 - alpha_next).sqrt() * eps_pred

            if return_intermediates and intermediates is not None:
                intermediates.append(xt.cpu())

        x_T = xt
        return x_T, intermediates

    # ------------------------------------------------------------------
    # Mock mode
    # ------------------------------------------------------------------

    @torch.no_grad()
    def mock_invert(
        self,
        x0: torch.Tensor,
        return_intermediates: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """Mock inversion that returns Gaussian noise without a real model.

        Useful for unit-testing feature extractors and the training pipeline
        without downloading the 3 GB ADM checkpoint.

        Args:
            x0: Input image batch ``[B, 3, H, W]``.  Values are ignored;
                only the shape is used to construct the output.
            return_intermediates: If ``True``, generate mock intermediate
                tensors whose count matches ``self.ddim_steps``.

        Returns:
            A tuple ``(x_T, intermediates)`` where both tensors contain
            standard Gaussian noise sampled on ``x0``'s device.
        """
        x_T = torch.randn_like(x0)

        intermediates: Optional[List[torch.Tensor]] = None
        if return_intermediates:
            intermediates = [
                torch.randn_like(x0) for _ in range(self.ddim_steps)
            ]

        return x_T, intermediates

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_mock(self) -> bool:
        """``True`` if operating in mock mode (no real checkpoint loaded)."""
        return self._mock_mode

    def __repr__(self) -> str:
        return (
            f"ADMBackbone("
            f"model_path={self.model_path!r}, "
            f"device={str(self.device)!r}, "
            f"ddim_steps={self.ddim_steps}, "
            f"image_size={self.image_size}, "
            f"mock={self._mock_mode}"
            f")"
        )
