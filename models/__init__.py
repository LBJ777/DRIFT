# models/__init__.py
# Convenience re-exports for the models sub-package.
# Sub-modules are imported lazily to avoid hard-failing when optional
# dependencies (e.g. scikit-learn) are not installed.

from .backbone.adm_wrapper import ADMBackbone
from .features.base import FeatureExtractor
from .heads.binary import BinaryDetectionHead

try:
    from .heads.attribution import GeneratorAttributionHead
except ImportError:
    pass  # scikit-learn may not be installed; attribution head unavailable

__all__ = [
    "ADMBackbone",
    "FeatureExtractor",
    "BinaryDetectionHead",
    "GeneratorAttributionHead",
]
