# models/heads/__init__.py

from .binary import BinaryDetectionHead
from .attribution import GeneratorAttributionHead

__all__ = ["BinaryDetectionHead", "GeneratorAttributionHead"]
