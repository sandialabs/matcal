"""
Public API for SIERRA models.

This package replaces the legacy monolithic `matcal.sierra.models` module
while preserving the same import paths for public classes.

Examples:
    from matcal.sierra.models import RoundUniaxialTensionModel
"""

from .base import UserDefinedSierraModel
from .material_point import UniaxialLoadingMaterialPointModel
from .tension import (
    RoundUniaxialTensionModel,
    RectangularUniaxialTensionModel,
    RoundNotchedTensionModel,
)
from .shear import SolidBarTorsionModel, TopHatShearModel
from .vfm import VFMUniaxialTensionHexModel, VFMUniaxialTensionConnectedHexModel

__all__ = [
    "UserDefinedSierraModel",
    "UniaxialLoadingMaterialPointModel",
    "RoundUniaxialTensionModel",
    "RectangularUniaxialTensionModel",
    "RoundNotchedTensionModel",
    "SolidBarTorsionModel",
    "TopHatShearModel",
    "VFMUniaxialTensionHexModel",
    "VFMUniaxialTensionConnectedHexModel",
]