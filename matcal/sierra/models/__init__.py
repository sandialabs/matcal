"""
Public API for SIERRA models.

This package replaces the legacy monolithic `matcal.sierra.models` module
while preserving the same import paths for public classes.

Examples:
    from matcal.sierra.models import RoundUniaxialTensionModel
"""

from .base import UserDefinedSierraModel
from .material_point import (
    SimpleShearMaterialPointModel,
    UniaxialLoadingMaterialPointModel,
    UniaxialStrainMaterialPointModel,
    UniaxialStressMaterialPointModel,
)
from .shear import SolidBarTorsionModel, TopHatShearModel
from .tension import (
    RectangularUniaxialTensionModel,
    RoundNotchedTensionModel,
    RoundUniaxialTensionModel,
)
from .vfm import VFMUniaxialTensionConnectedHexModel, VFMUniaxialTensionHexModel

__all__ = [
    "RectangularUniaxialTensionModel",
    "RoundNotchedTensionModel",
    "RoundUniaxialTensionModel",
    "SimpleShearMaterialPointModel",
    "SolidBarTorsionModel",
    "TopHatShearModel",
    "UniaxialLoadingMaterialPointModel",
    "UniaxialStrainMaterialPointModel",
    "UniaxialStressMaterialPointModel",
    "UserDefinedSierraModel",
    "VFMUniaxialTensionConnectedHexModel",
    "VFMUniaxialTensionHexModel",
]
