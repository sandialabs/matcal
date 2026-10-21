"""
Material point SIERRA models.

This module contains concrete material point models that use the standard-model
base infrastructure from :mod:`matcal.sierra.models.base`.
"""

import warnings
from collections.abc import Callable
from typing import Any, ClassVar

from matcal.core.boundary_condition_calculators import (
    get_displacement_function_from_strain_data_collection,
)
from matcal.core.constants import (
    DISPLACEMENT_KEY,
    ENG_STRAIN_KEY,
    ENG_STRESS_KEY,
    LOAD_KEY,
    TEMPERATURE_KEY,
    TRUE_STRAIN_KEY,
    TRUE_STRESS_KEY,
)
from matcal.core.state import State
from matcal.cubit.geometry import MaterialPointGeometry
from matcal.sierra.input_file_writer import SolidMechanicsUserOutput
from matcal.sierra.material import Material

from .base import _StandardSierraModelWithDeathBase


class _UniaxialMaterialPointModelBase(_StandardSierraModelWithDeathBase):
    """
    Shared implementation for material point models loaded along the Z axis.

    Derived classes define the lateral constraint through the fixed boundary
    condition node sets and directions, and may add constraint specific output.
    The prescribed displacement is applied in Z to the nodes on the top face.
    """

    _geometry_creator_class: ClassVar[type[MaterialPointGeometry]] = (
        MaterialPointGeometry
    )
    _death_blocks: ClassVar[list[str]] = ["material_point_block"]
    _model_blocks: ClassVar[list[str]] = ["material_point_block"]

    _loading_bc_node_sets: ClassVar[list[str]] = ["ns_positive_z"]
    _loading_bc_directions: ClassVar[list[str]] = ["z"]
    _loading_bc_direction_keys: ClassVar[list[str]] = ["component"]

    def __init__(self, material: Material) -> None:
        super().__init__(material=material, executable="adagio")

    def _additional_boundary_condition_setup(self, state: State) -> None:
        """Additional boundary condition setup (none needed for uniaxial loading)."""

    def _get_loading_boundary_condition_displacement_function(
        self, state: State, params_by_precedent: dict[str, Any]
    ) -> Callable:
        # For a material point, displacement BC is derived
        # from strain history (eng strain by default).
        func, metadata = get_displacement_function_from_strain_data_collection(
            self._boundary_condition_data,
            state,
            params_by_precedent,
            return_metadata=True,
        )
        self._set_last_loading_bc_comment(metadata)
        return func

    def _create_user_output_blocks(self, state: State) -> None:
        self._add_load_outputs()
        self._add_true_stress_strain_outputs()

    def _add_load_outputs(self) -> None:
        load_output = SolidMechanicsUserOutput(
            "global_stress_strain_load_disp",
            "ns_positive_z",
            "node set",
        )
        self._input_file._solid_mechanics_region.add_subblock(load_output)

        load_output.add_compute_global_from_nodal_field(
            DISPLACEMENT_KEY, "displacement(z)"
        )
        self._input_file._add_heartbeat_global_variable(DISPLACEMENT_KEY)
        self._input_file._add_heartbeat_global_variable(
            DISPLACEMENT_KEY, ENG_STRAIN_KEY
        )

        load_output.add_compute_global_from_nodal_field(
            LOAD_KEY, "force_external(z)", "sum"
        )
        self._input_file._add_heartbeat_global_variable(LOAD_KEY)
        self._input_file._add_heartbeat_global_variable(LOAD_KEY, ENG_STRESS_KEY)

    def _add_true_stress_strain_outputs(self) -> None:
        true_stress_strain_output = SolidMechanicsUserOutput(
            "true_stress_strain",
            "include all blocks",
        )
        self._input_file._solid_mechanics_region.add_subblock(true_stress_strain_output)

        self._add_element_global_output(
            true_stress_strain_output, TRUE_STRAIN_KEY, "log_strain(zz)"
        )
        self._add_element_global_output(
            true_stress_strain_output, TRUE_STRESS_KEY, "cauchy_stress(zz)"
        )
        self._add_element_global_output(
            true_stress_strain_output, "log_strain_xx", "log_strain(xx)"
        )
        self._add_element_global_output(
            true_stress_strain_output, "log_strain_yy", "log_strain(yy)"
        )

        if self.coupling is not None:
            self._add_element_global_output(
                true_stress_strain_output, TEMPERATURE_KEY, TEMPERATURE_KEY
            )

    def _add_element_global_output(
        self, user_output: SolidMechanicsUserOutput, name: str, field: str
    ) -> None:
        user_output.add_compute_global_from_element_field(name, field)
        self._input_file._add_heartbeat_global_variable(name)


class UniaxialStressMaterialPointModel(_UniaxialMaterialPointModelBase):
    """
    MatCal generated material point model for uniaxial stress loading.

    A single hexahedral element is displaced in Z on its top face while its
    lateral faces are free to contract, producing a uniaxial stress state.

    This model replaces :class:`UniaxialLoadingMaterialPointModel`, which
    is soon to be deprecated and behaves identically apart from its name and
    ``model_type``.
    """

    model_type: ClassVar[str] = "uniaxial_stress_material_point"

    _fixed_bc_node_sets: ClassVar[list[str]] = [
        "ns_negative_x",
        "ns_negative_y",
        "ns_negative_z",
    ]
    _fixed_bc_directions: ClassVar[list[str]] = ["x", "y", "z"]

    def _create_user_output_blocks(self, state: State) -> None:
        super()._create_user_output_blocks(state)
        self._add_contraction_output()

    def _add_contraction_output(self) -> None:
        contraction_output = SolidMechanicsUserOutput(
            "contraction", "ns_positive_x", "node set"
        )
        self._input_file._solid_mechanics_region.add_subblock(contraction_output)
        contraction_output.add_compute_global_from_nodal_field(
            "contraction", "displacement(x)"
        )
        self._input_file._add_heartbeat_global_variable("contraction")


class UniaxialLoadingMaterialPointModel(UniaxialStressMaterialPointModel):
    """
    MatCal generated material point model for uniaxial stress loading.

    .. deprecated::
        Use :class:`UniaxialStressMaterialPointModel` instead. This class
        will be removed in a future release. It keeps the
        ``"uniaxial_loading_material_point"`` ``model_type`` so existing
        model names and result directories are unchanged.
    """

    model_type: ClassVar[str] = "uniaxial_loading_material_point"

    def __init__(self, material: Material) -> None:
        warnings.warn(
            "UniaxialLoadingMaterialPointModel is deprecated and will be "
            "removed in a future release. Use UniaxialStressMaterialPointModel "
            "instead.",
            FutureWarning,
            stacklevel=2,
        )
        super().__init__(material)


class UniaxialStrainMaterialPointModel(_UniaxialMaterialPointModelBase):
    """
    MatCal generated material point model for uniaxial strain loading.

    A single hexahedral element is displaced in Z on its top face while all
    four lateral faces are fixed in their normal directions, so the only
    nonzero strain component is in Z. The lateral confinement produces
    nonzero lateral stresses, which are output as ``cauchy_stress_xx`` and
    ``cauchy_stress_yy``.
    """

    model_type: ClassVar[str] = "uniaxial_strain_material_point"

    _fixed_bc_node_sets: ClassVar[list[str]] = [
        "ns_negative_x",
        "ns_negative_y",
        "ns_negative_z",
        "ns_positive_x",
        "ns_positive_y",
    ]
    _fixed_bc_directions: ClassVar[list[str]] = ["x", "y", "z", "x", "y"]

    def _create_user_output_blocks(self, state: State) -> None:
        super()._create_user_output_blocks(state)
        self._add_lateral_stress_outputs()

    def _add_lateral_stress_outputs(self) -> None:
        lateral_stress_output = SolidMechanicsUserOutput(
            "lateral_stress",
            "include all blocks",
        )
        self._input_file._solid_mechanics_region.add_subblock(lateral_stress_output)
        self._add_element_global_output(
            lateral_stress_output, "cauchy_stress_xx", "cauchy_stress(xx)"
        )
        self._add_element_global_output(
            lateral_stress_output, "cauchy_stress_yy", "cauchy_stress(yy)"
        )


class SimpleShearMaterialPointModel(_StandardSierraModelWithDeathBase):
    """
    MatCal generated material point model for simple shear loading.

    Boundary conditions:
    - Bottom surface (ns_negative_z): Fixed in all directions (X, Y, Z)
    - Top surface (ns_positive_z): Displaced in X direction only, fixed in Y and Z

    The model is a single hex element (1x1x1). By default, it is fully integrated in
    deviatoric stress and under-integrated in pressure (controlled by material model).
    """

    model_type: ClassVar[str] = "simple_shear_material_point"

    _geometry_creator_class: ClassVar[type[MaterialPointGeometry]] = (
        MaterialPointGeometry
    )
    _death_blocks: ClassVar[list[str]] = ["material_point_block"]
    _model_blocks: ClassVar[list[str]] = ["material_point_block"]

    _loading_bc_node_sets: ClassVar[list[str]] = ["ns_positive_z"]
    _loading_bc_directions: ClassVar[list[str]] = ["x"]
    _loading_bc_direction_keys: ClassVar[list[str]] = ["component"]

    _fixed_bc_node_sets: ClassVar[list[str]] = ["ns_negative_z"] * 3 + [
        "ns_positive_z"
    ] * 2
    _fixed_bc_directions: ClassVar[list[str]] = ["x", "y", "z", "y", "z"]

    def __init__(self, material: Material) -> None:
        super().__init__(material=material, executable="adagio")

    def _additional_boundary_condition_setup(self, state: State) -> None:
        """Additional boundary condition setup for simple shear (none needed)."""

    def _get_loading_boundary_condition_displacement_function(
        self, state: State, params_by_precedent: dict[str, Any]
    ) -> Callable:
        """
        Get displacement function for simple shear loading.

        For simple shear, the displacement BC is derived from strain history.
        """
        func, metadata = get_displacement_function_from_strain_data_collection(
            self._boundary_condition_data,
            state,
            params_by_precedent,
            return_metadata=True,
        )
        self._set_last_loading_bc_comment(metadata)
        return func

    def _create_user_output_blocks(self, state: State) -> None:
        """Create user output blocks for simple shear."""
        self._add_shear_load_outputs()
        self._add_true_stress_strain_outputs()

    def _add_shear_load_outputs(self) -> None:
        """Add engineering stress/strain outputs from nodal data."""
        load_output = SolidMechanicsUserOutput(
            "global_shear_stress_strain_load_disp",
            "ns_positive_z",
            "node set",
        )
        self._input_file._solid_mechanics_region.add_subblock(load_output)

        # Displacement in X direction
        load_output.add_compute_global_from_nodal_field(
            DISPLACEMENT_KEY, "displacement(x)"
        )
        self._input_file._add_heartbeat_global_variable(DISPLACEMENT_KEY)

        # Engineering shear strain: gamma = displacement_x / height
        # For material point with height = 1.0, this simplifies to displacement_x
        self._input_file._add_heartbeat_global_variable(
            DISPLACEMENT_KEY, ENG_STRAIN_KEY
        )

        # Force in X direction
        load_output.add_compute_global_from_nodal_field(
            LOAD_KEY, "force_external(x)", "sum"
        )
        self._input_file._add_heartbeat_global_variable(LOAD_KEY)

        # Engineering shear stress: tau = sum(force_x) / area
        # For material point with area = 1.0, this simplifies to sum(force_x)
        self._input_file._add_heartbeat_global_variable(LOAD_KEY, ENG_STRESS_KEY)

    def _add_true_stress_strain_outputs(self) -> None:
        """Add true stress and strain outputs from element fields."""
        true_output = SolidMechanicsUserOutput(
            "true_shear_stress_strain",
            "include all blocks",
        )
        self._input_file._solid_mechanics_region.add_subblock(true_output)

        # Primary shear component (XZ)
        true_output.add_compute_global_from_element_field(
            TRUE_STRAIN_KEY, "log_strain(xz)"
        )
        self._input_file._add_heartbeat_global_variable(TRUE_STRAIN_KEY)

        true_output.add_compute_global_from_element_field(
            TRUE_STRESS_KEY, "cauchy_stress(xz)"
        )
        self._input_file._add_heartbeat_global_variable(TRUE_STRESS_KEY)

        # Normal strain components
        true_output.add_compute_global_from_element_field(
            "log_strain_xx", "log_strain(xx)"
        )
        self._input_file._add_heartbeat_global_variable("log_strain_xx")

        true_output.add_compute_global_from_element_field(
            "log_strain_yy", "log_strain(yy)"
        )
        self._input_file._add_heartbeat_global_variable("log_strain_yy")

        true_output.add_compute_global_from_element_field(
            "log_strain_zz", "log_strain(zz)"
        )
        self._input_file._add_heartbeat_global_variable("log_strain_zz")

        # Normal stress components
        true_output.add_compute_global_from_element_field(
            "cauchy_stress_xx", "cauchy_stress(xx)"
        )
        self._input_file._add_heartbeat_global_variable("cauchy_stress_xx")

        true_output.add_compute_global_from_element_field(
            "cauchy_stress_yy", "cauchy_stress(yy)"
        )
        self._input_file._add_heartbeat_global_variable("cauchy_stress_yy")

        true_output.add_compute_global_from_element_field(
            "cauchy_stress_zz", "cauchy_stress(zz)"
        )
        self._input_file._add_heartbeat_global_variable("cauchy_stress_zz")

        # Add temperature output if thermal coupling is active
        if self.coupling is not None:
            true_output.add_compute_global_from_element_field(
                TEMPERATURE_KEY, TEMPERATURE_KEY
            )
            self._input_file._add_heartbeat_global_variable(TEMPERATURE_KEY)
