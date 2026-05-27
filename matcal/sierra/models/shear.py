"""
Shear-family models: torsion and top-hat shear.

Concrete models in this module:
- SolidBarTorsionModel
- TopHatShearModel
"""

import numpy as np

from matcal.core.boundary_condition_calculators import get_rotation_function_from_data_collection
from matcal.core.constants import DISPLACEMENT_KEY, ROTATION_KEY, TORQUE_KEY
from matcal.sierra.input_file_writer import (
    SolidMechanicsUserOutput, SolidMechanicsUserVariable
)

from matcal.cubit.geometry import SolidBarTorsionGeometry, TopHatShearGeometry

from .base import _SymmetricUniaxiallyLoadedModelContactBase
from .tension import _TensionDerivedModelBase

class SolidBarTorsionModel(_TensionDerivedModelBase):
    """
    MatCal generated SIERRA/SM solid bar torsion test model.
    """

    model_type = "solid_bar_torsion_model"
    _geometry_creator_class = SolidBarTorsionGeometry

    _loading_bc_node_sets = ["ns_side_grip"]
    _loading_bc_directions = ["cylindrical_axis"]
    _loading_bc_direction_keys = ["cylindrical axis"]

    _fixed_bc_node_sets = ["ns_y_symmetry"]
    _fixed_bc_directions = ["y"]

    _solution_termination_variable = TORQUE_KEY

    def _additional_boundary_condition_setup(self, state):
        ifile = self._input_file
        ifile._add_prescribed_displacement_boundary_condition(
            "sierra_constant_function_zero",
            self._fixed_bc_node_sets,
            self._loading_bc_directions,
            self._loading_bc_direction_keys,
        )

    def _create_derived_user_output_blocks(self, state):
        self._add_rotation_user_variables()

        torque_rotation_output = SolidMechanicsUserOutput(
            "global_torque_rotation", "ns_side_grip", "node set"
        )
        self._input_file.solid_mechanics_region.add_subblock(torque_rotation_output, replace=True)

        self._add_variable_transforms(torque_rotation_output)
        self._add_torque_calculations(torque_rotation_output)
        self._add_rotation_calculations(torque_rotation_output)

    def _add_rotation_user_variables(self):
        quarter_rotation_count_var = SolidMechanicsUserVariable(
            "quarter_rotation_count", "global", "real", 0
        )
        self._input_file.solid_mechanics_region.add_subblock(
            quarter_rotation_count_var, replace=True
        )

        grip_rotation_var = SolidMechanicsUserVariable("grip_rotation", "global", "real", 0)
        self._input_file.solid_mechanics_region.add_subblock(grip_rotation_var, replace=True)

        half_grip_rotation_var = SolidMechanicsUserVariable(
            "add_grip_half_rotation", "global", "real", 0
        )
        self._input_file.solid_mechanics_region.add_subblock(half_grip_rotation_var, replace=True)

        previous_quadrant_var = SolidMechanicsUserVariable("previous_quadrant", "global", "real", 0)
        self._input_file.solid_mechanics_region.add_subblock(previous_quadrant_var, replace=True)

        current_quadrant_var = SolidMechanicsUserVariable("current_quadrant", "global", "real", 1)
        self._input_file.solid_mechanics_region.add_subblock(current_quadrant_var, replace=True)

    def _add_variable_transforms(self, torque_rotation_output):
        torque_rotation_output.add_nodal_variable_transformation(
            "displacement",
            "cylindrical_displacement",
            "cylindrical_coordinate_system",
        )
        torque_rotation_output.add_nodal_variable_transformation(
            "force_external",
            "cylindrical_force_external",
            "cylindrical_coordinate_system",
        )

    def _add_torque_calculations(self, torque_rotation_output):
        torque_rotation_output.add_compute_global_from_nodal_field(
            f"partial_{TORQUE_KEY}",
            "cylindrical_force_external(y)",
            "sum",
        )
        grip_radius = self._current_state_geo_params["grip_radius"]
        torque_rotation_output.add_compute_global_from_expression(
            TORQUE_KEY, f"partial_{TORQUE_KEY}*{grip_radius}"
        )
        self._input_file._add_heartbeat_global_variable(TORQUE_KEY)

    def _add_rotation_calculations(self, torque_rotation_output):
        torque_rotation_output.add_compute_global_from_nodal_field(
            "grip_cylindrical_x_disp", "cylindrical_displacement(x)"
        )
        torque_rotation_output.add_compute_global_from_nodal_field(
            "grip_cylindrical_y_disp", "cylindrical_displacement(y)"
        )

        torque_rotation_output.add_compute_global_as_function(
            "applied_rotation_radians", self._input_file._load_bc_function_name
        )
        torque_rotation_output.add_compute_global_from_expression(
            "applied_rotation", "applied_rotation_radians*180/{PI}*2"
        )

        grip_radius = self._current_state_geo_params["grip_radius"]
        torque_rotation_output.add_compute_global_from_expression(
            "tangent_denominator", f"{grip_radius} - grip_cylindrical_x_disp"
        )

        torque_rotation_output.add_compute_global_from_expression(
            "previous_quadrant", "current_quadrant"
        )

        current_quadrant_expression = (
            "(grip_cylindrical_y_disp > -1e-15 ) ? "
            "((tangent_denominator > -1e-15) ? 1 : 2) : "
            "((tangent_denominator > -1e-15 ) ? 4 : 3)"
        )
        torque_rotation_output.add_compute_global_from_expression(
            "current_quadrant", current_quadrant_expression
        )
        torque_rotation_output.add_compute_global_from_expression(
            "add_grip_half_rotation",
            "((current_quadrant == 2) || (current_quadrant == 4)) ? 180 : 0",
        )

        quarter_rotation_count_exp = (
            "((previous_quadrant != current_quadrant )) ? "
            "quarter_rotation_count+1 : quarter_rotation_count"
        )
        torque_rotation_output.add_compute_global_from_expression(
            "quarter_rotation_count", quarter_rotation_count_exp
        )

        partial_grip_rotation_exp = "(atan(grip_cylindrical_y_disp/(tangent_denominator))*180/{PI})"
        torque_rotation_output.add_compute_global_from_expression(
            f"partial_{ROTATION_KEY}", partial_grip_rotation_exp
        )

        grip_rotation_exp = (
            f"partial_{ROTATION_KEY}*2+90*2*(quarter_rotation_count) + add_grip_half_rotation"
        )
        torque_rotation_output.add_compute_global_from_expression(ROTATION_KEY, grip_rotation_exp)

        self._input_file._add_heartbeat_global_variable("applied_rotation")
        self._input_file._add_heartbeat_global_variable(ROTATION_KEY)

    def _get_loading_boundary_condition_displacement_function(self, state, params_by_precedent):
        rot_function, metadata = get_rotation_function_from_data_collection(
            self._boundary_condition_data,
            state,
            params_by_precedent,
            scale_factor=1.0,
            return_metadata=True,
        )

        # Accounting for symmetry across the gauge length and conversion of degrees to radians
        rot_function[ROTATION_KEY] *= 0.5 * np.pi / 180.0
        rot_function.rename_field(ROTATION_KEY, DISPLACEMENT_KEY)

        extra_lines = [
            "Applied symmetry factor of 0.5 to the prescribed rotation.",
            'Converted rotation from degrees to radians.',
            f'Renamed "{ROTATION_KEY}" to "{DISPLACEMENT_KEY}" for the SIERRA prescribed displacement boundary condition.',
        ]
        self._set_last_loading_bc_comment(metadata, extra_lines)
        return rot_function


class TopHatShearModel(_SymmetricUniaxiallyLoadedModelContactBase):
    """
    MatCal generated SIERRA/SM top hat shear test model.
    """

    model_type = "top_hat_shear_model"
    _geometry_creator_class = TopHatShearGeometry

    _death_blocks = ["localization_section"]

    _loading_bc_node_sets = ["ns_y_bottom"]

    _model_blocks = ["localization_section", "platten_interface_section"]
    _temperature_blocks = ["localization_section"]
    _thermal_bc_nodesets = ["ns_y_bottom", "ns_load"]

    _fixed_bc_node_sets = ["ns_x_symmetry", "ns_load", "ns_z_symmetry"]
    _fixed_bc_directions = ["x", "y", "z"]

    @property
    def self_contact(self):
        """
        Returns True if self contact is on for the model. Otherwise, returns False.
        """
        return self._input_file._self_contact()

    def _create_derived_user_output_blocks(self, state):
        self._add_disp_outputs(self._loading_bc_node_sets[0], 1)
        self._add_load_outputs(self._loading_bc_node_sets[0], 4)

    def activate_full_field_data_output(self):
        """
        Not implemented for this model.
        """
        raise AttributeError(
            f'Cannot use full field data output with the MatCal top hat shear model "{self.name}".'
        )