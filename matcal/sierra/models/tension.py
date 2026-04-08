"""
Uniaxial tension model family (round, rectangular, notched).

Concrete models in this module:
- RoundUniaxialTensionModel
- RectangularUniaxialTensionModel
- RoundNotchedTensionModel
"""

import numpy as np

from matcal.core.boundary_condition_calculators import (
    get_displacement_function_from_load_displacement_data_collection,
    get_displacement_function_from_strain_data_collection,
    raise_required_fields_not_found_error,
)
from matcal.core.constants import (
    DISPLACEMENT_KEY,
    DISPLACEMENT_RATE_KEY,
    ENG_STRAIN_KEY,
    ENG_STRESS_KEY,
    LOAD_KEY,
    STRAIN_RATE_KEY,
)
from matcal.sierra.input_file_writer import SolidMechanicsUserOutput

from matcal.cubit.geometry import (
    RoundUniaxialTensionGeometry,
    RectangularUniaxialTensionGeometry,
    RoundNotchedTensionGeometry,
)


class _TensionDerivedModelBase(_SymmetricUniaxiallyLoadedModelBase):
    """
    Shared base for gripped symmetric specimens (tension family and torsion family).
    """

    _loading_bc_node_sets = ["ns_side_grip"]

    _death_blocks = ["necking_section"]
    _model_blocks = ["grip_section", "gauge_section", "necking_section"]
    _temperature_blocks = ["gauge_section", "necking_section"]
    _thermal_bc_nodesets = ["ns_side_grip"]

    def __init__(self, material, executable="adagio", **kwargs):
        _all_input_params = {"mesh_method": 3}
        _all_input_params.update(kwargs)
        super().__init__(material, executable=executable, **_all_input_params)

    def _create_derived_user_output_blocks(self, state):
        self._add_disp_outputs("extensometer_surf", 2)
        self._add_load_outputs(self._loading_bc_node_sets[0], 4)
        self._add_strain_outputs()
        self._add_stress_outputs()

    def _add_strain_outputs(self):
        strain_output = SolidMechanicsUserOutput("global_strain", "extensometer_surf", "node set")
        self._input_file._solid_mechanics_region.add_subblock(strain_output)

        extensometer_len = self._current_state_geo_params["extensometer_length"]
        strain_output.add_compute_global_from_expression(ENG_STRAIN_KEY, f"{DISPLACEMENT_KEY}/{extensometer_len};")
        self._input_file._add_heartbeat_global_variable(ENG_STRAIN_KEY)

    def _add_stress_outputs(self):
        stress_output = SolidMechanicsUserOutput("global_stress", "ns_side_grip", "node set")
        self._input_file._solid_mechanics_region.add_subblock(stress_output)

        reference_area = self.reference_area
        stress_output.add_compute_global_from_expression(
            ENG_STRESS_KEY, f"{LOAD_KEY}/{reference_area};"
        )
        self._input_file._add_heartbeat_global_variable(ENG_STRESS_KEY)


class _UniaxialTensionModelBase(_TensionDerivedModelBase):
    """
    Shared base for uniaxial tension models that compute engineering stress/strain
    from displacement and load outputs.
    """

    def _get_loading_boundary_condition_displacement_function(self, state, params_by_precedent):
        bc_data = self._boundary_condition_data

        # Allow STRAIN_RATE or DISPLACEMENT_RATE to be used interchangeably by deriving the other.
        if STRAIN_RATE_KEY in params_by_precedent.keys():
            disp_rate = params_by_precedent[STRAIN_RATE_KEY] * self._current_state_geo_params.extensometer_length
            params_by_precedent.update({DISPLACEMENT_RATE_KEY: disp_rate})
        elif DISPLACEMENT_RATE_KEY in params_by_precedent.keys():
            eng_strain_rate = (
                params_by_precedent[DISPLACEMENT_RATE_KEY] / self._current_state_geo_params.extensometer_length
            )
            params_by_precedent.update({STRAIN_RATE_KEY: eng_strain_rate})

        common_state_field_names = bc_data.state_common_field_names(state.name)

        # Choose how to interpret boundary condition data
        if DISPLACEMENT_KEY in common_state_field_names:
            disp_func_calculator = get_displacement_function_from_load_displacement_data_collection
            scale_factor = self._current_state_geo_params.gauge_length / self._current_state_geo_params.extensometer_length
        elif ENG_STRAIN_KEY in common_state_field_names:
            disp_func_calculator = get_displacement_function_from_strain_data_collection
            scale_factor = self._current_state_geo_params.gauge_length
        else:
            raise_required_fields_not_found_error(
                state,
                DISPLACEMENT_KEY + ", " + ENG_STRAIN_KEY,
                bc_data.name,
            )

        disp_function = disp_func_calculator(
            bc_data,
            state,
            params_by_precedent,
            scale_factor=scale_factor,
        )

        # Account for symmetry across the gauge length (model is built with symmetry)
        disp_function[DISPLACEMENT_KEY] *= 0.5
        return disp_function

    def _create_derived_user_output_blocks(self, state):
        super()._create_derived_user_output_blocks(state)
        self._add_contraction_output()

    @property
    def reference_area(self):
        raise NotImplementedError

    def _add_contraction_output(self):
        raise NotImplementedError


class RoundUniaxialTensionModel(_UniaxialTensionModelBase):
    """
    MatCal generated SIERRA/SM uniaxial tension test model with a round cross section.
    """

    model_type = "round_uniaxial_tension_model"
    _geometry_creator_class = RoundUniaxialTensionGeometry

    def _add_contraction_output(self):
        contraction_output_z = SolidMechanicsUserOutput(
            "z_contraction", "z_radial_node", "node set"
        )
        self._input_file._solid_mechanics_region.add_subblock(contraction_output_z)
        contraction_output_z.add_compute_global_from_nodal_field(
            "z_radial_node_z_displacement", "displacement(z)"
        )
        contraction_output_z.add_compute_global_from_expression(
            "z_contraction", "2*z_radial_node_z_displacement;"
        )
        self._input_file._add_heartbeat_global_variable("z_contraction")

        contraction_output_x = SolidMechanicsUserOutput(
            "x_contraction", "x_radial_node", "node set"
        )
        self._input_file._solid_mechanics_region.add_subblock(contraction_output_x)
        contraction_output_x.add_compute_global_from_nodal_field(
            "x_radial_node_x_displacement", "displacement(x)"
        )
        contraction_output_x.add_compute_global_from_expression(
            "x_contraction", "2*x_radial_node_x_displacement;"
        )
        self._input_file._add_heartbeat_global_variable("x_contraction")

    @property
    def reference_area(self):
        r = self._current_state_geo_params["gauge_radius"]
        return np.pi * np.double(r) ** 2


class RectangularUniaxialTensionModel(_UniaxialTensionModelBase):
    """
    MatCal generated SIERRA/SM uniaxial tension test model with a rectangular cross section.
    """

    model_type = "rectangular_uniaxial_tension_model"
    _geometry_creator_class = RectangularUniaxialTensionGeometry

    def _add_contraction_output(self):
        contraction_output_z = SolidMechanicsUserOutput(
            "z_contraction", "thickness_center_node", "node set"
        )
        self._input_file._solid_mechanics_region.add_subblock(contraction_output_z)
        contraction_output_z.add_compute_global_from_nodal_field(
            "thickness_node_z_displacement", "displacement(z)"
        )
        contraction_output_z.add_compute_global_from_expression(
            "z_contraction", "2*thickness_node_z_displacement;"
        )
        self._input_file._add_heartbeat_global_variable("z_contraction")

        contraction_output_x = SolidMechanicsUserOutput(
            "x_contraction", "gauge_width_center_node", "node set"
        )
        self._input_file._solid_mechanics_region.add_subblock(contraction_output_x)
        contraction_output_x.add_compute_global_from_nodal_field(
            "width_node_x_displacement", "displacement(x)"
        )
        contraction_output_x.add_compute_global_from_expression(
            "x_contraction", "2*width_node_x_displacement;"
        )
        self._input_file._add_heartbeat_global_variable("x_contraction")

    @property
    def reference_area(self):
        return (
            self._current_state_geo_params["gauge_width"] * 
            self._current_state_geo_params["thickness"]
        )


class RoundNotchedTensionModel(_TensionDerivedModelBase):
    """
    MatCal generated SIERRA/SM notched tension test model with a round cross section.
    """

    model_type = "round_notched_tension_model"
    _geometry_creator_class = RoundNotchedTensionGeometry

    @property
    def reference_area(self):
        r = self._current_state_geo_params["notch_gauge_radius"]
        return np.pi * np.double(r) ** 2

    def _get_loading_boundary_condition_displacement_function(self, state, params_by_precedent):
        bc_data = self._boundary_condition_data
        common_state_field_names = bc_data.state_common_field_names(state.name)

        # Allow either displacement or engineering_strain as BC input
        if DISPLACEMENT_KEY in common_state_field_names:
            disp_func_calculator = get_displacement_function_from_load_displacement_data_collection
            scale_factor = 1.0
        elif ENG_STRAIN_KEY in common_state_field_names:
            disp_func_calculator = get_displacement_function_from_strain_data_collection
            scale_factor = self._current_state_geo_params.extensometer_length
        else:
            raise_required_fields_not_found_error(
                state,
                DISPLACEMENT_KEY + ", " + ENG_STRAIN_KEY,
                bc_data.name,
            )

        disp_function = disp_func_calculator(
            bc_data,
            state,
            params_by_precedent,
            scale_factor=scale_factor,
        )

        # Account for symmetry across gauge section
        disp_function[DISPLACEMENT_KEY] *= 0.5
        return disp_function