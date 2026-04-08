"""
Material point SIERRA models.

This module contains concrete material point models that use the standard-model
base infrastructure from :mod:`matcal.sierra.models.base`.
"""

import numpy as np

from matcal.core.boundary_condition_calculators import (
    get_displacement_function_from_strain_data_collection
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
from matcal.sierra.input_file_writer import SolidMechanicsUserOutput

from matcal.cubit.geometry import MaterialPointGeometry

from .base import _StandardSierraModelWithDeathBase


class UniaxialLoadingMaterialPointModel(_StandardSierraModelWithDeathBase):
    """
    MatCal generated material point model for uniaxial loading.
    """

    model_type = "uniaxial_loading_material_point"

    _geometry_creator_class = MaterialPointGeometry
    _death_blocks = ["material_point_block"]
    _model_blocks = ["material_point_block"]

    _loading_bc_node_sets = ["ns_positive_z"]
    _loading_bc_directions = ["z"]
    _loading_bc_direction_keys = ["component"]

    _fixed_bc_node_sets = ["ns_negative_x", "ns_negative_y", "ns_negative_z"]
    _fixed_bc_directions = ["x", "y", "z"]

    def __init__(self, material):
        super().__init__(material=material, executable="adagio")

    def _additional_boundary_condition_setup(self, state):
        """"""

    def _get_loading_boundary_condition_displacement_function(self, state, params_by_precedent):
        # For a material point, displacement BC is derived 
        # from strain history (eng strain by default).
        # Note: scale_factor is handled by the BC calculator as needed.
        func = get_displacement_function_from_strain_data_collection(
            self._boundary_condition_data, state, params_by_precedent
        )
        return func

    def _create_user_output_blocks(self, state):
        self._add_load_outputs()
        self._add_true_stress_strain_outputs()
        self._add_contraction_output()

    def _add_load_outputs(self):
        load_output = SolidMechanicsUserOutput(
            "global_stress_strain_load_disp",
            "ns_positive_z",
            "node set",
        )
        self._input_file._solid_mechanics_region.add_subblock(load_output)

        load_output.add_compute_global_from_nodal_field(DISPLACEMENT_KEY, "displacement(z)")
        self._input_file._add_heartbeat_global_variable(DISPLACEMENT_KEY)
        self._input_file._add_heartbeat_global_variable(DISPLACEMENT_KEY, ENG_STRAIN_KEY)

        load_output.add_compute_global_from_nodal_field(LOAD_KEY, "force_external(z)", "sum")
        self._input_file._add_heartbeat_global_variable(LOAD_KEY)
        self._input_file._add_heartbeat_global_variable(LOAD_KEY, ENG_STRESS_KEY)

    def _add_true_stress_strain_outputs(self):
        true_stress_strain_output = SolidMechanicsUserOutput(
            "true_stress_strain",
            "include all blocks",
        )
        self._input_file._solid_mechanics_region.add_subblock(true_stress_strain_output)

        true_stress_strain_output.add_compute_global_from_element_field(
            TRUE_STRAIN_KEY, "log_strain(zz)"
        )
        self._input_file._add_heartbeat_global_variable(TRUE_STRAIN_KEY)

        true_stress_strain_output.add_compute_global_from_element_field(
            TRUE_STRESS_KEY, "cauchy_stress(zz)"
        )
        self._input_file._add_heartbeat_global_variable(TRUE_STRESS_KEY)

        true_stress_strain_output.add_compute_global_from_element_field(
            "log_strain_xx", "log_strain(xx)"
        )
        self._input_file._add_heartbeat_global_variable("log_strain_xx")

        true_stress_strain_output.add_compute_global_from_element_field(
            "log_strain_yy", "log_strain(yy)"
         )
        self._input_file._add_heartbeat_global_variable("log_strain_yy")

        if self.coupling is not None:
            true_stress_strain_output.add_compute_global_from_element_field(
                TEMPERATURE_KEY, TEMPERATURE_KEY
            )
            self._input_file._add_heartbeat_global_variable(TEMPERATURE_KEY)

    def _add_contraction_output(self):
        contraction_output = SolidMechanicsUserOutput("contraction", "ns_positive_x", "node set")
        self._input_file._solid_mechanics_region.add_subblock(contraction_output)
        contraction_output.add_compute_global_from_nodal_field("contraction", "displacement(x)")
        self._input_file._add_heartbeat_global_variable("contraction")