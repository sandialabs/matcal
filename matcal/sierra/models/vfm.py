"""
Virtual Fields Method (VFM) SIERRA/SM models.

Concrete models in this module:
- VFMUniaxialTensionHexModel
- VFMUniaxialTensionConnectedHexModel

Internal:
- _vfm_field_series_data
- _VFMStandardSierraModel
"""

import os

from matcal.core.boundary_condition_calculators import BoundaryConditionDeterminationError
from matcal.core.constants import TIME_KEY
from matcal.core.utilities import (
    check_value_is_positive_integer,
    check_value_is_positive_real,
    check_value_is_nonempty_str,
)

from matcal.exodus.geometry import ExodusHexGeometryCreator
from matcal.exodus.mesh_modifications import ExodusHex8MeshExploder, _ExodusFieldInterpPreprocessor

from matcal.full_field.TwoDimensionalFieldGrid import FieldGridBase
from matcal.full_field.data import FieldData
from matcal.full_field.data_importer import FieldSeriesData, ImportedTwoDimensionalMesh
from matcal.full_field.field_mappers import MeshlessMapperGMLS

from matcal.sierra.input_file_writer import AnalyticSierraFunction, SolidMechanicsUserOutput

from .base import _CoupledStandardSierraModelBase


def _vfm_field_series_data(*args, **kwargs):
    """
    Wrapper around FieldSeriesData that filters fields down to those needed by VFM.
    """
    data = FieldSeriesData(*args, **kwargs)
    fields_to_keep = []
    for field in data.field_names:
        if "volume" in field:
            fields_to_keep.append(field)
        elif "first_pk_stress" in field:
            fields_to_keep.append(field)
        elif "centroid" in field:
            fields_to_keep.append(field)
        elif "time" in field:
            fields_to_keep.append(field)
    return data[fields_to_keep]


class _VFMStandardSierraModel(_CoupledStandardSierraModelBase):
    """
    Base class for MatCal VFM models.
    """

    model_type = "VFM"

    _death_blocks = ["block_main"]
    _model_blocks = ["block_main"]
    _temperature_blocks = ["block_main"]
    _thermal_bc_nodesets = []

    _geometry_creator_class = ExodusHexGeometryCreator

    _loading_bc_node_sets = ["front_node_set", "front_node_set", "back_node_set", "back_node_set"]
    _loading_bc_directions = ["x", "y", "x", "y"]
    _loading_bc_direction_keys = ["component", "component", "component", "component"]
    _loading_bc_read_variables = []

    _fixed_bc_node_sets = ["back_node_set"]
    _fixed_bc_directions = ["z"]

    def __init__(self, material, mesh, thickness):
        """
        :param material: Material model for the solid mechanics region.
        :type material: matcal.sierra.material.Material

        :param mesh: surface mesh filename (or FieldGridBase) used for VFM meshing.
        :type mesh: str or FieldGridBase

        :param thickness: specimen thickness.
        :type thickness: float
        """
        self._reference_mesh_grid = self._import_mesh(mesh)
        check_value_is_positive_real(thickness, "thickness")
        self._thickness = thickness

        super().__init__(
            material,
            executable="adagio",
            thickness=self._thickness,
            reference_mesh_grid=self._reference_mesh_grid,
        )

        self._input_file._clear_default_element_output_field_names()
        self._set_results_reader_object(_vfm_field_series_data)
        self._results_information.results_filename = os.path.join("results", "results.e")

        self._x_displacement_field_name = "U"
        self._y_displacement_field_name = "V"
        self._build_read_variables_list()

        self._polynomial_order = None
        self._search_radius_multiplier = None

        self.set_mapping_parameters()
        self.activate_exodus_output()
        self.add_element_output_variable("first_pk_stress")
        self.add_element_output_variable("centroid", volume_average=False)

    def _build_read_variables_list(self):
        self._loading_bc_read_variables = []
        self._loading_bc_read_variables.append(self._x_displacement_field_name)
        self._loading_bc_read_variables.append(self._y_displacement_field_name)
        self._loading_bc_read_variables.append(self._x_displacement_field_name)
        self._loading_bc_read_variables.append(self._y_displacement_field_name)

    def _create_user_output_blocks(self, state):
        self._add_temperature_global_outputs()

    def _set_state_loading_boundary_condition(self, state):
        params_by_precedent, _source_by_precedence = self._get_parameters_by_precedence(state)
        bc_func = self._prepare_loading_boundary_condition_displacement_function(
            state, params_by_precedent
        )
        self._input_file._set_time_parameters_to_loading_function(bc_func, scale_factor=1)

    def _additional_boundary_condition_setup(self, state):
        self._input_file._add_prescribed_displacement_boundary_condition(
            None,
            self._loading_bc_node_sets,
            self._loading_bc_directions,
            self._loading_bc_direction_keys,
            self._loading_bc_read_variables,
        )

    def _import_mesh(self, mesh):
        if isinstance(mesh, str) and os.path.exists(mesh):
            return ImportedTwoDimensionalMesh(mesh)
        elif isinstance(mesh, FieldGridBase):
            return mesh
        else:
            raise FileNotFoundError(f'Could not find the mesh file named "{mesh}". Check input.')

    def get_thickness(self):
        """
        Returns thickness as provided on instantiation.
        """
        return self._thickness

    def set_mapping_parameters(
        self,
        polynomial_order: int = MeshlessMapperGMLS.default_polynomial_order,
        search_radius_multiplier: float = MeshlessMapperGMLS.default_epsilon_multiplier,
    ):
        """
        Set the mapping parameters for Compadre GMLS.
        """
        check_value_is_positive_integer(polynomial_order, "polynomial_order")
        check_value_is_positive_real(search_radius_multiplier, "search_radius_multiplier")
        self._polynomial_order = polynomial_order
        self._search_radius_multiplier = search_radius_multiplier

    def set_displacement_field_names(self, x_displacement, y_displacement):
        """
        Set the field names in the boundary FieldData used for x/y displacements.
        """
        check_value_is_nonempty_str(x_displacement, "x_displacement")
        check_value_is_nonempty_str(y_displacement, "y_displacement")
        self._x_displacement_field_name = x_displacement
        self._y_displacement_field_name = y_displacement
        self._build_read_variables_list()

    def add_boundary_condition_data(self, data):
        super().add_boundary_condition_data(data)
        self._verify_valid_boundary_condition_data()

    def _verify_valid_boundary_condition_data(self):
        for state_name in self._boundary_condition_data.state_names:
            if len(self._boundary_condition_data[state_name]) > 1:
                raise BoundaryConditionDeterminationError(
                    "Currently VFM models only allow one boundary condition "
                     "data set per state. Check input."
                )
            if not isinstance(self._boundary_condition_data[state_name][0], FieldData):
                raise BoundaryConditionDeterminationError(
                    "Data passed to VFM models for boundary condition "
                    "generation must be FieldData types. "
                    "Received variable of type "
                    f"'{type(self._boundary_condition_data[state_name][0])}' "
                    f"for state '{state_name}'."
                )

    def _get_loading_boundary_condition_displacement_function(self, state, params_by_precedent):
        bc_data = self._boundary_condition_data[state.name][0]
        self._verify_fields_in_data(bc_data)
        disp_function_data = bc_data[[
            TIME_KEY, self._x_displacement_field_name, self._y_displacement_field_name
        ]]
        return disp_function_data

    def _get_fields_to_map(self):
        fields_to_project = [self._x_displacement_field_name, self._y_displacement_field_name]
        if self._temperature_field_from_boundary_data is not None:
            fields_to_project.append(self._temperature_field_from_boundary_data)
        return fields_to_project

    def _verify_fields_in_data(self, bc_data):
        fields_to_project = self._get_fields_to_map()
        fields_not_in_data = []
        for field in fields_to_project:
            if field not in bc_data.field_names:
                fields_not_in_data.append(field)

        if fields_not_in_data:
            raise BoundaryConditionDeterminationError(
                "The correct fields are not in the boundary condition data "
                f'for state "{bc_data.state.name}" for model "{self.name}". '
                f"The data has fields:{bc_data.field_names}\n"
                f"The following fields were expected: {fields_to_project}"
            )

    def activate_exodus_output(self):
        """
        By default, output is activated and output at every step.
        Output step interval cannot be adjusted.
        """
        super().activate_exodus_output(output_step_interval=1)

    def set_boundary_condition_scale_factor(self, *args, **kwargs):
        raise AttributeError(
            "Calling 'set_boundary_condition_scale_factor' is not allowed for MatCal VFM models."
        )

    def activate_element_death(self, *args, **kwargs):
        """
        Element death is allowed but not recommended due to loss of material.
        """
        super().activate_element_death(*args, **kwargs)

    def _generate_mesh(self, state_template_dir, mesh_filename, state):
        super()._generate_mesh(state_template_dir, mesh_filename, state)
        interp_preprocessor = _ExodusFieldInterpPreprocessor()
        interp_preprocessor.process(
            state_template_dir,
            mesh_filename,
            self._boundary_condition_data[state][0],
            fields_to_map=self._get_fields_to_map(),
            polynomial_order=self._polynomial_order,
            search_radius_multiplier=self._search_radius_multiplier,
        )


class VFMUniaxialTensionHexModel(_VFMStandardSierraModel):
    """
    VFM model using a disconnected (exploded) hex mesh.
    """

    def __init__(self, material, surface_mesh_filename, thickness, **kwargs):
        super().__init__(material, surface_mesh_filename, thickness, **kwargs)
        self.add_element_output_variable("volume", volume_average=False)

    def _generate_mesh(self, state_template_dir, mesh_filename, state):
        super()._generate_mesh(state_template_dir, mesh_filename, state)
        mesh_exploder = ExodusHex8MeshExploder(mesh_filename)
        mesh_exploder.boom()

    def activate_thermal_coupling(self, *args, **kwargs):
        """
        Disconnected elements support only adiabatic heating (no conduction).
        """
        if args or kwargs:
            raise AttributeError(
                "Calling 'activate_thermal_coupling' with arguments is not "
                 "allowed for MatCal VFM disconnected hex models. "
                "Only adiabatic heating is supported. If conduction is desired use the "
                "VFMUniaxialTensionConnectedHexModel."
            )
        super().activate_thermal_coupling()

    def use_iterative_coupling(self, *args, **kwargs):
        raise AttributeError(
            "Calling 'use_iterative_coupling' is not allowed for this MatCal VFM hex model. "
            "Use the MatCal VFMUniaxialTensionConnectedHexModel."
        )


class VFMUniaxialTensionConnectedHexModel(_VFMStandardSierraModel):
    """
    VFM model using a connected hex mesh and half-thickness symmetry, enabling conduction coupling.
    """

    _geometry_creator_class = ExodusHexGeometryCreator

    def __init__(self, material, surface_mesh_filename, thickness, **kwargs):
        super().__init__(material, surface_mesh_filename, thickness, **kwargs)

        # Use symmetry plane: represent half-thickness in the mesh; report full thickness to user.
        self._thickness /= 2
        self._base_geo_params["thickness"] /= 2.0

        self._double_vol_func = None
        self._double_vol_user_output = None
        self._build_double_volume_input_blocks()

    def _build_double_volume_input_blocks(self):
        double_vol_func = AnalyticSierraFunction("double_volume_func")
        double_vol_func.add_evaluation_expression("2*volume")
        double_vol_func.add_expression_variable("volume", "element", "volume")
        self._double_vol_func = double_vol_func

        double_vol_output = SolidMechanicsUserOutput("double_volume", "include all blocks")
        double_vol_output.add_compute_element_as_function("double_volume", "double_volume_func")
        self._double_vol_user_output = double_vol_output

    def _add_double_volume_output(self):
        if self._double_vol_func not in self.input_file.subblocks.values():
            self._input_file.add_subblock(self._double_vol_func)

        sm_region = self.input_file.solid_mechanics_region
        if self._double_vol_user_output not in sm_region.subblocks.values():
            self._input_file._solid_mechanics_region.add_subblock(self._double_vol_user_output)

        if not self._input_file.exodus_output.has_element_output("double_volume", "volume"):
            self._input_file.exodus_output.add_element_output("double_volume", "volume")

    def _create_user_output_blocks(self, state):
        super()._create_user_output_blocks(state)
        self._add_double_volume_output()

    def get_thickness(self):
        """
        Return full thickness (undo symmetry halving).
        """
        return self._thickness * 2