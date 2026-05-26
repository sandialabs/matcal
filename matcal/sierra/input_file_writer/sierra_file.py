"""
Top-level SIERRA input-deck builders for MatCal.

This module defines:
- SierraFileBase
- SierraFileWithCoupling
- SierraFileThreeDimensional
- SierraFileThreeDimensionalContact

It is largely a refactor of the original monolithic input_file_writer.py with imports
updated to the package layout.

Note:
- This file includes a small enhancement: optional insertion of a comment line
  above the prescribed displacement function block for provenance/debugging.
"""

from collections import OrderedDict

from matcal.core.boundary_condition_calculators import (
    format_bc_function_comment_lines,
    get_temperature_function_from_data_collection
)
from matcal.core.constants import TEMPERATURE_KEY, TIME_KEY, DISPLACEMENT_KEY
from matcal.core.input_file_writer import InputFileLine
from matcal.core.logger import initialize_matcal_logger
from matcal.core.utilities import (
    check_value_is_nonempty_str,
    get_string_from_text_file,
)

from .blocks_base import (
    SierraGlobalDefinitions,
    PiecewiseLinearFunction,
    AnalyticSierraFunction,
    _BaseSierraInputFileBlock,
    _get_default_solid_mechanics_region_name,
    _get_default_solid_mechanics_procedure_name,
    _get_default_thermal_region_name,
)
from .sections import TotalLagrangeSection, SolidSectionDefault, _SectionNames
from .materials import ThermalMaterial
from .solvers import (
    FetiSolver,
    GdswSolver,
    TpetraSolver,
    SolidMechanicsFullTangentPreconditioner,
    SolidMechanicsConjugateGradient,
    SolidMechanicsNonlinearSolverContainer,
    SolidMechanicsControlContact,
    SolidMechanicsLoadstepPredictor,
)
from .time_control import SolidMechanicsProcedure
from .boundary_conditions import (
    SolidMechanicsFixedDisplacement,
    SolidMechanicsPrescribedDisplacement,
    SolidMechanicsPrescribedTemperature,
    SolidMechanicsInitialTemperature,
)
from .outputs import (
    SolidMechanicsUserOutput,
    SolidMechanicsUserVariable,
    SolidMechanicsResultsOutput,
    SolidMechanicsHeartbeatOutput,
    SolidMechanicsSolutionTermination,
    SolidMechanicsAdaptiveTimeStepping,
    SolidMechanicsNonlocalDamageAverage,
)
from .regions_models import (
    FiniteElementModel,
    FiniteElementParameters,
    SolidMechanicsFiniteElementParameters,
    SolidMechanicsRegion,
    ThermalRegion,
    SolidMechanicsDeath,
    SolidMechanicsImplicitDynamics,
    _FiniteElementModelNames,
)
from .coupling import (
    _Coupling,
    _Failure,
    ArpeggioTransfer,
    CoupledInitialize,
    CoupledTransient,
    CoupledSystem,
    CoupledTransientParameters,
    SolutionControl,
    Procedure,
    NonlinearParameters,
)
from .contact import (
    SolidMechanicsConstantFrictionModel,
    SolidMechanicsContactDefinitions,
)

logger = initialize_matcal_logger(__name__)


class SierraFileBase(_BaseSierraInputFileBlock):
    """
    Base SIERRA input file builder for MatCal-generated solid mechanics decks.
    """

    required_keys = []
    default_values = {}
    type = "sierra"

    _load_bc_function_name = "prescribed_displacement"
    _temperature_bc_function_name = "prescribed_temperature"
    _material_model_line_name = "material_file_line"

    def __init__(self, material, death_blocks, **kwargs):
        super().__init__("matcal_generated_SIERRA_model", begin_end=True, **kwargs)

        self._death_blocks = death_blocks
        self._add_material_file_to_input(material)

        self._sm_finite_element_model = FiniteElementModel()
        self.add_subblock(self._sm_finite_element_model)

        global_defs = SierraGlobalDefinitions()
        self.add_lines(*global_defs.lines.values())

        self.add_subblock(FetiSolver())
        self.add_subblock(GdswSolver())

        self.add_subblock(TotalLagrangeSection())
        self.add_subblock(SolidSectionDefault())

        ct_section = TotalLagrangeSection()
        ct_section.use_composite_tet()
        self.add_subblock(ct_section)

        self._adaptive_time_stepping = None
        self._cg = None
        self._solver = None
        self._full_tangent_preconditioner = None

        self._exodus_output = None
        self._default_element_output = ["hydrostatic_stress", "von_mises", "log_strain"]
        self._default_nodal_output = ["displacement"]

        self._vol_average_user_output = SolidMechanicsUserOutput("vol_average", "include all blocks")

        self._coupling = None
        self._failure = None
        self._death = None
        self._initial_temp = None
        self._solution_termination = None

        self._solid_mechanics_region = self._build_default_solid_mechanics_region()
        sm_procedure_name = _get_default_solid_mechanics_procedure_name()
        self._solid_mechanics_procedure = SolidMechanicsProcedure(
            sm_procedure_name,
            self._solid_mechanics_region,
            0,
            1,
            300,
        )
        self.add_subblock(self._solid_mechanics_procedure)

        self._heartbeat_output = None
        self._reset_heartbeat_output()

        self._start_time_user_supplied = False
        self._end_time_user_supplied = False

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def cg(self):
        return self._cg

    @property
    def adative_time_stepping(self):
        return self._adaptive_time_stepping

    @property
    def full_tangent_preconditioner(self):
        return self._full_tangent_preconditioner

    @property
    def solid_mechanics_region(self):
        return self._solid_mechanics_region

    @property
    def solid_mechanics_procedure(self):
        return self._solid_mechanics_procedure

    @property
    def death(self):
        return self._death

    @property
    def solid_mechanics_finite_element_model(self):
        return self._sm_finite_element_model

    @property
    def solid_mechanics_element_section(self):
        return self._get_section_subblock()

    @property
    def exodus_output(self):
        return self._exodus_output

    @property
    def heartbeat_output(self):
        return self._heartbeat_output

    @property
    def element_type(self):
        return self._sm_finite_element_model.get_element_section()

    @property
    def prescribed_loading_boundary_condition(self):
        if self._load_bc_function_name in self.subblocks:
            return self.subblocks[self._load_bc_function_name]
        return None

    @property
    def prescribed_temperature_boundary_condition(self):
        if self._temperature_bc_function_name in self.subblocks:
            return self.subblocks[self._temperature_bc_function_name]
        return None

    @property
    def initial_temperature(self):
        return self._initial_temp

    @property
    def solution_termination(self):
        return self._solution_termination

    @property
    def exodus_output_active(self):
        return self._exodus_output is not None and self._exodus_output.has_output_variables

    @property
    def coupling(self):
        return self._coupling

    @property
    def failure(self):
        return self._failure

    # -------------------------------------------------------------------------
    # Internal construction helpers
    # -------------------------------------------------------------------------

    def _add_material_file_to_input(self, material):
        material_str = get_string_from_text_file(material.filename)
        self._material_file_line = InputFileLine(material_str, name=self._material_model_line_name)
        self._material_file_line.suppress_symbol()
        self.add_line(self._material_file_line)

    def _build_default_solid_mechanics_region(self):
        region_name = _get_default_solid_mechanics_region_name()
        region = SolidMechanicsRegion(region_name, self._sm_finite_element_model.name)

        self._adaptive_time_stepping = SolidMechanicsAdaptiveTimeStepping(1e-8, 1)
        self._adaptive_time_stepping.set_iteration_target()
        region.add_subblock(self._adaptive_time_stepping)

        self._solution_termination = SolidMechanicsSolutionTermination()
        region.add_subblock(self._solution_termination)

        gdsw = GdswSolver()
        self._full_tangent_preconditioner = SolidMechanicsFullTangentPreconditioner(gdsw)

        self._cg = SolidMechanicsConjugateGradient(full_tangent_preconditioner=self._full_tangent_preconditioner)

        self._solver = SolidMechanicsNonlinearSolverContainer()
        self._solver.add_subblock(self._cg)
        region.add_subblock(self._solver)

        return region

    def _reset_heartbeat_output(self):
        self._heartbeat_output = SolidMechanicsHeartbeatOutput(1, "time")
        self._solid_mechanics_region.add_subblock(self._heartbeat_output, replace=True)

    def _get_section_subblock(self):
        section_name = self._sm_finite_element_model.get_element_section()
        if section_name in self.subblocks:
            return self.get_subblock(section_name)
        return None

    def _reset_state_boundary_conditions_and_output(self):
        self._reset_state_displacement_conditions()
        self._reset_state_temperature_conditions()

        sm_region = self.solid_mechanics_region
        sm_region.remove_subblocks_by_type(SolidMechanicsUserOutput.type)

        self._reset_heartbeat_output()
        self._solid_mechanics_region.add_subblock(self._vol_average_user_output)

    # -------------------------------------------------------------------------
    # Comment injection helper 
    # -------------------------------------------------------------------------

    def _add_comment_to_function_block(self, function_name: str, comment: str):
        """
        Prepend one or more comment lines inside a SIERRA `function` block.

        Each emitted line begins with '# '.
        """
        check_value_is_nonempty_str(function_name, "function_name", call_depth=1)
        check_value_is_nonempty_str(comment, "comment", call_depth=1)

        if function_name not in self.subblocks:
            raise KeyError(f"Function subblock '{function_name}' not found; cannot add comment.")

        func_block = self.subblocks[function_name]

        new_lines = OrderedDict()
        for i, raw_line in enumerate(comment.splitlines()):
            text = raw_line.rstrip()
            if not text.startswith("#"):
                text = f"# {text}" if text else "#"
            comment_line = InputFileLine(text, name=f"comment_{function_name}_{i}")
            comment_line.suppress_symbol()
            new_lines[comment_line.name] = comment_line

        for k, v in func_block._lines.items():
            new_lines[k] = v

        func_block._lines = new_lines

    # -------------------------------------------------------------------------
    # Output activation / variable additions
    # -------------------------------------------------------------------------

    def _clear_default_element_output_field_names(self):
        self._default_element_output = []

    def _activate_exodus_output(self, output_step_interval=20):
        if self._exodus_output is None:
            self._exodus_output = SolidMechanicsResultsOutput(output_step_interval)
            self._solid_mechanics_region.add_subblock(self._exodus_output)

            for ele_output in self._default_element_output:
                self._add_element_output_variable(ele_output)

            for nodal_output in self._default_nodal_output:
                self._add_nodal_output_variable(nodal_output)
        else:
            self._exodus_output.set_output_step_increment(output_step_interval)

    def _add_volume_averaged_element_output(self, element_variable_name):
        save_name = element_variable_name
        element_variable_name = self._get_volume_averaged_name(element_variable_name)
        self._vol_average_user_output.add_compute_element_from_element(
            element_variable_name,
            save_name,
            replace=True,
        )
        return element_variable_name, save_name

    def _get_volume_averaged_name(self, element_variable_name):
        return element_variable_name + "_vol_avg"

    def _element_variable_in_mesh_output(self, element_variable_name, save_as_name=None):
        if self._exodus_output is not None:
            line_name = self._exodus_output._get_element_variable_line_name(element_variable_name, save_as_name)
            return line_name in self._exodus_output.lines
        return False

    def _remove_element_mesh_output(self, element_variable_name, save_name=None):
        if self._element_variable_in_mesh_output(element_variable_name, save_name):
            line_name = self._exodus_output._get_element_variable_line_name(element_variable_name, save_name)
            self._exodus_output._lines.pop(line_name)

    def _add_element_output(self, element_variable_name, save_name):
        if save_name is not None:
            self._remove_element_mesh_output(save_name)
        else:
            to_remove = self._get_volume_averaged_name(element_variable_name)
            self._remove_element_mesh_output(to_remove, element_variable_name)

        if not self._element_variable_in_mesh_output(element_variable_name, save_name):
            self._exodus_output.add_element_output(element_variable_name, save_name)
        else:
            logger.warning(f"Element variable '{element_variable_name}' already output. Not adding again.")

    def _add_element_output_variable(self, *element_variable_names, volume_average=True):
        if self._exodus_output is None:
            self._activate_exodus_output()

        for element_variable_name in element_variable_names:
            check_value_is_nonempty_str(element_variable_name, "element_variable_name", call_depth=1)
            save_name = None
            if volume_average:
                element_variable_name, save_name = self._add_volume_averaged_element_output(element_variable_name)
            self._add_element_output(element_variable_name, save_name)

    def _nodal_variable_in_mesh_output(self, nodal_variable_name):
        if self._exodus_output is not None:
            line_name = self._exodus_output._get_nodal_variable_line_name(nodal_variable_name)
            return line_name in self._exodus_output.lines
        return False

    def _add_nodal_output_variable(self, *nodal_variable_names):
        if self._exodus_output is None:
            self._activate_exodus_output()

        for nodal_variable_name in nodal_variable_names:
            check_value_is_nonempty_str(nodal_variable_name, "nodal_variable_name", call_depth=1)
            if not self._nodal_variable_in_mesh_output(nodal_variable_name):
                self._exodus_output.add_nodal_output(nodal_variable_name)
            else:
                logger.warning(f"Nodal variable '{nodal_variable_name}' already output. Not adding again.")

    def _add_new_default_mesh_output(self, variable_name, nodal=False):
        if self.exodus_output_active:
            if nodal:
                self._add_nodal_output_variable(variable_name)
            else:
                self._add_element_output_variable(variable_name)
        elif nodal and variable_name not in self._default_nodal_output:
            self._default_nodal_output.append(variable_name)
        elif (not nodal) and variable_name not in self._default_element_output:
            self._default_element_output.append(variable_name)

    def _add_temperature_output(self, nodal=False):
        self._add_new_default_mesh_output(TEMPERATURE_KEY, nodal)

    # -------------------------------------------------------------------------
    # Time / element selection
    # -------------------------------------------------------------------------

    def _use_under_integrated_element(self):
        self._sm_finite_element_model.set_element_section(_SectionNames.uniform_gradient)

    def _use_total_lagrange_element(self):
        self._sm_finite_element_model.set_element_section(_SectionNames.total_lagrange)

    def _add_solid_mechanics_finite_element_parameters(self, material_name, material_model, *blocks):
        fem_params = SolidMechanicsFiniteElementParameters(material_name, material_model, *blocks)
        self._sm_finite_element_model.add_subblock(fem_params)

    def _set_start_time(self, start_time):
        self._start_time_user_supplied = True
        self._update_time_parameters(start_time=start_time)

    def _set_end_time(self, end_time):
        self._end_time_user_supplied = True
        self._update_time_parameters(termination_time=end_time)

    def _set_number_of_time_steps(self, number_of_steps):
        self._update_time_parameters(number_of_steps=number_of_steps)

    def _update_time_parameters(self, start_time=None, termination_time=None, number_of_steps=None):
        if start_time is not None:
            self._solid_mechanics_procedure.set_start_time(start_time)
        if termination_time is not None:
            self._solid_mechanics_procedure.set_end_time(termination_time)
        if number_of_steps is not None:
            self._solid_mechanics_procedure.set_number_of_time_steps(number_of_steps)

    # -------------------------------------------------------------------------
    # Temperature handling
    # -------------------------------------------------------------------------

    def _set_state_prescribed_temperature_from_boundary_data(self, boundary_data, state, temperature_key):
        full_field_temp_data = self._get_temperature_boundary_data_full_field(boundary_data, state, temperature_key)
        if full_field_temp_data is not None:
            self._set_temperature_from_mesh_full_field_data(full_field_temp_data, temperature_key)
        else:
            self._set_temperature_from_temperature_time_history(boundary_data, state, temperature_key)

    def _get_temperature_boundary_data_full_field(self, boundary_data, state, temperature_key):
        full_field_temp_data = None
        data_list = boundary_data[state]
        for data in data_list:
            if len(data[temperature_key].shape) == 2:
                full_field_temp_data = data
                if len(data_list) > 1:
                    logger.warning(
                        "More than one data set supplied and at least one is full field. "
                        "Only the first full field data set will be used."
                    )
                self._add_temperature_output(nodal=True)
                break
        return full_field_temp_data

    def _set_temperature_from_mesh_full_field_data(self, full_field_temp_data, temperature_key):
        prescribed_temp = SolidMechanicsPrescribedTemperature("include all blocks")
        prescribed_temp.read_from_mesh(temperature_key)
        self._solid_mechanics_region.add_subblock(prescribed_temp)

    def _set_temperature_from_temperature_time_history(self, boundary_data, state, temperature_key):
        temperature_function, metadata = get_temperature_function_from_data_collection(
            boundary_data,
            state,
            temperature_key=temperature_key,
            return_metadata=True,
        )
        func_name = self._temperature_bc_function_name

        prescribed_temp = SolidMechanicsPrescribedTemperature(
            "include all blocks",
            function_name=func_name,
        )
        self._solid_mechanics_region.add_subblock(prescribed_temp)

        temp_function = PiecewiseLinearFunction(
            func_name,
            temperature_function[TIME_KEY],
            temperature_function[temperature_key],
        )
        self.add_subblock(temp_function)

        comment = "\n".join(format_bc_function_comment_lines(metadata))
        self._add_comment_to_function_block(func_name, comment)

        self._add_temperature_output(nodal=True)

    def _set_initial_temperature_from_parameters(self, model_parameters_by_precedent):
        if TEMPERATURE_KEY in model_parameters_by_precedent.keys():
            initial_temp_value = model_parameters_by_precedent[TEMPERATURE_KEY]
            self._initial_temp = SolidMechanicsInitialTemperature("include all blocks", initial_temp_value)
            self._solid_mechanics_region.add_subblock(self._initial_temp, replace=True)
            self._add_temperature_output()
        elif self._coupling is not None:
            raise RuntimeError(
                "When running a coupled simulation, a model/state/study parameter named "
                f'"{TEMPERATURE_KEY}" is required for all states.'
            )

    # -------------------------------------------------------------------------
    # Displacement BC handling
    # -------------------------------------------------------------------------

    def _set_time_parameters_to_loading_function(self, bc_function_array, scale_factor):
        if not self._start_time_user_supplied:
            self._update_time_parameters(start_time=min(bc_function_array[TIME_KEY]) * scale_factor)
        if not self._end_time_user_supplied:
            end_time = max(bc_function_array[TIME_KEY]) * scale_factor
            self._update_time_parameters(termination_time=end_time)

    def _add_prescribed_loading_boundary_condition_with_displacement_function(
        self,
        displacement_function,
        node_sets,
        directions,
        direction_keys,
        scale_factor,
        bc_comment=None,
    ):
        """
        Add the prescribed displacement function and associated prescribed displacement BC blocks.

        bc_comment (optional) will be inserted as a comment line inside the SIERRA function block.
        """
        self._set_time_parameters_to_loading_function(displacement_function, scale_factor)

        func = PiecewiseLinearFunction(
            self._load_bc_function_name,
            displacement_function[TIME_KEY],
            displacement_function[DISPLACEMENT_KEY],
        )
        func.scale_function(x=scale_factor, y=scale_factor)
        self.add_subblock(func)

        if bc_comment:
            self._add_comment_to_function_block(self._load_bc_function_name, bc_comment)

        self._add_prescribed_displacement_boundary_condition(
            self._load_bc_function_name,
            node_sets,
            directions,
            direction_keys,
        )

    def _add_prescribed_displacement_boundary_condition(
        self,
        function_name,
        node_sets,
        directions,
        direction_keys,
        read_variables=None,
    ):
        if read_variables is None:
            read_variables = [None] * len(node_sets)

        for node_set, direction, direction_key, read_variable in zip(node_sets, directions, direction_keys, read_variables):
            prescribed_disp = SolidMechanicsPrescribedDisplacement(function_name, node_set, direction, direction_key=direction_key)
            if read_variable is not None:
                prescribed_disp.read_from_mesh(read_variable)
            self._solid_mechanics_region.add_subblock(prescribed_disp)

    def _set_local_mesh_filename(self, mesh_name):
        self._sm_finite_element_model.set_mesh_filename(mesh_name)

    def _set_fixed_boundary_conditions(self, node_sets, directions):
        for node_set, direction in zip(node_sets, directions):
            fixed_disp = SolidMechanicsFixedDisplacement(node_set, direction)
            self._solid_mechanics_region.add_subblock(fixed_disp)

    def _reset_state_temperature_conditions(self):
        if self._temperature_bc_function_name in self.subblocks:
            self.remove_subblock(self._temperature_bc_function_name)

        self._solid_mechanics_region.remove_subblocks_by_type(
            SolidMechanicsPrescribedTemperature.type
        )
        self._solid_mechanics_region.remove_subblocks_by_type(
            SolidMechanicsInitialTemperature.type
        )

    def _reset_state_displacement_conditions(self):
        if self._load_bc_function_name in self.subblocks:
            self.remove_subblock(self._load_bc_function_name)

        self._solid_mechanics_region.remove_subblocks_by_type(SolidMechanicsPrescribedDisplacement.type)

    def _add_heartbeat_global_variable(self, global_var, save_as_name=None):
        if not self._heartbeat_output.has_global_output(global_var, save_as_name):
            self._heartbeat_output.add_global_output(global_var, save_as_name)
        else:
            logger.warning(f"Global variable '{global_var}' already added. Not adding again.")

    # -------------------------------------------------------------------------
    # Element death / coupling / convergence
    # -------------------------------------------------------------------------

    def _activate_adiabatic_heating(self):
        self._coupling = _Coupling.adiabatic
        self._add_temperature_output()

    def _activate_element_death(self, death_variable="damage", critical_value=0.15):
        self._failure = _Failure.local_failure
        self._death = SolidMechanicsDeath(death_variable, critical_value, *self._death_blocks)
        self._solid_mechanics_region.add_subblock(self._death, replace=True)
        self._add_new_default_mesh_output(death_variable)
        self._add_new_default_mesh_output("death_status")

    def _set_convergence_tolerance(
        self,
        nonlinear_solver,
        target_relative_residual,
        target_residual=None,
        acceptable_relative_residual=None,
        acceptable_residual=None,
        target_resid_factor=100,
        acceptable_relative_resid_factor=10,
    ):
        nonlinear_solver.set_target_relative_residual(target_relative_residual)

        if target_residual is None:
            nonlinear_solver.set_target_residual(target_relative_residual * target_resid_factor)
        else:
            nonlinear_solver.set_target_residual(target_residual)

        if acceptable_relative_residual is None:
            nonlinear_solver.set_acceptable_relative_residual(target_relative_residual * acceptable_relative_resid_factor)
        else:
            nonlinear_solver.set_acceptable_relative_residual(acceptable_relative_residual)

        if acceptable_residual is not None:
            nonlinear_solver.set_acceptable_residual(acceptable_residual)

    def _set_cg_convergence_tolerance(
        self,
        target_relative_residual,
        target_residual=None,
        acceptable_relative_residual=None,
        acceptable_residual=None,
    ):
        self._set_convergence_tolerance(
            self._cg,
            target_relative_residual,
            target_residual,
            acceptable_relative_residual,
            acceptable_residual,
        )


class SierraFileWithCoupling(SierraFileBase):
    """
    Extends SierraFileBase to support Arpeggio staggered/iterative coupling with Aria.
    """

    def __init__(self, material, death_blocks, **kwargs):
        super().__init__(material, death_blocks, **kwargs)
        self._thermal_bc_nodesets = None
        self._coupling_transfers = []
        self._coupled_procedure = None
        self._thermal_model = None
        self._thermal_region = None
        self._thermal_material = None
        self._transient_params_1 = None
        self._transient_params_2 = None
        self._transient1 = None
        self._transient2 = None
        self._thermal_material = None
        self._death_transfer = None
        self._plastic_work_variable = None
        self._work_transfer = None

    def _set_thermal_bc_nodesets(self, nodesets):
        self._thermal_bc_nodesets=nodesets

    def _add_heating_source_element_variables(self, plastic_work_variable):
        if self._thermal_region is not None:
            self._reset_plastic_work_transfer_lines(plastic_work_variable)
            if self.element_type is _SectionNames.composite_tet:
                res  = self._add_volume_averaged_element_output(plastic_work_variable)
                vol_averaged_name, save_name = res
                num_integration_points = 1
                self._work_transfer.add_field_to_send(vol_averaged_name, plastic_work_variable)
                self._thermal_region.add_equations_to_solve("Q2")
            else:
                num_integration_points = self._get_num_integration_points()             
                self._work_transfer.add_field_to_send(plastic_work_variable, 
                                                      plastic_work_variable, 'new')
                self._thermal_region.add_equations_to_solve("Q1")
            self._thermal_region.add_heating_source(plastic_work_variable, num_integration_points)

    def _reset_plastic_work_transfer_lines(self, plastic_work_variable):
            for line_name in self._work_transfer.lines:
                if plastic_work_variable in line_name:
                    self._work_transfer.lines.pop(line_name)


    def _activate_thermal_coupling(self, thermal_conductivity, 
                                    density, specific_heat, 
                                    plastic_work_variable):
        self._plastic_work_variable = plastic_work_variable
        self._thermal_material = ThermalMaterial(density, thermal_conductivity, 
                                           specific_heat)
        self.add_subblock(self._thermal_material)
        thermal_solver = TpetraSolver()
        self.add_subblock(thermal_solver)
        self._thermal_model = FiniteElementModel(name=_FiniteElementModelNames.thermal)
        self._add_sm_model_mesh_name_line_to_thermal_model()
        self._add_thermal_finite_element_parameters(*self._sm_finite_element_model.get_blocks())
        region_name = _get_default_thermal_region_name()
        self._thermal_region = ThermalRegion(region_name, self._thermal_model.name, 
                                          thermal_solver)
        arpeggio_init=self._get_arpeggio_initialize()
        transients = self._get_arpeggio_transients(plastic_work_variable)        
        
        arpeggio_sys = CoupledSystem("main", arpeggio_init.name, *transients)

        transient_params = self._get_arpeggio_transient_parameters()
        sltn_ctl = SolutionControl("coupling", arpeggio_sys, arpeggio_init, 
                                   *transient_params)
        self._coupled_procedure = Procedure(sltn_ctl, *self._coupling_transfers)
        self._add_heating_source_element_variables(plastic_work_variable)
        self.remove_subblock(self._solid_mechanics_procedure)
        self.add_subblock(self._coupled_procedure)
        self._coupled_procedure.add_subblock(self._solid_mechanics_region)
        self._coupled_procedure.add_subblock(self._thermal_region)
        self._coupling = _Coupling.staggered
        self._add_temperature_output(nodal=True)

    def _add_sm_model_mesh_name_line_to_thermal_model(self):
        mesh_name_line_name = FiniteElementModel.required_keys[0]
        sm_mesh_name_line = self._sm_finite_element_model.lines[mesh_name_line_name]
        self._thermal_model.add_line(sm_mesh_name_line) 
        self.add_subblock(self._thermal_model)

    def _get_arpeggio_initialize(self):
        arpeggio_init = CoupledInitialize("initialization", self._solid_mechanics_region.name, 
                                                  self._thermal_region.name)
        disp_transfer_init = ArpeggioTransfer("solid_mechanics_to_thermal_disps_initial")
        disp_transfer_init.set_nodal_copy_transfer(self._solid_mechanics_region.name, 
            self._thermal_region.name)
        disp_transfer_init.add_field_to_send("DISPLACEMENT", "solution->MESH_DISPLACEMENTS", 
                                        "NEW", "OLD")
        disp_transfer_init.add_field_to_send("DISPLACEMENT", "solution->MESH_DISPLACEMENTS", 
                                        "NEW", "NEW")
        self._coupling_transfers.append(disp_transfer_init)
        
        temp_transfer_init = ArpeggioTransfer("thermal_to_solid_mechanics_initial")
        temp_transfer_init.set_nodal_copy_transfer(self._thermal_region.name, 
                                                   self._solid_mechanics_region.name)
        temp_transfer_init.add_field_to_send("solution->TEMPERATURE", "TEMPERATURE", 
                                             "NEW", "OLD")
        temp_transfer_init.add_field_to_send("solution->TEMPERATURE", "TEMPERATURE", 
                                             "NEW", "NEW")
        self._coupling_transfers.append(temp_transfer_init)

        arpeggio_init.add_transfer_post_solid_mechanics(disp_transfer_init.name)
        arpeggio_init.add_transfer_post_thermal(temp_transfer_init.name)
        return arpeggio_init
    
    def _add_post_solid_mechanics_transfer_to_transients(self, transfer):
        if transfer not in self._transient1.lines:
            self._transient1.add_transfer_post_solid_mechanics(transfer)
            self._transient1._setup_lines()
        
        if transfer not in self._transient2.lines:
            self._transient2.add_transfer_post_solid_mechanics(transfer)
            self._transient2._setup_lines()

    def _add_post_thermal_transfer_to_transients(self, transfer):
        if transfer not in self._transient1.lines:
            self._transient1.add_transfer_post_thermal(transfer)
        if transfer not in self._transient2.lines:
            self._transient2.add_transfer_post_thermal(transfer)
        self._transient1._setup_lines()
        self._transient2._setup_lines()
        
    def _get_arpeggio_transients(self, plastic_work_variable_name):
        self._transient1 = CoupledTransient("transient1", self._solid_mechanics_region.name, 
            self._thermal_region.name)
        self._transient2 = CoupledTransient("transient2", self._solid_mechanics_region.name, 
            self._thermal_region.name)
        self._add_transfers_to_transients(plastic_work_variable_name)

        return self._transient1, self._transient2
    
    def _add_transfers_to_transients(self, plastic_work_variable_name):
        transfers = self._get_transient_transfers(plastic_work_variable_name)
        disp_transfer, work_transfer, temp_transfer, death_transfer = transfers

        self._add_post_solid_mechanics_transfer_to_transients(disp_transfer.name)
        self._add_post_solid_mechanics_transfer_to_transients(work_transfer.name)
        self._add_post_thermal_transfer_to_transients(temp_transfer.name)

        if death_transfer is not None:
            self._add_post_solid_mechanics_transfer_to_transients(death_transfer.name)

    def _get_transient_transfers(self, plastic_work_variable_name):
        disp_transfer = ArpeggioTransfer("solid_mechanics_to_thermal_disps")
        disp_transfer.set_nodal_copy_transfer(self._solid_mechanics_region.name, 
            self._thermal_region.name)
        disp_transfer.add_field_to_send("DISPLACEMENT", "solution->MESH_DISPLACEMENTS", 
                                        "NEW", "New")
        self._coupling_transfers.append(disp_transfer)

        self._work_transfer = ArpeggioTransfer("solid_mechanics_to_thermal_work")
        self._work_transfer.set_element_copy_transfer(self._solid_mechanics_region.name, 
            self._thermal_region.name)
        self._work_transfer.add_field_to_send(plastic_work_variable_name, 
                                        plastic_work_variable_name, "new")
        self._coupling_transfers.append(self._work_transfer)

        temp_transfer = ArpeggioTransfer("thermal_to_solid_mechanics")
        temp_transfer.set_nodal_copy_transfer(self._thermal_region.name, 
                                                   self._solid_mechanics_region.name)
        temp_transfer.add_field_to_send("solution->TEMPERATURE", "TEMPERATURE", 
                                             "NEW", "OLD")
        self._coupling_transfers.append(temp_transfer)
        self._set_death_transfer()
        return disp_transfer, self._work_transfer, temp_transfer, self._death_transfer

    def _set_death_transfer(self):
        if self._death is not None and self._death_transfer is None:
            self._death_transfer = ArpeggioTransfer("solid_mechanics_to_thermal_death_status")
            self._death_transfer.set_element_copy_transfer(self._solid_mechanics_region.name, 
                                                   self._thermal_region.name)
            self._death_transfer.add_field_to_send("death_status", "death_status_aria", 
                                             "NEW")
            self._death_transfer.add_send_blocks(*self._death_blocks)
            self._coupling_transfers.append(self._death_transfer)
            self._thermal_region.add_element_death(*self._death_blocks)  
        return self._death_transfer

    def _activate_element_death(self, *args, **kwargs):
        super()._activate_element_death(*args, **kwargs)
        if self._thermal_region is not None:
            self._set_death_transfer()
            self._coupled_procedure.add_subblock(self._death_transfer, replace=True)
            self._add_post_solid_mechanics_transfer_to_transients(self._death_transfer.name)

    def _get_arpeggio_transient_parameters(self):
        time_params = self._get_solid_mechanics_time_params()
        start_time, termination_time, init_time_step, time_step = time_params
        self._transient_params_1 = CoupledTransientParameters(self._transient1.name, 
                                                         self._thermal_region.name, 
                                                         self._solid_mechanics_region.name,
                                                         start_time, 
                                                         start_time+init_time_step, 
                                                         init_time_step)
        self._transient_params_2 = CoupledTransientParameters(self._transient2.name, 
                                                         self._thermal_region.name,
                                                         self._solid_mechanics_region.name, 
                                                         start_time+init_time_step, 
                                                         termination_time, 
                                                         time_step)
        return self._transient_params_1, self._transient_params_2
        
    def _get_solid_mechanics_time_params(self):
        start_time = self._solid_mechanics_procedure._start_time
        termination_time = self._solid_mechanics_procedure._termination_time
        init_time_step_scale_factor = self._solid_mechanics_procedure._init_time_step_scale_factor
        time_step = self._solid_mechanics_procedure._time_step
        init_time_step = time_step*init_time_step_scale_factor
        return start_time, termination_time, init_time_step, time_step

    @property
    def is_coupled_simulation(self):
        return self._coupling is not None and self._coupling != _Coupling.adiabatic

    def _update_time_parameters(self, start_time=None, termination_time=None, number_of_steps=None):
        super()._update_time_parameters(start_time=start_time, 
                                        termination_time=termination_time, 
                                        number_of_steps=number_of_steps)
        if self.is_coupled_simulation:
            time_params = self._get_solid_mechanics_time_params()
            start_time, termination_time, init_time_step, time_step = time_params
            sltn_ctrl = self._coupled_procedure._solution_control
            sltn_ctrl.set_transient_time_parameters(self._transient1.name, 
                start_time, start_time+init_time_step, init_time_step)
            sltn_ctrl.set_transient_time_parameters(self._transient2.name, 
                start_time+init_time_step, termination_time, time_step)

    def _add_thermal_finite_element_parameters(self, *blocks):
        new_finite_element_parameters = FiniteElementParameters(self._thermal_material.name,
            *blocks)
        self._thermal_model.add_subblock(new_finite_element_parameters)

    def _set_initial_temperature_from_parameters(self, model_parameters_by_precedent):
        super()._set_initial_temperature_from_parameters(model_parameters_by_precedent)
        if self.is_coupled_simulation:
            temperature = model_parameters_by_precedent[TEMPERATURE_KEY]
            self._set_dirichlet_temperature_boundary_conditions(self._thermal_bc_nodesets, 
                temperature)
            self._thermal_region.add_initial_condition(temperature)

    def _set_dirichlet_temperature_boundary_conditions(self, node_sets, magnitude):
        for node_set in node_sets:
            self._thermal_region.add_dirichlet_temperature_boundary_condition(node_set, magnitude)

    def _activate_iterative_coupling(self):
        self._coupling = _Coupling.iterative
        nonlinear_step1_params = NonlinearParameters(name="converge_step_1")
        self._transient1.set_nonlinear_step_name(nonlinear_step1_params.name)
        self._coupled_procedure._solution_control.add_subblock(nonlinear_step1_params)
        nonlinear_step2_params = NonlinearParameters(name="converge_step_2")
        self._transient2.set_nonlinear_step_name(nonlinear_step2_params.name)
        self._coupled_procedure._solution_control.add_subblock(nonlinear_step2_params)

    def _use_total_lagrange_element(self, composite_tet=False):
        if composite_tet:
            self._sm_finite_element_model.set_element_section(_SectionNames.composite_tet)
        else:
            super()._use_total_lagrange_element()
        self._add_heating_source_element_variables(self._plastic_work_variable)

    def _use_under_integrated_element(self):
        super()._use_under_integrated_element()
        self._add_heating_source_element_variables(self._plastic_work_variable)

    def _get_num_integration_points(self):
        integration_pts = 8
        if self.element_type == _SectionNames.composite_tet:
            integration_pts = 4
        elif self.element_type == _SectionNames.uniform_gradient:
            integration_pts = 1
        return integration_pts


class SierraFileThreeDimensional(SierraFileWithCoupling):
    def __init__(self, material, death_blocks, **kwargs):
        super().__init__(material, death_blocks, **kwargs)
        self._implicit_dynamics = None
        self._solution_termination_output = None
        self._setup_solution_termination()
        self._full_field_output = None
        self._nonlocal_functions = []
        self._damage_increment_user_output = None
        self._nonlocal_average_output = None
        self._nonlocal_damage_increment_user_output = None
        self._nonlocal_damage_user_variables = []
    
    def _setup_solution_termination(self):
        self._solution_termination.add_global_termination_criteria("terminate_solution", 0.5, ">")

    def _add_solution_termination_user_output(self, termination_variable, max_var_drop):
        max_var_name = f"max_{termination_variable}"
        max_user_var = SolidMechanicsUserVariable(max_var_name, "global", "real", 0.0)
        max_user_var.add_global_operator()
        self._solid_mechanics_region.add_subblock(max_user_var, replace=True)
        self._add_max_termination_user_output(termination_variable, max_var_drop, 
                                      max_var_name)

    def _add_max_termination_user_output(self, termination_variable, max_var_drop, 
        max_var_name):
        self._solution_termination_output = SolidMechanicsUserOutput("solution_termination")
        self._solution_termination_output.add_compute_global_from_expression(max_var_name, 
            f"({termination_variable} >= {max_var_name}) ? {termination_variable} "+
            f": {max_var_name};")                                               
        end_time = self._solid_mechanics_procedure._termination_time
        start_time = self._solid_mechanics_procedure._start_time
        one_tenth_run_time = (end_time-start_time)/10+start_time
        self._solution_termination_output.add_compute_global_from_expression("terminate_solution", 
            f"(time >= {one_tenth_run_time}) ? "+
             f"({termination_variable} < {max_var_name}*(1-{max_var_drop})) ? 1 : 0"+
            f": 0;")     
        self._solid_mechanics_region.add_subblock(self._solution_termination_output)

    def _activate_full_field_results_output(self, results_file, *model_blocks):
        self._full_field_output = SolidMechanicsResultsOutput(1, results_file, 
            name="full_field_output")
        self._full_field_output.add_include_surface("full_field_data_surface")
        self._full_field_output.add_exclude_blocks(*model_blocks)
        global_outputs = self._heartbeat_output.get_global_outputs()
        for global_output in global_outputs:
            self._full_field_output.add_line(global_output)
        if self.coupling == _Coupling.adiabatic:
            self._full_field_output.add_element_output("temperature")
        elif self.coupling is not None:
            self._full_field_output.add_nodal_output("temperature")            
        self._full_field_output.add_nodal_output("displacement")
        self._full_field_output.set_output_exposed_surface()
        self._solid_mechanics_region.add_subblock(self._full_field_output, replace=True)

    def _add_nonlocal_user_output(self, death_variable, nonlocal_radius):
        self._reset_nonlocal_input()
        self._failure = _Failure.nonlocal_failure
        num_integration_pts = self._get_num_integration_points()
        self._add_nonlocal_damage_user_variables(num_integration_pts)
        self._add_damage_increment_user_output(death_variable, num_integration_pts)
        self._add_nonlocal_average_output(nonlocal_radius)
        self._add_nonlocal_damage_increment_user_output(death_variable, num_integration_pts)

    def _reset_nonlocal_input(self):
        for func in self._nonlocal_functions:
            self.remove_subblock(func)
        self._nonlocal_functions = []

        sm_subblocks = self._solid_mechanics_region.subblocks
        for user_var in self._nonlocal_damage_user_variables:
            if user_var.name in sm_subblocks:
                self._solid_mechanics_region.remove_subblock(user_var)
        self._nonlocal_damage_user_variables = []

        if self._damage_increment_user_output is not None:
            if self._damage_increment_user_output.name in sm_subblocks:
                self._solid_mechanics_region.remove_subblock(self._damage_increment_user_output)
                self._damage_increment_user_output = None
        if self._nonlocal_average_output is not None:
            if self._nonlocal_average_output.name in sm_subblocks:
                self._solid_mechanics_region.remove_subblock(self._nonlocal_average_output)
                self._nonlocal_average_output = None
        if self._nonlocal_damage_increment_user_output is not None:
            if self._nonlocal_damage_increment_user_output.name in sm_subblocks:
                self._solid_mechanics_region.remove_subblock(self._nonlocal_damage_increment_user_output)
                self._nonlocal_damage_increment_user_output = None
    
    def _add_nonlocal_damage_user_variables(self, num_integration_pts):
        initial_values = [0]*num_integration_pts
        damage_increment_var = SolidMechanicsUserVariable("damage_increment", "element",
                                                          "real", *initial_values)
        damage_increment_var.add_blocks(*self._death_blocks)
        self._nonlocal_damage_user_variables.append(damage_increment_var)
        self._solid_mechanics_region.add_subblock(damage_increment_var)
       
        nl_damage_increment_var = SolidMechanicsUserVariable("nonlocal_damage_increment", "element",
                                                          "real", *initial_values)
        nl_damage_increment_var.add_blocks(*self._death_blocks)
        self._nonlocal_damage_user_variables.append(nl_damage_increment_var)
        self._solid_mechanics_region.add_subblock(nl_damage_increment_var)

    def _get_death_blocks_string(self):
        return " ".join(self._death_blocks)

    def _add_damage_increment_user_output(self, death_variable, num_integration_pts):
        damage_inc_output = SolidMechanicsUserOutput("damage_increment_output", 
            self._get_death_blocks_string(), "block")
        for int_pt in range(num_integration_pts):
            int_pt +=1
            func_name = f"get_damage_increment_{int_pt}"
            damage_inc_output.add_compute_element_as_function(f"damage_increment({int_pt})", 
                                                              func_name)
            func = AnalyticSierraFunction(func_name)
            func.add_expression_variable("d_cur", "element", f"{death_variable}({int_pt})", 
                                         "state", "new")
            func.add_expression_variable("d_old", "element", f"{death_variable}({int_pt})", 
                                         "state", "old")
            func.add_evaluation_expression("d_cur - d_old")
            self._nonlocal_functions.append(func)
            self.add_subblock(func)
        self._damage_increment_user_output = damage_inc_output
        self._solid_mechanics_region.add_subblock(damage_inc_output)

    def _add_nonlocal_average_output(self, nonlocal_radius):
        nl_damage_avg_output = SolidMechanicsUserOutput("nonlocal_damage_average",
            self._get_death_blocks_string(), "block")
        nonlocal_avg = SolidMechanicsNonlocalDamageAverage(nonlocal_radius)
        nl_damage_avg_output.add_subblock(nonlocal_avg)
        self._nonlocal_average_output = nl_damage_avg_output
        self._solid_mechanics_region.add_subblock(nl_damage_avg_output)

    def _add_nonlocal_damage_increment_user_output(self, death_variable, num_integration_pts):
        critical_value = self._death.get_critical_value()
        expression = (f"(d_old + nl_damage_inc) < {critical_value} ? d_old + nl_damage_inc :"+
                f"{critical_value};")
        nl_damage_inc_output = SolidMechanicsUserOutput("nonlocal_damage_increment_output",
            self._get_death_blocks_string(), "block")
        for int_pt in range(num_integration_pts):
            int_pt+=1
            func_name = f"apply_nonlocal_damage_increment_{int_pt}"
            nl_damage_inc_output.add_compute_element_as_function(f"damage({int_pt})", 
                                                              func_name)
            func = AnalyticSierraFunction(func_name)
            func.add_expression_variable("nl_damage_inc", "element", 
                                         f"nonlocal_damage_increment({int_pt})", 
                                         "state", "new")
            func.add_expression_variable("d_old", "element", f"{death_variable}({int_pt})", 
                                         "state", "old")
            func.add_evaluation_expression(expression)
            self._nonlocal_functions.append(func)
            self.add_subblock(func)
        self._nonlocal_damage_increment_user_output = nl_damage_inc_output
        self._solid_mechanics_region.add_subblock(nl_damage_inc_output)

    def _activate_implicit_dynamics(self):
        self._implicit_dynamics = SolidMechanicsImplicitDynamics()
        self._solid_mechanics_region.add_subblock(self._implicit_dynamics)


class SierraFileThreeDimensionalContact(SierraFileThreeDimensional):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._contact_definitions = None
        self._friction_model = None
        self._control_contact = None
        self._contact_target_relative_residual = 1e-3
        self._contact_target_residual = 1e-2
        self._contact_acceptable_relative_residual = 1e-2
        self._contact_acceptable_residual = None
    
    def _self_contact(self):
        return self._contact_definitions is not None
        
    def _activate_self_contact(self, friction_coefficient):

        if not self._self_contact():
            self._friction_model = SolidMechanicsConstantFrictionModel('self_contact_friction', 
                friction_coefficient)
            self._contact_definitions = SolidMechanicsContactDefinitions(self._friction_model)
            self._solid_mechanics_region.add_subblock(self._contact_definitions)
            self._setup_contact_solver()
        else:
            logger.warning(f"Self contact already active. Updating friction coefficient "+
                           "to {friction_coefficient}.")
            self._friction_model.set_friction_coefficient(friction_coefficient)

    def _setup_contact_solver(self):
        self._add_control_contact_block()
        self._load_step_predictor = SolidMechanicsLoadstepPredictor()
        self._solver.add_subblock(self._load_step_predictor)

    def _add_control_contact_block(self):
        self._control_contact = SolidMechanicsControlContact()
        self._set_contact_convergence_tolerance(self._contact_target_relative_residual, 
            self._contact_target_residual, self._contact_acceptable_relative_residual, 
            self._contact_acceptable_residual)
        self._solver.add_subblock(self._control_contact)

    def _set_contact_convergence_tolerance(self, target_relative_residual, target_residual=None, 
        acceptable_relative_residual=None, acceptable_residual=None):
        self._contact_target_relative_residual = target_relative_residual
        if target_residual is not None:
            self._contact_target_residual = target_residual
        if acceptable_relative_residual is not None:
            self._contact_acceptable_relative_residual = acceptable_relative_residual
        if acceptable_residual  is not None:
            self._contact_acceptable_residual = acceptable_residual
        if self._control_contact is not None:
            self._set_convergence_tolerance(self._control_contact, target_relative_residual, 
                target_residual,acceptable_relative_residual, acceptable_residual, 
                target_resid_factor=10)
            self._update_cg_solver_convergence_tolerances()

    def _update_cg_solver_convergence_tolerances(self):
        contact_target_resid = self._control_contact.get_target_relative_residual()
        super()._set_cg_convergence_tolerance(target_relative_residual=contact_target_resid/10, 
                                        target_residual=contact_target_resid*10, 
                                        acceptable_relative_residual=10)

    