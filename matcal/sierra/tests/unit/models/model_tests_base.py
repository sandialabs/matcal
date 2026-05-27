# model_tests_base.py
import numpy as np
import os

from matcal.core.boundary_condition_calculators import (
    max_state_values,
)
from matcal.core.constants import (
    TIME_KEY,
    TEMPERATURE_KEY,
    ENG_STRAIN_KEY,
    DISPLACEMENT_KEY,
    STRAIN_RATE_KEY,
    DISPLACEMENT_RATE_KEY,
    LOAD_KEY,
)
from matcal.core.data import convert_dictionary_to_data
from matcal.core.state import SolitaryState, State
from matcal.core.tests.unit.test_models import ModelTestBase

from matcal.full_field.data_importer import FieldSeriesData

from matcal.sierra.input_file_writer import SierraFileBase, _Coupling
from matcal.sierra.input_file_writer.boundary_conditions import (
    SolidMechanicsInitialTemperature,
    SolidMechanicsPrescribedTemperature,
)
from matcal.sierra.input_file_writer.sections import _SectionNames
from matcal.sierra.input_file_writer.sierra_file import _Failure
from matcal.sierra.material import Material
from matcal.sierra.tests.utilities import write_linear_elastic_material_file


class MatcalStandardModelUnitTestNewBase:
    def __init__():
        pass

    class CommonTests(ModelTestBase.CommonTests):

        def setUp(self):
            super().setUp(__file__)

            self._example_material_file = write_linear_elastic_material_file()
            self._example_material = Material(
                "matcal_test", self._example_material_file, "linear_elastic"
            )
            self._has_temperature_function_in_input = True

        def _get_temp_data(self):
            temp_dict = {
                "temperature": np.linspace(298, 500, 2),
                "time": np.linspace(0, 10, 2),
                "displacement": np.linspace(0, 1, 2),
                "engineering_strain": np.linspace(0, 1, 2),
                "grip_rotation": np.linspace(0, 200, 2),
            }
            return convert_dictionary_to_data(temp_dict)

        def test_prepare_loading_boundary_condition_state_not_in_BCs(self):
            model = self.init_model()
            bc_data_cols = self.boundary_condition_data_sets
            bad_state = State("not in bc dc")
            for data_col in bc_data_cols:
                model.add_boundary_condition_data(data_col)
                with self.assertRaises(KeyError):
                    model._prepare_loading_boundary_condition_displacement_function(
                        bad_state, "./"
                    )
                break

        def test_prepare_loading_boundary_condition_state_no_bc_data_added(self):
            model = self.init_model()
            bc_data_cols = self.boundary_condition_data_sets
            bad_state = State("not in bc dc")
            for data_col in bc_data_cols:
                with self.assertRaises(RuntimeError):
                    model._prepare_loading_boundary_condition_displacement_function(
                        bad_state, "./"
                    )
                break

        def test_setup_state_all_states(self):
            model = self.init_model()
            bc_data_cols = self.boundary_condition_data_sets
            for data_col in bc_data_cols:
                model.reset_boundary_condition_data()
                model.add_boundary_condition_data(data_col)
                for state in data_col.states.values():
                    model._setup_state(state, build_mesh=False)
                    self.assertTrue(os.path.exists(model._input_filename))

        def test_raise_error_without_correct_material(self):
            with self.assertRaises(TypeError):
                self._model_class("Text")

        def test_add_boundary_condition_add_data(self):
            model = self.init_model()
            with self.assertRaises(TypeError):
                model.add_boundary_condition_data(1)

            bc_data_cols = self.boundary_condition_data_sets
            for data_col in bc_data_cols:
                for state in data_col.state_names:
                    for data in data_col[state]:
                        model.add_boundary_condition_data(data)
                bc_dc = model._boundary_condition_data
                self.assertEqual(data_col.state_names, bc_dc.state_names)
                self._compare_datas_in_two_data_collections(data_col, bc_dc)
                model.reset_boundary_condition_data()

        def test_add_boundary_condition_add_data_collections(self):
            model = self.init_model()

            bc_data_cols = self.boundary_condition_data_sets
            for data_col in bc_data_cols:
                model.add_boundary_condition_data(data_col)
                bc_dc = model._boundary_condition_data
                self.assertEqual(data_col.state_names, bc_dc.state_names)
                self._compare_datas_in_two_data_collections(data_col, bc_dc)
                model.reset_boundary_condition_data()

        def _compare_datas_in_two_data_collections(self, dc1, dc2):
            for state in dc1.state_names:
                for id, data in enumerate(dc1[state]):
                    self.assert_close_arrays(data, dc2[state][id])

        def test_reset_boundary_condition(self):
            model = self.init_model()
            for data_col in self.boundary_condition_data_sets:
                model.add_boundary_condition_data(data_col)
            self.assertTrue(model._boundary_condition_data.state_names != [])
            model.reset_boundary_condition_data()
            self.assertEqual(model._boundary_condition_data.state_names, [])

        def test_activate_element_death(self):
            model = self.init_model()
            self.assertIsNone(model.failure)
            with self.assertRaises(TypeError):
                model.activate_element_death(1, 1)
            with self.assertRaises(TypeError):
                model.activate_element_death("death_var", None)
            model.activate_element_death("eqps", "{kill_eqps}")
            self.assertEqual(model.failure, _Failure.local_failure)

        def test_get_boundary_condition_function(self):
            model = self.init_model()
            for bc_dc in self.boundary_condition_data_sets:
                model.add_boundary_condition_data(bc_dc)
                for state in bc_dc.states.values():
                    model._setup_state(state, build_mesh=False)
                    disp_func = model._get_loading_boundary_condition_displacement_function(
                        state, {}
                    )
                    self.assertTrue(TIME_KEY in disp_func.field_names)
                    self.assertTrue(len(disp_func.field_names) == 2)
                model.reset_boundary_condition_data()

        def test_set_state_model_temperature_bc_data(self):
            model = self.init_model()
            temp_data = self._get_temp_data()

            model.add_boundary_condition_data(temp_data)
            model.read_temperature_from_boundary_condition_data()
            model._set_state_model_temperature(temp_data.state)

            ifile = model.input_file
            ad_region = ifile.solid_mechanics_region
            prescribe_temp_type = SolidMechanicsPrescribedTemperature.type

            self.assertIsNotNone(ad_region.get_subblock_by_type(prescribe_temp_type))
            self.assertIsNone(
                ad_region.get_subblock_by_type(SolidMechanicsInitialTemperature.type)
            )
            if self._has_temperature_function_in_input:
                self.assertTrue(
                    SierraFileBase._temperature_bc_function_name in ifile.subblocks
                )

        def test_set_state_model_temperature_changing_source(self):
            model = self.init_model()
            temp_data = self._get_temp_data()

            model.add_boundary_condition_data(temp_data)
            model.read_temperature_from_boundary_condition_data()
            model._set_state_model_temperature(temp_data.state)

            ifile = model.input_file
            ad_region = ifile.solid_mechanics_region
            prescribe_temp_type = SolidMechanicsPrescribedTemperature.type

            self.assertIsNotNone(ad_region.get_subblock_by_type(prescribe_temp_type))
            self.assertIsNone(
                ad_region.get_subblock_by_type(SolidMechanicsInitialTemperature.type)
            )
            if self._has_temperature_function_in_input:
                self.assertTrue(
                    SierraFileBase._temperature_bc_function_name in ifile.subblocks
                )

            model.reset_boundary_condition_data()
            model._set_state_model_temperature(temp_data.state)

            self.assertIsNone(ad_region.get_subblock_by_type(prescribe_temp_type))
            self.assertIsNone(
                ad_region.get_subblock_by_type(SolidMechanicsInitialTemperature.type)
            )
            if self._has_temperature_function_in_input:
                self.assertFalse(
                    SierraFileBase._temperature_bc_function_name in ifile.subblocks
                )

            state = State("test", temperature=298)
            temp_data.set_state(state)
            model.add_boundary_condition_data(temp_data)
            model._set_state_model_temperature(temp_data.state)

            self.assertIsNone(ad_region.get_subblock_by_type(prescribe_temp_type))
            init_temp_block = ad_region.get_subblock_by_type(
                SolidMechanicsInitialTemperature.type
            )
            self.assertIsNotNone(init_temp_block)
            self.assertEqual(init_temp_block.get_line_value("magnitude"), 298)
            if self._has_temperature_function_in_input:
                self.assertFalse(
                    SierraFileBase._temperature_bc_function_name in ifile.subblocks
                )

        def test_set_state_model_temperature_no_temp(self):
            model = self.init_model()
            model._set_state_model_temperature(SolitaryState())
            ifile = model.input_file
            ad_region = ifile.solid_mechanics_region
            prescribe_temp_type = SolidMechanicsPrescribedTemperature.type

            self.assertIsNone(ad_region.get_subblock_by_type(prescribe_temp_type))
            self.assertIsNone(
                ad_region.get_subblock_by_type(SolidMechanicsInitialTemperature.type)
            )
            self.assertFalse(SierraFileBase._temperature_bc_function_name in ifile.subblocks)

        def test_set_state_model_temperature_state(self):
            model = self.init_model()
            temp_data = self._get_temp_data()
            temp_state = State("400k", temperature=400)
            temp_data.set_state(temp_state)

            model._set_state_model_temperature(temp_data.state)

            sm_region = model.input_file.solid_mechanics_region
            prescribed_temp_type = SolidMechanicsPrescribedTemperature.type
            self.assertIsNone(sm_region.get_subblock_by_type(prescribed_temp_type))

            init_temp = sm_region.get_subblock_by_type(SolidMechanicsInitialTemperature.type)
            self.assertIsNotNone(init_temp)
            self.assertEqual(init_temp.get_line_value("magnitude"), 400)
            self.assertFalse(
                SierraFileBase._temperature_bc_function_name in model.input_file.subblocks
            )

        def test_set_state_model_temperature_model_constants(self):
            model = self.init_model()
            temp_data = self._get_temp_data()
            temp_state = State("400k", temperature=400)
            temp_data.set_state(temp_state)

            model.add_constants(temperature=385)
            model._set_state_model_temperature(temp_data.state)

            sm_region = model.input_file.solid_mechanics_region
            prescribed_temp_type = SolidMechanicsPrescribedTemperature.type
            self.assertIsNone(sm_region.get_subblock_by_type(prescribed_temp_type))

            init_temp = sm_region.get_subblock_by_type(SolidMechanicsInitialTemperature.type)
            self.assertIsNotNone(init_temp)
            self.assertEqual(init_temp.get_line_value("magnitude"), 385)
            self.assertFalse(
                SierraFileBase._temperature_bc_function_name in model.input_file.subblocks
            )

        def test_temp_from_BC_data_bad_data(self):
            model = self.init_model()
            temp_data = self._get_temp_data()

            model.add_boundary_condition_data(temp_data)
            model.read_temperature_from_boundary_condition_data("temp")
            with self.assertRaises(model.TemperatureFieldNotPresentError):
                model._set_state_model_temperature(temp_data.state)

            model.reset_boundary_condition_data()
            model.read_temperature_from_boundary_condition_data("temperature")

            temp_data = temp_data.remove_field("time")
            model.add_boundary_condition_data(temp_data)
            with self.assertRaises(model.TemperatureFieldNotPresentError):
                model._set_state_model_temperature(temp_data.state)

            model.read_temperature_from_boundary_condition_data("temperature")
            with self.assertRaises(TypeError):
                model.read_temperature_from_boundary_condition_data(0.1)

        def test_no_temp_in_state_when_coupled_raises_error(self):
            data = self._get_temp_data()
            model = self.init_model()
            model.add_boundary_condition_data(data)
            model.activate_thermal_coupling()
            with self.assertRaises(RuntimeError):
                model._set_state_model_temperature(data.state)

        def test_raises_error_if_activating_adiabatic_heating_and_reading_temp_from_data(self):
            model = self.init_model()
            data = self._get_temp_data()
            model.add_boundary_condition_data(data)
            model.read_temperature_from_boundary_condition_data()
            with self.assertRaises(RuntimeError):
                model.activate_thermal_coupling()

        def test_set_results_filename(self):
            model = self.init_model()
            with self.assertRaises(AttributeError):
                model.set_results_filename("a fn")

        def test_epu_results(self):
            model = self.init_model()
            self.assertFalse(model._epu_results())

        def test_get_simulator_class_inputs(self):
            model = self.init_model()
            state_names = self.boundary_condition_data_sets[0].state_names
            data = self.boundary_condition_data_sets[0][state_names[0]][0]

            args, kwargs = model._get_simulator_class_inputs(data.state)
            self.assertEqual(args[0], model.name)
            self.assertEqual(args[1], model._simulation_information)
            self.assertEqual(args[2], model._results_information)
            self.assertEqual(args[3], data.state)
            self.assertEqual(args[4], model._input_filename)
            self.assertEqual(kwargs["custom_commands"], [])
            self.assertEqual(kwargs["epu_results"], model._epu_results())
            self.assertIsInstance(kwargs["model_constants"], dict)
            self.assertEqual(kwargs["model_constants"], model.get_model_constants())

        def test_add_executable_argument(self):
            model = self.init_model()
            with self.assertRaises(TypeError):
                model.add_executable_argument(1)
            model.add_executable_argument("--nosignal")
            model.add_executable_argument("--aprepro on")

            state_names = self.boundary_condition_data_sets[0].state_names
            data = self.boundary_condition_data_sets[0][state_names[0]][0]
            args, kwargs = model._get_simulator_class_inputs(data.state)
            self.assertEqual(["--nosignal", "--aprepro on"], kwargs["custom_commands"])

        def test_activate_exodus_output(self):
            model = self.init_model()
            self.assertFalse(model.exodus_output)
            model.activate_exodus_output()
            self.assertTrue(model.exodus_output)
            with self.assertRaises(TypeError):
                model.activate_exodus_output("yay")
            with self.assertRaises(ValueError):
                model.activate_exodus_output(-1)

        def test_set_number_of_time_steps(self):
            model = self.init_model()
            model.set_number_of_time_steps(1)
            self.assertEqual(model.input_file._solid_mechanics_procedure._time_steps, 1)
            model.set_number_of_time_steps(1000)
            self.assertEqual(model.input_file._solid_mechanics_procedure._time_steps, 1000)
            with self.assertRaises(ValueError):
                model.set_number_of_time_steps(-1)
            with self.assertRaises(TypeError):
                model.set_number_of_time_steps("error")

        def test_set_start_time(self):
            model = self.init_model()
            model.set_start_time(1)
            self.assertEqual(model.input_file._solid_mechanics_procedure._start_time, 1)
            model.set_start_time(1000)
            self.assertEqual(model.input_file._solid_mechanics_procedure._start_time, 1000)
            with self.assertRaises(TypeError):
                model.set_start_time("error")

        def test_set_end_time(self):
            model = self.init_model()
            model.set_end_time(1)
            self.assertEqual(model.input_file._solid_mechanics_procedure._termination_time, 1)
            model.set_end_time(1000)
            self.assertEqual(model.input_file._solid_mechanics_procedure._termination_time, 1000)
            with self.assertRaises(TypeError):
                model.set_end_time("error")

        def test_use_total_lagrange_element(self):
            model = self.init_model()
            section_block = model.input_file.solid_mechanics_element_section
            self.assertEqual(section_block.name, _SectionNames.total_lagrange)

            model.use_under_integrated_element()
            section_block = model.input_file.solid_mechanics_element_section
            self.assertEqual(section_block.name, _SectionNames.uniform_gradient)

            model.use_total_lagrange_element()
            section_block = model.input_file.solid_mechanics_element_section
            self.assertEqual(section_block.name, _SectionNames.total_lagrange)

        def test_element_type(self):
            model = self.init_model()
            model.use_under_integrated_element()
            self.assertEqual(model.element_type, _SectionNames.uniform_gradient)

        def test_use_under_integrated_element(self):
            model = self.init_model()
            model.use_under_integrated_element()
            section_block = model.input_file.solid_mechanics_element_section
            self.assertEqual(section_block.name, _SectionNames.uniform_gradient)

        def test_activate_exodus_output_interval_adjust(self, initial_false=True):
            model = self.init_model()
            if initial_false:
                self.assertFalse(model.exodus_output)

            model.activate_exodus_output()
            self.assertTrue(model.exodus_output)

            exo_output = model.input_file._exodus_output
            model.activate_exodus_output(10)
            self.assertEqual(exo_output.get_line_value("at step", -1), 10)

            with self.assertRaises(ValueError):
                model.activate_exodus_output(-1)
            with self.assertRaises(TypeError):
                model.activate_exodus_output("error")

        def test_add_element_output_variable(self, initial_false=True):
            model = self.init_model()
            if initial_false:
                self.assertFalse(model.exodus_output)

            model.add_element_output_variable("stress")
            self.assertTrue(model.exodus_output)

            ifile = model.input_file
            self.assertTrue(ifile._element_variable_in_mesh_output("stress_vol_avg", "stress"))

            model.add_element_output_variable("stress")
            model.add_element_output_variable("stress", volume_average=False)
            self.assertTrue(ifile._element_variable_in_mesh_output("stress"))
            self.assertFalse(ifile._element_variable_in_mesh_output("stress_vol_avg", "stress"))

        def test_add_nodal_output_variable(self, initial_false=True):
            model = self.init_model()
            if initial_false:
                self.assertFalse(model.exodus_output)

            ifile = model.input_file
            self.assertFalse(ifile._nodal_variable_in_mesh_output(TEMPERATURE_KEY))

            model.add_nodal_output_variable(TEMPERATURE_KEY)
            self.assertEqual(model.exodus_output, True)
            self.assertTrue(ifile._nodal_variable_in_mesh_output(TEMPERATURE_KEY))

            model.add_nodal_output_variable("velocity")
            model.add_nodal_output_variable("velocity")
            self.assertTrue(ifile._nodal_variable_in_mesh_output("velocity"))

        def _check_bc_calc_displacement(self, model, bc_dc, state):
            _, max_data_set, _ = max_state_values(bc_dc[state], DISPLACEMENT_KEY)
            model._setup_state(state, build_mesh=False)
            params_by_precedent, _ = model._get_parameters_by_precedence(state)
            func = model._prepare_loading_boundary_condition_displacement_function(
                state, params_by_precedent
            )
            self.assertAlmostEqual(
                func[TIME_KEY][-1],
                max_data_set[DISPLACEMENT_KEY][-1] / (params_by_precedent[DISPLACEMENT_RATE_KEY]),
            )

        def _check_bc_calc_eng_strain(self, model, bc_dc, state):
            _, max_data_set, _ = max_state_values(bc_dc[state], ENG_STRAIN_KEY)
            model._setup_state(state, build_mesh=False)
            params_by_precedent, _ = model._get_parameters_by_precedence(state)
            func = model._prepare_loading_boundary_condition_displacement_function(
                state, params_by_precedent
            )
            self.assertAlmostEqual(
                func[TIME_KEY][-1],
                max_data_set[ENG_STRAIN_KEY][-1] / params_by_precedent[STRAIN_RATE_KEY],
            )

        def test_boundary_condition_function_calculations(self):
            model = self.init_model()
            for bc_dc in self.boundary_condition_data_sets:
                model.reset_boundary_condition_data()
                bc_dc.remove_field("time")
                model.add_boundary_condition_data(bc_dc)
                for state in bc_dc.states.values():
                    if TIME_KEY not in bc_dc.field_names:
                        if DISPLACEMENT_KEY in bc_dc.state_common_field_names(state.name):
                            self._check_bc_calc_displacement(model, bc_dc, state)
                        elif ENG_STRAIN_KEY in bc_dc.state_common_field_names(state.name):
                            self._check_bc_calc_eng_strain(model, bc_dc, state)

        def test_boundary_condition_scale_factor_makes_it_to_input(self):
            model = self.init_model()
            for bc_dc in self.boundary_condition_data_sets:
                model.reset_boundary_condition_data()
                bc_dc.remove_field("time")
                model.add_boundary_condition_data(bc_dc)

                sf = 2
                for state in bc_dc.states.values():
                    model.set_boundary_condition_scale_factor(sf)
                    model._setup_state(state, build_mesh=False)

                    ifile = model.input_file
                    disp_func = model._get_loading_boundary_condition_displacement_function
                    max_disp_time = disp_func(state, state.params)[TIME_KEY][-1]

                    self.assertEqual(
                        ifile.solid_mechanics_procedure._termination_time, max_disp_time * sf
                    )

                    bc_func = ifile.prescribed_loading_boundary_condition
                    self.assertEqual(bc_func.get_line_value("x scale"), sf)
                    self.assertEqual(bc_func.get_line_value("y scale"), sf)
                    sf += 1

        def test_set_minimum_timestep(self):
            model = self.init_model()
            with self.assertRaises(ValueError):
                model.set_minimum_timestep(-1)

            sol_term = model.input_file.solution_termination
            self.assertNotIn("global timestep", sol_term.lines)

            model.set_minimum_timestep(1e-3)
            self.assertIn("global timestep", sol_term.lines)
            self.assertEqual(sol_term.get_line_value("global timestep", -1), 1e-3)

        def test_set_convergence_tolerance(self):
            model = self.init_model()
            cg = model.input_file._cg
            self.assertEqual(cg.get_target_relative_residual(), 1e-9)
            self.assertAlmostEqual(cg.get_target_residual(), 1e-7)
            self.assertAlmostEqual(cg.get_acceptable_relative_residual(), 1e-8)

            with self.assertRaises(ValueError):
                model.set_convergence_tolerance(1)

            model.set_convergence_tolerance(1e-6)
            self.assertEqual(cg.get_target_relative_residual(), 1e-6)
            self.assertAlmostEqual(cg.get_target_residual(), 1e-4)
            self.assertAlmostEqual(cg.get_acceptable_relative_residual(), 1e-5)

            with self.assertRaises(ValueError):
                model.set_convergence_tolerance(1e-6, -1.0)

            model.set_convergence_tolerance(1e-6, 1e-5)
            self.assertEqual(cg.get_target_relative_residual(), 1e-6)
            self.assertAlmostEqual(cg.get_target_residual(), 1e-5)
            self.assertAlmostEqual(cg.get_acceptable_relative_residual(), 1e-5)

            with self.assertRaises(ValueError):
                model.set_convergence_tolerance(1e-6, acceptable_relative_residual=1e-7)

            model.set_convergence_tolerance(1e-6, acceptable_relative_residual=1e-4)
            self.assertAlmostEqual(cg.get_acceptable_relative_residual(), 1e-4)

            with self.assertRaises(ValueError):
                model.set_convergence_tolerance(1e-8, acceptable_residual=-1.0)

            model.set_convergence_tolerance(1e-8, acceptable_residual=1e-5)
            self.assertAlmostEqual(cg.get_acceptable_residual(), 1e-5)

        def test_set_boundary_condition_scale_factor(self):
            model = self.init_model()
            model.set_boundary_condition_scale_factor(1.25)
            self.assertEqual(1.25, model._boundary_condition_scale_factor)

            with self.assertRaises(ValueError):
                model.set_boundary_condition_scale_factor(0.1)
            with self.assertRaises(ValueError):
                model.set_boundary_condition_scale_factor(10.1)
            with self.assertRaises(TypeError):
                model.set_boundary_condition_scale_factor("invalid type")

        def _get_loading_function_block_string(self, model):
            func_block = model.input_file.subblocks[model.input_file._load_bc_function_name]
            return func_block.get_string()

        
class MatcalThreeDimensionalStandardModelUnitTestNewBase:
    def __init__():
        pass

    class CommonTests(MatcalStandardModelUnitTestNewBase.CommonTests):
        _load_var = LOAD_KEY
        _displacement_var = DISPLACEMENT_KEY
        _displacement_user_output_block_name = "global_disp"
        _load_user_output_block_name = "global_load"

        def test_staggered_coupling_input(self):
            model = self.init_model()
            ifile = model.input_file
            self.assertIsNone(ifile._coupled_procedure)

            model.activate_thermal_coupling(
                thermal_conductivity=1,
                density=1,
                specific_heat=1,
                plastic_work_variable="my_var",
            )
            self.assertIsNotNone(ifile._coupled_procedure)

            with self.assertRaises(ValueError):
                model.activate_thermal_coupling(
                    thermal_conductivity=-1,
                    density=1,
                    specific_heat=1,
                    plastic_work_variable="my_var",
                )
            with self.assertRaises(ValueError):
                model.activate_thermal_coupling(
                    thermal_conductivity=1,
                    density=0,
                    specific_heat=1,
                    plastic_work_variable="my_var",
                )
            with self.assertRaises(ValueError):
                model.activate_thermal_coupling(
                    thermal_conductivity=1,
                    density=1,
                    specific_heat=0,
                    plastic_work_variable="my_var",
                )
            with self.assertRaises(TypeError):
                model.activate_thermal_coupling(
                    thermal_conductivity=1,
                    density=1,
                    specific_heat=1,
                    plastic_work_variable=0,
                )
            with self.assertRaises(ValueError):
                model.activate_thermal_coupling(
                    thermal_conductivity=1, specific_heat=1, plastic_work_variable=0
                )

        def test_raises_error_if_activating_thermal_coupling_and_reading_temp_from_data(self):
            model = self.init_model()
            data = convert_dictionary_to_data({self._displacement_var: np.linspace(0, 1, 2)})
            model.add_boundary_condition_data(data)
            model.read_temperature_from_boundary_condition_data()
            with self.assertRaises(RuntimeError):
                model.activate_thermal_coupling()

        def test_iterative_coupling(self):
            model = self.init_model()
            with self.assertRaises(RuntimeError):
                model.use_iterative_coupling()

            model.activate_thermal_coupling(
                thermal_conductivity=1,
                density=1,
                specific_heat=1,
                plastic_work_variable="my_var",
            )
            self.assertEqual(model.coupling, _Coupling.staggered)
            model.use_iterative_coupling()
            self.assertEqual(model.coupling, _Coupling.iterative)

            data = convert_dictionary_to_data({self._displacement_var: [0, 1]})
            state = State("with temp", temperature=1)
            data.set_state(state)

            model.add_boundary_condition_data(data)
            model._set_state_model_temperature(state)

            self.assertIsNotNone(model._input_file._initial_temp)
            self.assertTrue("temperature" in model._input_file._default_nodal_output)

        def test_set_allowable_load_drop_factor(self):
            model = self.init_model()
            model.set_allowable_load_drop_factor(0.25)
            self.assertEqual(0.25, model._allowable_load_drop_factor)

            with self.assertRaises(ValueError):
                model.set_allowable_load_drop_factor(-0.1)
            with self.assertRaises(ValueError):
                model.set_allowable_load_drop_factor(1.1)
            with self.assertRaises(TypeError):
                model.set_allowable_load_drop_factor("invalid type")
                model = self.init_model()

            count = 0
            for idx, bc_dc in enumerate(self.boundary_condition_data_sets):
                model.add_boundary_condition_data(bc_dc)
                for state in bc_dc.states.values():
                    model._setup_state(state, build_mesh=False)
                    self.assertIsNotNone(model._input_file._solution_termination)
                    self.assertIsNotNone(model._input_file._solution_termination_output)
                    self.assertIn(
                        "1-0.25", model._input_file._solution_termination_output.get_string()
                    )
                    if idx > 1:
                        break
                    count += 1

        def test_set_allowable_load_drop_factor_update_from_state(self):
            model = self.init_model()
            model.set_allowable_load_drop_factor(0.25)
            self.assertEqual(0.25, model._allowable_load_drop_factor)

            state = State("new_load_drop", allowable_load_drop_factor=0.1)
            data = convert_dictionary_to_data({self._displacement_var: [0, 1]})
            data.set_state(state)

            model.add_boundary_condition_data(data)
            model._setup_state(state, build_mesh=False)
            self.assertIn(
                "1-0.1", model._input_file._solution_termination_output.get_string()
            )

        def test_set_allowable_load_drop_factor_update_from_model_constants(self):
            model = self.init_model()
            model.set_allowable_load_drop_factor(0.25)
            self.assertEqual(0.25, model._allowable_load_drop_factor)

            state = State("new_load_drop", allowable_load_drop_factor=0.1)
            data = convert_dictionary_to_data({self._displacement_var: [0, 1]})
            data.set_state(state)

            model.add_boundary_condition_data(data)
            model.add_constants(allowable_load_drop_factor=0.3)
            model._setup_state(state, build_mesh=False)
            self.assertIn(
                "1-0.3", model._input_file._solution_termination_output.get_string()
            )

        def test_epu_results_full_field_data(self):
            model = self.init_model()
            model.activate_full_field_data_output(0.01, 0.1)
            self.assertFalse(model._epu_results())
            model.set_number_of_cores(2)
            self.assertTrue(model._epu_results())

        def test_composite_tet(self):
            model = self.init_model()
            model.use_composite_tet_element()
            self.assertEqual(model.element_type, _SectionNames.composite_tet)
            self.assertEqual(model._base_geo_params["element_type"], "tet10")

        def test_add_full_field_output(self):
            model = self.init_model()
            model.activate_full_field_data_output(
                full_field_window_width=0.5 * 0.0254 / 2,
                full_field_window_height=1.25 * 0.0254 / 2,
            )
            self.assertEqual(
                model._base_geo_params["full_field_window_width"], 0.5 * 0.0254 / 2
            )
            self.assertEqual(
                model._base_geo_params["full_field_window_height"], 1.25 * 0.0254 / 2
            )
            self.assertTrue(model._full_field_output)
            self.assertEqual(model.results_filename, "results/full_field_results.e")
            self.assertEqual(model._results_information.results_reader_object, FieldSeriesData)

            count = 0
            for idx, bc_dc in enumerate(self.boundary_condition_data_sets):
                model.add_boundary_condition_data(bc_dc)
                for state in bc_dc.states.values():
                    model._setup_state(state, build_mesh=False)
                    self.assertIsNotNone(model._input_file._full_field_output)
                    if count > 1:
                        break
                    count += 1

        def test_add_full_field_output_wrong_input(self):
            model = self.init_model()
            with self.assertRaises(TypeError):
                model.activate_full_field_data_output()
            with self.assertRaises(TypeError):
                model.activate_full_field_data_output(1)
            with self.assertRaises(TypeError):
                model.activate_full_field_data_output("a", "b")
            with self.assertRaises(TypeError):
                model.activate_full_field_data_output(1, "b")
            with self.assertRaises(TypeError):
                model.activate_full_field_data_output("a", 1)

        def test_activate_element_death_nonlocal(self):
            model = self.init_model()
            self.assertIsNone(model._input_file._death)
            self.assertIsNone(model._nonlocal_radius)
            self.assertIsNone(model.failure)

            model.activate_element_death(nonlocal_radius=0.1)
            self.assertEqual(model._nonlocal_radius, 0.1)
            self.assertEqual(model.failure, _Failure.nonlocal_failure)

        def test_activate_element_death_nonlocal_change_element_type(self):
            model = self.init_model()
            model.activate_element_death(nonlocal_radius=0.1)

            for user_var in model._input_file._nonlocal_damage_user_variables:
                self.assertEqual(user_var.get_line_value("type", -1), 8)
            self.assertEqual(len(model._input_file._nonlocal_functions), 16)

            model.use_composite_tet_element()
            for user_var in model._input_file._nonlocal_damage_user_variables:
                self.assertEqual(user_var.get_line_value("type", -1), 4)
            self.assertEqual(len(model._input_file._nonlocal_functions), 8)

            model.use_under_integrated_element()
            for user_var in model._input_file._nonlocal_damage_user_variables:
                self.assertEqual(user_var.get_line_value("type", -1), 1)
            self.assertEqual(len(model._input_file._nonlocal_functions), 2)

        def test_activate_implicit_dynamics(self):
            model = self.init_model()
            self.assertIsNone(model._input_file._implicit_dynamics)
            model.activate_implicit_dynamics()
            self.assertIsNotNone(model._input_file._implicit_dynamics)

        def test_common_outputs_added(self):
            model = self.init_model()
            data = convert_dictionary_to_data({self._displacement_var: [0, 1]})
            model.add_boundary_condition_data(data)
            model._setup_state(SolitaryState(), build_mesh=False)

            sm_region = model._input_file.solid_mechanics_region
            self.assertIn(self._displacement_user_output_block_name, sm_region.subblocks)
            lines = sm_region.subblocks[self._displacement_user_output_block_name].lines
            self.assertIn(f"global {self._displacement_var}", lines)
            self.assertIn(f"global partial_{self._displacement_var}", lines)

            self.assertIn(self._load_user_output_block_name, sm_region.subblocks)
            lines = sm_region.subblocks[self._load_user_output_block_name].lines
            self.assertIn(f"global {self._load_var}", lines)
            self.assertIn(f"global partial_{self._load_var}", lines)

            hb_output = model._input_file.heartbeat_output
            self.assertTrue(hb_output.has_global_output(self._load_var))
            self.assertTrue(hb_output.has_global_output(self._displacement_var))
            self.assertTrue(hb_output.has_global_output("time"))

        def test_common_outputs_added_full_field(self):
            model = self.init_model()
            data = convert_dictionary_to_data({self._displacement_var: [0, 1]})
            model.add_boundary_condition_data(data)
            model.activate_full_field_data_output(0.1, 0.1)
            model._setup_state(SolitaryState(), build_mesh=False)

            ff_output = model._input_file._full_field_output
            self.assertTrue(ff_output.has_global_output(self._load_var))
            self.assertTrue(ff_output.has_global_output(self._displacement_var))
            self.assertTrue(ff_output.has_global_output("time"))

        def test_outputs_added_adiabatic(self):
            model = self.init_model()
            data = convert_dictionary_to_data({self._displacement_var: [0, 1]})
            state = State("temp", temperature=100)
            data.set_state(state)

            model.add_boundary_condition_data(data)
            model.activate_thermal_coupling()
            model._setup_state(state, build_mesh=False)

            sm_region = model._input_file.solid_mechanics_region
            self.assertIn("global_temperature_output", sm_region.subblocks)

            lines = sm_region.subblocks["global_temperature_output"].lines
            self.assertIn("global low_temperature", lines)
            self.assertIn("global med_temperature", lines)
            self.assertIn("global high_temperature", lines)
            self.assertIn("element temperature", lines["global med_temperature"].get_string())

            hb_output = model._input_file.heartbeat_output
            self.assertTrue(hb_output.has_global_output("low_temperature"))
            self.assertTrue(hb_output.has_global_output("med_temperature"))
            self.assertTrue(hb_output.has_global_output("high_temperature"))

        def test_outputs_added_coupled(self):
            model = self.init_model()
            data = convert_dictionary_to_data({self._displacement_var: [0, 1]})
            state = State("temp", temperature=100)
            data.set_state(state)

            model.add_boundary_condition_data(data)
            model.activate_thermal_coupling(1, 1, 1, "work_var")
            model._setup_state(state, build_mesh=False)

            sm_region = model._input_file.solid_mechanics_region
            self.assertIn("global_temperature_output", sm_region.subblocks)

            lines = sm_region.subblocks["global_temperature_output"].lines
            self.assertIn("global low_temperature", lines)
            self.assertIn("global med_temperature", lines)
            self.assertIn("global high_temperature", lines)
            self.assertIn("nodal temperature", lines["global med_temperature"].get_string())

            hb_output = model._input_file.heartbeat_output
            self.assertTrue(hb_output.has_global_output("low_temperature"))
            self.assertTrue(hb_output.has_global_output("med_temperature"))
            self.assertTrue(hb_output.has_global_output("high_temperature"))


def _parse_denominator_value_from_expression_string(expr_string):
    """
    Parse strings where the quoted expression is of the form:
        "some_string/the_number_of_interest"
    There may be spaces. The quotes are the only quotes in the string.

    We extract the numeric token after the last '/' and before ';' (or before closing quote).
    Returns: float
    """
    s = expr_string

    q1 = s.find('"')
    q2 = s.rfind('"')
    if q1 == -1 or q2 == -1 or q2 <= q1:
        raise AssertionError(f"Expected expression to contain quotes. Got: {s}")

    expr = s[q1 + 1 : q2].strip()
    if expr.endswith(";"):
        expr = expr[:-1].strip()

    slash = expr.rfind("/")
    if slash == -1:
        raise AssertionError(f'Expected "/" in quoted expression. Got: "{expr}"')

    denom_str = expr[slash + 1 :].strip()

    try:
        return float(denom_str)
    except ValueError as e:
        raise AssertionError(
            f'Failed parsing denominator "{denom_str}" from "{expr}"'
        ) from e