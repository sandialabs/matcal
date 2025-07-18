from collections import OrderedDict
import numpy as np
import os

from matcal.core.boundary_condition_calculators import (max_state_values, 
    BoundaryConditionDeterminationError)
from matcal.core.constants import (TIME_KEY, TEMPERATURE_KEY, ENG_STRAIN_KEY, ENG_STRESS_KEY,
                                   DISPLACEMENT_KEY, STRAIN_RATE_KEY, 
                                   DISPLACEMENT_RATE_KEY, LOAD_KEY, ROTATION_KEY, 
                                   TORQUE_KEY, TRUE_STRAIN_KEY, TRUE_STRESS_KEY)
from matcal.core.data import convert_dictionary_to_data, DataCollection
from matcal.core.serializer_wrapper import matcal_save
from matcal.core.state import SolitaryState, State
from matcal.core.tests.unit.test_models import ModelTestBase
from matcal.core.utilities import matcal_name_format

from matcal.cubit.geometry import GeometryParameters

from matcal.full_field.data import convert_dictionary_to_field_data
from matcal.full_field.data_exporter import export_full_field_data_to_json
from matcal.full_field.data_importer import FieldSeriesData

from matcal.sierra.input_file_writer import (SolidMechanicsPrescribedTemperature,
    SolidMechanicsInitialTemperature, SierraFileBase, _Coupling, _Failure, 
    _SectionNames, SolidMechanicsPrescribedDisplacement, SolidMechanicsFixedDisplacement)
from matcal.sierra.material import Material
from matcal.sierra.models import _vfm_field_series_data
from matcal.sierra.tests.utilities import write_linear_elastic_material_file


class MatcalStandardModelUnitTestNewBase:
    def __init__():
        pass
    class CommonTests(ModelTestBase.CommonTests):

        def setUp(self):
            super().setUp(__file__)

            self._example_material_file = write_linear_elastic_material_file()
            self._example_material = Material("matcal_test", self._example_material_file,
                                               "linear_elastic")
            self._has_temperature_function_in_input = True

        def _get_temp_data(self):
            temp_dict = {"temperature":np.linspace(298, 500, 2), 
                        "time":np.linspace(0,10,2), 
                        "displacement":np.linspace(0,1,2),
                        "engineering_strain":np.linspace(0,1,2), 
                        "grip_rotation":np.linspace(0,200,2)}

            temp_data = convert_dictionary_to_data(temp_dict)
            return temp_data

        def test_prepare_loading_boundary_condition_state_not_in_BCs(self):
            model = self.init_model()
            bc_data_cols = self.boundary_condition_data_sets
            bad_state = State("not in bc dc")
            for data_col in bc_data_cols:
                model.add_boundary_condition_data(data_col)
                with self.assertRaises(KeyError):
                    model._prepare_loading_boundary_condition_displacement_function(bad_state, "./")
                break    

        def test_prepare_loading_boundary_condition_state_no_bc_data_added(self):
            model = self.init_model()
            bc_data_cols = self.boundary_condition_data_sets
            bad_state = State("not in bc dc")
            for data_col in bc_data_cols:
                with self.assertRaises(RuntimeError):
                    model._prepare_loading_boundary_condition_displacement_function(bad_state, "./")
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
                bc_dc  = model._boundary_condition_data
                self.assertEqual(data_col.state_names, bc_dc.state_names)
                self._compare_datas_in_two_data_collections(data_col, bc_dc)
                model.reset_boundary_condition_data()

        def test_add_boundary_condition_add_data_collections(self):
            model = self.init_model()

            bc_data_cols = self.boundary_condition_data_sets
            for data_col in bc_data_cols:
                model.add_boundary_condition_data(data_col)              
                bc_dc  = model._boundary_condition_data
                self.assertEqual(data_col.state_names, bc_dc.state_names)
                self._compare_datas_in_two_data_collections(data_col, bc_dc)
                model.reset_boundary_condition_data()

        def _compare_datas_in_two_data_collections(self, dc1, dc2):
            for state in dc1.state_names:
                    for id, data in enumerate(dc1[state]):
                        self.assert_close_arrays(data, dc2[state][id])

        def test_reset_boundary_condition(self):
            model = self.init_model()

            bc_data_cols = self.boundary_condition_data_sets
            for data_col in bc_data_cols:
                model.add_boundary_condition_data(data_col)   
            bc_dc  = model._boundary_condition_data
            self.assertTrue(bc_dc.state_names != [])           
            model.reset_boundary_condition_data()
            bc_dc  = model._boundary_condition_data
            self.assertEqual(bc_dc.state_names, [])

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
                    disp_func = model._get_loading_boundary_condition_displacement_function(state, 
                                                                                            {})            
                    self.assertTrue(TIME_KEY in disp_func.field_names)
                    self.assertTrue(len(disp_func.field_names) == 2)
                model.reset_boundary_condition_data()

        def test_set_state_model_temperature_bc_data(self):
            import numpy as np
            model = self.init_model()
            temp_data = self._get_temp_data()

            model.add_boundary_condition_data(temp_data)
            model.read_temperature_from_boundary_condition_data()
            model._set_state_model_temperature(temp_data.state)
            ifile = model.input_file
            ad_region = ifile.solid_mechanics_region
            prescribe_temp_type = SolidMechanicsPrescribedTemperature.type
            self.assertIsNotNone(ad_region.get_subblock_by_type(prescribe_temp_type))
            self.assertIsNone(ad_region.get_subblock_by_type(SolidMechanicsInitialTemperature.type))
            if self._has_temperature_function_in_input:
                self.assertTrue(SierraFileBase._temperature_bc_function_name in ifile.subblocks)
         
        def test_set_state_model_temperature_changing_source(self):
            import numpy as np
            model = self.init_model()
            temp_data = self._get_temp_data()
            model.add_boundary_condition_data(temp_data)
            model.read_temperature_from_boundary_condition_data()
            model._set_state_model_temperature(temp_data.state)
            ifile = model.input_file
            ad_region = ifile.solid_mechanics_region
            prescribe_temp_type = SolidMechanicsPrescribedTemperature.type
            self.assertIsNotNone(ad_region.get_subblock_by_type(prescribe_temp_type))
            self.assertIsNone(ad_region.get_subblock_by_type(SolidMechanicsInitialTemperature.type))

            if self._has_temperature_function_in_input:
                self.assertTrue(SierraFileBase._temperature_bc_function_name in ifile.subblocks)

            model.reset_boundary_condition_data()
            model._set_state_model_temperature(temp_data.state)
            self.assertIsNone(ad_region.get_subblock_by_type(prescribe_temp_type))
            self.assertIsNone(ad_region.get_subblock_by_type(SolidMechanicsInitialTemperature.type))
            if self._has_temperature_function_in_input:
                self.assertFalse(SierraFileBase._temperature_bc_function_name in ifile.subblocks)

            state = State("test", temperature=298)
            temp_data.set_state(state)
            model.add_boundary_condition_data(temp_data)
            model._set_state_model_temperature(temp_data.state)
            self.assertIsNone(ad_region.get_subblock_by_type(prescribe_temp_type))
            init_temp_block = ad_region.get_subblock_by_type(SolidMechanicsInitialTemperature.type)
            self.assertIsNotNone(init_temp_block)
            self.assertEqual(init_temp_block.get_line_value("magnitude"), 298)
            if self._has_temperature_function_in_input:
                self.assertFalse(SierraFileBase._temperature_bc_function_name in ifile.subblocks)

        def test_set_state_model_temperature_no_temp(self):
            model = self.init_model()
            model._set_state_model_temperature(SolitaryState())
            ifile = model.input_file
            ad_region = ifile.solid_mechanics_region
            prescribe_temp_type = SolidMechanicsPrescribedTemperature.type
            self.assertIsNone(ad_region.get_subblock_by_type(prescribe_temp_type))
            self.assertIsNone(ad_region.get_subblock_by_type(SolidMechanicsInitialTemperature.type))
            self.assertFalse(SierraFileBase._temperature_bc_function_name in ifile.subblocks)

        def test_set_state_model_temperature_state(self):
            model = self.init_model()
            import numpy as np
            temp_data = self._get_temp_data()
            from matcal.core.state import State
            temp_state = State("400k", temperature=400)
            temp_data.set_state(temp_state)
            model._set_state_model_temperature(temp_data.state)
            ifile = model.input_file
            sm_region = ifile.solid_mechanics_region
            prescribed_temp_type = SolidMechanicsPrescribedTemperature.type
            self.assertIsNone(sm_region.get_subblock_by_type(prescribed_temp_type))
            init_temp = sm_region.get_subblock_by_type(SolidMechanicsInitialTemperature.type)
            self.assertIsNotNone(init_temp)
            self.assertEqual(init_temp.get_line_value("magnitude"), 400)
            self.assertFalse(SierraFileBase._temperature_bc_function_name in ifile.subblocks)

        def test_set_state_model_temperature_model_constants(self):
            model = self.init_model()
            import numpy as np
            temp_data = self._get_temp_data()
            temp_state = State("400k", temperature=400)
            temp_data.set_state(temp_state)
            model.add_constants(temperature=385)
            model._set_state_model_temperature(temp_data.state)
            ifile = model.input_file
            sm_region = ifile.solid_mechanics_region
            prescribed_temp_type = SolidMechanicsPrescribedTemperature.type
            self.assertIsNone(sm_region.get_subblock_by_type(prescribed_temp_type))
            init_temp = sm_region.get_subblock_by_type(SolidMechanicsInitialTemperature.type)
            self.assertIsNotNone(init_temp)
            self.assertEqual(init_temp.get_line_value("magnitude"), 385)
            self.assertFalse(SierraFileBase._temperature_bc_function_name in ifile.subblocks)

        def test_temp_from_BC_data_bad_data(self):
            import numpy as np
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
                temp_prepocessor_args = model._set_state_model_temperature(temp_data.state)
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
                model.set_number_of_time_steps('error')

        def test_set_start_time(self):
            model = self.init_model()
            model.set_start_time(1)
            self.assertEqual(model.input_file._solid_mechanics_procedure._start_time, 1)
            model.set_start_time(1000)
            self.assertEqual(model.input_file._solid_mechanics_procedure._start_time, 1000)
            with self.assertRaises(TypeError):
                model.set_start_time('error')

        def test_set_end_time(self):
            model = self.init_model()
            model.set_end_time(1)
            self.assertEqual(model.input_file._solid_mechanics_procedure._termination_time, 1)
            model.set_end_time(1000)
            self.assertEqual(model.input_file._solid_mechanics_procedure._termination_time, 1000)
            with self.assertRaises(TypeError):
                model.set_end_time('error')

        def test_use_total_lagrange_element(self):
            model = self.init_model()
            ifile = model.input_file
            section_block = ifile.solid_mechanics_element_section
            self.assertEqual(section_block.name, _SectionNames.total_lagrange)
            model.use_under_integrated_element()
            section_block = ifile.solid_mechanics_element_section
            self.assertEqual(section_block.name, _SectionNames.uniform_gradient)
            model.use_total_lagrange_element()
            section_block = ifile.solid_mechanics_element_section
            self.assertEqual(section_block.name, _SectionNames.total_lagrange)

        def test_element_type(self):
            model = self.init_model()
            model.use_under_integrated_element()
            ifile = model.input_file
            element_type = model.element_type
            self.assertEqual(element_type, _SectionNames.uniform_gradient)

        def test_use_under_integrated_element(self):
            model = self.init_model()
            model.use_under_integrated_element()
            ifile = model.input_file
            section_block = ifile.solid_mechanics_element_section
            self.assertEqual(section_block.name, _SectionNames.uniform_gradient)

        def test_activate_exodus_output_interval_adjust(self, initial_false=True):
            model = self.init_model()
            ifile = model.input_file
            if initial_false:
                self.assertFalse(model.exodus_output)
            model.activate_exodus_output()
            self.assertTrue(model.exodus_output)
            exo_output = ifile._exodus_output
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

        def _check_bc_calc_displacement(self, model, bc_dc, state, ):
            max_data_index, max_data_set, max_value_index = max_state_values(bc_dc[state], 
                DISPLACEMENT_KEY)
            model._setup_state(state, build_mesh=False)
            params_by_precedent, source_by_precedence = model._get_parameters_by_precedence(state)
            func = model._prepare_loading_boundary_condition_displacement_function(state, 
                params_by_precedent)
            self.assertAlmostEqual(func[TIME_KEY][-1], 
                max_data_set[DISPLACEMENT_KEY][-1]/(params_by_precedent[DISPLACEMENT_RATE_KEY]))

        def _check_bc_calc_eng_strain(self, model, bc_dc, state, ):
            max_data_index, max_data_set, max_value_index = max_state_values(bc_dc[state], 
                ENG_STRAIN_KEY)
            model._setup_state(state, build_mesh=False)
            params_by_precedent, source_by_precedence = model._get_parameters_by_precedence(state)
            func = model._prepare_loading_boundary_condition_displacement_function(state, 
                params_by_precedent)
            self.assertAlmostEqual(func[TIME_KEY][-1], 
                max_data_set[ENG_STRAIN_KEY][-1]/params_by_precedent[STRAIN_RATE_KEY])

        def test_boundary_condition_function_calculations(self):
            model = self.init_model()
            for bc_dc in self.boundary_condition_data_sets:
                model.reset_boundary_condition_data()
                bc_dc.remove_field("time")
                model.add_boundary_condition_data(bc_dc)
                for id, state in enumerate(bc_dc.states.values()):
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
                model.set_boundary_condition_scale_factor(2)
                sf = 2
                for id, state in enumerate(bc_dc.states.values()):
                    model.set_boundary_condition_scale_factor(sf)
                    model._setup_state(state, build_mesh=False)
                    ifile = model.input_file
                    disp_func = model._get_loading_boundary_condition_displacement_function
                    max_disp_time = disp_func(state, state.params)[TIME_KEY][-1]
                    self.assertEqual(ifile.solid_mechanics_procedure._termination_time,
                                      max_disp_time*sf)
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
                model.set_convergence_tolerance(1e-6, 1e-7)
            model.set_convergence_tolerance(1e-6, 1e-5)
            self.assertEqual(cg.get_target_relative_residual(), 1e-6)
            self.assertAlmostEqual(cg.get_target_residual(), 1e-5)
            self.assertAlmostEqual(cg.get_acceptable_relative_residual(), 1e-5)
            with self.assertRaises(ValueError):
                model.set_convergence_tolerance(1e-6, acceptable_relative_residual=1e-7)
            model.set_convergence_tolerance(1e-6, acceptable_relative_residual=1e-4)
            self.assertAlmostEqual(cg.get_acceptable_relative_residual(), 1e-4)
            with self.assertRaises(ValueError):
                model.set_convergence_tolerance(1e-8, acceptable_residual=1e-6)
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


from matcal.sierra.tests.sierra_sm_models_for_tests import UniaxialLoadingMaterialPointModelForTests
class UniaxialLoadingMaterialPointModelTests(MatcalStandardModelUnitTestNewBase.CommonTests, 
    UniaxialLoadingMaterialPointModelForTests):
    def test_setup_state_all_states_with_build_mesh(self):
        model = self.init_model()
        bc_data_cols = self.boundary_condition_data_sets
        for data_col in bc_data_cols:
            model.reset_boundary_condition_data()
            model.add_boundary_condition_data(data_col)
            for state in data_col.states.values():
                model._setup_state(state, build_mesh=True)
                self.assertTrue(os.path.exists(model._mesh_filename))
                os.remove(model._mesh_filename)

    def _check_outputs_default(self, model):
        sm_region = model._input_file.solid_mechanics_region
        self.assertIn("global_stress_strain_load_disp", 
            sm_region.subblocks)
        lines =  sm_region.subblocks["global_stress_strain_load_disp"].lines
        self.assertIn(f"global {DISPLACEMENT_KEY}", lines)
        self.assertIn(f"global {LOAD_KEY}", lines)
        
        self.assertIn("true_stress_strain", sm_region.subblocks)
        lines =  sm_region.subblocks["true_stress_strain"].lines
        self.assertIn(f"global {TRUE_STRAIN_KEY}", lines)
        self.assertIn(f"global log_strain_xx", lines)
        self.assertIn(f"global log_strain_yy", lines)
        self.assertIn(f"global {TRUE_STRESS_KEY}", lines)

        hb_output = model._input_file.heartbeat_output
        self.assertTrue(hb_output.has_global_output(DISPLACEMENT_KEY))
        self.assertTrue(hb_output.has_global_output(LOAD_KEY))
        self.assertTrue(hb_output.has_global_output(DISPLACEMENT_KEY, ENG_STRAIN_KEY))
        self.assertTrue(hb_output.has_global_output(LOAD_KEY, ENG_STRESS_KEY))
        self.assertTrue(hb_output.has_global_output(TRUE_STRAIN_KEY))
        self.assertTrue(hb_output.has_global_output(TRUE_STRESS_KEY))
        self.assertTrue(hb_output.has_global_output("log_strain_xx"))
        self.assertTrue(hb_output.has_global_output("log_strain_yy"))
        self.assertTrue(hb_output.has_global_output("time"))    
        self.assertTrue(hb_output.has_global_output("contraction"))    

    def test_outputs_added(self):
        model = self.init_model()
        data = convert_dictionary_to_data({"engineering_strain":[0,1]})
        model.add_boundary_condition_data(data)
        model._setup_state(SolitaryState(), build_mesh=False)
        self._check_outputs_default(model)
        sm_region = model._input_file.solid_mechanics_region
        lines =  sm_region.subblocks["true_stress_strain"].lines
        self.assertNotIn(f"global {TEMPERATURE_KEY}", lines)

        hb_output = model._input_file.heartbeat_output
        self.assertFalse(hb_output.has_global_output(TEMPERATURE_KEY))

    def test_outputs_added_adiabatic(self):
        model = self.init_model()
        data = convert_dictionary_to_data({"engineering_strain":[0,1]})
        data.set_state(State("temp", temperature=100))
        
        model.add_boundary_condition_data(data)
        model.activate_thermal_coupling()
        model._setup_state(data.state, build_mesh=False)
        self._check_outputs_default(model)
        sm_region = model._input_file.solid_mechanics_region
        lines =  sm_region.subblocks["true_stress_strain"].lines
        self.assertIn(f"global {TEMPERATURE_KEY}", lines)

        hb_output = model._input_file.heartbeat_output
        self.assertTrue(hb_output.has_global_output(TEMPERATURE_KEY))


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
            model.activate_thermal_coupling(thermal_conductivity=1, density=1, 
                                            specific_heat=1, plastic_work_variable="my_var")
            self.assertIsNotNone(ifile._coupled_procedure)

            with self.assertRaises(ValueError):
                model.activate_thermal_coupling(thermal_conductivity=-1, density=1, 
                                                specific_heat=1, plastic_work_variable="my_var")
            with self.assertRaises(ValueError):
                model.activate_thermal_coupling(thermal_conductivity=1, density=0, 
                                                specific_heat=1, plastic_work_variable="my_var")
            with self.assertRaises(ValueError):
                model.activate_thermal_coupling(thermal_conductivity=1, density=1, 
                                                specific_heat=0, plastic_work_variable="my_var")
            with self.assertRaises(TypeError):
                model.activate_thermal_coupling(thermal_conductivity=1, density=1, 
                                                specific_heat=1, plastic_work_variable=0)
            with self.assertRaises(ValueError):
                model.activate_thermal_coupling(thermal_conductivity=1, 
                                                specific_heat=1, plastic_work_variable=0)

        def test_raises_error_if_activating_thermal_coupling_and_reading_temp_from_data(self):
            import numpy as np
            model = self.init_model()
            data_dict = {self._displacement_var:np.linspace(0,1,2)}
            data = convert_dictionary_to_data(data_dict)
            model.add_boundary_condition_data(data)
            model.read_temperature_from_boundary_condition_data()
            with self.assertRaises(RuntimeError):
                model.activate_thermal_coupling()

        def test_iterative_coupling(self):
            model = self.init_model()

            with self.assertRaises(RuntimeError):
                model.use_iterative_coupling()
            model.activate_thermal_coupling(thermal_conductivity=1, density=1, 
                                            specific_heat=1, plastic_work_variable="my_var")
            self.assertEqual(model.coupling, _Coupling.staggered)
            model.use_iterative_coupling()
            self.assertEqual(model.coupling, _Coupling.iterative)
            data_dict = {self._displacement_var:[0,1]}
            data = convert_dictionary_to_data(data_dict)
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
                    self.assertIn("1-0.25", 
                                  model._input_file._solution_termination_output.get_string())
                    if idx > 1:
                        break
                    count += 1

        def test_set_allowable_load_drop_factor_update_from_state(self):
            model = self.init_model()
            model.set_allowable_load_drop_factor(0.25)
            self.assertEqual(0.25, model._allowable_load_drop_factor)

            state = State("new_load_drop", allowable_load_drop_factor=0.1)
            data = convert_dictionary_to_data({self._displacement_var:[0,1]})
            data.set_state(state)
            model.add_boundary_condition_data(data)
            model._setup_state(state, build_mesh=False)
            self.assertIn("1-0.1", model._input_file._solution_termination_output.get_string())

        def test_set_allowable_load_drop_factor_update_from_model_constants(self):
            model = self.init_model()
            model.set_allowable_load_drop_factor(0.25)
            self.assertEqual(0.25, model._allowable_load_drop_factor)

            state = State("new_load_drop", allowable_load_drop_factor=0.1)
            data = convert_dictionary_to_data({self._displacement_var:[0,1]})
            data.set_state(state)
            model.add_boundary_condition_data(data)
            model.add_constants(allowable_load_drop_factor=0.3)
            model._setup_state(state, build_mesh=False)
            self.assertIn("1-0.3", model._input_file._solution_termination_output.get_string())

        def test_epu_results_full_field_data(self):
            model = self.init_model()
            model.activate_full_field_data_output(.01, 0.1)
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
            model.activate_full_field_data_output(full_field_window_width=0.5*0.0254/2,
                                                  full_field_window_height=1.25*0.0254/2)
            self.assertEqual(model._base_geo_params["full_field_window_width"],0.5*0.0254/2)
            self.assertEqual(model._base_geo_params["full_field_window_height"],1.25*0.0254/2)
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
                    count +=1

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
            self.assertIsNone(model._input_file._death)
            self.assertIsNone(model._nonlocal_radius)
            self.assertIsNone(model.failure)
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
            data = convert_dictionary_to_data({self._displacement_var:[0,1]})
            model.add_boundary_condition_data(data)
            model._setup_state(SolitaryState(), build_mesh=False)
            sm_region = model._input_file.solid_mechanics_region
            self.assertIn(self._displacement_user_output_block_name, 
                sm_region.subblocks)
            lines =  sm_region.subblocks[self._displacement_user_output_block_name].lines
            self.assertIn(f"global {self._displacement_var}", lines)
            self.assertIn(f"global partial_{self._displacement_var}", lines)
            
            self.assertIn(self._load_user_output_block_name, sm_region.subblocks)
            lines =  sm_region.subblocks[self._load_user_output_block_name].lines
            self.assertIn(f"global {self._load_var}", lines)
            self.assertIn(f"global partial_{self._load_var}", lines)
            
            hb_output = model._input_file.heartbeat_output
            self.assertTrue(hb_output.has_global_output(self._load_var))
            self.assertTrue(hb_output.has_global_output(self._displacement_var))
            self.assertTrue(hb_output.has_global_output("time"))

        def test_common_outputs_added_full_field(self):
            model = self.init_model()
            data = convert_dictionary_to_data({self._displacement_var:[0,1]})
            model.add_boundary_condition_data(data)
            model.activate_full_field_data_output(0.1, 0.1)
            model._setup_state(SolitaryState(), build_mesh=False)
            ff_output = model._input_file._full_field_output
            self.assertTrue(ff_output.has_global_output(self._load_var))
            self.assertTrue(ff_output.has_global_output(self._displacement_var))
            self.assertTrue(ff_output.has_global_output("time"))
            return ff_output

        def test_outputs_added_adiabatic(self):
            model = self.init_model()
            data = convert_dictionary_to_data({self._displacement_var:[0,1]})
            state = State("temp", temperature=100)
            data.set_state(state)
            model.add_boundary_condition_data(data)
            model.activate_thermal_coupling()
            model._setup_state(state, build_mesh=False)
            sm_region = model._input_file.solid_mechanics_region
            self.assertIn("global_temperature_output", sm_region.subblocks)
            lines =  sm_region.subblocks["global_temperature_output"].lines
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
            data = convert_dictionary_to_data({self._displacement_var:[0,1]})
            state = State("temp", temperature=100)
            data.set_state(state)            
            model.add_boundary_condition_data(data)
            model.activate_thermal_coupling(1,1,1,"work_var")
            model._setup_state(state, build_mesh=False)
            sm_region = model._input_file.solid_mechanics_region
            self.assertIn("global_temperature_output", sm_region.subblocks)
            lines =  sm_region.subblocks["global_temperature_output"].lines
            self.assertIn("global low_temperature", lines)
            self.assertIn("global med_temperature", lines)
            self.assertIn("global high_temperature", lines)
            self.assertIn("nodal temperature", lines["global med_temperature"].get_string())
            hb_output = model._input_file.heartbeat_output
            self.assertTrue(hb_output.has_global_output("low_temperature"))
            self.assertTrue(hb_output.has_global_output("med_temperature"))
            self.assertTrue(hb_output.has_global_output("high_temperature"))


class EigthSymmetryModelTests:
    def __init__():
        pass
    class CommonTests(MatcalThreeDimensionalStandardModelUnitTestNewBase.CommonTests):

        def test_add_boundary_condition_mixed_boundary_data_fields(self):
            model = self.init_model()
            eng_data_dict = {"engineering_stress":np.linspace(0,100,10), 
                            "engineering_strain":np.linspace(0,1,10)}
            data_stress_strain = convert_dictionary_to_data(eng_data_dict)
            data_dict = {"load":np.linspace(0,100,10), 
                          "displacement":np.linspace(0,1,10)}
            data_load_disp = convert_dictionary_to_data(data_dict)
            data_col = DataCollection("test", data_load_disp, data_stress_strain)
            model.add_boundary_condition_data(data_col)              
            with self.assertRaises(BoundaryConditionDeterminationError):
                model._prepare_loading_boundary_condition_displacement_function(SolitaryState(), {})

        def test_state_geo_param_override(self):
            model = self.init_model()
            data_dict = {"displacement":np.linspace(0,100,10)}
            data = convert_dictionary_to_data(data_dict)
            state = State("test", extensometer_length=0.5*0.0254)
            data.set_state(state)
            data2 = convert_dictionary_to_data(data_dict)
            default_state = data2.state
            dc = DataCollection("test", data, data2)

            model.add_boundary_condition_data(dc)              
            model._setup_state(state, ".", build_mesh=False)
            self.assertEqual(model._current_state_geo_params["extensometer_length"], 0.5*0.0254)
            self.assertEqual(model._base_geo_params["extensometer_length"], 1.*0.0254)

            model._setup_state(default_state, ".", build_mesh=False)
            self.assertEqual(model._current_state_geo_params["extensometer_length"], 1.*0.0254)

        def test_state_geo_param_override_model_constants_take_precedent(self):
            model = self.init_model()
            data_dict = {"displacement":np.linspace(0,100,10)}
            data = convert_dictionary_to_data(data_dict)
            state = State("test", extensometer_length=0.5*0.0254, element_size=0.01*0.0254)
            data.set_state(state)
            data2 = convert_dictionary_to_data(data_dict)
            default_state = data2.state
            dc = DataCollection("test", data, data2)

            model.add_boundary_condition_data(dc)  
            model.add_state_constants(state, extensometer_length = 0.25*0.0254, 
                                      element_size=0.02*0.0254)            
            model._setup_state(state, ".", build_mesh=False)

            self.assertEqual(model._current_state_geo_params["extensometer_length"], 0.25*0.0254)
            self.assertEqual(model._base_geo_params["extensometer_length"], 1.*0.0254)

            model._setup_state(default_state, ".", build_mesh=False)
            self.assertEqual(model._current_state_geo_params["extensometer_length"], 1.*0.0254)
    

class UniaxialTensionStandardModelUnitTestBase:
    def __init__():
        pass
    class CommonTests(EigthSymmetryModelTests.CommonTests):

        def test_fail_if_mixed_bc_fields_in_a_state(self):
            model = self.init_model()
            disp_data = convert_dictionary_to_data({"displacement":[0,1]})
            strain_data = convert_dictionary_to_data({"engineering_strain":[0,1]})
            bc_dc = DataCollection("test", disp_data, strain_data)

            model.add_boundary_condition_data(bc_dc)
            with self.assertRaises(BoundaryConditionDeterminationError):
                model._setup_state(disp_data.state, build_mesh=False)
    
        def test_specific_outputs_added(self):
            model = self.init_model()
            data = convert_dictionary_to_data({"displacement":[0,1]})
            model.add_boundary_condition_data(data)
            model._setup_state(SolitaryState(), build_mesh=False)
            sm_region = model._input_file.solid_mechanics_region
            self.assertIn("global_strain", sm_region.subblocks)
            lines =  sm_region.subblocks["global_strain"].lines
            self.assertIn("global engineering_strain", lines)

            self.assertIn("global_stress", sm_region.subblocks)
            lines =  sm_region.subblocks["global_stress"].lines
            self.assertIn("global engineering_stress", lines)

            self.assertIn("x_contraction", sm_region.subblocks)
            lines =  sm_region.subblocks["x_contraction"].lines
            self.assertIn("global x_contraction", lines)

            self.assertIn("z_contraction", sm_region.subblocks)
            lines =  sm_region.subblocks["z_contraction"].lines
            self.assertIn("global z_contraction", lines)

            hb_output = model._input_file.heartbeat_output
            self.assertTrue(hb_output.has_global_output("engineering_stress"))
            self.assertTrue(hb_output.has_global_output("engineering_strain"))
            self.assertTrue(hb_output.has_global_output("x_contraction"))
            self.assertTrue(hb_output.has_global_output("z_contraction"))

        def test_derived_outputs_added_full_field(self):
            ff_output = super().test_common_outputs_added_full_field()
            self.assertTrue(ff_output.has_global_output("engineering_strain"))
            self.assertTrue(ff_output.has_global_output("engineering_stress"))
            self.assertTrue(ff_output.has_global_output("x_contraction"))
            self.assertTrue(ff_output.has_global_output("z_contraction"))

            
from matcal.sierra.tests.sierra_sm_models_for_tests import RoundUniaxialTensionModelForTests
class RoundUniaxialTensionModelUnitTests(UniaxialTensionStandardModelUnitTestBase.CommonTests, 
    RoundUniaxialTensionModelForTests):
    """"""
    def test_bad_geo_caught_on_setup_state_and_init(self):
        mat = self._get_material(plasticity=True)
        geo_params_bad = {"extensometer_length": 1.5,
                        "gauge_length": 1.25,
                        "gauge_radius": 0.125,
                        "grip_radius": 0.25,
                        "total_length": 4,
                        "fillet_radius": 0.188,
                        "taper": 0.0015,
                        "necking_region":0.375,
                        "element_size": 0.0125,
                        "mesh_method":3,
                        "grip_contact_length":1}
        with self.assertRaises(GeometryParameters.ValueError):
            model = self._model_class(mat, **geo_params_bad)
        geo_params_good = geo_params_bad
        geo_params_good["extensometer_length"] = 1.0
        model = self._model_class(mat, **geo_params_good)
        #geo params add in order that could result in failure. Need to 
        #check params only after all parameters have been processed.
        model.add_constants(extensometer_length=1.5, gauge_length=1.55)
        params_by_precedence, param_source = model._get_parameters_by_precedence(SolitaryState())
        model._update_geometry_parameters(params_by_precedence, param_source)
        

from matcal.sierra.tests.sierra_sm_models_for_tests import RectangularUniaxialTensionModelForTests
class RectangularUniaxialTensionModelUnitTests(UniaxialTensionStandardModelUnitTestBase.CommonTests, 
    RectangularUniaxialTensionModelForTests):
    """"""


from matcal.sierra.tests.sierra_sm_models_for_tests import RoundNotchedTensionModelForTests
class RoundNotchedTensionModelUnitTests(EigthSymmetryModelTests.CommonTests, 
    RoundNotchedTensionModelForTests):
    """"""

from matcal.sierra.tests.sierra_sm_models_for_tests import SolidBarTorsionModelForTests
class SolidBarTorsionModelUnitTests(MatcalThreeDimensionalStandardModelUnitTestNewBase.CommonTests, 
    SolidBarTorsionModelForTests):
    _load_var = TORQUE_KEY
    _displacement_var = ROTATION_KEY
    _displacement_user_output_block_name = "global_torque_rotation"
    _load_user_output_block_name = "global_torque_rotation"

    def test_specific_outputs_added(self):
            model = self.init_model()
            data = convert_dictionary_to_data({self._displacement_var:[0,1]})
            model.add_boundary_condition_data(data)
            model._setup_state(SolitaryState(), build_mesh=False)
            sm_region = model._input_file.solid_mechanics_region
            lines =  sm_region.subblocks[self._displacement_user_output_block_name].lines
            self.assertIn("global applied_rotation", lines)
            hb_output = model._input_file.heartbeat_output
            self.assertTrue(hb_output.has_global_output("applied_rotation"))

    def test_prescribed_zero_displacement_added(self):
        model = self.init_model()
        data = convert_dictionary_to_data({self._displacement_var:[0,1]})
        model.add_boundary_condition_data(data)
        model._setup_state(SolitaryState(), build_mesh=False)
        sm_region = model._input_file.solid_mechanics_region
        zero_func_block_name = "ns_y_symmetry cylindrical_axis sierra_constant_function_zero"
        self.assertIn(zero_func_block_name, sm_region.subblocks)


from matcal.sierra.tests.sierra_sm_models_for_tests import TopHatShearModelForTests
class TopHatShearModelUnitTests(MatcalThreeDimensionalStandardModelUnitTestNewBase.CommonTests, 
    TopHatShearModelForTests):
    """"""

    def test_add_full_field_output(self):
        """"""

    def test_add_full_field_output_wrong_input(self):
        """"""

    def test_common_outputs_added_full_field(self):
        """"""

    def test_epu_results_full_field_data(self):
        """"""

    def test_errors_with_ff_output_request(self):
        model = self.init_model()
        with self.assertRaises(AttributeError):
            model.activate_full_field_data_output()

    def test_activate_self_contact(self):
        model = self.init_model()
        self.assertFalse(model.self_contact)
        model.activate_self_contact()
        self.assertTrue(model.self_contact)
        
        with self.assertRaises(TypeError):
            model.activate_self_contact("not valid input")
        self.assertIsNotNone(model.input_file._control_contact)
        self.assertEqual(model.input_file._friction_model.get_friction_coefficient(), 0.3)
        model.activate_self_contact(0.2)
        self.assertEqual(model.input_file._friction_model.get_friction_coefficient(), 0.2)

    def test_set_contact_convergence_tolerance(self):
        model = self.init_model()
        model.activate_self_contact()
        with self.assertRaises(ValueError):
            model.set_contact_convergence_tolerance(1)
        contact = model.input_file._control_contact
        model.set_contact_convergence_tolerance(1e-6)
        cg = model.input_file._cg

        self.assertEqual(contact.get_target_relative_residual(), 1e-6)
        self.assertAlmostEqual(contact.get_target_residual(), 1e-5)
        self.assertAlmostEqual(contact.get_acceptable_relative_residual(), 1e-5)
        self.assertEqual(cg.get_target_relative_residual(), 1e-7)
        self.assertAlmostEqual(cg.get_target_residual(), 1e-5)
        self.assertAlmostEqual(cg.get_acceptable_relative_residual(), 10)

    def test_set_contact_convergence_tolerance_different_options(self):
        model = self.init_model()
        model.activate_self_contact()
        with self.assertRaises(ValueError):
            model.set_contact_convergence_tolerance(1)
        contact = model.input_file._control_contact
        model.set_contact_convergence_tolerance(1e-6)
        with self.assertRaises(ValueError):
            model.set_contact_convergence_tolerance(1e-6, 1e-7)
        model.set_contact_convergence_tolerance(1e-6, 1e-5)
        self.assertEqual(contact.get_target_relative_residual(), 1e-6)
        self.assertAlmostEqual(contact.get_target_residual(), 1e-5)
        self.assertAlmostEqual(contact.get_acceptable_relative_residual(), 1e-5)
        with self.assertRaises(ValueError):
            model.set_contact_convergence_tolerance(1e-6, acceptable_relative_residual=1e-7)
        model.set_contact_convergence_tolerance(1e-6, acceptable_relative_residual=1e-4)
        self.assertAlmostEqual(contact.get_acceptable_relative_residual(), 1e-4)
        with self.assertRaises(ValueError):
            model.set_contact_convergence_tolerance(1e-8, target_residual=1e-6, acceptable_residual=1e-7)
        model.set_contact_convergence_tolerance(1e-8, acceptable_residual=1e-5)
        self.assertAlmostEqual(contact.get_acceptable_residual(), 1e-5)
    
    def test_activate_contact_different_cg_options(self):
        model = self.init_model()
        model.activate_self_contact()

        model.set_convergence_tolerance(1e-6)
        cg = model.input_file._cg
        self.assertEqual(cg.get_target_relative_residual(), 1e-6)
        self.assertAlmostEqual(cg.get_target_residual(), 1e-4)
        self.assertAlmostEqual(cg.get_acceptable_relative_residual(), 1e-5)


class TestVFMUniaxialTensionModelCommon:
    def __init__():
        pass
    class VFMCommonTests(MatcalStandardModelUnitTestNewBase.CommonTests):
        
        def setUp(self):
            super().setUp()
            self._has_temperature_function_in_input = False

        def test_boundary_condition_function_calculations(self):
            """"Not needed, reads from mesh"""

        def test_add_boundary_condition_add_data_collections(self):
            """skipping due to shared infrastructure"""

        def test_add_boundary_condition_add_data(self):
            """skipping due to shared infrastructure"""

        def test_add_nodal_output_variable(self):
            super().test_add_nodal_output_variable(initial_false=False)

        def test_add_element_output_variable(self):
            super().test_add_element_output_variable(initial_false=False)

        def test_activate_exodus_output_interval_adjust(self):
            with self.assertRaises(TypeError):
                super().test_activate_exodus_output_interval_adjust(initial_false=False)

        def test_activate_exodus_output(self):
            model = self.init_model()
            model.activate_exodus_output()
            self.assertTrue(model.exodus_output)
            with self.assertRaises(TypeError):
                model.activate_exodus_output("yay")
            with self.assertRaises(TypeError):
                model.activate_exodus_output(-1)
        
        def test_boundary_condition_scale_factor_makes_it_to_input(self):
            with self.assertRaises(AttributeError):
                super().test_boundary_condition_scale_factor_makes_it_to_input()

        def test_set_boundary_condition_scale_factor(self):
            with self.assertRaises(AttributeError):
                super().test_set_boundary_condition_scale_factor()

        def _get_temp_data(self):
            return self._field_data

        def test_set_field_names(self):
            model = self.init_model(x_disp_name=None, y_disp_name=None)
            self.assertEqual(model._x_displacement_field_name, "U")
            self.assertEqual(model._y_displacement_field_name, "V")
            model.set_displacement_field_names("X", "Y")
            self.assertEqual(model._x_displacement_field_name, "X")
            self.assertEqual(model._y_displacement_field_name, "Y")

        def test_repeat_boundary_data(self):
            model = self.init_model()
            from copy import deepcopy
            from matcal import DataCollection
            field_data2 =  deepcopy(self._field_data)
            field_data2.set_name("new_data")

            my_dc = DataCollection("too many field datas", self._field_data, field_data2)

            with self.assertRaises(BoundaryConditionDeterminationError):
                model.add_boundary_condition_data(my_dc)

        def test_not_field_boundary_data(self):
            from matcal import convert_dictionary_to_data
            model = self.init_model()
            data_dict = {"x":[0,1], "y":[0,1]}
            data = convert_dictionary_to_data(data_dict)
            with self.assertRaises(BoundaryConditionDeterminationError):
                model.add_boundary_condition_data(data)

        def test_activate_element_death(self):
            model = self.init_model()
            model.activate_element_death()

        def test_mesh_not_found(self):
            with self.assertRaises(FileNotFoundError):
                self._model_class(self._example_material, "no_mesh.g", thickness=.1)

        def test_thickness(self):
            model = self.init_model()   
            thickness = model.get_thickness()
            self.assertAlmostEqual(0.1, thickness)

        def test_bad_inits(self):
            self.init_model()
            with self.assertRaises(TypeError):
                model = self._model_class(self._example_material, self._mesh_grid, thickness="a") 
            with self.assertRaises(FileNotFoundError):
                model = self._model_class(self._example_material, "not a mesh", thickness=0.1) 
            with self.assertRaises(TypeError):
                model = self._model_class("not a material class", self._mesh_grid, thickness=0.1) 

        def test_use_under_integrated_element(self):
            model = self.init_model()
            model.add_boundary_condition_data(self._field_data)
            model.set_displacement_field_names("Ux", "Uy")
            exo_output = model._input_file.exodus_output
            self.assertTrue(exo_output.has_element_output("first_pk_stress_vol_avg", 
                                                           "first_pk_stress"))
            model.use_under_integrated_element()
            self.assertTrue(exo_output.has_element_output("first_pk_stress_vol_avg", 
                                                           "first_pk_stress"))

        def test_change_map(self):
            model = self.init_model()
            self.assertAlmostEqual(model._polynomial_order, 1)
            self.assertAlmostEqual(model._search_radius_multiplier, 2.75)
            model.set_mapping_parameters(3, 5)
            self.assertAlmostEqual(model._polynomial_order, 3)
            self.assertAlmostEqual(model._search_radius_multiplier, 5)

        def test_generate_input_deck_with_temperature(self):
            model = self.init_model()
            model.add_boundary_condition_data(self._field_data)
            model.set_displacement_field_names("Ux", "Uy")
            model.read_temperature_from_boundary_condition_data('temperature')
            state = SolitaryState()
            model._setup_state(state, build_mesh=False)
            self.assertTrue(os.path.exists(matcal_name_format(model.name)+".i"))

        def test_generate_input_deck(self):
            model = self.init_model()
            model.add_boundary_condition_data(self._field_data)
            model.set_displacement_field_names("Ux", "Uy")
            state = SolitaryState()
            model._setup_state(state, build_mesh=False)
            self.assertTrue(os.path.exists(matcal_name_format(model.name)+".i"))

        def test_generate_fail_wrong_bc_fields(self):
            model = self.init_model(x_disp_name="bad_x", y_disp_name="bad_y")
            model.add_boundary_condition_data(self._field_data)
            state = SolitaryState()
            with self.assertRaises(BoundaryConditionDeterminationError):
                model._setup_state(state, build_mesh=False)

        def test_generate_input_bad_field_errors_before_mesh_build(self):
            model = self.init_model(x_disp_name="bad_x", y_disp_name="bad_y")
            model.add_boundary_condition_data(self._field_data)
            state = SolitaryState()
            with self.assertRaises(BoundaryConditionDeterminationError):
                ## with pycompadre this will segfault and error out if 
                # the missing fields error isnt thrown
                model._setup_state(state, build_mesh=False)

        def test_activate_coupling(self):
            model = self.init_model()
            model.add_boundary_condition_data(self._field_data)
            model.set_displacement_field_names("Ux", "Uy")
            model.activate_thermal_coupling()

        def test_get_boundary_condition_function(self):
            model = self.init_model(x_disp_name="Ux", y_disp_name="Uy")
            for bc_dc in self.boundary_condition_data_sets:
                model.add_boundary_condition_data(bc_dc)
                for state in bc_dc.states.values():
                    model._setup_state(state, build_mesh=False)
                    disp_func = model._get_loading_boundary_condition_displacement_function(state, 
                                                                                            {})            
                    self.assertTrue(TIME_KEY in disp_func.field_names)
                model.reset_boundary_condition_data()

        def test_check_input_bcs(self):
            model = self.init_model(x_disp_name="Ux", y_disp_name="Uy")
            for bc_dc in self.boundary_condition_data_sets:
                model.add_boundary_condition_data(bc_dc)
                for state in bc_dc.states.values():
                    model._setup_state(state, build_mesh=False)
                    sm_region= model.input_file.solid_mechanics_region
                    prescribed_disp_type = SolidMechanicsPrescribedDisplacement.type
                    self.assertTrue(len(sm_region.get_subblocks_by_type(prescribed_disp_type)), 4)
                    fixed_disp_type = SolidMechanicsFixedDisplacement.type
                    self.assertTrue(len(sm_region.get_subblocks_by_type(fixed_disp_type)), 1)
                    fixed_disp = sm_region.get_subblock_by_type(fixed_disp_type)
                    self.assertEqual(fixed_disp.get_line_value("component"), "z")
                    self.assertEqual(fixed_disp.get_line_value("node set"), "back_node_set")
                    bc_inputs = zip(model._loading_bc_node_sets, model._loading_bc_directions, 
                                    model._loading_bc_direction_keys, model._loading_bc_read_variables)
                    for node_set, direction, direction_key, read_var in bc_inputs:
                        bc_name = node_set+' '+direction
                        bc = sm_region.subblocks[bc_name]
                        self.assertIn(direction_key, bc.lines)
                        self.assertEqual(bc.get_line_value("read variable"), read_var)
                    
                model.reset_boundary_condition_data()


from matcal.sierra.tests.sierra_sm_models_for_tests import VFMUniaxialTensionHexModelForTests
class TestVFMUniaxialTensionHexModel(VFMUniaxialTensionHexModelForTests, 
                                     TestVFMUniaxialTensionModelCommon.VFMCommonTests):

    def test_staggered_coupling_input(self):
        model = self.init_model()
        with self.assertRaises(AttributeError):
            model.activate_thermal_coupling(1)

    def test_use_iterative_coupling(self):
        model = self.init_model()
        with self.assertRaises(AttributeError):
            model.use_iterative_coupling()

    def test_input_has_output(self):
            model = self.init_model(x_disp_name="Ux", y_disp_name="Uy")
            for bc_dc in self.boundary_condition_data_sets:
                model.add_boundary_condition_data(bc_dc)
                for state in bc_dc.states.values():
                    model._setup_state(state, build_mesh=False)
                    exo_output= model.input_file.exodus_output
                    self.assertTrue(exo_output.has_element_output("first_pk_stress_vol_avg", 
                                                                  "first_pk_stress"))
                    self.assertTrue(exo_output.has_element_output("centroid"))
                    self.assertTrue(exo_output.has_element_output("volume"))
                   
                model.reset_boundary_condition_data()

    def _make_simple_mesh_with_info(self):
        n_time = 3
        n_loc = 4
        time = np.linspace(0, 1, n_time)
        T = np.random.uniform(0,1,[n_time, n_loc])
        
        ref_ff_data = OrderedDict({'T':T, "first_pk_stress":T, "centroid":T,
                       "volume":T,
                       "first_pk_stress_xx":T,
                       'time':time, 'x':np.array([0, 1, 1, 0]), 'y':np.array([0, 0, 1, 1]),
                       'con':[[0, 1, 2, 3]]})
        ref_ff_data = convert_dictionary_to_field_data(ref_ff_data, ['x', 'y'], 'con')
        return ref_ff_data
    
    def test_vfm_field_series_data(self):
        data = self._make_simple_mesh_with_info()
        export_full_field_data_to_json("data.json", data)
        res = _vfm_field_series_data("data.json")
        sorted_fields = sorted(res.field_names)
        goal = sorted(["first_pk_stress", "centroid", "volume", "first_pk_stress_xx", "time"])
        self.assertEqual(goal, sorted_fields)


from matcal.sierra.tests.sierra_sm_models_for_tests import VFMUniaxialTensionConnectedHexModelForTests
class TestVFMUniaxialTensionConnectedHexModel(VFMUniaxialTensionConnectedHexModelForTests, 
                                TestVFMUniaxialTensionModelCommon.VFMCommonTests):
        def test_input_has_output(self):
            model = self.init_model(x_disp_name="Ux", y_disp_name="Uy")
            for bc_dc in self.boundary_condition_data_sets:
                model.add_boundary_condition_data(bc_dc)
                for state in bc_dc.states.values():
                    model._setup_state(state, build_mesh=False)
                    exo_output= model.input_file.exodus_output
                    self.assertTrue(exo_output.has_element_output("first_pk_stress_vol_avg", 
                                                                  "first_pk_stress"))
                    self.assertTrue(exo_output.has_element_output("centroid"))
                    self.assertTrue(exo_output.has_element_output("double_volume", "volume"))
                    
                model.reset_boundary_condition_data()


from matcal.sierra.tests.sierra_sm_models_for_tests import UserDefinedSierraModelForTests
from matcal.sierra.tests.utilities import write_empty_file

class UserDefinedSierraModelTests(ModelTestBase.CommonTests, 
    UserDefinedSierraModelForTests):

    def setUp(self):
        super().setUp(__file__)
        
    def test_basic_init(self):
        model = self.init_model()
        self.assertEqual(model._input_filename, os.path.abspath(self._input_file))
        self.assertEqual(model._mesh_filename, os.path.abspath(self._mesh_file))
        self.assertEqual(model.executable, "adagio")

    def test_extra_files_needed_init(self):
        apr_files = ["fake_apr.inc", "fake_apr2.inc", "test_dir"]
        for f in apr_files:
            write_empty_file(f)
        write_empty_file(self._input_file)
        write_empty_file(self._mesh_file)
                
        model = self._model_class('aria', self._input_file, self._mesh_file, *apr_files)
        for apr_file in apr_files:
            self.assertIn(os.path.abspath(apr_file), model._additional_sources_to_copy)
        model._setup_state(SolitaryState(), build_mesh=False)

    def test_read_full_field_data(self):
        model = self.init_model()
        model.read_full_field_data("test.e")
        self.assertTrue(model._results_information.results_reader_object == FieldSeriesData)
        self.assertEqual(model._results_information.results_filename, "test.e")
