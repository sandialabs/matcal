from matcal.core.constants import DISPLACEMENT_KEY, TEMPERATURE_KEY, TIME_KEY
from matcal.core.data import (DataCollection, convert_dictionary_to_data)
from matcal.core.tests.unit.comment_test_helpers import (
    assert_data_set_index_comment,
    assert_data_set_name_comment,
    assert_selection_reason_comment,
    assert_source_collection_comment,
    assert_source_fields_comment,
)

from matcal.core.tests.MatcalUnitTest import MatcalUnitTest

from matcal.full_field.data import convert_dictionary_to_field_data

from matcal.sierra.input_file_writer import (
    AnalyticSierraFunction, 
    _Coupling, SolidMechanicsUserOutput, SolidMechanicsUserVariable, 
    SierraFileBase, SierraFileThreeDimensional,
)

from matcal.sierra.input_file_writer.boundary_conditions import (
    SolidMechanicsPrescribedDisplacement, 
    SolidMechanicsPrescribedTemperature,  
)
from matcal.sierra.input_file_writer.coupling import (
    _Failure, 
)
from matcal.sierra.input_file_writer.outputs import (
    SolidMechanicsResultsOutput
)
from matcal.sierra.input_file_writer.regions_models import (
    FiniteElementModel, SolidMechanicsImplicitDynamics, 
)
from matcal.sierra.input_file_writer.sections import(
    _SectionNames
)
from matcal.sierra.material import Material


class TestSierraInputFile(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)

    def _make_input_deck(self):
        mat_filename = "matfile.inc"
        with open(mat_filename, "w") as f:
            f.write("material...")
        mat = Material("test_mat", mat_filename, "j2_plasticity")
        return SierraFileBase(mat, ["block_to_kill"])

    def test_get_solid_mechanics_finite_element_model_input_block(self):
        ifile = self._make_input_deck()
        sm_fe_model_block = ifile.solid_mechanics_finite_element_model
        self.assertIsInstance(sm_fe_model_block, FiniteElementModel)
        self.assertEqual(sm_fe_model_block, ifile._sm_finite_element_model)

    def test_get_section_subblock(self):
        ifile = self._make_input_deck()
        ifile._add_solid_mechanics_finite_element_parameters("test_mat", 
                                                             "j2_plasticity", 
                                                             "block1")
        section_block = ifile._get_section_subblock()
        self.assertEqual(section_block.name, _SectionNames.total_lagrange)
        ifile._sm_finite_element_model.set_element_section("not valid")
        section_block = ifile._get_section_subblock()

        self.assertEqual(section_block, None)
        ifile._use_under_integrated_element()
        section_block = ifile._get_section_subblock()
        self.assertEqual(section_block.name, _SectionNames.uniform_gradient)
        ifile._use_total_lagrange_element()
        section_block = ifile._get_section_subblock()
        self.assertEqual(section_block.name, _SectionNames.total_lagrange)
        ifile._sm_finite_element_model.set_element_section(_SectionNames.composite_tet)
        section_block = ifile._get_section_subblock()
        self.assertEqual(section_block.name, _SectionNames.composite_tet)

        
    def test_use_under_integrated_element(self):
        ifile = self._make_input_deck()
        ifile._add_solid_mechanics_finite_element_parameters("test_mat", 
                                                             "j2_plasticity", 
                                                             "block1")
        ifile._use_under_integrated_element()
        section_block = ifile._get_section_subblock()
        self.assertEqual(section_block.name, _SectionNames.uniform_gradient)

    def test_use_total_lagrange_element(self):
        ifile = self._make_input_deck()
        ifile._add_solid_mechanics_finite_element_parameters("test_mat", 
                                                             "j2_plasticity", 
                                                             "block1")
        ifile._use_under_integrated_element()
        ifile._use_total_lagrange_element()
        section_block = ifile._get_section_subblock()
        self.assertEqual(section_block.name, _SectionNames.total_lagrange)

    def test_element_type(self):
        ifile = self._make_input_deck()
        ifile._add_solid_mechanics_finite_element_parameters("test_mat", 
                                                             "j2_plasticity", 
                                                             "block1")
        ifile._use_under_integrated_element()
        element_type = ifile.element_type
        self.assertEqual(element_type, _SectionNames.uniform_gradient)

        ifile._use_total_lagrange_element()
        element_type = ifile.element_type
        self.assertEqual(element_type, _SectionNames.total_lagrange)

    def test_add_sm_fe_model_parameters(self):
        ifile = self._make_input_deck()
        ifile._add_solid_mechanics_finite_element_parameters("test_mat", 
                                                            "j2_plasticity", 
                                                            "block1", "block2")
        fe_model_block = ifile.solid_mechanics_finite_element_model
        fe_params_block = fe_model_block.get_subblock("block1 block2")
        self.assertEqual(fe_params_block.get_line_value("model"), 
                         "j2_plasticity")
        self.assertEqual(fe_params_block.get_line_value("material"), 
                         "test_mat")
        
    def test_get_input_string(self):
        ifile = self._make_input_deck()
        ifile._set_local_mesh_filename("test.g")
        input_str = ifile.get_input_string()
        self.assertIsInstance(input_str, str)
    
    def test_activate_exodus_output_interval_adjust(self):
        ifile = self._make_input_deck()
        self.assertFalse(ifile.exodus_output)
        ifile._activate_exodus_output()
        self.assertTrue(ifile.exodus_output)
        exo_output = ifile._exodus_output
        ifile._activate_exodus_output(10)
        self.assertEqual(exo_output.get_line_value("at step", -1), 10)

    def test_add_element_output_variable_invalid(self):
        ifile = self._make_input_deck()
        with self.assertRaises(TypeError):
            ifile._add_element_output_variable(1)
        with self.assertRaises(TypeError):
            ifile._add_element_output_variable(TEMPERATURE_KEY,1)    
        with self.assertRaises(TypeError):
            ifile._add_element_output_variable(TEMPERATURE_KEY,1, volume_average=False)    
    
    def test_add_element_out_not_averaged_removes_averaged(self):
        ifile = self._make_input_deck()
        ifile._add_element_output_variable("stress")
        ifile._add_element_output_variable("stress")
        ifile._add_element_output_variable("stress", volume_average=False)
        self.assertTrue(ifile._element_variable_in_mesh_output("stress"))    
        self.assertFalse(ifile._element_variable_in_mesh_output("stress_vol_avg", "stress"))  

    def test_add_element_averaged_removes_not_averaged(self):
        ifile = self._make_input_deck()
        no_output = ifile._element_variable_in_mesh_output("no_output")
        self.assertFalse(no_output)
        ifile._add_element_output_variable("stress", volume_average=False)
        ifile._add_element_output_variable("stress", volume_average=False)
        ifile._add_element_output_variable("stress")
        self.assertTrue(ifile._element_variable_in_mesh_output("stress_vol_avg", "stress"))  
        self.assertFalse(ifile._element_variable_in_mesh_output("stress"))    
        bad_evar_name = ifile._element_variable_in_mesh_output("not_in_mesh_output")
        self.assertFalse(bad_evar_name)

    def test_add_nodal_output_variable_invalid(self):
        ifile = self._make_input_deck()
        with self.assertRaises(TypeError):
            ifile._add_nodal_output_variable(1)

    def test_add_nodal_output_variable(self):
        ifile = self._make_input_deck()
        self.assertEqual(ifile.exodus_output_active, False)
        self.assertFalse(ifile._nodal_variable_in_mesh_output(TEMPERATURE_KEY))
        ifile._add_nodal_output_variable(TEMPERATURE_KEY)
        self.assertEqual(ifile.exodus_output_active, True)
        self.assertTrue(ifile._nodal_variable_in_mesh_output(TEMPERATURE_KEY))    
        ifile._add_nodal_output_variable("velocity")
        ifile._add_nodal_output_variable("velocity")
        self.assertTrue(ifile._nodal_variable_in_mesh_output("velocity")) 

    def test_activate_exodus_output(self):
        ifile = self._make_input_deck()
        self.assertEqual(ifile.exodus_output_active, False)
        ifile._activate_exodus_output()
        self.assertEqual(ifile.exodus_output_active, True)
        self.assertTrue(ifile._element_variable_in_mesh_output("hydrostatic_stress_vol_avg", 
                                                               save_as_name="hydrostatic_stress"))    
        self.assertTrue(ifile._element_variable_in_mesh_output("von_mises_vol_avg", 
                                                               save_as_name="von_mises"))    
        self.assertTrue(ifile._element_variable_in_mesh_output("log_strain_vol_avg", 
                                                               save_as_name="log_strain"))

    def test_clear_default_element_output_field_names(self):
        ifile = self._make_input_deck()
        self.assertTrue(len(ifile._default_element_output) > 0)
        self.assertEqual(ifile._default_element_output, ["hydrostatic_stress", "von_mises", 
                                                         "log_strain"])
        ifile._clear_default_element_output_field_names()

        self.assertTrue(len(ifile._default_element_output) == 0)
        self.assertEqual(ifile._default_element_output, [])
        
    def test_activate_adiabatic_heating(self):
        ifile = self._make_input_deck()
        self.assertNotIn(TEMPERATURE_KEY, ifile._default_element_output)
        self.assertNotIn(TEMPERATURE_KEY, ifile._default_nodal_output)
        self.assertIsNone(ifile.coupling)
        ifile._activate_adiabatic_heating()
        self.assertIn(TEMPERATURE_KEY, ifile._default_element_output)
        self.assertNotIn(TEMPERATURE_KEY, ifile._default_nodal_output)
        self.assertEqual(ifile.coupling, _Coupling.adiabatic)
        ifile._activate_exodus_output()
        self.assertTrue(ifile._element_variable_in_mesh_output(TEMPERATURE_KEY+"_vol_avg", 
                                                               TEMPERATURE_KEY))
                
    def test_set_state_prescribed_temperature_from_boundary_data(self):
        ifile = self._make_input_deck()
        bc_data = convert_dictionary_to_data({"time":[0,1], TEMPERATURE_KEY:[298,500]})
        dc = DataCollection("test", bc_data)
        ifile._set_state_prescribed_temperature_from_boundary_data(dc, 
                                                                   bc_data.state,
                                                                   TEMPERATURE_KEY)
        self.assertIn(SierraFileBase._temperature_bc_function_name, ifile.subblocks)
        self.assertIn("include all blocks prescribed_temperature", ifile.solid_mechanics_region.subblocks )
        self.assertIn(TEMPERATURE_KEY, ifile._default_nodal_output)
        ifile._activate_exodus_output()
        self.assertTrue(ifile._nodal_variable_in_mesh_output(TEMPERATURE_KEY))

    def test_prescribed_temperature_boundary_condition_property(self):
        ifile = self._make_input_deck()
        self.assertIsNone(ifile.prescribed_temperature_boundary_condition)
        bc_data = convert_dictionary_to_data({"time":[0,1], TEMPERATURE_KEY:[298,500]})
        dc = DataCollection("test", bc_data)

        ifile._set_state_prescribed_temperature_from_boundary_data(dc, 
                                                                   bc_data.state,
                                                                   TEMPERATURE_KEY)
        temp_func = ifile.subblocks[SierraFileBase._temperature_bc_function_name]
        self.assertEqual(temp_func, ifile.prescribed_temperature_boundary_condition)

    def test_set_state_prescribed_temperature_from_boundary_data_ff(self):
        ifile = self._make_input_deck()
        ff_temp_data = {"time":[0,1], TEMPERATURE_KEY:[[298, 298],
                                                    [500, 500]]}
        ff_temp_data['X'] = [0, 1]
        ff_temp_data['Y'] = [0, 1]
        bc_data = convert_dictionary_to_field_data(ff_temp_data, ["X", "Y"])

        ff_temp_data2 = {"time":[0,1], TEMPERATURE_KEY:[[297, 297],
                                                    [510, 501]]}
        ff_temp_data2['X'] = [0, 1]
        ff_temp_data2['Y'] = [0, 1]
        bc_data2 = convert_dictionary_to_field_data(ff_temp_data2, ["X", "Y"])

        dc = DataCollection("test", bc_data, bc_data2)
        ifile._set_state_prescribed_temperature_from_boundary_data(dc, 
                                                                   bc_data.state,
                                                                   TEMPERATURE_KEY)
        self.assertIn("include all blocks read temperature from mesh",
                       ifile.solid_mechanics_region.subblocks )
        self.assertIn(TEMPERATURE_KEY, ifile._default_nodal_output)
        ifile._activate_exodus_output()
        self.assertTrue(ifile._nodal_variable_in_mesh_output(TEMPERATURE_KEY))
    
    def test_add_temperature_output(self):
        ifile = self._make_input_deck()
        ifile._activate_exodus_output()
        self.assertFalse(ifile._element_variable_in_mesh_output(TEMPERATURE_KEY+"_vol_avg",
                                                                TEMPERATURE_KEY))
        self.assertFalse(ifile._nodal_variable_in_mesh_output(TEMPERATURE_KEY))
        ifile._add_temperature_output()
        self.assertTrue(ifile._element_variable_in_mesh_output(TEMPERATURE_KEY+"_vol_avg", 
                                                               TEMPERATURE_KEY))
        ifile._add_temperature_output(nodal=True)
        self.assertTrue(ifile._nodal_variable_in_mesh_output(TEMPERATURE_KEY))
        
    def test_set_initial_temp_from_params(self):
        ifile = self._make_input_deck()
        ifile._activate_adiabatic_heating()
        with self.assertRaises(RuntimeError):
            ifile._set_initial_temperature_from_parameters({})
        ifile._set_initial_temperature_from_parameters({TEMPERATURE_KEY:100})
        initial_temp_block = ifile.solid_mechanics_region.get_subblock("initial temperature")
        self.assertEqual(initial_temp_block.get_line_value("magnitude"), 100)

    def test_add_prescribed_loading_boundary_condition_with_displacement_function(self):
        ifile = self._make_input_deck()
        disp_func = convert_dictionary_to_data({TIME_KEY:[0.1, 10], 
                                                DISPLACEMENT_KEY:[0,1]})
        ifile._add_prescribed_loading_boundary_condition_with_displacement_function(disp_func, 
            ["top_nodeset", "side_nodeset"], ["x", "y"], ["component", "component"], 1.0)
        self.assertIn(f"top_nodeset x {SierraFileBase._load_bc_function_name}",
                       ifile.solid_mechanics_region.subblocks)
        self.assertIn(f"side_nodeset y {SierraFileBase._load_bc_function_name}", 
                      ifile.solid_mechanics_region.subblocks)
        self.assertEqual(ifile.solid_mechanics_procedure._start_time, 0.1)
        self.assertEqual(ifile.solid_mechanics_procedure._termination_time, 10)

    def test_add_prescribed_displacement_boundary_condition_with_read_var(self):
        ifile = self._make_input_deck()
        ifile._add_prescribed_displacement_boundary_condition(None, 
            ["top_nodeset", "side_nodeset"], ["x", "y"], ["component", "component"], 
            ["U", "V"])
        blocks = ifile.solid_mechanics_region.subblocks
        self.assertIn(f"top_nodeset x", blocks)
        self.assertEqual(blocks["top_nodeset x"].get_line_value("read variable"), "U")
        self.assertIn(f"side_nodeset y", blocks)
        self.assertEqual(blocks["side_nodeset y"].get_line_value("read variable"), "V")

    def test_write_input_file(self):
        ifile = self._make_input_deck()
        ifile._set_local_mesh_filename("test.g")
        ifile.write_input_to_file("my_filename.txt")
        self.assert_file_exists("my_filename.txt")

    def test_user_end_time(self):
        ifile = self._make_input_deck()
        disp_func = convert_dictionary_to_data({TIME_KEY:[0.1, 10], 
                                                DISPLACEMENT_KEY:[0,1]})
        ifile._add_prescribed_loading_boundary_condition_with_displacement_function(disp_func, 
            ["top_nodeset", "side_nodeset"],   ["x", "y"], ["component", "component"], 1.0)
        self.assertFalse(ifile._end_time_user_supplied)       
        ifile._set_end_time(5)
        self.assertTrue(ifile._end_time_user_supplied)       
        self.assertEqual(ifile.solid_mechanics_procedure._start_time, 0.1)
        self.assertEqual(ifile.solid_mechanics_procedure._termination_time, 5)

    def test_user_start_time(self):
        ifile = self._make_input_deck()
        disp_func = convert_dictionary_to_data({TIME_KEY:[0.1, 10], 
                                                DISPLACEMENT_KEY:[0,1]})
        ifile._add_prescribed_loading_boundary_condition_with_displacement_function(disp_func, 
            ["top_nodeset", "side_nodeset"], ["x", "y"], ["component", "component"], 1.0)
        self.assertFalse(ifile._start_time_user_supplied)       
        ifile._set_start_time(5)
        self.assertTrue(ifile._start_time_user_supplied)       
        self.assertEqual(ifile.solid_mechanics_procedure._start_time, 5)
        self.assertEqual(ifile.solid_mechanics_procedure._termination_time, 10)

    def test_set_number_of_time_steps(self):
        ifile = self._make_input_deck()
        disp_func = convert_dictionary_to_data({TIME_KEY:[0.1, 10], 
                                                DISPLACEMENT_KEY:[0,1]})
        ifile._add_prescribed_loading_boundary_condition_with_displacement_function(disp_func, 
            ["top_nodeset", "side_nodeset"], ["x", "y"], ["component", "component"], 1.0)
        self.assertEqual(ifile.solid_mechanics_procedure._time_steps, 300)
        ifile._set_number_of_time_steps(1000)
        self.assertEqual(ifile.solid_mechanics_procedure._time_steps, 1000)

    def test_set_fixed_boundary_conditions(self):
        nsets = ["fixed_x_ns", "fixed_y_ns"]
        dirs = ["x", "y"]
        ifile = self._make_input_deck()
        ifile._set_fixed_boundary_conditions(nsets, dirs)
        self.assertIn("fixed_x_ns x", ifile.solid_mechanics_region.subblocks)
        self.assertIn("fixed_y_ns y", ifile.solid_mechanics_region.subblocks)
        
    def test_use_boundary_condition_scale_factor(self):
        ifile = self._make_input_deck()
        disp_func = convert_dictionary_to_data({TIME_KEY:[0.1, 10], 
                                                DISPLACEMENT_KEY:[0,1]})
        ifile._add_prescribed_loading_boundary_condition_with_displacement_function(disp_func, 
                                                              ["top_nodeset"], 
                                                              ["x"], ["component"],  1.25)
        precribed_disp_func = ifile.subblocks[SierraFileBase._load_bc_function_name]
        self.assertEqual(precribed_disp_func.get_line_value("x scale"), 1.25)
        self.assertEqual(precribed_disp_func.get_line_value("y scale"), 1.25)

    def test_prescribed_loading_boundary_condition_property(self):
        ifile = self._make_input_deck()
        self.assertIsNone(ifile.prescribed_loading_boundary_condition)
        
        disp_func = convert_dictionary_to_data({TIME_KEY:[0.1, 10], 
                                                DISPLACEMENT_KEY:[0,1]})
        ifile._add_prescribed_loading_boundary_condition_with_displacement_function(disp_func, 
                                                              ["top_nodeset"], 
                                                              ["x"], ["component"], 1.25)
        precribed_disp_func = ifile.subblocks[SierraFileBase._load_bc_function_name]
        self.assertEqual(precribed_disp_func, ifile.prescribed_loading_boundary_condition)
        
    def test_reset_state_temperature_conditions(self):
        ifile = self._make_input_deck()
        bc_data = convert_dictionary_to_data({"time":[0,1], TEMPERATURE_KEY:[298,500]})
        dc = DataCollection("test", bc_data)
        ifile._set_state_prescribed_temperature_from_boundary_data(dc, 
                                                                   bc_data.state,
                                                                   TEMPERATURE_KEY)
        self.assertIn(SierraFileBase._temperature_bc_function_name, ifile.subblocks)
        self.assertIn("include all blocks prescribed_temperature", ifile.solid_mechanics_region.subblocks )
        block_type = SolidMechanicsPrescribedTemperature.type
        self.assertIsNotNone(ifile.solid_mechanics_region.get_subblock_by_type(block_type))
        ifile._reset_state_temperature_conditions()
        self.assertNotIn(SierraFileBase._temperature_bc_function_name, ifile.subblocks)
        self.assertIsNone(ifile.solid_mechanics_region.get_subblock_by_type(block_type))

    def test_reset_state_displacement_conditions(self):
        ifile = self._make_input_deck()
        disp_func = convert_dictionary_to_data({TIME_KEY:[0.1, 10], 
                                                DISPLACEMENT_KEY:[0,1]})
        ifile._add_prescribed_loading_boundary_condition_with_displacement_function(disp_func, 
            ["top_nodeset", "side_nodeset"], ["x", "y"], ["component", "component"], 1.0)
        self.assertIn(f"top_nodeset x {SierraFileBase._load_bc_function_name}",
                       ifile.solid_mechanics_region.subblocks)
        self.assertIn(f"side_nodeset y {SierraFileBase._load_bc_function_name}", 
                      ifile.solid_mechanics_region.subblocks)
        block_type = SolidMechanicsPrescribedDisplacement.type
        self.assertIsNotNone(ifile.solid_mechanics_region.get_subblock_by_type(block_type))
        ifile._reset_state_displacement_conditions()
        self.assertIsNone(ifile.solid_mechanics_region.get_subblock_by_type(block_type))
        self.assertNotIn(SierraFileBase._load_bc_function_name, ifile.subblocks)

    def test_add_heartbeat_global_variable(self):
        ifile = self._make_input_deck()
        ifile._add_heartbeat_global_variable("disp")
        has_time = ifile._heartbeat_output.has_global_output("time")
        self.assertTrue(has_time)
        has_disp = ifile._heartbeat_output.has_global_output("disp")
        self.assertTrue(has_disp)
        ifile._add_heartbeat_global_variable("disp")
        
    def test_activate_element_death(self):
        ifile = self._make_input_deck()
        self.assertIsNone(ifile.failure)
        ifile._activate_element_death()
        death_block = ifile._death
        self.assertEqual(death_block.get_line_value("block"), "block_to_kill")
        self.assertEqual(death_block.get_line_value("criterion", 2), "damage")
        self.assertEqual(death_block.get_line_value("criterion", 4), 0.15)
        self.assertEqual(ifile.failure, _Failure.local_failure)

    def test_cg_property(self):
        ifile = self._make_input_deck()
        cg = ifile.cg
        self.assertEqual(ifile._cg, cg)
    
    def test_adaptive_time_stepping_property(self):
        ifile = self._make_input_deck()
        adaptive_time_stepping = ifile.adative_time_stepping
        self.assertEqual(ifile._adaptive_time_stepping, adaptive_time_stepping)

    def test_full_tangent_preconditioner_property(self):
        ifile = self._make_input_deck()
        full_tangent_preconditioner = ifile.full_tangent_preconditioner
        self.assertEqual(ifile._full_tangent_preconditioner, full_tangent_preconditioner)

    def test_death_property(self):
        ifile = self._make_input_deck()
        ifile._activate_element_death()
        death = ifile.death
        self.assertEqual(ifile._death, death)

    def test_solid_mechanics_element_section_property(self):
        ifile = self._make_input_deck()
        ifile._add_solid_mechanics_finite_element_parameters("test_mat", 
                                                             "j2_plasticity", 
                                                             "block1")
        solid_mechanics_element_section = ifile.solid_mechanics_element_section
        self.assertEqual(ifile._get_section_subblock(), solid_mechanics_element_section)

    def test_exodus_output_property(self):
        ifile = self._make_input_deck()
        exodus_output = ifile.exodus_output
        self.assertEqual(ifile._exodus_output, exodus_output)

    def test_heartbeat_output_property(self):
        ifile = self._make_input_deck()
        heartbeat_output = ifile.heartbeat_output
        self.assertEqual(ifile._heartbeat_output, heartbeat_output)

    def test_element_type_property(self):
        ifile = self._make_input_deck()
        ifile._add_solid_mechanics_finite_element_parameters("mat1", "j2_plasticity", 
                                                             "block1", "block2")
        element_type = ifile.element_type
        self.assertEqual(ifile._sm_finite_element_model.get_element_section()
            , element_type)
        self.assertEqual(element_type, _SectionNames.total_lagrange)

    def test_heartbeat_output_property(self):
        ifile = self._make_input_deck()
        heartbeat_output = ifile.heartbeat_output
        self.assertEqual(ifile._heartbeat_output, heartbeat_output)

    def test_solution_termination_property(self):
        ifile = self._make_input_deck()
        sol_term = ifile.solution_termination
        self.assertEqual(ifile._solution_termination, sol_term)

    def test_set_initial_temp_property(self):
        ifile = self._make_input_deck()
        ifile._activate_adiabatic_heating()
        ifile._set_initial_temperature_from_parameters({TEMPERATURE_KEY:100})
        initial_temp_block = ifile.solid_mechanics_region.get_subblock("initial temperature")
        self.assertEqual(initial_temp_block, ifile.initial_temperature)

    def test_reset_heartbeat_output(self):
        ifile = self._make_input_deck()
        heartbeat1 = ifile.heartbeat_output
        ifile._reset_heartbeat_output()
        self.assertNotEqual(ifile.heartbeat_output, heartbeat1)
        self.assertIsNotNone(ifile.heartbeat_output)
    
    def test_reset_state_boundary_conditions_and_output(self):
        ifile = self._make_input_deck()
        bc_data = convert_dictionary_to_data({"time":[0,1], TEMPERATURE_KEY:[298,500]})
        dc = DataCollection("test", bc_data)
        ifile._set_state_prescribed_temperature_from_boundary_data(dc, 
                                                                   bc_data.state,
                                                                   TEMPERATURE_KEY)
        disp_func = convert_dictionary_to_data({TIME_KEY:[0.1, 10], 
                                                DISPLACEMENT_KEY:[0,1]})
        ifile._add_prescribed_loading_boundary_condition_with_displacement_function(disp_func, 
            ["top_nodeset", "side_nodeset"], ["x", "y"], ["component", "component"], 1.0)
        ifile._add_heartbeat_global_variable("disp")

        ifile._reset_state_boundary_conditions_and_output()
        sm_region = ifile.solid_mechanics_region
        self.assertFalse(sm_region.get_subblock_by_type(SolidMechanicsResultsOutput.type))
        self.assertFalse(sm_region.get_subblock_by_type(SolidMechanicsUserVariable.type))
        self.assertIn(ifile._vol_average_user_output, sm_region.subblocks.values())
        self.assertEqual(len(sm_region.get_subblocks_by_type(SolidMechanicsUserOutput.type)), 1)

    def test_set_cg_convergence_tolerance_only_target_relative(self):
        ifile = self._make_input_deck()
        ifile._set_cg_convergence_tolerance(1e-8)
        self.assertAlmostEqual(ifile._cg.get_target_relative_residual(), 1e-8)
        self.assertAlmostEqual(ifile._cg.get_target_residual(), 1e-6)
        self.assertAlmostEqual(ifile._cg.get_acceptable_relative_residual(), 1e-7)

    def test_set_cg_convergence_tolerance_target_relative_and_target(self):
        ifile = self._make_input_deck()
        ifile._set_cg_convergence_tolerance(1e-8, target_residual=1e-7)
        self.assertAlmostEqual(ifile._cg.get_target_relative_residual(), 1e-8)
        self.assertAlmostEqual(ifile._cg.get_target_residual(), 1e-7)
        self.assertAlmostEqual(ifile._cg.get_acceptable_relative_residual(), 1e-7)
        self.assertIsNone(ifile._cg.get_acceptable_residual())

    def test_set_cg_convergence_tolerance_target_relative_and_acceptable_relative(self):
        ifile = self._make_input_deck()
        ifile._set_cg_convergence_tolerance(1e-8, acceptable_relative_residual=1e-6)
        self.assertAlmostEqual(ifile._cg.get_target_relative_residual(), 1e-8)
        self.assertAlmostEqual(ifile._cg.get_target_residual(), 1e-6)
        self.assertAlmostEqual(ifile._cg.get_acceptable_relative_residual(), 1e-6)
        self.assertIsNone(ifile._cg.get_acceptable_residual())

    def test_set_cg_convergence_tolerance_target_relative_and_acceptable(self):
        ifile = self._make_input_deck()
        ifile._set_cg_convergence_tolerance(1e-8, acceptable_residual=1e-5)
        self.assertAlmostEqual(ifile._cg.get_target_relative_residual(), 1e-8)
        self.assertAlmostEqual(ifile._cg.get_target_residual(), 1e-6)
        self.assertAlmostEqual(ifile._cg.get_acceptable_relative_residual(), 1e-7)
        self.assertAlmostEqual(ifile._cg.get_acceptable_residual(), 1e-5)

    def test_set_time_parameters_to_loading_function(self):
        ifile = self._make_input_deck()
        data = convert_dictionary_to_data({"time":[1, 4], "displacement":[0,4]})
        ifile._set_time_parameters_to_loading_function(data, 2)
        self.assertAlmostEqual(ifile.solid_mechanics_procedure._start_time, 2)
        self.assertAlmostEqual(ifile.solid_mechanics_procedure._termination_time, 8)
        self.assertAlmostEqual(ifile.solid_mechanics_procedure._time_step, 6/300)
        self.assertAlmostEqual(ifile.solid_mechanics_procedure._small_time_step, 6/300*1e-3)

    def _get_function_block_string(self, ifile, function_name):
        return ifile.subblocks[function_name].get_string()
    
    def test_add_comment_to_function_block_single_line(self):
        ifile = self._make_input_deck()
        disp_func = convert_dictionary_to_data({
            TIME_KEY: [0.0, 1.0],
            DISPLACEMENT_KEY: [0.0, 1.0],
        })
        ifile._add_prescribed_loading_boundary_condition_with_displacement_function(
            disp_func,
            ["top_nodeset"],
            ["x"],
            ["component"],
            1.0,
        )

        ifile._add_comment_to_function_block(
            SierraFileBase._load_bc_function_name,
            f'Using tabulated "{TIME_KEY}" and "{DISPLACEMENT_KEY}" fields from source data set.',
        )

        func_block = ifile.subblocks[SierraFileBase._load_bc_function_name]
        line_names = list(func_block._lines.keys())
        self.assertTrue(line_names[0].startswith("comment_prescribed_displacement_0"))

        func_str = self._get_function_block_string(
            ifile,
            SierraFileBase._load_bc_function_name,
        )
        assert_source_fields_comment(self, func_str, DISPLACEMENT_KEY, uses_time=True)

    def test_add_comment_to_function_block_multi_line(self):
        ifile = self._make_input_deck()
        disp_func = convert_dictionary_to_data({
            TIME_KEY: [0.0, 1.0],
            DISPLACEMENT_KEY: [0.0, 1.0],
        })
        ifile._add_prescribed_loading_boundary_condition_with_displacement_function(
            disp_func,
            ["top_nodeset"],
            ["x"],
            ["component"],
            1.0,
        )

        comment = "\n".join([
            f'Using tabulated "{TIME_KEY}" and "{DISPLACEMENT_KEY}" fields from source data set.',
            'Source data collection: "boundary conditions".',
            'Selected data set index: 0.',
        ])
        ifile._add_comment_to_function_block(
            SierraFileBase._load_bc_function_name,
            comment,
        )

        func_block = ifile.subblocks[SierraFileBase._load_bc_function_name]
        line_names = list(func_block._lines.keys())
        self.assertTrue(line_names[0].startswith("comment_prescribed_displacement_0"))
        self.assertTrue(line_names[1].startswith("comment_prescribed_displacement_1"))
        self.assertTrue(line_names[2].startswith("comment_prescribed_displacement_2"))

        func_str = self._get_function_block_string(
            ifile,
            SierraFileBase._load_bc_function_name,
        )
        assert_source_fields_comment(self, func_str, DISPLACEMENT_KEY, uses_time=True)
        assert_source_collection_comment(self, func_str, "boundary conditions")
        assert_data_set_index_comment(self, func_str, 0)

    def test_add_prescribed_loading_boundary_condition_with_displacement_function_adds_comment(self):
        ifile = self._make_input_deck()
        disp_func = convert_dictionary_to_data({
            TIME_KEY: [0.0, 1.0],
            DISPLACEMENT_KEY: [0.0, 1.0],
        })
        comment = "\n".join([
            f'Using tabulated "{TIME_KEY}" and "{DISPLACEMENT_KEY}" fields from source data set.',
            'Source data collection: "boundary conditions".',
        ])

        ifile._add_prescribed_loading_boundary_condition_with_displacement_function(
            disp_func,
            ["top_nodeset"],
            ["x"],
            ["component"],
            1.0,
            bc_comment=comment,
        )

        func_str = self._get_function_block_string(
            ifile,
            SierraFileBase._load_bc_function_name,
        )
        assert_source_fields_comment(self, func_str, DISPLACEMENT_KEY, uses_time=True)
        assert_source_collection_comment(self, func_str, "boundary conditions")

    def test_set_state_prescribed_temperature_from_boundary_data_adds_comment_to_temperature_function(self):
        ifile = self._make_input_deck()
        bc_data = convert_dictionary_to_data({
            TIME_KEY: [0.0, 1.0],
            TEMPERATURE_KEY: [298.0, 500.0],
        })
        bc_data.set_name("thermal_history")
        dc = DataCollection("boundary conditions", bc_data)

        ifile._set_state_prescribed_temperature_from_boundary_data(
            dc,
            bc_data.state,
            TEMPERATURE_KEY,
        )

        func_str = self._get_function_block_string(
            ifile,
            SierraFileBase._temperature_bc_function_name,
        )
        assert_source_fields_comment(self, func_str, TEMPERATURE_KEY, uses_time=True)
        assert_source_collection_comment(self, func_str, "boundary conditions")
        assert_data_set_index_comment(self, func_str, 0)
        assert_data_set_name_comment(self, func_str, "thermal_history")
        assert_selection_reason_comment(self, func_str, TEMPERATURE_KEY, bc_data.state.name)    


class TestSierraFileThreeDimensional(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)

    def _make_input_deck(self):
        mat_filename = "matfile.inc"
        with open(mat_filename, "w") as f:
            f.write("material...")
        mat = Material("test_mat", mat_filename, "j2_plasticity")
        ifile = SierraFileThreeDimensional(mat, ["block_to_kill"])
        ifile._add_solid_mechanics_finite_element_parameters("test_mat", 
                                                             "j2_plasticity", 
                                                             "block1")
        ifile._set_local_mesh_filename("test.g")

        return ifile

    def test_activate_implicit_dynamics(self):
        ifile = self._make_input_deck()
        self.assertIsNone(ifile._implicit_dynamics)
        ifile._activate_implicit_dynamics()
        self.assertIsInstance(ifile._implicit_dynamics, SolidMechanicsImplicitDynamics)
        self.assertIn(ifile._implicit_dynamics.name, ifile._solid_mechanics_region.subblocks)
    
    def test_use_total_lagrange_element_composite_tet(self):
        ifile = self._make_input_deck()
        ifile._use_total_lagrange_element(composite_tet=True)
        self.assertEqual(ifile.element_type, _SectionNames.composite_tet)

    def test_change_element_type_back(self):
        ifile = self._make_input_deck()
        self.assertEqual(ifile.element_type, _SectionNames.total_lagrange)
        ifile._use_total_lagrange_element(composite_tet=True)
        self.assertEqual(ifile.element_type, _SectionNames.composite_tet)
        ifile._use_total_lagrange_element(composite_tet=False)
        self.assertEqual(ifile.element_type, _SectionNames.total_lagrange)

    def test_add_solution_termination_user_output(self):
        ifile = self._make_input_deck()
        self.assertIsNone(ifile._solution_termination_output)
        self.assertEqual(ifile._solution_termination.get_line_value("global terminate_solution", -3), 
            "terminate_solution")
        self.assertEqual(ifile._solution_termination.get_line_value("global terminate_solution", -1), 0.5)
        self.assertEqual(ifile._solution_termination.get_line_value("global terminate_solution", -2), ">")
        ifile._add_solution_termination_user_output("load", 0.25)
        self.assertIn("load < max_load*(1-0.25)", ifile._solution_termination_output.get_string())         
        
    def test_activate_full_field_results_output_uncoupled(self):
        ifile = self._make_input_deck()
        self.assertIsNone(ifile._full_field_output)
        ifile._add_heartbeat_global_variable("load")
        ifile._add_heartbeat_global_variable("displacement")
        
        ifile._activate_full_field_results_output("results/full_fields_results.e", 
                                                  "block1", "block2")
        self.assertIsInstance(ifile._full_field_output, SolidMechanicsResultsOutput)
        ffo = ifile._full_field_output
        self.assertEqual(ffo.name, "full_field_output")
        self.assertEqual(ffo.get_line_value("exclude"), "block1")
        self.assertEqual(ffo.get_line_value("exclude", -1), "block2")
        self.assertEqual(ffo.get_line_value("output mesh"), "exposed surface")
        self.assertEqual(ffo.get_line_value("include"), "full_field_data_surface")
        self.assertEqual(ffo.get_line_value("database name"), "results/full_fields_results.e")
        self.assertIn("nodal displacement", ffo.lines)
        self.assertIn("global time", ffo.lines)
        self.assertIn("global displacement", ffo.lines)
        self.assertIn("global load", ffo.lines)

    def test_activate_full_field_results_output_adiabatic(self):
        ifile = self._make_input_deck()
        self.assertIsNone(ifile._full_field_output)
        ifile._add_heartbeat_global_variable("load")
        ifile._add_heartbeat_global_variable("displacement")
        ifile._activate_adiabatic_heating()
        ifile._activate_full_field_results_output("results/full_fields_results.e", 
                                                  "block1", "block2")
        self.assertIsInstance(ifile._full_field_output, SolidMechanicsResultsOutput)
        ffo = ifile._full_field_output
        self.assertEqual(ffo.name, "full_field_output")
        self.assertEqual(ffo.get_line_value("exclude"), "block1")
        self.assertEqual(ffo.get_line_value("exclude", -1), "block2")
        self.assertEqual(ffo.get_line_value("output mesh"), "exposed surface")
        self.assertEqual(ffo.get_line_value("include"), "full_field_data_surface")
        self.assertEqual(ffo.get_line_value("database name"), "results/full_fields_results.e")
        self.assertIn("nodal displacement", ffo.lines)
        self.assertIn("element temperature", ffo.lines)
        self.assertIn("global time", ffo.lines)
        self.assertIn("global displacement", ffo.lines)
        self.assertIn("global load", ffo.lines)

    def test_activate_full_field_results_output_coupled(self):
        ifile = self._make_input_deck()
        self.assertIsNone(ifile._full_field_output)
        ifile._add_heartbeat_global_variable("load")
        ifile._add_heartbeat_global_variable("displacement")
        ifile._activate_thermal_coupling(1,1,1,"work_var")
        ifile._activate_full_field_results_output("results/full_fields_results.e", 
                                                  "block1", "block2")
        self.assertIsInstance(ifile._full_field_output, SolidMechanicsResultsOutput)
        ffo = ifile._full_field_output
        self.assertEqual(ffo.name, "full_field_output")
        self.assertEqual(ffo.get_line_value("exclude"), "block1")
        self.assertEqual(ffo.get_line_value("exclude", -1), "block2")
        self.assertEqual(ffo.get_line_value("output mesh"), "exposed surface")
        self.assertEqual(ffo.get_line_value("include"), "full_field_data_surface")
        self.assertEqual(ffo.get_line_value("database name"), "results/full_fields_results.e")
        self.assertIn("nodal displacement", ffo.lines)
        self.assertIn("nodal temperature", ffo.lines)
        self.assertIn("global time", ffo.lines)
        self.assertIn("global displacement", ffo.lines)
        self.assertIn("global load", ffo.lines)

    def test_add_nonlocal_user_output_functions_added(self):
        ifile = self._make_input_deck()
        ifile._activate_element_death()
        ifile._add_nonlocal_user_output("damage", 0.1)

        self.assertEqual(ifile.failure, _Failure.nonlocal_failure)
        for i in range(8):
            self.assertIn(f"apply_nonlocal_damage_increment_{i+1}", ifile.subblocks)
            self.assertIn(f"get_damage_increment_{i+1}", ifile.subblocks)

        apply_nonlocal_func = ifile.subblocks["apply_nonlocal_damage_increment_1"]
        self.assertIsInstance(apply_nonlocal_func, AnalyticSierraFunction)
        self.assertIn("nl_damage_inc", apply_nonlocal_func.lines)
        self.assertIn("d_old", apply_nonlocal_func.lines)
        self.assertIn("(d_old + nl_damage_inc) < 0.15 ? d_old + nl_damage_inc :0.15;", 
            apply_nonlocal_func.get_string())

        get_dam_inc_func = ifile.subblocks["get_damage_increment_1"]
        self.assertIsInstance(get_dam_inc_func, AnalyticSierraFunction)
        self.assertIn("d_cur", get_dam_inc_func.lines)
        self.assertIn("d_old", get_dam_inc_func.lines)
        self.assertIn("d_cur - d_old",get_dam_inc_func.get_string())

    def test_add_nonlocal_user_output_output_added(self):
        ifile = self._make_input_deck()
        ifile._activate_element_death()
        ifile._add_nonlocal_user_output("damage", 0.1)

        damage_inc_found = False
        nonlocal_damage_found = False
        apply_nonlocal_found = False
        
        for block in ifile.solid_mechanics_region.get_subblocks_by_type("user output"):
            if "element damage(1)" in block.lines:
                apply_nonlocal_found = True

            if "nonlocal average" in block.subblocks:
                nonlocal_damage_found = True
            if "element damage_increment(1)" in block.lines:
                damage_inc_found = True
        
        self.assertTrue(damage_inc_found)
        self.assertTrue(nonlocal_damage_found)
        self.assertTrue(apply_nonlocal_found)

    def test_add_nonlocal_user_output_user_variables_added(self):
        ifile = self._make_input_deck()
        ifile._activate_element_death()
        ifile._add_nonlocal_user_output("damage", 0.1) 
        self.assertIn("damage_increment", ifile.solid_mechanics_region.subblocks)
        self.assertIn("nonlocal_damage_increment", ifile.solid_mechanics_region.subblocks)
        damage_inc_var = ifile.solid_mechanics_region.subblocks["damage_increment"]
        self.assertEqual(damage_inc_var.get_line_value("block"), "block_to_kill")
        self.assertEqual(damage_inc_var.get_line_value("initial value"), 0.0)
        self.assertEqual(damage_inc_var.get_line_value("initial value", -1), 0.0)
        self.assertEqual(damage_inc_var.get_line_value("type"), "element")
        self.assertEqual(damage_inc_var.get_line_value("type", -1), 8)
        nonlocal_dam_inc = ifile.solid_mechanics_region.subblocks["nonlocal_damage_increment"]
        self.assertEqual(nonlocal_dam_inc.get_line_value("block"), "block_to_kill")
        self.assertEqual(nonlocal_dam_inc.get_line_value("initial value"), 0.0)
        self.assertEqual(nonlocal_dam_inc.get_line_value("initial value", -1), 0.0)
        self.assertEqual(nonlocal_dam_inc.get_line_value("type"), "element")
        self.assertEqual(nonlocal_dam_inc.get_line_value("type", -1), 8)

    def test_add_nonlocal_user_output_user_variables_added_get_string_twice(self):
        ifile = self._make_input_deck()
        ifile._activate_element_death()
        ifile._add_nonlocal_user_output("damage", 0.1) 
        ifile.get_input_string()
        ifile.get_input_string()
        self.assertIn("damage_increment", ifile.solid_mechanics_region.subblocks)
        self.assertIn("nonlocal_damage_increment", ifile.solid_mechanics_region.subblocks)
        damage_inc_var = ifile.solid_mechanics_region.subblocks["damage_increment"]
        self.assertEqual(damage_inc_var.get_line_value("block"), "block_to_kill")
        self.assertEqual(damage_inc_var.get_line_value("initial value"), 0.0)
        self.assertEqual(damage_inc_var.get_line_value("initial value", -1), 0.0)
        self.assertEqual(damage_inc_var.get_line_value("type"), "element")
        self.assertEqual(damage_inc_var.get_line_value("type", -1), 8)
        nonlocal_dam_inc = ifile.solid_mechanics_region.subblocks["nonlocal_damage_increment"]
        self.assertEqual(nonlocal_dam_inc.get_line_value("block"), "block_to_kill")
        self.assertEqual(nonlocal_dam_inc.get_line_value("initial value"), 0.0)
        self.assertEqual(nonlocal_dam_inc.get_line_value("initial value", -1), 0.0)
        self.assertEqual(nonlocal_dam_inc.get_line_value("type"), "element")
        self.assertEqual(nonlocal_dam_inc.get_line_value("type", -1), 8)

    def test_add_nonlocal_user_output_functions_added_composite_tet(self):
        ifile = self._make_input_deck()
        ifile._activate_element_death()
        ifile._use_total_lagrange_element(composite_tet=True)
        ifile._add_nonlocal_user_output("damage", 0.1)

        self.assertEqual(ifile.failure, _Failure.nonlocal_failure)
        for i in range(4):
            self.assertIn(f"apply_nonlocal_damage_increment_{i+1}", ifile.subblocks)
            self.assertIn(f"get_damage_increment_{i+1}", ifile.subblocks)

        with self.assertRaises(KeyError):
            ifile.subblocks["apply_nonlocal_damage_increment_5"]
            ifile.subblocks["get_damage_increment_5"]       
        damage_inc_var = ifile.solid_mechanics_region.subblocks["damage_increment"]
        self.assertEqual(damage_inc_var.get_line_value("block"), "block_to_kill")
        self.assertEqual(damage_inc_var.get_line_value("initial value"), 0.0)
        self.assertEqual(damage_inc_var.get_line_value("type"), "element")
        self.assertEqual(damage_inc_var.get_line_value("type", -1), 4)
        nonlocal_dam_inc = ifile.solid_mechanics_region.subblocks["nonlocal_damage_increment"]
        self.assertEqual(nonlocal_dam_inc.get_line_value("block"), "block_to_kill")
        self.assertEqual(nonlocal_dam_inc.get_line_value("initial value"), 0.0)
        self.assertEqual(nonlocal_dam_inc.get_line_value("type"), "element")
        self.assertEqual(nonlocal_dam_inc.get_line_value("type", -1), 4)

    def test_add_nonlocal_user_output_functions_added_uniform_gradient(self):
        ifile = self._make_input_deck()
        ifile._activate_element_death()
        ifile._use_under_integrated_element()
        ifile._add_nonlocal_user_output("damage", 0.1)

        self.assertEqual(ifile.failure, _Failure.nonlocal_failure)
        self.assertIn(f"apply_nonlocal_damage_increment_1", ifile.subblocks)
        self.assertIn(f"get_damage_increment_1", ifile.subblocks)

        with self.assertRaises(KeyError):
            ifile.subblocks["apply_nonlocal_damage_increment_2"]
            ifile.subblocks["get_damage_increment_2"]     

        damage_inc_var = ifile.solid_mechanics_region.subblocks["damage_increment"]
        self.assertEqual(ifile._nonlocal_damage_user_variables[0], damage_inc_var)

        self.assertEqual(damage_inc_var.get_line_value("block"), "block_to_kill")
        self.assertEqual(damage_inc_var.get_line_value("initial value"), 0.0)
        self.assertEqual(damage_inc_var.get_line_value("type"), "element")
        self.assertEqual(damage_inc_var.get_line_value("type", -1), 1)
        nonlocal_dam_inc = ifile.solid_mechanics_region.subblocks["nonlocal_damage_increment"]
        self.assertEqual(ifile._nonlocal_damage_user_variables[1], nonlocal_dam_inc)

        self.assertEqual(nonlocal_dam_inc.get_line_value("block"), "block_to_kill")
        self.assertEqual(nonlocal_dam_inc.get_line_value("initial value"), 0.0)
        self.assertEqual(nonlocal_dam_inc.get_line_value("type"), "element")
        self.assertEqual(nonlocal_dam_inc.get_line_value("type", -1), 1)
        self.assertEqual(len(ifile._nonlocal_functions), 2)
        dam_inc_output = ifile.solid_mechanics_region.subblocks["damage_increment_output"]
        self.assertEqual(ifile._damage_increment_user_output, dam_inc_output)
        nonlocal_dam_average = ifile.solid_mechanics_region.subblocks["nonlocal_damage_average"]

        self.assertEqual(ifile._nonlocal_average_output,nonlocal_dam_average)
        nonlocal_dam_inc_output = ifile.solid_mechanics_region.subblocks["nonlocal_damage_increment_output"]
        self.assertEqual(ifile._nonlocal_damage_increment_user_output, nonlocal_dam_inc_output)

    def test_reset_nonlocal_input(self):
        ifile = self._make_input_deck()
        ifile._activate_element_death()
        ifile._add_nonlocal_user_output("damage", 0.1)
        ifile._reset_nonlocal_input()

        self.assertNotIn(f"apply_nonlocal_damage_increment_0", ifile.subblocks)
        self.assertNotIn(f"get_damage_increment_0", ifile.subblocks)
        
        self.assertNotIn("damage_increment", ifile.solid_mechanics_region.subblocks)
        self.assertNotIn("nonlocal_damage_increment", ifile.solid_mechanics_region.subblocks)
        self.assertEqual(ifile._nonlocal_functions, [])
        self.assertIsNone(ifile._damage_increment_user_output)
        self.assertIsNone(ifile._nonlocal_average_output)
        self.assertIsNone(ifile._nonlocal_damage_increment_user_output)
        self.assertEqual(ifile._nonlocal_damage_user_variables, [])
