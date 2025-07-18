from matcal.core.constants import DISPLACEMENT_KEY, TEMPERATURE_KEY, TIME_KEY
from matcal.core.data import (DataCollection, convert_dictionary_to_data)
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest

from matcal.full_field.data import convert_dictionary_to_field_data

from matcal.sierra.input_file_writer import (SolidMechanicsProcedure,
                                             SierraGlobalDefinitions,
                                             SolidMechanicsRegion, 
                                             AnalyticSierraFunction, 
                                             PiecewiseLinearFunction,
                                             ThermalMaterial,
                                             ThermalTimeParameters,
                                             FetiSolver, 
                                             _Coupling,
                                             GdswSolver, 
                                             SolidSectionDefault,
                                             TpetraSolver, 
                                             ThermalRegion,
                                             TotalLagrangeSection, 
                                             ArpeggioTransfer, 
                                             CoupledTransientParameters, 
                                             CoupledTransient, 
                                             CoupledSystem, 
                                             CoupledInitialize,
                                             NonlinearParameters, 
                                             SolutionControl, 
                                             Procedure, 
                                             _Failure,
                                             SolidMechanicsFiniteElementParameters, 
                                             FiniteElementModel,
                                             _SectionNames,
                                             SolidMechanicsImplicitDynamics, 
                                             SolidMechanicsDeath, 
                                             SolidMechanicsFixedDisplacement,
                                             SolidMechanicsPrescribedDisplacement,
                                             SolidMechanicsUserOutput, 
                                             SolidMechanicsPrescribedTemperature, 
                                             SolidMechanicsInitialTemperature, 
                                             SolidMechanicsUserVariable, 
                                             SolidMechanicsNonlocalDamageAverage, 
                                             SolidMechanicsResultsOutput, 
                                             SolidMechanicsHeartbeatOutput, 
                                             SolidMechanicsAdaptiveTimeStepping, 
                                             SolidMechanicsInteractionDefaults, 
                                             SolidMechanicsConstantFrictionModel, 
                                             SolidMechanicsRemoveInitialOverlap, 
                                             SolidMechanicsContactDefinitions, 
                                             SolidMechanicsControlContact, 
                                             SolidMechanicsLoadstepPredictor, 
                                             SolidMechanicsFullTangentPreconditioner, 
                                             SolidMechanicsConjugateGradient, 
                                             SolidMechanicsSolutionTermination,
                                             SierraFileBase, 
                                             FiniteElementModelNames, 
                                             SierraFileWithCoupling, 
                                             SierraFileThreeDimensional,
                                             SierraFileThreeDimensionalContact,
                                             ThermalDeath, 
                                             get_default_solid_mechanics_procedure_name, 
                                             get_default_solid_mechanics_region_name, 
                                             get_default_coupled_procedure_name, 
                                             get_default_thermal_region_name, 
                                             )
from matcal.sierra.material import Material


class TestSierraInputFileWriter(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def test_sierra_global_defs_bloc(self):
        global_defs = SierraGlobalDefinitions()
        test_str = global_defs.get_string()
        self.assertEqual(len(test_str.split("\n")), 11)
        self.assertTrue("cylindrical" in test_str)
        self.assertTrue("rectangular" in test_str)
        self.assertTrue("axis" in test_str)
        
    def test_analytic_function_(self):
        func = AnalyticSierraFunction("double_volume")
        func.add_expression_variable("volume", "element", "volume")
        func.add_evaluation_expression("2*volume")
        test_str = func.get_string()
        self.assertTrue("Begin" in test_str)
        self.assertTrue("End" in test_str)
        self.assertTrue("function" in test_str)
        self.assertTrue("double_volume" in test_str)
        self.assertTrue("2*volume" in test_str)
        self.assertTrue("expression variable" in test_str)
        self.assertTrue("evaluate expression" in test_str)
        self.assertTrue("type is analytic" in test_str)
        self.assertTrue("volume = element volume" in test_str)
    
    def test_piecewise_linear_function_(self):
        func = PiecewiseLinearFunction("piecewise_linear", [0,1], [0, 1])
        test_str = func.get_string()
        self.assertTrue("Begin function piecewise_linear" in test_str)
        self.assertTrue("End function piecewise_linear" in test_str)
        self.assertTrue("type is piecewise linear" in test_str)
        self.assertTrue("Begin Values" in test_str)
        self.assertTrue("End Values" in test_str)
        self.assertTrue("0 0" in test_str)
        self.assertTrue("1 1" in test_str)
        
        func.scale_function(x = 2)
        test_str = func.get_string()
        self.assertTrue("x scale = 2" in test_str)
        func.scale_function(y = 3)
        test_str = func.get_string()
        self.assertTrue("y scale = 3" in test_str)
        self.assertTrue("x scale" not in test_str)

    def test_aria_material_(self):
        mat = ThermalMaterial(1, 2, 3)
        test_str = mat.get_string()
        self.assertTrue("Begin aria material matcal_thermal" in test_str)
        self.assertTrue("density = constant rho = 1" in test_str)
        self.assertTrue("thermal conductivity = constant K = 2" in test_str)
        self.assertTrue("specific heat = constant cp = 3" in test_str)
        self.assertTrue("heat conduction = basic" in test_str)

    def test_feti_solver_(self):
        input_ = FetiSolver()
        test_str = input_.get_string()
        self.assertTrue("Begin feti equation solver feti" in test_str)

    def test_gdsw_solver_(self):
        input_ = GdswSolver()
        test_str = input_.get_string()
        self.assertTrue("Begin gdsw equation solver gdsw" in test_str)

    def test_tpetra_solver_(self):
        input_ = TpetraSolver()
        test_str = input_.get_string()
        self.assertTrue("Begin tpetra equation solver tpetra" in test_str)

    def test_total_lagrange_(self):
        input_ = TotalLagrangeSection()
        test_str = input_.get_string()
        self.assertTrue("Begin total lagrange section total_lagrange" in test_str)
        self.assertTrue("volume average J = on" in test_str)
        self.assertTrue("composite_tet" not in test_str)
        input_.use_composite_tet()
        test_str = input_.get_string()
        self.assertTrue("composite_tet" in test_str)
        self.assertTrue("total_lagrange" not in test_str)
        input_.use_composite_tet(False)
        test_str = input_.get_string()
        self.assertTrue("composite_tet" not in test_str)
        self.assertTrue("total_lagrange" in test_str)
    
    def test_default_section_(self):
        input_ = SolidSectionDefault()
        test_str = input_.get_string()
        self.assertTrue("Begin solid section uniform_gradient" in test_str)
        self.assertTrue("strain incrementation = strongly_objective" in test_str)

    def test_adagio_proc(self):
        region_block = SolidMechanicsRegion("adagio_region", 
                                    FiniteElementModelNames.solid_mechanics)
        input_subblock = SolidMechanicsProcedure("adagio_proc", region_block, 
                                           0, 1, 100)
        test_str = input_subblock.get_string()
        self.assertTrue("Begin adagio procedure adagio_proc\n" in test_str)
        self.assertTrue("Begin time control\n" in test_str)
        self.assertTrue("termination time =" in test_str)
        self.assertTrue("start time =" in test_str)
        self.assertTrue("time increment =" in test_str)
        self.assertTrue("time stepping block elastic_init" in test_str)
        self.assertTrue("time stepping block load" in test_str)
        self.assertTrue("Begin parameters for adagio region adagio_region" in test_str)
        tc_subblock = input_subblock.get_subblock("time control")
        elastic_init_subblock = tc_subblock.get_subblock("elastic_init")
        self.assertEqual(elastic_init_subblock.get_line_value("start time"), 0)
        load_subblock = tc_subblock.get_subblock("load")
        self.assertEqual(load_subblock.get_line_value("start time"), 
                         0.01*1e-3)
        elatic_init_params = elastic_init_subblock.get_subblock("adagio_region")
        self.assertEqual(elatic_init_params.get_line_value("time increment"), 0.01*1e-3)
        load_params = load_subblock.get_subblock("adagio_region")
        self.assertEqual(load_params.get_line_value("time increment"), 0.01)

    def test_adagio_proc_set_number_of_time_steps(self):
        region_block = SolidMechanicsRegion("adagio_region", 
                                    FiniteElementModelNames.solid_mechanics)
        input_subblock = SolidMechanicsProcedure("adagio_proc", region_block, 
                                           0, 1, 100)
        self.assertEqual(input_subblock._time_steps, 100)
        input_subblock.set_number_of_time_steps(1000)
        self.assertEqual(input_subblock._time_steps, 1000)
        
        tc_subblock = input_subblock.get_subblock("time control")
        elastic_init_subblock = tc_subblock.get_subblock("elastic_init")
        self.assertEqual(elastic_init_subblock.get_line_value("start time"), 0)
        load_subblock = tc_subblock.get_subblock("load")
        self.assertEqual(load_subblock.get_line_value("start time"), 
                         0.001*1e-3)
        elatic_init_params = elastic_init_subblock.get_subblock("adagio_region")
        self.assertEqual(elatic_init_params.get_line_value("time increment"), 0.001*1e-3)
        load_params = load_subblock.get_subblock("adagio_region")
        self.assertEqual(load_params.get_line_value("time increment"), 0.001)

    def test_adagio_proc_set_start_time(self):
        region_block = SolidMechanicsRegion("adagio_region", 
                                    FiniteElementModelNames.solid_mechanics)
        input_subblock = SolidMechanicsProcedure("adagio_proc", region_block, 
                                           0, 1, 100)
        self.assertEqual(input_subblock._start_time, 0)
        input_subblock.set_start_time(0.1)
        self.assertEqual(input_subblock._start_time, 0.1)
        
        tc_subblock = input_subblock.get_subblock("time control")
        elastic_init_subblock = tc_subblock.get_subblock("elastic_init")
        self.assertEqual(elastic_init_subblock.get_line_value("start time"), 0.1)
        ts = (1-0.1)/100
        load_subblock = tc_subblock.get_subblock("load")
        self.assertEqual(load_subblock.get_line_value("start time"), 
                         0.1+ts*1e-3)
        elatic_init_params = elastic_init_subblock.get_subblock("adagio_region")
        self.assertEqual(elatic_init_params.get_line_value("time increment"),ts*1e-3)
        load_params = load_subblock.get_subblock("adagio_region")
        self.assertEqual(load_params.get_line_value("time increment"), ts)

    def test_adagio_proc_set_end_time(self):
        region_block = SolidMechanicsRegion("adagio_region", 
                                    FiniteElementModelNames.solid_mechanics)
        input_subblock = SolidMechanicsProcedure("adagio_proc", region_block, 
                                           0, 1, 100)
        self.assertEqual(input_subblock._termination_time, 1)
        input_subblock.set_end_time(10)
        self.assertEqual(input_subblock._termination_time, 10)
        
        tc_subblock = input_subblock.get_subblock("time control")
        elastic_init_subblock = tc_subblock.get_subblock("elastic_init")
        self.assertEqual(elastic_init_subblock.get_line_value("start time"), 0.0)
        ts = (10-0.0)/100
        load_subblock = tc_subblock.get_subblock("load")
        self.assertEqual(load_subblock.get_line_value("start time"), 
                         0+ts*1e-3)
        elatic_init_params = elastic_init_subblock.get_subblock("adagio_region")
        self.assertEqual(elatic_init_params.get_line_value("time increment"),ts*1e-3)
        load_params = load_subblock.get_subblock("adagio_region")
        self.assertEqual(load_params.get_line_value("time increment"), ts)

    def test_arpeggio_transfer_block_nodal_field_to_send(self):
        input_block = ArpeggioTransfer("test")
        with self.assertRaises(ValueError):
            test_str = input_block.get_string()
        input_block.add_field_to_send("displacement", "solution->mesh_displacements")
        input_block.add_field_to_send("displacement", "solution->mesh_displacements", 
                                      sending_state="new", receiving_state="old")
        with self.assertRaises(ValueError):
            #needs copy keyword - next line outside of this with
            test_str = input_block.get_string()
        input_block.set_nodal_copy_transfer("solid_mechanics_region", "thermal_region")
        test_str = input_block.get_string()
        self.assertTrue("Begin transfer test" in test_str)
        send_state_none_str = ("send field displacement state none to "+
                               "solution->mesh_displacements state none")
        self.assertTrue(send_state_none_str in test_str)
        send_state_new_old_str = ("send field displacement state new to "+
                               "solution->mesh_displacements state old")
        self.assertTrue(send_state_new_old_str in test_str)

    def test_arpeggio_transfer_block_element_field_to_send(self):
        input_block = ArpeggioTransfer("test")
        with self.assertRaises(ValueError):
            test_str = input_block.get_string()
        input_block.add_field_to_send("plastic_work_variable", "plastic_work_variable")
        with self.assertRaises(ValueError):
            #needs copy keyword - next line outside of this with
            test_str = input_block.get_string()
        input_block.set_element_copy_transfer("solid_mechanics_region", "thermal_region")
        test_str = input_block.get_string()
        self.assertTrue("Begin transfer test" in test_str)
        send_state_none_str = ("send field plastic_work_variable state none to "+
                               "plastic_work_variable state none")
        self.assertTrue(send_state_none_str in test_str)

    def test_arpeggio_transfer_add_send_blocks(self):
        input_block = ArpeggioTransfer("test")
        input_block.add_field_to_send("avg_plastic_work_variable", "plastic_work_variable")
        input_block.set_element_copy_transfer("solid_mechanics_region", "thermal_region")
        input_block.add_send_blocks("block1", "block2")
        send_blocks = input_block.get_line_value("send_blocks", 2)
        receive_blocks = input_block.get_line_value("send_blocks", 4)
        self.assertEqual(send_blocks, receive_blocks)
        self.assertEqual(send_blocks, "block1 block2")
        test_str = input_block.get_string()
        
    def _get_arpeggio_parameter_block(self, name):
        input_block = CoupledTransientParameters(name, "thermal_region", 
                                            "adagio_region", 
                                            0, 1, 0.01)
        return input_block
    
    def test_arpeggio_transient_parameter_block(self):
        input_block = self._get_arpeggio_parameter_block("test")
        test_str = input_block.get_string()
        self.assertTrue("thermal_region" in input_block.subblocks)
        self.assertTrue("adagio_region" in input_block.subblocks)
        self.assertEqual(input_block.get_line_value("start time"), 0)
        self.assertEqual(input_block.get_line_value("termination time"), 1)
        self.assertTrue("Begin parameters for transient test" in test_str)

    def test_arpeggio_transient_parameter_block_set_start_time(self):
        input_block = self._get_arpeggio_parameter_block("test")
        test_str = input_block.get_string()
        self.assertEqual(input_block.start_time, 0)
        input_block.set_start_time(0.5)
        self.assertEqual(input_block.start_time, 0.5)

    def test_arpeggio_transient_parameter_block_set_termination_time(self):
        input_block = self._get_arpeggio_parameter_block("test")
        test_str = input_block.get_string()
        self.assertEqual(input_block.termination_time, 1)
        input_block.set_termination_time(0.5)
        self.assertEqual(input_block.termination_time, 0.5)

    def test_arpeggio_transient_parameter_block_set_time_increment(self):
        input_block = self._get_arpeggio_parameter_block("test")
        test_str = input_block.get_string()
        thermal_time_param_block = input_block.subblocks["thermal_region"]
        solid_time_param_block = input_block.subblocks["adagio_region"]
        self.assertEqual(thermal_time_param_block.get_line_value("initial time step size"), 0.01)
        self.assertEqual(solid_time_param_block.get_line_value("time increment"), 0.01)
        input_block.set_time_increment(0.5)
        self.assertEqual(thermal_time_param_block.get_line_value("initial time step size"), 0.5)
        self.assertEqual(solid_time_param_block.get_line_value("time increment"), 0.5)
        self.assertEqual(input_block.time_increment, 0.5)
        
    def test_arpeggio_transient_parameter_block_time_increment_unequal_error(self):
        input_block = self._get_arpeggio_parameter_block("test")
        test_str = input_block.get_string()
        thermal_time_param_block = input_block.subblocks["thermal_region"]
        solid_time_param_block = input_block.subblocks["adagio_region"]
        solid_time_param_block.set_time_increment(0.1)
        with self.assertRaises(ValueError):
            input_block.time_increment

    def test_arpeggio_transient_block(self):
        input_block = CoupledTransient("transient_test", "solid_mechanics_region", 
                                             "thermal_region")
        input_block.add_transfer_post_solid_mechanics("solid_mechanics_to_thermal_disps")
        input_block.add_transfer_post_thermal("thermal_to_solid_mechanics")
        
        test_strs = input_block.get_string().split("\n")
        self.assertTrue("transient transient_test" in test_strs[0])
        self.assertTrue("advance solid_mechanics_region" in test_strs[1])
        self.assertTrue("transfer solid_mechanics_to_thermal_disps" in test_strs[2])
        self.assertTrue("advance thermal_region" in test_strs[3])
        self.assertTrue("transfer thermal_to_solid_mechanics" in test_strs[4])
        
    def test_arpeggio_transient_block_nonlinear(self):
        input_block = CoupledTransient("transient_test", "solid_mechanics_region", 
                                             "thermal_region")
        input_block.set_nonlinear_step_name("converge_step_1")
        input_block.add_transfer_post_solid_mechanics("solid_mechanics_to_thermal_disps")
        input_block.add_transfer_post_thermal("thermal_to_solid_mechanics")
        test_strs = input_block.get_string().split("\n")
        self.assertTrue("nonlinear converge_step_1" in test_strs[2])
        self.assertTrue("advance solid_mechanics_region" in test_strs[3])
        self.assertTrue("transfer solid_mechanics_to_thermal_disps" in test_strs[4])
    
    def test_arpeggio_transient_block_nonlinear_get_string_twice(self):
        input_block = CoupledTransient("transient_test", "solid_mechanics_region", 
                                             "thermal_region")
        input_block.set_nonlinear_step_name("converge_step_1")
        input_block.add_transfer_post_solid_mechanics("solid_mechanics_to_thermal_disps")
        input_block.add_transfer_post_thermal("thermal_to_solid_mechanics")
        test_strs = input_block.get_string().split("\n")
        test_strs = input_block.get_string().split("\n")
        self.assertTrue("nonlinear converge_step_1" in test_strs[2])
        self.assertTrue("advance solid_mechanics_region" in test_strs[3])
        self.assertTrue("transfer solid_mechanics_to_thermal_disps" in test_strs[4])

    def test_arpeggio_transient_block_set_nonlinear(self):
        input_block = CoupledTransient("transient_test", "solid_mechanics_region", 
                                             "thermal_region", "converge_step_1")
        input_block.add_transfer_post_solid_mechanics("solid_mechanics_to_thermal_disps")
        input_block.add_transfer_post_thermal("thermal_to_solid_mechanics")
        test_strs = input_block.get_string().split("\n")
        self.assertTrue("nonlinear converge_step_1" in test_strs[2])
        self.assertTrue("advance solid_mechanics_region" in test_strs[3])
        self.assertTrue("transfer solid_mechanics_to_thermal_disps" in test_strs[4])

    def _get_system_block(self):
        transient_1 = CoupledTransient("transient_test", "solid_mechanics_region", 
                                             "thermal_region")
        transient_1.add_transfer_post_solid_mechanics("solid_mechanics_to_thermal_disps")
        transient_1.add_transfer_post_thermal("thermal_to_solid_mechanics")
        
        transient_2 = CoupledTransient("transient_test2", "solid_mechanics_region", 
                                             "thermal_region")
        transient_2.add_transfer_post_solid_mechanics("solid_mechanics_to_thermal_disps")
        transient_2.add_transfer_post_thermal("thermal_to_solid_mechanics")

        input_block = CoupledSystem("main", "initialization", transient_1, 
                                          transient_2)
        return input_block

    def test_arpeggio_system_block(self):
        input_block = self._get_system_block()
        
        test_str = input_block.get_string()
        self.assertTrue("use initialize initialization" in test_str)
        self.assertTrue("transient_test" in input_block.subblocks)
        self.assertTrue("transient_test2" in input_block.subblocks)

    def _get_initialize_block(self):
        input_block = CoupledInitialize("initialization", "solid_mechanics_region", 
                                             "thermal_region")
        input_block.add_transfer_post_solid_mechanics("solid_mechanics_to_thermal_disps")
        input_block.add_transfer_post_thermal("thermal_to_solid_mechanics")
        return input_block

    def test_arpeggio_initialize_block(self):
        input_block = self._get_initialize_block()
        test_strs = input_block.get_string().split("\n")
        self.assertTrue("initialize initialization" in test_strs[0])

    def test_nonlinear_parameters_block(self):
        input_block = NonlinearParameters("test")
        test_string = input_block.get_string()
        goal_string = ('converged when "thermal_region.MaxInitialNonlinearResidual(0)'+
                       ' < 1.0e-8  || CURRENT_STEP > 20"')
        self.assertEqual(goal_string, test_string.split("\n")[1].strip())

    def _get_solution_control_block(self):
        sys_block = self._get_system_block()
        init = self._get_initialize_block()
        sltn_ctl = SolutionControl("coupling", sys_block, init, 
                                   self._get_arpeggio_parameter_block("test"))
        sltn_ctl.add_subblock(self._get_arpeggio_parameter_block("test2"))

        return sltn_ctl
    
    def test_solution_control_block(self):
        sltn_ctl = self._get_solution_control_block()
        
        test_str = sltn_ctl.get_string()
        self.assertTrue("Begin parameters for transient test2" in test_str)
        self.assertTrue("Begin parameters for transient test" in test_str)
        self.assertTrue("Begin transient transient_test" in test_str)
        self.assertTrue("Begin transient transient_test2" in test_str)
        self.assertTrue("Begin solution control description coupling" in test_str)
        self.assertTrue("use system main" in test_str)
        self.assertTrue("Begin system main" in test_str)

    def test_solution_control_block_set_transient_time_parameters(self):
        sltn_ctl = self._get_solution_control_block()
        sltn_ctl.set_transient_time_parameters("test2", 1, 2, 0.3)
        test2_transient = sltn_ctl.subblocks["test2"]
        self.assertEqual(test2_transient.start_time, 1)
        self.assertEqual(test2_transient.termination_time, 2)
        self.assertEqual(test2_transient.time_increment, 0.3)
    
    def test_procedure_block(self):
        sltn_ctl = self._get_solution_control_block()
        tnsfr = ArpeggioTransfer("test")
        tnsfr.add_field_to_send("displacement", "solution->mesh_displacements")
        tnsfr.add_field_to_send("displacement", "solution->mesh_displacements", 
                                    sending_state="new", receiving_state="old")
        tnsfr.set_nodal_copy_transfer("solid_mechanics_region", "thermal_region")

        tnsfr2 = ArpeggioTransfer("test2")
        tnsfr2.add_field_to_send("displacement", "solution->mesh_displacements", 
                                    sending_state="new", receiving_state="old")
        tnsfr2.set_nodal_copy_transfer("solid_mechanics_region", "thermal_region")
        procedure_blk = Procedure(sltn_ctl, tnsfr, tnsfr2)

        test_str  = procedure_blk.get_string()
        self.assertTrue("Begin transfer test" in test_str)
        self.assertTrue("Begin transfer test2" in test_str)
        self.assertTrue("Begin solution control description coupling" in test_str)

    def test_aria_time_parameters(self):
        input_block = ThermalTimeParameters("my_region", 0.01)
        self.assertEqual(input_block.name, "my_region")
        self.assertEqual(input_block.lines["initial time step size"].get_values()[-1], 0.01)
        input_block.set_time_increment(1e-3)
        self.assertEqual(input_block.lines["initial time step size"].get_values()[-1], 0.001)
        
    def _get_finite_element_params_block(self, blocks=["block_1", "block_2"]):
        input_block = SolidMechanicsFiniteElementParameters("test_material", 
                                                "elastic", 
                                                *blocks)
        return input_block
    
    def test_finite_element_block_parameters(self):
        input_block = self._get_finite_element_params_block()
        test_str = input_block.get_string()
        self.assertTrue("Begin parameters for block block_1 block_2" in test_str)
        self.assertTrue("section = total_lagrange" in test_str)
    
    def test_finite_element_block_parameters_UG(self):
        input_block = SolidMechanicsFiniteElementParameters("test_material", 
                                                "elastic", 
                                                "block_1", "block_2")
        input_block.set_section(_SectionNames.uniform_gradient)
        test_str = input_block.get_string()
        self.assertTrue("Begin parameters for block block_1 block_2" in test_str)
        self.assertTrue("section = uniform_gradient" in test_str)

    def test_finite_element_block_parameters_get_section(self):
        input_block = SolidMechanicsFiniteElementParameters("test_material", 
                                                "elastic", 
                                                "block_1", "block_2")
        input_block.set_section(_SectionNames.uniform_gradient)
        section = input_block.get_section()
        self.assertEqual(section, _SectionNames.uniform_gradient)
        
    def test_finite_element_model_block(self):
        FE_params_block = self._get_finite_element_params_block()
        input_block = FiniteElementModel(FE_params_block)
        input_block.set_mesh_filename("test_mesh.g")
        test_str = input_block.get_string()
        self.assertTrue("Begin finite element model matcal_solid_mechanics" in test_str)
        database_values = input_block.lines[FiniteElementModel.required_keys[1]].get_values()
        self.assertEqual("exodusII", database_values[-1])
        mesh_name_values = input_block.lines[FiniteElementModel.required_keys[0]].get_values()
        self.assertEqual("test_mesh.g", mesh_name_values[-1])
        self.assertTrue("Begin parameters for block block_1 block_2" in test_str)

    def test_finite_element_model_block_raise_val_error_no_params(self):
        input_block = FiniteElementModel()
        input_block.set_mesh_filename("test_mesh.g")
        with self.assertRaises(ValueError):
            input_block.get_element_section()
            
    def test_finite_element_model_block_set_element_section(self):
        FE_params_block = self._get_finite_element_params_block()
        input_block = FiniteElementModel(FE_params_block)
        input_block.set_element_section(_SectionNames.uniform_gradient)
        input_block.set_mesh_filename("test_mesh.g")
        section = input_block.get_element_section()
        self.assertEqual(section, _SectionNames.uniform_gradient)

    def test_finite_element_model_block_get_element_section_two_sections(self):
        FE_params_block1 = self._get_finite_element_params_block()
        FE_params_block2 = self._get_finite_element_params_block(blocks=["block3", "block_4"])
        FE_params_block2.set_section(_SectionNames.uniform_gradient)
        input_block = FiniteElementModel(FE_params_block1, FE_params_block2)
        input_block.set_mesh_filename("test_mesh.g")
        section = input_block.get_element_section()
        self.assertEqual(section, set([_SectionNames.total_lagrange,
                                   _SectionNames.uniform_gradient]))

    def test_SM_implicit_dynamics_block(self):
        input_block = SolidMechanicsImplicitDynamics()
        test_str = input_block.get_string()

        self.assertEqual("Begin implicit dynamics", test_str.split("\n")[0])
        self.assertEqual("contact timestep = off", test_str.split("\n")[1].strip())

    def test_solid_mechanics_death_block(self):
        input_block = SolidMechanicsDeath("eqps", 0.15, "necking_section", 
                                               "gauge_section")
        test_str = input_block.get_string()
        self.assertEqual("Begin element death hades", test_str.split("\n")[0])
        block_values = input_block.lines["block"].get_values()
        self.assertEqual("necking_section", block_values[1])
        self.assertEqual("gauge_section", block_values[2])
        criterion_values = input_block.lines["criterion"].get_values()
        self.assertEqual("eqps", criterion_values[-3])
        self.assertEqual(">=", criterion_values[-2])
        self.assertEqual(0.15, criterion_values[-1])
        self.assertEqual(0.15, input_block.get_critical_value())
    
    def test_solid_mechanics_fixed_disp_block(self):
        input_block = SolidMechanicsFixedDisplacement("fixed_x_ns", "x")
        test_str = input_block.get_string()
        self.assertEqual("Begin fixed displacement", test_str.split("\n")[0])
        node_set_values = input_block.lines["node set"].get_values()
        self.assertEqual("fixed_x_ns", node_set_values[-1])
        dir_values = input_block.lines["component"].get_values()
        self.assertEqual("x", dir_values[-1])
        
    def test_solid_mechanics_prescribed_disp_block(self):
        input_block = SolidMechanicsPrescribedDisplacement("disp_func", "grip_ns", 
                                                                "x")
        function_values = input_block.lines["function"].get_values()
        self.assertEqual("disp_func", function_values[-1])
        self.assertNotIn("scale factor", input_block.lines)

        input_block = SolidMechanicsPrescribedDisplacement("disp_func", "grip_ns", 
                                                                "x", scale_factor=0.5)
        scale_factor_values = input_block.lines["scale factor"].get_values()
        self.assertEqual(0.5, scale_factor_values[-1])

    def test_solid_mechanics_user_output_block_nodal_field(self):
        input_block = SolidMechanicsUserOutput("load_disp_output", "grip_ns", 
                                                    "node set")
        
        input_block.add_compute_global_from_nodal_field("load", "external_force(y)", 
                                                        calculation="sum")
        input_block.add_compute_global_from_nodal_field("displacement", "displacement(y)", 
                                                        calculation="sum")
        test_strs = input_block.get_string().split("\n")
        self.assertTrue("node set = grip_ns" in test_strs[1].strip())
        self.assertTrue("compute at every step" in test_strs[2].strip())
        self.assertTrue(("compute global load as sum of nodal " +
                        "external_force(y)") in test_strs[3].strip())
        self.assertTrue(("compute global displacement as sum of " +
                        "nodal displacement(y)") in test_strs[4].strip())

    def test_solid_mechanics_user_output_block_element_field(self):
        input_block = SolidMechanicsUserOutput("temps_output", "necking_section", 
                                                    "block")
        input_block.add_compute_global_from_element_field("high_temp", TEMPERATURE_KEY, 
                                                        calculation="max")
        input_block.add_compute_global_from_element_field("avg_temp", TEMPERATURE_KEY, 
                                                        calculation="average")
        test_strs = input_block.get_string().split("\n")
        self.assertTrue("block = necking_section" in test_strs[1].strip())
        self.assertTrue("compute at every step" in test_strs[2].strip())
        self.assertTrue(("compute global high_temp as max of element " +
                        TEMPERATURE_KEY) in test_strs[3].strip())
        self.assertTrue(("compute global avg_temp as average of " +
                        "element temperature") in test_strs[4].strip())
        
    def test_solid_mechanics_user_output_block_expression(self):
        input_block = SolidMechanicsUserOutput("load_disp_output", "grip_ns", "node set")
        input_block.add_compute_global_from_expression("displacement", "partial_displacement*2;")
        input_block.add_compute_global_from_expression("load", "partial_load*4;")
        test_strs = input_block.get_string().split("\n")
        self.assertTrue("node set = grip_ns" in test_strs[1].strip())
        self.assertTrue("compute at every step" in test_strs[2].strip())
        
        self.assertTrue(("compute global displacement from expression " +
                        "\" partial_displacement*2; \"") in test_strs[3].strip())
        self.assertTrue(("compute global load from expression " +
                        "\" partial_load*4; \"") in test_strs[4].strip())
        
    def test_solid_mechanics_user_output_block_element_function(self):
        input_block = SolidMechanicsUserOutput("element_outputs", "include all blocks")
        input_block.add_compute_element_as_function("test", "test_function")
        test_strs = input_block.get_string().split("\n")
        element_test = input_block.get_line_value("element test", -1)
        self.assertEqual(element_test, "test_function")
        self.assertEqual(test_strs[1].strip(), "include all blocks")

    def test_solid_mechanics_user_output_block_global_function(self):
        input_block = SolidMechanicsUserOutput("global_function", "include all blocks")
        input_block.add_compute_global_as_function("test", "test_function")
        test_strs = input_block.get_string().split("\n")
        global_test = input_block.get_line_value("global test", -1)
        self.assertEqual(global_test, "test_function")
        self.assertEqual(test_strs[1].strip(), "include all blocks")

    def test_solid_mechanics_user_output_block_element_from_element(self):
        input_block = SolidMechanicsUserOutput("element_outputs", "include all blocks")
        input_block.add_compute_element_from_element("test_avg", "test")
        test_strs = input_block.get_string().split("\n")
        self.assertEqual(test_strs[3].strip(), "compute element test_avg as volume weighted "
                         "average of element test")

    def test_solid_mechanics_user_output_block_add_nodal_variable_transformation(self):
        input_block = SolidMechanicsUserOutput("transform", "include all blocks")
        input_block.add_nodal_variable_transformation("disp", "cyl_disp", "cyl_coord_sys")
        
        self.assertEqual(input_block.lines["cyl_disp"].get_string().strip(), 
            "transform nodal variable disp to coordinate system cyl_coord_sys as cyl_disp")

    def test_solid_mechanics_prescribed_temperature_block(self):
        input_block = SolidMechanicsPrescribedTemperature("include all blocks", 
                                                               function_name="temp_func")
        test_strs = input_block.get_string().splitlines()

        self.assertTrue("include all blocks" in test_strs[1].strip())
        self.assertTrue("function" in test_strs[2].strip())
        
        input_block = SolidMechanicsPrescribedTemperature("temp_nodes", 
                                                               function_name="temp_func")
        test_str = input_block.get_string()
        test_strs = test_str.splitlines()

        self.assertEqual("node set = temp_nodes", test_strs[1].strip())
        self.assertTrue("receive from transfer" not in test_str)
        input_block = SolidMechanicsPrescribedTemperature("temp_nodes", 
                                                               transfer=True)
        test_str = input_block.get_string()
        test_strs = test_str.splitlines()
        
        self.assertTrue("receive from transfer" in test_str)
        self.assertTrue("function" not in test_str)

    def test_solid_mechanics_prescribed_temperature_read_from_mesh(self):
        input_block = SolidMechanicsPrescribedTemperature("include all blocks",)
        input_block.read_from_mesh("temp")
        test_strs = input_block.get_string().splitlines()
        self.assertTrue("include all blocks" in test_strs[1].strip())
        self.assertTrue("read variable = temp" in test_strs[2].strip())

    def test_solid_mechanics_initial_temperature_block(self):
        input_block = SolidMechanicsInitialTemperature("include all blocks", 
                                                               20)
        test_str = input_block.get_string()
        test_strs = test_str.splitlines()
        self.assertEqual("include all blocks", test_strs[1].strip())
        self.assertEqual("magnitude = 20", test_strs[2].strip())
        
        input_block = SolidMechanicsInitialTemperature("temp_block", 
                                                               20)
        test_str = input_block.get_string()
        test_strs = test_str.splitlines()
        self.assertEqual("block = temp_block", test_strs[1].strip())
        
    def test_solid_mechanics_user_variable(self):
        input_block = SolidMechanicsUserVariable("damage_increment", "element", 
                                                      "real", 1e-4, 1e-4, 1e-4, 1e-4)
        test_str = input_block.get_string()
        test_strs = test_str.splitlines()
        self.assertEqual("type = element real length = 4", test_strs[1].strip())
        initial_values = input_block.get_line("initial value").get_values()
        self.assertEqual(initial_values[1:], [1e-4]*4)

    def test_solid_mechanics_user_variable_add_blocks(self):
        input_block = SolidMechanicsUserVariable("damage_increment", "element", 
                                                      "real", 1e-4, 1e-4, 1e-4, 1e-4)
        input_block.add_blocks("block1", "block2")
        test_str = input_block.get_string()
        self.assertIn("block = block1 block2", test_str)

    def test_solid_mechanics_nonlocal_damage_average_block(self):
        input_block = SolidMechanicsNonlocalDamageAverage(0.01)
        test_str = input_block.get_string()
        test_strs = test_str.splitlines()
        self.assertEqual("source variable = element damage_increment", test_strs[1].strip())
        self.assertEqual("target_variable = element nonlocal_damage_increment", test_strs[2].strip())
        self.assertEqual("radius = 0.01", test_strs[3].strip())
        self.assertEqual("distance algorithm = euclidean_graph", test_strs[4].strip())

    def test_solid_mechanics_user_output_block(self):
        input_block = SolidMechanicsResultsOutput(20)
        input_block.add_element_output("eqps_avg")
        input_block.add_element_output("test", "test_out")

        input_block.add_global_output("load")
        input_block.add_global_output("displacement")

        input_block.add_nodal_output("displacement", "displ")
        test_str = input_block.get_string()
        output_name = input_block.get_line_value("database name", -1)
        self.assertEqual(output_name, "./results/results.e")
        eqps = input_block.get_line_value("element eqps_avg", -1)
        self.assertEqual(eqps, "eqps_avg")
        self.assertTrue(input_block.has_element_output("eqps_avg"))
        self.assertTrue(input_block.has_element_output("test", "test_out"))

        displ_line_name = input_block._get_nodal_variable_line_name("displacement", "displ")
        displ = input_block.get_line_value(displ_line_name, -1)
        self.assertEqual(displ, "displ")
        load = input_block.get_line_value("global load", -1)
        self.assertEqual(load, "load")
        displacement = input_block.get_line_value("global displacement", -1)
        self.assertEqual(displacement, "displacement")
        
    def test_solid_mechanics_user_output_block_exposed_surf(self):
        input_block = SolidMechanicsResultsOutput(20)
        input_block.add_element_output("eqps_avg")
        input_block.add_include_surface("DIC_surf")
        input_block.add_exclude_blocks("necking_block")
        input_block.set_output_exposed_surface()
        test_str = input_block.get_string()
        includes = input_block.get_line_value("include", -1)
        self.assertEqual(includes, "DIC_surf")
        excludes = input_block.get_line_value("exclude", -1)
        self.assertEqual(excludes, "necking_block")
        output_mesh = input_block.get_line_value("output mesh", -1)
        self.assertEqual(output_mesh, "exposed surface")
        input_block.set_output_exposed_surface(False)
        self.assertNotIn("output mesh", input_block.lines)

    def test_solid_mechanics_heartbeat_output(self):
        input_block = SolidMechanicsHeartbeatOutput(1, "load", 
                                                         "displacement")
        test_str = input_block.get_string()
        timestamp = input_block.get_line_value("timestamp", -1)
        self.assertEqual(timestamp, "''")
        timestamp_format = input_block.get_line_value("timestamp", -2)
        self.assertEqual(timestamp_format, "format")
        self.assertIn("global load", input_block.lines)
        self.assertIn("global displacement", input_block.lines)

    def test_solid_mechanics_heartbeat_output_get_global_output(self):
        input_block = SolidMechanicsHeartbeatOutput(1, "load", 
                                                         "displacement")
        g_outputs = input_block.get_global_outputs()
        g_output_names = []
        for g_output in g_outputs:
            g_output_names.append(g_output.name)
        self.assertTrue(len(g_outputs), 2)
        self.assertIn("global load", g_output_names)
        self.assertIn("global displacement", g_output_names)
        
    def test_solid_mechanics_adaptive_time_stepping(self):
        input_block = SolidMechanicsAdaptiveTimeStepping()
        min_mult = input_block.get_line_value("minimum multiplier")
        self.assertEqual(min_mult, 1e-8)
        max_mult = input_block.get_line_value("maximum multiplier")
        self.assertEqual(max_mult, 1)
        self.assertEqual(len(input_block.lines), 3)
        input_block.set_cutback_factor(0.75)
        cutback = input_block.get_line_value("cutback factor")
        self.assertEqual(cutback, 0.75)
        input_block.set_growth_factor(1.25)
        growth = input_block.get_line_value("growth factor")
        self.assertEqual(growth, 1.25)
        self.assertEqual(len(input_block.lines), 5)
        input_block.set_iteration_target()
        target_its = input_block.get_line_value("target iterations")
        self.assertEqual(target_its, 75)
        window = input_block.get_line_value("iteration window")
        self.assertEqual(window, 5)
        self.assertEqual(len(input_block.lines), 7)
        input_block.set_adaptive_time_stepping_method("solver_average")
        method = input_block.get_line_value("method")
        self.assertEqual(method, "solver_average")
        
    def test_solid_mechanics_interaction_defaults(self):
        input_block = SolidMechanicsInteractionDefaults("friction")
        self.assertTrue("friction model" in input_block.lines)
        self.assertTrue("self contact" in input_block.lines)
        self.assertTrue("general contact" in input_block.lines)
        self.assertEqual(len(input_block.lines), 3)
        
        input_block.set_self_contact(False)
        self_contact = input_block.get_line_value("self contact")
        self.assertEqual(self_contact, "off")
        gen_contact = input_block.get_line_value("general contact")
        self.assertEqual(gen_contact, "on")
        friction_mod = input_block.get_line_value("friction model")
        self.assertEqual(friction_mod, "friction")

    def test_solid_mechanics_constant_friction_model(self):
        input_block = SolidMechanicsConstantFrictionModel("friction")
        self.assertTrue("friction coefficient" in input_block.lines)
        self.assertEqual(len(input_block.lines), 1)
        fric_coeff =input_block.get_friction_coefficient()
        self.assertEqual(fric_coeff, 0.3)
    
    def test_solid_mechanics_remove_initial_overlap(self):
        input_block = SolidMechanicsRemoveInitialOverlap()
        self.assertEqual(len(input_block.lines), 0)
        
    def test_solid_mechanics_contact_definitions(self):
        friction_model_block = SolidMechanicsConstantFrictionModel("friction")
        input_block = SolidMechanicsContactDefinitions(friction_model_block)
        
        interactions_block = input_block.get_interaction_defaults_block()
        self.assertIsInstance(interactions_block, SolidMechanicsInteractionDefaults)

        obtained_friction_mod_block = input_block.get_constant_friction_model_block()
        self.assertEqual(obtained_friction_mod_block, friction_model_block)

        remove_overlap_block = input_block.get_remove_initial_overlap_block()
        self.assertIsInstance(remove_overlap_block, SolidMechanicsRemoveInitialOverlap)

    def test_solid_mechanics_control_contact(self):
        input_block = SolidMechanicsControlContact()

        min_iters = input_block.get_line_value("minimum iterations")
        self.assertEqual(min_iters, 5)
        accept_rel_resid = input_block.get_line_value("acceptable relative residual")
        self.assertEqual(accept_rel_resid, 1e-2)
        target_rel_resid = input_block.get_line_value("target relative residual")
        self.assertEqual(target_rel_resid, 1e-3)

    def test_solid_mechanics_loadstep_predictor(self):
        input_block = SolidMechanicsLoadstepPredictor()
        scale_factor = input_block.get_line_value("scale factor")
        self.assertEqual(scale_factor, 0.0)
        input_block.set_scale_factor(1.0)
        self.assertEqual(scale_factor, 0.0)
                
    def test_solid_mechanics_full_tangent_preconditioner(self):
        input_block = SolidMechanicsFullTangentPreconditioner()
        
        small_num_iters = input_block.get_line_value("small number of iterations")
        self.assertEqual(small_num_iters, 20)
        
        min_smooth_iters = input_block.get_line_value("minimum smoothing iterations")
        self.assertEqual(min_smooth_iters, 15)
        
        iters_update = input_block.get_line_value("iteration update")
        self.assertEqual(iters_update, 25)
        self.assertEqual(len(input_block.lines), 3)
        solver = GdswSolver()
        input_block = SolidMechanicsFullTangentPreconditioner(solver)
        self.assertEqual(len(input_block.lines), 4)
        linear_solver = input_block.get_line_value("linear solver")
        self.assertEqual(linear_solver, "gdsw")

    def test_solid_mechanics_conjugate_gradient(self):
        input_block = SolidMechanicsConjugateGradient()
        self.assertEqual(len(input_block.lines), 6)
        reference = input_block.get_line_value("reference")
        self.assertEqual(reference, "Belytschko")
        self.assertFalse(input_block._print_name)

        full_tan_precond = SolidMechanicsFullTangentPreconditioner()
        input_block = SolidMechanicsConjugateGradient(full_tangent_preconditioner=full_tan_precond)
        self.assertIn(SolidMechanicsFullTangentPreconditioner.type, input_block.subblocks)
        input_block.set_full_tangent_preconditioner(None)
        self.assertNotIn(SolidMechanicsFullTangentPreconditioner, input_block.subblocks)
        
    def test_solid_mechanics_conjugate_gradient_set_tolerances(self):
        input_block = SolidMechanicsConjugateGradient()
        self.assertEqual(input_block.get_target_relative_residual(), 1e-9)
        input_block.set_target_relative_residual(1e-7)        
        self.assertEqual(input_block.get_target_relative_residual(), 1e-7)

        self.assertAlmostEqual(input_block.get_target_residual(), 1e-7)
        input_block.set_target_residual(1e-9)        
        self.assertEqual(input_block.get_target_residual(), 1e-9)
        
        self.assertEqual(input_block.get_acceptable_relative_residual(), 1e-8)
        input_block.set_acceptable_relative_residual(1e-6)        
        self.assertEqual(input_block.get_acceptable_relative_residual(), 1e-6)
        
        self.assertIsNone(input_block.get_acceptable_residual())
        input_block.set_acceptable_residual(1e-4)
        self.assertEqual(input_block.get_acceptable_residual(), 1e-4)

    def tests_olid_mechanics_region(self):
        region_block = SolidMechanicsRegion("adagio_region", 
                                    FiniteElementModelNames.solid_mechanics)
        self.assertEqual(len(region_block.lines), 1)
        finite_ele_model_name = region_block.get_line_value(SolidMechanicsRegion.required_keys[0])
        self.assertEqual(finite_ele_model_name, FiniteElementModelNames.solid_mechanics)

    def test_get_subblock_by_type(self):
        region_block = SolidMechanicsRegion("adagio_region", 
                                    FiniteElementModelNames.solid_mechanics)
        self.assertEqual(len(region_block.lines), 1)
        finite_ele_model_name = region_block.get_line_value(SolidMechanicsRegion.required_keys[0])
        self.assertEqual(finite_ele_model_name, FiniteElementModelNames.solid_mechanics)
        bc1 = SolidMechanicsPrescribedDisplacement("test", "test_ns", "X")
        bc2 = SolidMechanicsPrescribedDisplacement("test", "test_ns2", "X")
        bc3 = SolidMechanicsPrescribedDisplacement("test", "test_ns3", "X")
        bc4 = SolidMechanicsPrescribedDisplacement("test", "test_ns4", "X")
        subblock  = region_block.get_subblock_by_type(bc1.type)
        self.assertEqual(subblock, None)

        region_block.add_subblock(bc1)
        region_block.add_subblock(bc2)
        region_block.add_subblock(bc3)
        region_block.add_subblock(bc4)

        subblock  = region_block.get_subblock_by_type(bc1.type)
        self.assertEqual(subblock, bc1)
        region_block.remove_subblock(bc1)
        region_block.remove_subblock(bc2)
        
        subblock  = region_block.get_subblock_by_type(bc3.type)
        self.assertEqual(subblock, bc3)

    def test_remove_subblocks_by_type(self):
        region_block = SolidMechanicsRegion("adagio_region", 
                                    FiniteElementModelNames.solid_mechanics)
        self.assertEqual(len(region_block.lines), 1)
        finite_ele_model_name = region_block.get_line_value(SolidMechanicsRegion.required_keys[0])
        self.assertEqual(finite_ele_model_name, FiniteElementModelNames.solid_mechanics)
        bc1 = SolidMechanicsPrescribedDisplacement("test", "test_ns", "X")
        bc2 = SolidMechanicsPrescribedDisplacement("test", "test_ns2", "X")
        bc3 = SolidMechanicsPrescribedDisplacement("test", "test_ns3", "X")
        bc4 = SolidMechanicsPrescribedDisplacement("test", "test_ns4", "X")
        
        subblock  = region_block.get_subblock_by_type(bc1.type)
        region_block.add_subblock(bc1)
        region_block.add_subblock(bc2)
        region_block.add_subblock(bc3)
        region_block.add_subblock(bc4)

        subblock  = region_block.remove_subblocks_by_type(bc1.type)
        self.assertEqual(subblock, None)

    def test_thermal_death(self):
        input_block = ThermalDeath("death_status", 0.99, "block1", "block2", 
                                   criterion_eval_operator="<=")
        self.assertEqual(input_block.get_line_value("Add volume", 1), "block1")
        self.assertEqual(input_block.get_line_value("Add volume", 2), "block2")
        self.assertEqual(input_block.get_line_value("criterion", 2), "death_status")
        self.assertEqual(input_block.get_line_value("criterion", 3), "<=")
        self.assertEqual(input_block.get_line_value("criterion", 4), 0.99)
        
    def test_solution_termination(self):
        input_block = SolidMechanicsSolutionTermination()
        input_block.add_global_termination_criteria("test", 1)
        self.assertEqual(input_block.get_line_value("global test", 1), "global")
        self.assertEqual(input_block.get_line_value("global test", 2), "test")
        self.assertEqual(input_block.get_line_value("global test", 3), "<")
        self.assertEqual(input_block.get_line_value("global test", 4), 1)
        self.assertEqual(input_block.get_line_value("terminate type"), "entire_run")


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
        

class TestThermalRegion(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)
        self._region_name = get_default_thermal_region_name()
        self._fe_model = FiniteElementModelNames.thermal
        self._solver = TpetraSolver()

    def _make_thermal_region(self):
        thermal_region = ThermalRegion(self._region_name, self._fe_model, self._solver)
        return thermal_region
    
    def test_init(self):
        thermal_region = self._make_thermal_region()
        self.assertEqual(thermal_region.get_line_value("nonlinear solution strategy"), "NEWTON")
        self.assertEqual(thermal_region.get_line_value("use finite element model"),self._fe_model)
        self.assertEqual(thermal_region.get_line_value("use linear solver"), self._solver.name)
        test_str = thermal_region.get_input_string()
        self.assertIn("EQ energy", test_str)
        self.assertIn("EQ mesh", test_str)

    def test_add_heating_source(self):
        thermal_region = self._make_thermal_region()
        thermal_region.add_heating_source("plastic_work", 8)
        test_str = thermal_region.get_input_string()
        self.assertIn("source for energy", test_str)
        self.assertIn("plastic_work", thermal_region.subblocks)

    def test_add_element_death(self):
        thermal_region = self._make_thermal_region()
        thermal_region.add_element_death("death_block1", "death_block2")
        test_str = thermal_region.get_input_string()
        self.assertIn("User field real element scalar death_status_aria", test_str)
        self.assertIn("transfer element death", test_str)
        self.assertIn('hades', thermal_region.subblocks)

    def test_add_initial_condition(self):
        thermal_region = self._make_thermal_region()
        thermal_region.add_initial_condition(298)
        test_str = thermal_region.get_input_string()
        self.assertIn("IC const on all_blocks TEMPERATURE = 298", test_str)

    def test_add_dirichlet_temperature_boundary_condition(self):
        thermal_region = self._make_thermal_region()
        thermal_region.add_dirichlet_temperature_boundary_condition("grip", 298)
        test_str = thermal_region.get_input_string()
        self.assertIn("BC Const Dirichlet at grip Temperature = 298", test_str)


class TestSierraFileWithCoupling(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)

    def _make_input_deck(self):
        mat_filename = "matfile.inc"
        with open(mat_filename, "w") as f:
            f.write("material...")
        mat = Material("test_mat", mat_filename, "j2_plasticity")
        ifile = SierraFileWithCoupling(mat, ["block_to_kill"])
        ifile._set_thermal_bc_nodesets(["dirichlet_bc1", "dirichlet_bc2"])
        ifile._set_local_mesh_filename("test.g")
        ifile._add_solid_mechanics_finite_element_parameters("test_mat", 
                                                            "j2_plasticity", 
                                                            "block1", "block2")
        return ifile

    def test_activate_thermal_coupling(self):
        ifile = self._make_input_deck()
        self.assertIsNone(ifile.coupling)
        self.assertIsNone(ifile._coupled_procedure)
        self.assertIsNone(ifile._thermal_material)
        self.assertIsNone(ifile._thermal_model)
        self.assertIsNone(ifile._thermal_region)
        self.assertEqual(ifile._coupling_transfers, [])

        ifile._activate_thermal_coupling(1,1,1,"plastic_work")
        
        self.assertIn(TpetraSolver().name, ifile.subblocks)
        self.assertIn(FiniteElementModelNames.thermal, ifile.subblocks)
        self.assertIn(ThermalMaterial(1,1,1).name, ifile.subblocks)

        self.assertIn(get_default_coupled_procedure_name(), ifile.subblocks)
        self.assertEqual(len(ifile._coupling_transfers), 5)
        
    def test_activate_thermal_coupling_update_mesh(self):
        ifile = self._make_input_deck()
        ifile._activate_thermal_coupling(1,1,1,"plastic_work")
        self.assertEqual(ifile._thermal_model.mesh_filename, 
                         ifile._sm_finite_element_model.mesh_filename)     
        self.assertEqual(ifile._thermal_model.mesh_filename, "test.g")
        
        ifile._set_local_mesh_filename("test_2.g")
        self.assertEqual(ifile._thermal_model.mesh_filename, 
                         ifile._sm_finite_element_model.mesh_filename)     
        self.assertEqual(ifile._thermal_model.mesh_filename, "test_2.g")
    
    def test_activate_thermal_coupling_get_string_twice(self):
        ifile = self._make_input_deck()
        ifile._activate_thermal_coupling(1,1,1,"plastic_work")
        ifile.get_input_string()
        ifile.get_input_string()
        self.assertEqual(ifile._thermal_model.mesh_filename, 
                         ifile._sm_finite_element_model.mesh_filename)     
        self.assertEqual(ifile._thermal_model.mesh_filename, "test.g")
        
        ifile._set_local_mesh_filename("test_2.g")
        self.assertEqual(ifile._thermal_model.mesh_filename, 
                         ifile._sm_finite_element_model.mesh_filename)     
        self.assertEqual(ifile._thermal_model.mesh_filename, "test_2.g")

    def test_activate_thermal_coupling_update_element(self):
        ifile = self._make_input_deck()

        ifile._activate_thermal_coupling(1,1,1,"plastic_work")
        work_transfer = ifile._work_transfer

        plastic_work_var = ifile._thermal_region.subblocks["plastic_work"]
        self.assertEqual(plastic_work_var.get_line_value("type", -1), 8)     
        self.assertNotIn("plastic_work_vol_avg", work_transfer.get_string())

        ifile._use_under_integrated_element()
        plastic_work_var = ifile._thermal_region.subblocks["plastic_work"]
        self.assertEqual(plastic_work_var.get_line_value("type", -1), 1)     
        self.assertNotIn("plastic_work_vol_avg", work_transfer.get_string())

        ifile._use_total_lagrange_element(composite_tet=True)
        plastic_work_var = ifile._thermal_region.subblocks["plastic_work"]
        self.assertEqual(plastic_work_var.get_line_value("type", -1), 1)     
        self.assertIn("plastic_work_vol_avg", work_transfer.get_string())
        vol_avg_output = ifile._vol_average_user_output
        self.assertTrue("element plastic_work_vol_avg" in vol_avg_output.lines)
        enery_eq_key = "EQ energy for TEMPERATURE on all_blocks using"
        self.assertIn("Q2", ifile._thermal_region.get_line_value(enery_eq_key))
        disp_eq_key = "EQ mesh for MESH_DISPLACEMENTS on all_blocks using"
        self.assertIn("Q2", ifile._thermal_region.get_line_value(disp_eq_key))

        ifile._use_total_lagrange_element(composite_tet=False)
        plastic_work_var = ifile._thermal_region.subblocks["plastic_work"]
        self.assertEqual(plastic_work_var.get_line_value("type", -1), 8)     
        self.assertNotIn("plastic_work_vol_avg", work_transfer.get_string())
        self.assertIn("plastic_work", work_transfer.get_string())
        enery_eq_key = "EQ energy for TEMPERATURE on all_blocks using"
        self.assertIn("Q1", ifile._thermal_region.get_line_value(enery_eq_key))
        disp_eq_key = "EQ mesh for MESH_DISPLACEMENTS on all_blocks using"
        self.assertIn("Q1", ifile._thermal_region.get_line_value(disp_eq_key))

    def test_activate_thermal_coupling_under_integrated(self):
        ifile = self._make_input_deck()
        ifile._use_under_integrated_element()
        ifile._activate_thermal_coupling(1,1,1,"plastic_work")
        work_transfer = ifile._work_transfer
        plastic_work_var = ifile._thermal_region.subblocks["plastic_work"]
        self.assertEqual(plastic_work_var.get_line_value("type", -1), 1)     
        self.assertNotIn("plastic_work_vol_avg", work_transfer.get_string())
        self.assertIn("plastic_work", work_transfer.get_string())
        enery_eq_key = "EQ energy for TEMPERATURE on all_blocks using"
        self.assertIn("Q1", ifile._thermal_region.get_line_value(enery_eq_key))
        disp_eq_key = "EQ mesh for MESH_DISPLACEMENTS on all_blocks using"
        self.assertIn("Q1", ifile._thermal_region.get_line_value(disp_eq_key))

    def test_activate_thermal_coupling_total_lagrange(self):
        ifile = self._make_input_deck()
        ifile._use_total_lagrange_element()
        ifile._activate_thermal_coupling(1,1,1,"plastic_work")
        work_transfer = ifile._work_transfer
        plastic_work_var = ifile._thermal_region.subblocks["plastic_work"]
        self.assertEqual(plastic_work_var.get_line_value("type", -1), 8)     
        self.assertNotIn("plastic_work_vol_avg", work_transfer.get_string())
        self.assertIn("plastic_work", work_transfer.get_string())
        enery_eq_key = "EQ energy for TEMPERATURE on all_blocks using"
        self.assertIn("Q1", ifile._thermal_region.get_line_value(enery_eq_key))
        disp_eq_key = "EQ mesh for MESH_DISPLACEMENTS on all_blocks using"
        self.assertIn("Q1", ifile._thermal_region.get_line_value(disp_eq_key))

    def test_activate_thermal_coupling_composite_tet(self):
        ifile = self._make_input_deck()
        ifile._use_total_lagrange_element(composite_tet=True)
        ifile._activate_thermal_coupling(1,1,1,"plastic_work")
        work_transfer = ifile._work_transfer
        plastic_work_var = ifile._thermal_region.subblocks["plastic_work"]
        self.assertEqual(plastic_work_var.get_line_value("type", -1), 1)     
        self.assertIn("plastic_work_vol_avg", work_transfer.get_string())
        self.assertIn("plastic_work", work_transfer.get_string())
        enery_eq_key = "EQ energy for TEMPERATURE on all_blocks using"
        self.assertIn("Q2", ifile._thermal_region.get_line_value(enery_eq_key))
        disp_eq_key = "EQ mesh for MESH_DISPLACEMENTS on all_blocks using"
        self.assertIn("Q2", ifile._thermal_region.get_line_value(disp_eq_key))

    def test_coupling_set_time_parameters_to_loading_function(self):
        ifile = self._make_input_deck()
        self.assertIsNone(ifile._transient_params_1)
        self.assertIsNone(ifile._transient_params_2)
        ifile._activate_thermal_coupling(1,1,1,"plastic_work")
        self.assertEqual(ifile._transient_params_1.start_time, 0)
        self.assertEqual(ifile._transient_params_1.termination_time, 1.0/300*1e-3)
        self.assertEqual(ifile._transient_params_1.time_increment, 1.0/300*1e-3)
        self.assertEqual(ifile._transient_params_2.start_time, 1.0/300*1e-3)
        self.assertEqual(ifile._transient_params_2.termination_time, 1.0)
        self.assertEqual(ifile._transient_params_2.time_increment, 1.0/300)
        data = convert_dictionary_to_data({"time":[1, 4], "displacement":[0,4]})
        ifile._set_time_parameters_to_loading_function(data, 1)
        self.assertEqual(ifile._transient_params_1.start_time, 1)
        self.assertEqual(ifile._transient_params_1.termination_time, 1+3/300*1e-3)
        self.assertEqual(ifile._transient_params_1.time_increment, 3/300*1e-3)
        self.assertEqual(ifile._transient_params_2.start_time, 1+3/300*1e-3)
        self.assertEqual(ifile._transient_params_2.termination_time, 4.0)
        self.assertEqual(ifile._transient_params_2.time_increment, 3.0/300)

    def test_coupling_set_time_parameters_to_loading_function_with_scale_factor(self):
        ifile = self._make_input_deck()
        self.assertIsNone(ifile._transient_params_1)
        self.assertIsNone(ifile._transient_params_2)
        ifile._activate_thermal_coupling(1,1,1,"plastic_work")
        self.assertEqual(ifile._transient_params_1.start_time, 0)
        self.assertEqual(ifile._transient_params_1.termination_time, 1.0/300*1e-3)
        self.assertEqual(ifile._transient_params_1.time_increment, 1.0/300*1e-3)
        self.assertEqual(ifile._transient_params_2.start_time, 1.0/300*1e-3)
        self.assertEqual(ifile._transient_params_2.termination_time, 1.0)
        self.assertEqual(ifile._transient_params_2.time_increment, 1.0/300)
        data = convert_dictionary_to_data({"time":[1, 4], "displacement":[0,4]})
        ifile._set_time_parameters_to_loading_function(data, 2)
        self.assertEqual(ifile._transient_params_1.start_time, 2)
        self.assertEqual(ifile._transient_params_1.termination_time, 2+6/300*1e-3)
        self.assertEqual(ifile._transient_params_1.time_increment, 6/300*1e-3)
        self.assertEqual(ifile._transient_params_2.start_time, 2+6/300*1e-3)
        self.assertEqual(ifile._transient_params_2.termination_time, 8.0)
        self.assertEqual(ifile._transient_params_2.time_increment, 6.0/300)

    def test_coupling_set_time_steps(self):
        ifile = self._make_input_deck()
        ifile._activate_thermal_coupling(1,1,1,"plastic_work")
        ifile._set_number_of_time_steps(1000)
        self.assertEqual(ifile._transient_params_1.start_time, 0)
        self.assertEqual(ifile._transient_params_1.termination_time, 1.0/1000*1e-3)
        self.assertEqual(ifile._transient_params_1.time_increment, 1.0/1000*1e-3)
        self.assertEqual(ifile._transient_params_2.start_time, 1.0/1000*1e-3)
        self.assertEqual(ifile._transient_params_2.termination_time, 1.0)
        self.assertEqual(ifile._transient_params_2.time_increment, 1.0/1000)
    
    def test_add_thermal_finite_element_parameters(self):
        ifile = self._make_input_deck()
        ifile._activate_thermal_coupling(1,1,1,"plastic_work")
        ifile._add_thermal_finite_element_parameters("block3", "block4")
        self.assertIn("block1 block2", ifile._thermal_model.subblocks)
        self.assertIn("block3 block4", ifile._thermal_model.subblocks)
    
    def test_set_initial_temperature_from_parameters(self):
        ifile = self._make_input_deck()
        ifile._set_initial_temperature_from_parameters({TEMPERATURE_KEY:100})
        ifile._activate_thermal_coupling(1,1,1,"plastic_work")
        ifile._set_initial_temperature_from_parameters({TEMPERATURE_KEY:100})
        self.assertIn("BC Const Dirichlet at dirichlet_bc1 Temperature", 
                      ifile._thermal_region.lines )
        self.assertIn("BC Const Dirichlet at dirichlet_bc2 Temperature", 
                      ifile._thermal_region.lines )
        thermal_region = ifile._thermal_region
        self.assertIn("IC const on all_blocks TEMPERATURE", thermal_region.lines)
        self.assertEqual(thermal_region.get_line_value("IC const on all_blocks TEMPERATURE"), 100)

    def test_activate_death_after_activate_coupling(self):
        ifile = self._make_input_deck()
        ifile._activate_thermal_coupling(1,1,1,"plastic_work")
        self.assertIsNone(ifile._death_transfer)
        ifile._activate_element_death()
        self.assertIsNotNone(ifile._death_transfer)
        self.assertIn(ifile._death_transfer, ifile._coupled_procedure.subblocks.values())
        self.assertIn(ifile._death_transfer.name, ifile._transient1.lines)
        self.assertIn(ifile._death_transfer.name, ifile._transient2.lines)

    def test_activate_death_get_string_twice(self):
        ifile = self._make_input_deck()
        ifile._activate_thermal_coupling(1,1,1,"plastic_work")
        self.assertIsNone(ifile._death_transfer)
        ifile._activate_element_death()
        ifile.get_string()
        ifile.get_string()
        self.assertIsNotNone(ifile._death_transfer)
        self.assertIn(ifile._death_transfer, ifile._coupled_procedure.subblocks.values())
        self.assertIn(ifile._death_transfer.name, ifile._transient1.lines)
        self.assertIn(ifile._death_transfer.name, ifile._transient2.lines)

    def test_activate_death_before_activate_coupling(self):
        ifile = self._make_input_deck()
        ifile._activate_element_death()
        ifile._activate_thermal_coupling(1,1,1,"plastic_work")
        self.assertIsNotNone(ifile._death_transfer)
        self.assertIn(ifile._death_transfer, ifile._coupled_procedure.subblocks.values())
        self.assertIn(ifile._death_transfer.name, ifile._transient1.lines)
        self.assertIn(ifile._death_transfer.name, ifile._transient2.lines)

    def test_update_death(self):
        ifile = self._make_input_deck()
        ifile._activate_element_death()
        ifile._activate_thermal_coupling(1,1,1,"plastic_work")
        ifile._activate_element_death("eqps", 1)
        self.assertIsNotNone(ifile._death_transfer)
        self.assertIn(ifile._death_transfer, ifile._coupled_procedure.subblocks.values())
        self.assertIn(ifile._death_transfer.name, ifile._transient1.lines)
        self.assertIn(ifile._death_transfer.name, ifile._transient2.lines)

    def test_activate_iterative_coupling(self):
        ifile = self._make_input_deck()
        ifile._activate_thermal_coupling(1,1,1,"work_var") 
        self.assertEqual(ifile.coupling, _Coupling.staggered)

        ifile._activate_iterative_coupling()

        self.assertIn("converge_step_1", ifile._coupled_procedure._solution_control.subblocks)
        self.assertIn("converge_step_2", ifile._coupled_procedure._solution_control.subblocks)
        self.assertEqual(ifile._transient1._nonlinear_step_name, "converge_step_1")
        self.assertEqual(ifile._transient2._nonlinear_step_name, "converge_step_2")
        self.assertEqual(ifile.coupling, _Coupling.iterative)

    def test_activate_iterative_coupling_get_string_twice(self):
        ifile = self._make_input_deck()
        ifile._activate_thermal_coupling(1,1,1,"work_var") 
        self.assertEqual(ifile.coupling, _Coupling.staggered)

        ifile._activate_iterative_coupling()
        ifile.get_input_string()
        ifile.get_input_string()
        
        self.assertIn("converge_step_1", ifile._coupled_procedure._solution_control.subblocks)
        self.assertIn("converge_step_2", ifile._coupled_procedure._solution_control.subblocks)
        self.assertEqual(ifile._transient1._nonlinear_step_name, "converge_step_1")
        self.assertEqual(ifile._transient2._nonlinear_step_name, "converge_step_2")
        self.assertEqual(ifile.coupling, _Coupling.iterative)


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

class TestSierraFileThreeDimensionalContact(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)

    def _make_input_deck(self):
        mat_filename = "matfile.inc"
        with open(mat_filename, "w") as f:
            f.write("material...")
        mat = Material("test_mat", mat_filename, "j2_plasticity")
        ifile = SierraFileThreeDimensionalContact(mat, ["block_to_kill"])
        ifile._add_solid_mechanics_finite_element_parameters("test_mat", 
                                                             "j2_plasticity", 
                                                             "block1")
        ifile._set_local_mesh_filename("test.g")

        return ifile

    def test_activate_self_contact(self):
        ifile = self._make_input_deck()
        self.assertIsNone(ifile._contact_definitions)
        self.assertIsNone(ifile._friction_model)
        self.assertIsNone(ifile._control_contact)
        self.assertIsNone(ifile._contact_acceptable_residual)
        self.assertEqual(ifile._contact_target_relative_residual, 1e-3)
        self.assertEqual(ifile._contact_target_residual, 1e-2)
        self.assertEqual(ifile._contact_acceptable_relative_residual, 1e-2)

        ifile._activate_self_contact(0.1)
        self.assertEqual(ifile._friction_model.get_friction_coefficient(), 0.1)
        self.assertIn(ifile._load_step_predictor.name, ifile._solver.subblocks)        
        self.assertIn(ifile._friction_model.name, ifile._contact_definitions.subblocks)        
        self.assertIn(ifile._contact_definitions.name, ifile.solid_mechanics_region.subblocks)
        self.assertIn(ifile._control_contact.name, ifile._solver.subblocks)
        self.assertEqual(ifile._control_contact.get_target_relative_residual(), 1e-3)
        self.assertEqual(ifile._control_contact.get_target_residual(), 1e-2)
        self.assertEqual(ifile._control_contact.get_acceptable_relative_residual(), 1e-2)
        self.assertIsNone(ifile._control_contact.get_acceptable_residual())

        self.assertEqual(ifile._cg.get_target_relative_residual(), 1e-4)
        self.assertEqual(ifile._cg.get_target_residual(), 1e-2)
        self.assertEqual(ifile._cg.get_acceptable_relative_residual(), 10)
        self.assertIsNone(ifile._cg.get_acceptable_residual())

    def test_set_contact_convergence_tolerance_before_activate_contact(self):
        ifile = self._make_input_deck()
        ifile._set_contact_convergence_tolerance(1e-4, 1e-3, 5e-2, 5e-1)
        self.assertEqual(ifile._contact_acceptable_residual, 5e-1)
        self.assertEqual(ifile._contact_target_relative_residual, 1e-4)
        self.assertEqual(ifile._contact_target_residual, 1e-3)
        self.assertEqual(ifile._contact_acceptable_relative_residual, 5e-2)
       
        ifile._activate_self_contact(0.1)
        
        self.assertEqual(ifile._control_contact.get_target_relative_residual(), 1e-4)
        self.assertEqual(ifile._control_contact.get_target_residual(), 1e-3)
        self.assertEqual(ifile._control_contact.get_acceptable_relative_residual(), 5e-2)
        self.assertEqual(ifile._control_contact.get_acceptable_residual(), 5e-1)

        self.assertEqual(ifile._cg.get_target_relative_residual(), 1e-5)
        self.assertEqual(ifile._cg.get_target_residual(), 1e-3)
        self.assertEqual(ifile._cg.get_acceptable_relative_residual(), 10)
        self.assertIsNone(ifile._cg.get_acceptable_residual())
    
    def test_set_contact_convergence_tolerance_after_activate_contact(self):
        ifile = self._make_input_deck()
       
        ifile._activate_self_contact(0.1)
        ifile._set_contact_convergence_tolerance(1e-4, 1e-3, 5e-2, 5e-1)
        self.assertEqual(ifile._contact_acceptable_residual, 5e-1)
        self.assertEqual(ifile._contact_target_relative_residual, 1e-4)
        self.assertEqual(ifile._contact_target_residual, 1e-3)
        self.assertEqual(ifile._contact_acceptable_relative_residual, 5e-2)

        self.assertEqual(ifile._control_contact.get_target_relative_residual(), 1e-4)
        self.assertEqual(ifile._control_contact.get_target_residual(), 1e-3)
        self.assertEqual(ifile._control_contact.get_acceptable_relative_residual(), 5e-2)
        self.assertEqual(ifile._control_contact.get_acceptable_residual(), 5e-1)

        self.assertEqual(ifile._cg.get_target_relative_residual(), 1e-5)
        self.assertEqual(ifile._cg.get_target_residual(), 1e-3)
        self.assertEqual(ifile._cg.get_acceptable_relative_residual(), 10)
        self.assertIsNone(ifile._cg.get_acceptable_residual())

    def test_activate_self_contact_twice_updates_friction_coeff(self):
        ifile = self._make_input_deck()
        self.assertIsNone(ifile._contact_definitions)
        self.assertIsNone(ifile._friction_model)
        self.assertIsNone(ifile._control_contact)
        self.assertIsNone(ifile._contact_acceptable_residual)
        self.assertEqual(ifile._contact_target_relative_residual, 1e-3)
        self.assertEqual(ifile._contact_target_residual, 1e-2)
        self.assertEqual(ifile._contact_acceptable_relative_residual, 1e-2)

        ifile._activate_self_contact(0.1)
        self.assertEqual(ifile._friction_model.get_friction_coefficient(), 0.1)
        ifile._activate_self_contact(0.2)
        self.assertEqual(ifile._friction_model.get_friction_coefficient(), 0.2)
        