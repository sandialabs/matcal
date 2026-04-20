# test_coupling.py (PART 1/2)

from matcal.core.constants import TEMPERATURE_KEY
from matcal.core.data import convert_dictionary_to_data
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest

from matcal.sierra.input_file_writer.blocks_base import _get_default_coupled_procedure_name
from matcal.sierra.input_file_writer.coupling import (
    _Coupling,
    ArpeggioTransfer,
    CoupledTransientParameters,
    CoupledTransient,
    CoupledSystem,
    CoupledInitialize,
    NonlinearParameters,
    Procedure,
    SolutionControl,
)
from matcal.sierra.input_file_writer.materials import ThermalMaterial
from matcal.sierra.input_file_writer.regions_models import _FiniteElementModelNames
from matcal.sierra.input_file_writer.solvers import TpetraSolver
from matcal.sierra.input_file_writer.sierra_file import SierraFileWithCoupling
from matcal.sierra.material import Material


class TestArpeggioTransfer(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)

    def test_arpeggio_transfer_block_nodal_field_to_send(self):
        input_block = ArpeggioTransfer("test")
        with self.assertRaises(ValueError):
            input_block.get_string()

        input_block.add_field_to_send("displacement", "solution->mesh_displacements")
        input_block.add_field_to_send(
            "displacement",
            "solution->mesh_displacements",
            sending_state="new",
            receiving_state="old",
        )
        with self.assertRaises(ValueError):
            # needs copy keyword - next line outside of this with
            input_block.get_string()

        input_block.set_nodal_copy_transfer("solid_mechanics_region", "thermal_region")
        test_str = input_block.get_string()
        self.assertTrue("Begin transfer test" in test_str)

        send_state_none_str = (
            "send field displacement state none to "
            + "solution->mesh_displacements state none"
        )
        self.assertTrue(send_state_none_str in test_str)

        send_state_new_old_str = (
            "send field displacement state new to "
            + "solution->mesh_displacements state old"
        )
        self.assertTrue(send_state_new_old_str in test_str)

    def test_arpeggio_transfer_block_element_field_to_send(self):
        input_block = ArpeggioTransfer("test")
        with self.assertRaises(ValueError):
            input_block.get_string()

        input_block.add_field_to_send("plastic_work_variable", "plastic_work_variable")

        with self.assertRaises(ValueError):
            # needs copy keyword - next line outside of this with
            input_block.get_string()

        input_block.set_element_copy_transfer("solid_mechanics_region", "thermal_region")
        test_str = input_block.get_string()
        self.assertTrue("Begin transfer test" in test_str)

        send_state_none_str = (
            "send field plastic_work_variable state none to "
            + "plastic_work_variable state none"
        )
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
        input_block.get_string()


class TestCoupledTransientParameters(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)

    def _get_arpeggio_parameter_block(self, name):
        return CoupledTransientParameters(name, "thermal_region", "adagio_region", 0, 1, 0.01)

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
        self.assertEqual(input_block.start_time, 0)
        input_block.set_start_time(0.5)
        self.assertEqual(input_block.start_time, 0.5)

    def test_arpeggio_transient_parameter_block_set_termination_time(self):
        input_block = self._get_arpeggio_parameter_block("test")
        self.assertEqual(input_block.termination_time, 1)
        input_block.set_termination_time(0.5)
        self.assertEqual(input_block.termination_time, 0.5)

    def test_arpeggio_transient_parameter_block_set_time_increment(self):
        input_block = self._get_arpeggio_parameter_block("test")
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
        solid_time_param_block = input_block.subblocks["adagio_region"]
        solid_time_param_block.set_time_increment(0.1)
        with self.assertRaises(ValueError):
            _ = input_block.time_increment


class TestCoupledTransient(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)

    def test_arpeggio_transient_block(self):
        input_block = CoupledTransient("transient_test", "solid_mechanics_region", "thermal_region")
        input_block.add_transfer_post_solid_mechanics("solid_mechanics_to_thermal_disps")
        input_block.add_transfer_post_thermal("thermal_to_solid_mechanics")

        test_strs = input_block.get_string().split("\n")
        self.assertTrue("transient transient_test" in test_strs[0])
        self.assertTrue("advance solid_mechanics_region" in test_strs[1])
        self.assertTrue("transfer solid_mechanics_to_thermal_disps" in test_strs[2])
        self.assertTrue("advance thermal_region" in test_strs[3])
        self.assertTrue("transfer thermal_to_solid_mechanics" in test_strs[4])

    def test_arpeggio_transient_block_nonlinear(self):
        input_block = CoupledTransient("transient_test", "solid_mechanics_region", "thermal_region")
        input_block.set_nonlinear_step_name("converge_step_1")
        input_block.add_transfer_post_solid_mechanics("solid_mechanics_to_thermal_disps")
        input_block.add_transfer_post_thermal("thermal_to_solid_mechanics")
        test_strs = input_block.get_string().split("\n")
        self.assertTrue("nonlinear converge_step_1" in test_strs[2])
        self.assertTrue("advance solid_mechanics_region" in test_strs[3])
        self.assertTrue("transfer solid_mechanics_to_thermal_disps" in test_strs[4])

    def test_arpeggio_transient_block_nonlinear_get_string_twice(self):
        input_block = CoupledTransient("transient_test", "solid_mechanics_region", "thermal_region")
        input_block.set_nonlinear_step_name("converge_step_1")
        input_block.add_transfer_post_solid_mechanics("solid_mechanics_to_thermal_disps")
        input_block.add_transfer_post_thermal("thermal_to_solid_mechanics")
        _ = input_block.get_string().split("\n")
        test_strs = input_block.get_string().split("\n")
        self.assertTrue("nonlinear converge_step_1" in test_strs[2])
        self.assertTrue("advance solid_mechanics_region" in test_strs[3])
        self.assertTrue("transfer solid_mechanics_to_thermal_disps" in test_strs[4])

    def test_arpeggio_transient_block_set_nonlinear(self):
        input_block = CoupledTransient(
            "transient_test", "solid_mechanics_region", "thermal_region", "converge_step_1"
        )
        input_block.add_transfer_post_solid_mechanics("solid_mechanics_to_thermal_disps")
        input_block.add_transfer_post_thermal("thermal_to_solid_mechanics")
        test_strs = input_block.get_string().split("\n")
        self.assertTrue("nonlinear converge_step_1" in test_strs[2])
        self.assertTrue("advance solid_mechanics_region" in test_strs[3])
        self.assertTrue("transfer solid_mechanics_to_thermal_disps" in test_strs[4])


class TestCoupledSystemAndInitialize(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)

    def _get_system_block(self):
        transient_1 = CoupledTransient("transient_test", "solid_mechanics_region", "thermal_region")
        transient_1.add_transfer_post_solid_mechanics("solid_mechanics_to_thermal_disps")
        transient_1.add_transfer_post_thermal("thermal_to_solid_mechanics")

        transient_2 = CoupledTransient("transient_test2", "solid_mechanics_region", "thermal_region")
        transient_2.add_transfer_post_solid_mechanics("solid_mechanics_to_thermal_disps")
        transient_2.add_transfer_post_thermal("thermal_to_solid_mechanics")

        return CoupledSystem("main", "initialization", transient_1, transient_2)

    def test_arpeggio_system_block(self):
        input_block = self._get_system_block()
        test_str = input_block.get_string()
        self.assertTrue("use initialize initialization" in test_str)
        self.assertTrue("transient_test" in input_block.subblocks)
        self.assertTrue("transient_test2" in input_block.subblocks)

    def _get_initialize_block(self):
        input_block = CoupledInitialize("initialization", "solid_mechanics_region", "thermal_region")
        input_block.add_transfer_post_solid_mechanics("solid_mechanics_to_thermal_disps")
        input_block.add_transfer_post_thermal("thermal_to_solid_mechanics")
        return input_block

    def test_arpeggio_initialize_block(self):
        input_block = self._get_initialize_block()
        test_strs = input_block.get_string().split("\n")
        self.assertTrue("initialize initialization" in test_strs[0])


class TestNonlinearParametersSolutionControlProcedure(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)

    def test_nonlinear_parameters_block(self):
        input_block = NonlinearParameters("test")
        test_string = input_block.get_string()
        goal_string = (
            'converged when "thermal_region.MaxInitialNonlinearResidual(0)'
            ' < 1.0e-8  || CURRENT_STEP > 20"'
        )
        self.assertEqual(goal_string, test_string.split("\n")[1].strip())

    def _get_arpeggio_parameter_block(self, name):
        return CoupledTransientParameters(name, "thermal_region", "adagio_region", 0, 1, 0.01)

    def _get_system_block(self):
        transient_1 = CoupledTransient("transient_test", "solid_mechanics_region", "thermal_region")
        transient_1.add_transfer_post_solid_mechanics("solid_mechanics_to_thermal_disps")
        transient_1.add_transfer_post_thermal("thermal_to_solid_mechanics")

        transient_2 = CoupledTransient("transient_test2", "solid_mechanics_region", "thermal_region")
        transient_2.add_transfer_post_solid_mechanics("solid_mechanics_to_thermal_disps")
        transient_2.add_transfer_post_thermal("thermal_to_solid_mechanics")

        return CoupledSystem("main", "initialization", transient_1, transient_2)

    def _get_initialize_block(self):
        input_block = CoupledInitialize("initialization", "solid_mechanics_region", "thermal_region")
        input_block.add_transfer_post_solid_mechanics("solid_mechanics_to_thermal_disps")
        input_block.add_transfer_post_thermal("thermal_to_solid_mechanics")
        return input_block

    def _get_solution_control_block(self):
        sys_block = self._get_system_block()
        init = self._get_initialize_block()
        sltn_ctl = SolutionControl(
            "coupling", sys_block, init, self._get_arpeggio_parameter_block("test")
        )
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
        tnsfr.add_field_to_send(
            "displacement",
            "solution->mesh_displacements",
            sending_state="new",
            receiving_state="old",
        )
        tnsfr.set_nodal_copy_transfer("solid_mechanics_region", "thermal_region")

        tnsfr2 = ArpeggioTransfer("test2")
        tnsfr2.add_field_to_send(
            "displacement",
            "solution->mesh_displacements",
            sending_state="new",
            receiving_state="old",
        )
        tnsfr2.set_nodal_copy_transfer("solid_mechanics_region", "thermal_region")

        procedure_blk = Procedure(sltn_ctl, tnsfr, tnsfr2)

        test_str = procedure_blk.get_string()
        self.assertTrue("Begin transfer test" in test_str)
        self.assertTrue("Begin transfer test2" in test_str)
        self.assertTrue("Begin solution control description coupling" in test_str)


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
        ifile._add_solid_mechanics_finite_element_parameters(
            "test_mat", "j2_plasticity", "block1", "block2"
        )
        return ifile

    def test_activate_thermal_coupling(self):
        ifile = self._make_input_deck()
        self.assertIsNone(ifile.coupling)
        self.assertIsNone(ifile._coupled_procedure)
        self.assertIsNone(ifile._thermal_material)
        self.assertIsNone(ifile._thermal_model)
        self.assertIsNone(ifile._thermal_region)
        self.assertEqual(ifile._coupling_transfers, [])

        ifile._activate_thermal_coupling(1, 1, 1, "plastic_work")

        self.assertIn(TpetraSolver().name, ifile.subblocks)
        self.assertIn(_FiniteElementModelNames.thermal, ifile.subblocks)
        self.assertIn(ThermalMaterial(1, 1, 1).name, ifile.subblocks)

        self.assertIn(_get_default_coupled_procedure_name(), ifile.subblocks)
        self.assertEqual(len(ifile._coupling_transfers), 5)

    def test_activate_thermal_coupling_update_mesh(self):
        ifile = self._make_input_deck()
        ifile._activate_thermal_coupling(1, 1, 1, "plastic_work")
        self.assertEqual(
            ifile._thermal_model.mesh_filename, ifile._sm_finite_element_model.mesh_filename
        )
        self.assertEqual(ifile._thermal_model.mesh_filename, "test.g")

        ifile._set_local_mesh_filename("test_2.g")
        self.assertEqual(
            ifile._thermal_model.mesh_filename, ifile._sm_finite_element_model.mesh_filename
        )
        self.assertEqual(ifile._thermal_model.mesh_filename, "test_2.g")

    def test_activate_thermal_coupling_get_string_twice(self):
        ifile = self._make_input_deck()
        ifile._activate_thermal_coupling(1, 1, 1, "plastic_work")
        ifile.get_input_string()
        ifile.get_input_string()
        self.assertEqual(
            ifile._thermal_model.mesh_filename, ifile._sm_finite_element_model.mesh_filename
        )
        self.assertEqual(ifile._thermal_model.mesh_filename, "test.g")

        ifile._set_local_mesh_filename("test_2.g")
        self.assertEqual(
            ifile._thermal_model.mesh_filename, ifile._sm_finite_element_model.mesh_filename
        )
        self.assertEqual(ifile._thermal_model.mesh_filename, "test_2.g")

    def test_activate_thermal_coupling_update_element(self):
        ifile = self._make_input_deck()
        ifile._activate_thermal_coupling(1, 1, 1, "plastic_work")
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
        ifile._activate_thermal_coupling(1, 1, 1, "plastic_work")
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
        ifile._activate_thermal_coupling(1, 1, 1, "plastic_work")
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
        ifile._activate_thermal_coupling(1, 1, 1, "plastic_work")
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
        ifile._activate_thermal_coupling(1, 1, 1, "plastic_work")

        self.assertEqual(ifile._transient_params_1.start_time, 0)
        self.assertEqual(ifile._transient_params_1.termination_time, 1.0 / 300 * 1e-3)
        self.assertEqual(ifile._transient_params_1.time_increment, 1.0 / 300 * 1e-3)
        self.assertEqual(ifile._transient_params_2.start_time, 1.0 / 300 * 1e-3)
        self.assertEqual(ifile._transient_params_2.termination_time, 1.0)
        self.assertEqual(ifile._transient_params_2.time_increment, 1.0 / 300)

        data = convert_dictionary_to_data({"time": [1, 4], "displacement": [0, 4]})
        ifile._set_time_parameters_to_loading_function(data, 1)
        self.assertEqual(ifile._transient_params_1.start_time, 1)
        self.assertEqual(ifile._transient_params_1.termination_time, 1 + 3 / 300 * 1e-3)
        self.assertEqual(ifile._transient_params_1.time_increment, 3 / 300 * 1e-3)
        self.assertEqual(ifile._transient_params_2.start_time, 1 + 3 / 300 * 1e-3)
        self.assertEqual(ifile._transient_params_2.termination_time, 4.0)
        self.assertEqual(ifile._transient_params_2.time_increment, 3.0 / 300)

    def test_coupling_set_time_parameters_to_loading_function_with_scale_factor(self):
        ifile = self._make_input_deck()
        self.assertIsNone(ifile._transient_params_1)
        self.assertIsNone(ifile._transient_params_2)
        ifile._activate_thermal_coupling(1, 1, 1, "plastic_work")

        data = convert_dictionary_to_data({"time": [1, 4], "displacement": [0, 4]})
        ifile._set_time_parameters_to_loading_function(data, 2)
        self.assertEqual(ifile._transient_params_1.start_time, 2)
        self.assertEqual(ifile._transient_params_1.termination_time, 2 + 6 / 300 * 1e-3)
        self.assertEqual(ifile._transient_params_1.time_increment, 6 / 300 * 1e-3)
        self.assertEqual(ifile._transient_params_2.start_time, 2 + 6 / 300 * 1e-3)
        self.assertEqual(ifile._transient_params_2.termination_time, 8.0)
        self.assertEqual(ifile._transient_params_2.time_increment, 6.0 / 300)

    def test_coupling_set_time_steps(self):
        ifile = self._make_input_deck()
        ifile._activate_thermal_coupling(1, 1, 1, "plastic_work")
        ifile._set_number_of_time_steps(1000)
        self.assertEqual(ifile._transient_params_1.start_time, 0)
        self.assertEqual(ifile._transient_params_1.termination_time, 1.0 / 1000 * 1e-3)
        self.assertEqual(ifile._transient_params_1.time_increment, 1.0 / 1000 * 1e-3)
        self.assertEqual(ifile._transient_params_2.start_time, 1.0 / 1000 * 1e-3)
        self.assertEqual(ifile._transient_params_2.termination_time, 1.0)
        self.assertEqual(ifile._transient_params_2.time_increment, 1.0 / 1000)

    def test_add_thermal_finite_element_parameters(self):
        ifile = self._make_input_deck()
        ifile._activate_thermal_coupling(1, 1, 1, "plastic_work")
        ifile._add_thermal_finite_element_parameters("block3", "block4")
        self.assertIn("block1 block2", ifile._thermal_model.subblocks)
        self.assertIn("block3 block4", ifile._thermal_model.subblocks)

    def test_set_initial_temperature_from_parameters(self):
        ifile = self._make_input_deck()
        ifile._set_initial_temperature_from_parameters({TEMPERATURE_KEY: 100})
        ifile._activate_thermal_coupling(1, 1, 1, "plastic_work")
        ifile._set_initial_temperature_from_parameters({TEMPERATURE_KEY: 100})
        self.assertIn("BC Const Dirichlet at dirichlet_bc1 Temperature", ifile._thermal_region.lines)
        self.assertIn("BC Const Dirichlet at dirichlet_bc2 Temperature", ifile._thermal_region.lines)

        thermal_region = ifile._thermal_region
        self.assertIn("IC const on all_blocks TEMPERATURE", thermal_region.lines)
        self.assertEqual(
            thermal_region.get_line_value("IC const on all_blocks TEMPERATURE"),
            100,
        )

    def test_activate_death_after_activate_coupling(self):
        ifile = self._make_input_deck()
        ifile._activate_thermal_coupling(1, 1, 1, "plastic_work")
        self.assertIsNone(ifile._death_transfer)
        ifile._activate_element_death()
        self.assertIsNotNone(ifile._death_transfer)
        self.assertIn(ifile._death_transfer, ifile._coupled_procedure.subblocks.values())
        self.assertIn(ifile._death_transfer.name, ifile._transient1.lines)
        self.assertIn(ifile._death_transfer.name, ifile._transient2.lines)

    def test_activate_death_get_string_twice(self):
        ifile = self._make_input_deck()
        ifile._activate_thermal_coupling(1, 1, 1, "plastic_work")
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
        ifile._activate_thermal_coupling(1, 1, 1, "plastic_work")
        self.assertIsNotNone(ifile._death_transfer)
        self.assertIn(ifile._death_transfer, ifile._coupled_procedure.subblocks.values())
        self.assertIn(ifile._death_transfer.name, ifile._transient1.lines)
        self.assertIn(ifile._death_transfer.name, ifile._transient2.lines)

    def test_update_death(self):
        ifile = self._make_input_deck()
        ifile._activate_element_death()
        ifile._activate_thermal_coupling(1, 1, 1, "plastic_work")
        ifile._activate_element_death("eqps", 1)
        self.assertIsNotNone(ifile._death_transfer)
        self.assertIn(ifile._death_transfer, ifile._coupled_procedure.subblocks.values())
        self.assertIn(ifile._death_transfer.name, ifile._transient1.lines)
        self.assertIn(ifile._death_transfer.name, ifile._transient2.lines)

    def test_activate_iterative_coupling(self):
        ifile = self._make_input_deck()
        ifile._activate_thermal_coupling(1, 1, 1, "work_var")
        self.assertEqual(ifile.coupling, _Coupling.staggered)

        ifile._activate_iterative_coupling()

        self.assertIn("converge_step_1", ifile._coupled_procedure._solution_control.subblocks)
        self.assertIn("converge_step_2", ifile._coupled_procedure._solution_control.subblocks)
        self.assertEqual(ifile._transient1._nonlinear_step_name, "converge_step_1")
        self.assertEqual(ifile._transient2._nonlinear_step_name, "converge_step_2")
        self.assertEqual(ifile.coupling, _Coupling.iterative)

    def test_activate_iterative_coupling_get_string_twice(self):
        ifile = self._make_input_deck()
        ifile._activate_thermal_coupling(1, 1, 1, "work_var")
        self.assertEqual(ifile.coupling, _Coupling.staggered)

        ifile._activate_iterative_coupling()
        ifile.get_input_string()
        ifile.get_input_string()

        self.assertIn("converge_step_1", ifile._coupled_procedure._solution_control.subblocks)
        self.assertIn("converge_step_2", ifile._coupled_procedure._solution_control.subblocks)
        self.assertEqual(ifile._transient1._nonlinear_step_name, "converge_step_1")
        self.assertEqual(ifile._transient2._nonlinear_step_name, "converge_step_2")
        self.assertEqual(ifile.coupling, _Coupling.iterative)