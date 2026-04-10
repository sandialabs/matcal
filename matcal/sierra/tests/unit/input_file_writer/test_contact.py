from matcal.core.tests.MatcalUnitTest import MatcalUnitTest

from matcal.sierra.input_file_writer.contact import (
    SolidMechanicsInteractionDefaults,
    SolidMechanicsConstantFrictionModel,
    SolidMechanicsRemoveInitialOverlap,
    SolidMechanicsContactDefinitions,
)
from matcal.sierra.input_file_writer.sierra_file import SierraFileThreeDimensionalContact
from matcal.sierra.material import Material


class TestContactBlocks(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)

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
        fric_coeff = input_block.get_friction_coefficient()
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


class TestSierraFileThreeDimensionalContact(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)

    def _make_input_deck(self):
        mat_filename = "matfile.inc"
        with open(mat_filename, "w") as f:
            f.write("material...")
        mat = Material("test_mat", mat_filename, "j2_plasticity")
        ifile = SierraFileThreeDimensionalContact(mat, ["block_to_kill"])
        ifile._add_solid_mechanics_finite_element_parameters("test_mat", "j2_plasticity", "block1")
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