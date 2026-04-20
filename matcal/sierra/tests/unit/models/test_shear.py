# test_shear.py
from matcal.core.constants import TORQUE_KEY, ROTATION_KEY
from matcal.core.data import convert_dictionary_to_data
from matcal.core.state import SolitaryState

from matcal.sierra.tests.unit.models.model_tests_base import (
    MatcalThreeDimensionalStandardModelUnitTestNewBase
)
from matcal.sierra.tests.sierra_sm_models_for_tests import (
    SolidBarTorsionModelForTests,
    TopHatShearModelForTests,
)


class SolidBarTorsionModelUnitTests(
    MatcalThreeDimensionalStandardModelUnitTestNewBase.CommonTests,
    SolidBarTorsionModelForTests,
):
    _load_var = TORQUE_KEY
    _displacement_var = ROTATION_KEY
    _displacement_user_output_block_name = "global_torque_rotation"
    _load_user_output_block_name = "global_torque_rotation"

    def test_specific_outputs_added(self):
        model = self.init_model()
        data = convert_dictionary_to_data({self._displacement_var: [0, 1]})
        model.add_boundary_condition_data(data)
        model._setup_state(SolitaryState(), build_mesh=False)

        sm_region = model._input_file.solid_mechanics_region
        lines = sm_region.subblocks[self._displacement_user_output_block_name].lines
        self.assertIn("global applied_rotation", lines)

        hb_output = model._input_file.heartbeat_output
        self.assertTrue(hb_output.has_global_output("applied_rotation"))

    def test_prescribed_zero_displacement_added(self):
        model = self.init_model()
        data = convert_dictionary_to_data({self._displacement_var: [0, 1]})
        model.add_boundary_condition_data(data)
        model._setup_state(SolitaryState(), build_mesh=False)

        sm_region = model._input_file.solid_mechanics_region
        zero_func_block_name = "ns_y_symmetry cylindrical_axis sierra_constant_function_zero"
        self.assertIn(zero_func_block_name, sm_region.subblocks)


class TopHatShearModelUnitTests(
    MatcalThreeDimensionalStandardModelUnitTestNewBase.CommonTests,
    TopHatShearModelForTests,
):
    """"""

    # These are intentionally blank in the original file (placeholders to avoid inheriting
    # full-field tests that TopHat doesn't support). Keep them as no-ops.
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
            model.set_contact_convergence_tolerance(
                1e-6, acceptable_relative_residual=1e-7
            )
        model.set_contact_convergence_tolerance(1e-6, acceptable_relative_residual=1e-4)
        self.assertAlmostEqual(contact.get_acceptable_relative_residual(), 1e-4)

        with self.assertRaises(ValueError):
            model.set_contact_convergence_tolerance(
                1e-8, target_residual=1e-6, acceptable_residual=1e-7
            )
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