from xml.etree import ElementTree

import numpy as np

from matcal.core.parameters import Parameter, ParameterCollection
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest
from matcal.dakota.gradient_convergence import (
    ConvergenceSettings,
    MatCalConvergenceMonitor,
    MatCalConvergenceReached,
    batch_center,
    keywords_to_remove,
    matcal_checked_settings,
    method_specific_lines,
    native_lines,
    rol_options_xml,
    scale_to_bounds,
    total_objective,
)


class TestConvergenceSettings(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)

    def test_defaults_match_previous_deck_defaults(self):
        settings = ConvergenceSettings()
        self.assertEqual(settings.relative, 1e-3)
        self.assertEqual(settings.max_function_evaluations, 1000)
        self.assertIsNone(settings.absolute)
        self.assertIsNone(settings.variable)

    def test_repr_names_values(self):
        self.assertIn("relative=0.001", repr(ConvergenceSettings()))


class TestNativeLines(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)
        self.settings = ConvergenceSettings(
            relative=1e-6, absolute=1e-4, variable=1e-5, max_function_evaluations=50
        )

    def test_nl2sol_translates_absolute_to_half_sum_of_squares(self):
        lines = native_lines("nl2sol", self.settings)
        self.assertEqual(
            lines,
            {
                "convergence_tolerance": 1e-6,
                "absolute_conv_tol": 5e-5,
                "x_conv_tol": 1e-5,
                "max_function_evaluations": 50,
            },
        )

    def test_dot_and_conmin_relative_is_checked_by_matcal(self):
        for method in ["dot_sqp", "dot_slp", "dot_mmfd", "conmin_mfd"]:
            self.assertEqual(
                native_lines(method, self.settings), {"max_function_evaluations": 50}
            )
            checked = matcal_checked_settings(method, self.settings)
            self.assertEqual(checked.relative, 1e-6)

    def test_dot_and_conmin_get_tight_native_tolerance(self):
        for method in ["dot_sqp", "dot_slp", "dot_mmfd", "conmin_mfd"]:
            self.assertEqual(
                method_specific_lines(method), {"convergence_tolerance": 1e-12}
            )
            self.assertNotIn(
                "convergence_tolerance", keywords_to_remove(method, self.settings)
            )

    def test_common_methods_only_relative_and_evaluations(self):
        lines = native_lines("optpp_q_newton", self.settings)
        self.assertEqual(
            lines, {"convergence_tolerance": 1e-6, "max_function_evaluations": 50}
        )

    def test_rol_only_variable(self):
        self.assertEqual(
            native_lines("rol", self.settings), {"variable_tolerance": 1e-5}
        )

    def test_rol_default_variable_tolerance(self):
        self.assertEqual(
            native_lines("rol", ConvergenceSettings()), {"variable_tolerance": 1e-8}
        )

    def test_unset_settings_are_not_written(self):
        lines = native_lines("nl2sol", ConvergenceSettings())
        self.assertNotIn("absolute_conv_tol", lines)
        self.assertNotIn("x_conv_tol", lines)


class TestMatCalCheckedSettings(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)
        self.settings = ConvergenceSettings(
            relative=1e-6, absolute=1e-4, variable=1e-5, max_function_evaluations=50
        )

    def test_nl2sol_needs_no_matcal_checks(self):
        self.assertFalse(matcal_checked_settings("nl2sol", self.settings).is_active())

    def test_conmin_checks_absolute_and_variable(self):
        checked = matcal_checked_settings("conmin_mfd", self.settings)
        self.assertEqual(checked, ConvergenceSettings(1e-6, 1e-4, 1e-5, None))
        self.assertTrue(checked.is_active())

    def test_rol_checks_relative_absolute_and_evaluations(self):
        checked = matcal_checked_settings("rol", self.settings)
        self.assertEqual(checked, ConvergenceSettings(1e-6, 1e-4, None, 50))


class TestKeywordsToRemove(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)

    def test_stale_nl2sol_keywords_removed_for_other_methods(self):
        settings = ConvergenceSettings(absolute=1e-4, variable=1e-5)
        removed = keywords_to_remove("dot_sqp", settings)
        self.assertIn("absolute_conv_tol", removed)
        self.assertIn("x_conv_tol", removed)
        self.assertNotIn("convergence_tolerance", removed)

    def test_rol_removes_unsupported_defaults(self):
        removed = keywords_to_remove("rol", ConvergenceSettings())
        for keyword in [
            "convergence_tolerance",
            "max_function_evaluations",
            "speculative",
        ]:
            self.assertIn(keyword, removed)

    def test_rol_keeps_scaling_and_options_file(self):
        removed = keywords_to_remove("rol", ConvergenceSettings())
        self.assertNotIn("scaling", removed)
        self.assertNotIn("options_file", removed)

    def test_other_methods_remove_rol_options_file(self):
        self.assertIn(
            "options_file", keywords_to_remove("nl2sol", ConvergenceSettings())
        )

    def test_written_keywords_are_kept(self):
        removed = keywords_to_remove("nl2sol", ConvergenceSettings(absolute=1e-4))
        self.assertNotIn("absolute_conv_tol", removed)


class TestBatchHelpers(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)
        self.parameters = ParameterCollection(
            "p", Parameter("m", 0.0, 10.0), Parameter("b", -5.0, 5.0)
        )

    def test_batch_center_is_first_set(self):
        batch = [{"cv": [1.0, 0.5]}, {"cv": [1.00005, 0.5]}, {"cv": [1.0, 0.500275]}]
        self.assert_close_arrays(batch_center(batch), np.array([1.0, 0.5]))

    def test_batch_center_of_single_set(self):
        self.assert_close_arrays(batch_center({"cv": [2.0, 0.0]}), np.array([2.0, 0.0]))

    def test_scale_to_bounds(self):
        scaled = scale_to_bounds(np.array([1.0, 0.0]), self.parameters)
        self.assert_close_arrays(scaled, np.array([0.1, 0.5]))

    def test_scale_to_bounds_keeps_unbounded_values_raw(self):
        parameters = ParameterCollection(
            "p", Parameter("m", -np.inf, np.inf, 1.0), Parameter("b", 0.0, np.inf, 2.0)
        )
        scaled = scale_to_bounds(np.array([3.0, 4.0]), parameters)
        self.assert_close_arrays(scaled, np.array([3.0, 4.0]))

    def test_total_objective_for_residuals_is_sum_of_squares(self):
        self.assertAlmostEqual(total_objective(np.array([1.0, 2.0]), True), 5.0)

    def test_total_objective_for_objectives_is_sum(self):
        self.assertAlmostEqual(total_objective(np.array([1.0, 2.0]), False), 3.0)


class TestMatCalConvergenceMonitor(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)

    def _monitor(self, **settings):
        defaults = {
            "relative": None,
            "absolute": None,
            "variable": None,
            "max_function_evaluations": None,
        }
        defaults.update(settings)
        return MatCalConvergenceMonitor(ConvergenceSettings(**defaults))

    def test_no_settings_never_converges(self):
        monitor = self._monitor()
        self.assertIsNone(monitor.update(np.array([0.1]), 1.0, 3))
        self.assertIsNone(monitor.update(np.array([0.1]), 1.0, 6))

    def test_absolute_tolerance(self):
        monitor = self._monitor(absolute=1e-3)
        self.assertIsNone(monitor.update(np.array([0.1]), 1.0, 3))
        reason = monitor.update(np.array([0.2]), 1e-4, 6)
        self.assertTrue(reason.success)
        self.assertIn("absolute", reason.message)

    def test_absolute_tolerance_on_first_iterate(self):
        reason = self._monitor(absolute=1e-3).update(np.array([0.1]), 0.0, 3)
        self.assertIn("absolute", reason.message)

    def test_relative_tolerance_needs_two_consecutive_small_drops(self):
        monitor = self._monitor(relative=1e-2)
        self.assertIsNone(monitor.update(np.array([0.1]), 1.0, 3))
        self.assertIsNone(monitor.update(np.array([0.2]), 0.5, 6))
        self.assertIsNone(monitor.update(np.array([0.3]), 0.499, 9))
        reason = monitor.update(np.array([0.4]), 0.4985, 12)
        self.assertIn("relative", reason.message)

    def test_relative_count_resets_after_large_drop(self):
        monitor = self._monitor(relative=1e-2)
        monitor.update(np.array([0.1]), 1.0, 3)
        self.assertIsNone(monitor.update(np.array([0.2]), 0.999, 6))
        self.assertIsNone(monitor.update(np.array([0.3]), 0.5, 9))
        self.assertIsNone(monitor.update(np.array([0.4]), 0.4999, 12))

    def test_relative_tolerance_with_zero_objective(self):
        monitor = self._monitor(relative=1e-2)
        monitor.update(np.array([0.1]), 0.0, 3)
        self.assertIsNone(monitor.update(np.array([0.2]), 0.0, 6))

    def test_variable_tolerance(self):
        monitor = self._monitor(variable=1e-3)
        self.assertIsNone(monitor.update(np.array([0.1, 0.5]), 1.0, 3))
        self.assertIsNone(monitor.update(np.array([0.2, 0.5]), 0.5, 6))
        reason = monitor.update(np.array([0.2005, 0.5]), 0.4, 9)
        self.assertIn("variable", reason.message)

    def test_non_finite_change_is_not_convergence(self):
        monitor = self._monitor(variable=1e-3)
        monitor.update(np.array([np.nan]), 1.0, 3)
        self.assertIsNone(monitor.update(np.array([np.nan]), 0.5, 6))

    def test_non_improving_center_is_not_an_iterate(self):
        monitor = self._monitor(variable=1e-3)
        monitor.update(np.array([0.1]), 1.0, 3)
        self.assertIsNone(monitor.update(np.array([0.1001]), 2.0, 6))
        self.assertIsNone(monitor.update(np.array([0.3]), 0.5, 9))

    def test_max_function_evaluations_is_not_success(self):
        monitor = self._monitor(max_function_evaluations=5)
        self.assertIsNone(monitor.update(np.array([0.1]), 1.0, 3))
        reason = monitor.update(np.array([0.2]), 0.5, 6)
        self.assertFalse(reason.success)
        self.assertIn("function evaluations", reason.message)

    def test_reason_as_exception(self):
        reason = self._monitor(absolute=1.0).update(np.array([0.1]), 0.5, 3)
        error = MatCalConvergenceReached(reason)
        self.assertIs(error.reason, reason)
        self.assertIn("absolute", str(error))

    def test_monitor_repr(self):
        monitor = self._monitor(absolute=1.0)
        monitor.update(np.array([0.1]), 2.0, 3)
        self.assertIn("best_objective=2.0", repr(monitor))


class TestRolOptions(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)

    def test_rol_method_specific_lines_point_to_options_file(self):
        self.assertEqual(
            method_specific_lines("rol"), {"options_file": '"matcal_rol_options.xml"'}
        )

    def test_other_methods_have_no_specific_lines(self):
        self.assertEqual(method_specific_lines("nl2sol"), {})

    def test_rol_options_xml_sets_initial_radius(self):
        root = ElementTree.fromstring(rol_options_xml(0.25))
        radius = root.find(".//Parameter[@name='Initial Radius']")
        self.assertEqual(radius.get("type"), "double")
        self.assertEqual(float(radius.get("value")), 0.25)
