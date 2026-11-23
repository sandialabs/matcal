import os

import numpy as np

from matcal.core.data import convert_dictionary_to_data
from matcal.core.models import PythonModel
from matcal.core.parameters import (
    Parameter,
    ParameterCollection,
    UserDefinedParameterPreprocessor,
)
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest
from matcal.core.objective import CurveBasedInterpolatedObjective

from matcal.core.tests.unit.test_study_base import simple_func

from matcal.dakota.input_file_writer import (DakMethodKeys, DakGradientKeys, 
                                             NongradientResponseBlock)
from matcal.dakota.local_calibration_studies import (CobylaCalibrationStudy, 
                                                     GradientCalibrationStudy, 
                                                     MeshAdaptiveSearchCalibrationStudy, 
                                                     ParallelDirectSearchCalibrationStudy, 
                                                     PatternSearchCalibrationStudy, 
                                                     SolisWetsCalibrationStudy, 
                                                     _GradientDakotaFile, 
                                                     LeastSquaresResponseBlock, 
                                                     GradientResponseBlock, 
                                                     _MeshAdaptiveDakotaFile, 
                                                     _ColinyNongradientDakotaFile, 
                                                     _PatternSearchDakotaFile, 
                                                     _GradientMethodKeys,
                                                     _LeastSquaresMethodKeys, 
                                                     _OptppPdsDakotaFile)

from matcal.dakota.tests.unit.test_dakota_studies_base import TestDakotaStudyBase
from matcal.dakota.tests.unit.test_input_file_writer import TestDakotaFileBase


class TestCalibrationDakotaFile:
    def __init__():
        pass
    class CommonTests(TestDakotaFileBase.CommonTests):

        def test_set_convergence_tol(self):
            with self.assertRaises(TypeError):
                self.df.set_convergence_tolerance("100.0")
            with self.assertRaises(ValueError):
                self.df.set_convergence_tolerance(100.0)

            self.df.set_convergence_tolerance(1e-9)
            meth_type_b = self.df.get_method_type_block()
            self.assertEqual(meth_type_b.get_line_value(DakMethodKeys.convergence_tol), 
                             1e-9)
            
        def test_set_max_iterations(self):
            with self.assertRaises(TypeError):
                self.df.set_max_iterations(1e-9)
            with self.assertRaises(ValueError):
                self.df.set_max_iterations(-1)
            self.df.set_max_iterations(100000)

            meth_type_b = self.df.get_method_type_block()
            self.assertEqual(meth_type_b.get_line_value(DakMethodKeys.max_iterations), 
                             100000)

        def test_set_max_function_evals(self):
            with self.assertRaises(TypeError):
                self.df.set_max_function_evaluations(1e-9)
            with self.assertRaises(ValueError):
                self.df.set_max_function_evaluations(-1)
            self.df.set_max_function_evaluations(100000)

            meth_type_b = self.df.get_method_type_block()
            self.assertEqual(meth_type_b.get_line_value(DakMethodKeys.max_func_evals), 
                             100000)


def linear_function(**parameters):
    x = np.linspace(0, 10, 25)
    return {"x": x, "y": parameters["m"] * x + parameters["b"]}


def double_slope_and_add_table(parameters):
    parameters["m"] = 2.0 * parameters["m"]
    parameters["table"] = "0.0, 0.0\n1.0, 2.0"
    return parameters


def _linear_calibration(method, *setters, slope_bounds=(0.0, 10.0),
                        intercept_bounds=(-5.0, 5.0)):
    slope = Parameter("m", *slope_bounds, 1.0)
    intercept = Parameter("b", *intercept_bounds, 0.5)
    study = GradientCalibrationStudy(ParameterCollection("line", slope, intercept))
    study.set_method(method)
    goal = convert_dictionary_to_data(linear_function(m=2.0, b=0.0))
    study.add_evaluation_set(
        PythonModel(linear_function), CurveBasedInterpolatedObjective("x", "y"), goal
    )
    for setter in setters:
        setter(study)
    return study


class TestGradientDakotaFile(TestCalibrationDakotaFile.CommonTests):
    
    _dakota_file_class = _GradientDakotaFile

    def test_init_least_square_methods(self):

        for least_squ_meth in _GradientDakotaFile.least_squares_methods:
            grad_file = self._dakota_file_class()
            grad_file.set_method(least_squ_meth)
            method_type_block = grad_file.get_method_type_block()

            self.assertEqual(method_type_block.name, least_squ_meth)
            resp = grad_file.get_response_block()
            self.assertTrue(isinstance(resp, LeastSquaresResponseBlock))
            
    def test_init_grad_obj_methods(self):
        for grad_meth in _GradientDakotaFile.gradient_methods:

            grad_file = self._dakota_file_class()
            grad_file.set_method(grad_meth)
            method_type_block = grad_file.get_method_type_block()

            self.assertEqual(method_type_block.name, grad_meth)
            resp = grad_file.get_response_block()
            self.assertTrue(isinstance(resp, GradientResponseBlock))

    def test_set_step_size(self):
        grad_file = _GradientDakotaFile()
        grad_file.set_step_size(1e-8)
        grad_block = grad_file.get_gradient_block()
        step_size = grad_block.get_line_value(DakGradientKeys.fd_step_size)
        self.assertEqual(step_size, 1e-8)
        with self.assertRaises(TypeError):
            grad_file.set_step_size("a")
        with self.assertRaises(ValueError):
            grad_file.set_step_size(-1)        
        with self.assertRaises(ValueError):
            grad_file.set_step_size(0.1)        


class TestGradientDakotaFileSharedSettings(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)
        self.grad_file = _GradientDakotaFile()

    def _method_lines(self):
        return self.grad_file.get_method_type_block().lines

    def test_absolute_tolerance_is_native_for_nl2sol(self):
        self.grad_file.set_absolute_convergence_tolerance(1e-4)
        self.assertAlmostEqual(self._method_lines()["absolute_conv_tol"].get_values()[1], 5e-5)

    def test_variable_tolerance_is_native_for_nl2sol(self):
        self.grad_file.set_variable_tolerance(1e-5)
        self.assertEqual(self._method_lines()["x_conv_tol"].get_values()[1], 1e-5)

    def test_matcal_checked_settings_are_not_written(self):
        self.grad_file.set_method("conmin_mfd")
        self.grad_file.set_absolute_convergence_tolerance(1e-4)
        self.grad_file.set_variable_tolerance(1e-5)
        self.assertNotIn("absolute_conv_tol", self._method_lines())
        self.assertNotIn("x_conv_tol", self._method_lines())

    def test_settings_survive_set_method(self):
        self.grad_file.set_absolute_convergence_tolerance(1e-4)
        self.grad_file.set_convergence_tolerance(1e-7)
        self.grad_file.set_method("optpp_q_newton")
        self.assertNotIn("absolute_conv_tol", self._method_lines())
        self.assertEqual(self._method_lines()["convergence_tolerance"].get_values()[1], 1e-7)
        self.grad_file.set_method("nl2sol")
        self.assertIn("absolute_conv_tol", self._method_lines())
        self.assertEqual(self.grad_file.get_convergence_settings().absolute, 1e-4)

    def test_get_convergence_settings_returns_copy(self):
        self.grad_file.get_convergence_settings().absolute = 1.0
        self.assertIsNone(self.grad_file.get_convergence_settings().absolute)

    def test_rol_drops_unsupported_keywords(self):
        self.grad_file.set_method("rol")
        self.grad_file.set_variable_tolerance(1e-6)
        for keyword in ["convergence_tolerance", "max_function_evaluations",
                        "speculative"]:
            self.assertNotIn(keyword, self._method_lines())
        self.assertIn("scaling", self._method_lines())
        self.assertEqual(self._method_lines()["options_file"].get_values()[1],
                         '"matcal_rol_options.xml"')
        self.assertEqual(self._method_lines()["variable_tolerance"].get_values()[1], 1e-6)
        self.assertIn("max_iterations", self._method_lines())

    def test_leaving_rol_restores_common_keywords(self):
        self.grad_file.set_method("rol")
        self.grad_file.set_method("dot_sqp")
        for keyword in ["convergence_tolerance", "max_function_evaluations",
                        "speculative", "scaling"]:
            self.assertIn(keyword, self._method_lines())
        self.assertNotIn("options_file", self._method_lines())
        self.assertNotIn("variable_tolerance", self._method_lines())

    def test_invalid_absolute_tolerance(self):
        with self.assertRaises(TypeError):
            self.grad_file.set_absolute_convergence_tolerance("a")
        with self.assertRaises(ValueError):
            self.grad_file.set_absolute_convergence_tolerance(-1.0)

    def test_invalid_variable_tolerance(self):
        with self.assertRaises(TypeError):
            self.grad_file.set_variable_tolerance("a")
        with self.assertRaises(ValueError):
            self.grad_file.set_variable_tolerance(2.0)

    def test_dot_writes_tight_native_tolerance(self):
        self.grad_file.set_method("dot_sqp")
        self.grad_file.set_convergence_tolerance(1e-6)
        self.assertEqual(self._method_lines()["convergence_tolerance"].get_values()[1],
                         1e-12)
        self.assertEqual(self.grad_file.get_convergence_settings().relative, 1e-6)

    def test_leaving_conmin_restores_relative_tolerance(self):
        self.grad_file.set_convergence_tolerance(1e-6)
        self.grad_file.set_method("conmin_mfd")
        self.grad_file.set_method("optpp_q_newton")
        self.assertEqual(self._method_lines()["convergence_tolerance"].get_values()[1],
                         1e-6)

    def test_set_finite_difference_interval(self):
        for interval in ["central", "forward"]:
            self.grad_file.set_finite_difference_interval(interval)
            line = self.grad_file.get_gradient_block().get_line(
                DakGradientKeys.interval_type)
            self.assertEqual(line.get_values()[1], interval)

    def test_set_invalid_finite_difference_interval(self):
        with self.assertRaises(TypeError):
            self.grad_file.set_finite_difference_interval(1)
        with self.assertRaises(ValueError):
            self.grad_file.set_finite_difference_interval("backward")

    def test_set_step_size_type(self):
        for step_type in ["absolute", "bounds", "relative"]:
            self.grad_file.set_step_size_type(step_type)
            values = self.grad_file.get_gradient_block().get_line(
                DakGradientKeys.method_source).get_values()
            self.assertEqual(values[1:], ["dakota", step_type])

    def test_set_invalid_step_size_type(self):
        with self.assertRaises(TypeError):
            self.grad_file.set_step_size_type(1)
        with self.assertRaises(ValueError):
            self.grad_file.set_step_size_type("central")


class TestGradientCalibrationMatCalChecks(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)

    def test_matcal_absolute_tolerance_stops_conmin(self):
        study = _linear_calibration(
            "conmin_mfd", lambda study: study.set_absolute_convergence_tolerance(1e-2))
        results = study.launch()
        self.assertTrue(results.success)
        self.assertIn("MatCal", results.exit_message)
        self.assertIn("absolute", results.exit_message)
        self.assertLessEqual(results.best_total_objective, 1e-2)
        best_index = results.best_evaluation_index
        for name in ["m", "b"]:
            self.assertAlmostEqual(results.outcome[f"best:{name}"],
                                   results.parameter_history[name][best_index])

    def test_callback_error_propagates_and_restores_stdout(self):
        study = _linear_calibration("conmin_mfd")
        study._for_testing_fail_after_first_batch()
        stdout_before = os.fstat(1)
        with self.assertRaises(RuntimeError):
            study.launch()
        self.assertEqual(os.fstat(1).st_ino, stdout_before.st_ino)

    def test_preprocessor_outcome_is_in_dakota_parameters(self):
        study = _linear_calibration("nl2sol")
        study.add_parameter_preprocessor(
            UserDefinedParameterPreprocessor(double_slope_and_add_table))
        results = study.launch()
        self.assertEqual(set(results.outcome), {"best:m", "best:b"})
        self.assertAlmostEqual(results.outcome["best:m"], 1.0, delta=1e-6)

    def test_matcal_stop_with_preprocessor_reports_dakota_parameters(self):
        study = _linear_calibration(
            "conmin_mfd", lambda study: study.set_absolute_convergence_tolerance(1e-2))
        study.add_parameter_preprocessor(
            UserDefinedParameterPreprocessor(double_slope_and_add_table))
        results = study.launch()
        self.assertIn("MatCal", results.exit_message)
        self.assertEqual(set(results.outcome), {"best:m", "best:b"})
        self.assertAlmostEqual(results.outcome["best:m"], 1.0, delta=0.05)

    def test_matcal_function_evaluation_limit_for_rol(self):
        study = _linear_calibration(
            "rol", lambda study: study.set_max_function_evaluations(6))
        results = study.launch()
        self.assertFalse(results.success)
        self.assertIn("function evaluations", results.exit_message)
        self.assertEqual(results.number_of_evaluations, 6)

    def test_rol_calibration(self):
        study = _linear_calibration(
            "rol", lambda study: study.set_variable_tolerance(1e-8),
            lambda study: study.set_max_iterations(50))
        results = study.launch()
        self.assertAlmostEqual(results.outcome["best:m"], 2.0, delta=1e-3)
        self.assertAlmostEqual(results.outcome["best:b"], 0.0, delta=1e-2)
        self.assertTrue(os.path.exists("matcal_rol_options.xml"))

    def test_rol_calibration_from_far_start(self):
        study = GradientCalibrationStudy(ParameterCollection(
            "line", Parameter("m", 0.0, 100.0), Parameter("b", -5.0, 5.0, 0.505)))
        study.set_method("rol")
        goal = convert_dictionary_to_data(linear_function(m=2.0, b=0.0))
        study.add_evaluation_set(
            PythonModel(linear_function), CurveBasedInterpolatedObjective("x", "y"), goal)
        study.set_max_iterations(50)
        results = study.launch()
        self.assertTrue(results.success)
        self.assertAlmostEqual(results.outcome["best:m"], 2.0, delta=1e-2)
        self.assertAlmostEqual(results.outcome["best:b"], 0.0, delta=5e-2)


UNBOUNDED = (-np.inf, np.inf)


class TestGradientCalibrationUnboundedParameters(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)

    def _assert_line_found(self, results, slope_delta=1e-2, intercept_delta=5e-2):
        self.assertAlmostEqual(results.outcome["best:m"], 2.0, delta=slope_delta)
        self.assertAlmostEqual(results.outcome["best:b"], 0.0, delta=intercept_delta)

    def test_unbounded_calibration_converges(self):
        for method in ["nl2sol", "optpp_g_newton", "rol"]:
            with self.subTest(method=method):
                study = _linear_calibration(method, slope_bounds=UNBOUNDED,
                                            intercept_bounds=UNBOUNDED)
                self._assert_line_found(study.launch())

    def test_unbounded_optpp_objective_methods_converge(self):
        # As with bounded parameters, the default relative tolerance stops
        # these methods early on this problem (m near 1.9).
        for method in ["optpp_q_newton", "optpp_cg"]:
            with self.subTest(method=method):
                study = _linear_calibration(
                    method, lambda study: study.set_convergence_tolerance(1e-8),
                    slope_bounds=UNBOUNDED, intercept_bounds=UNBOUNDED)
                self._assert_line_found(study.launch())

    def test_unbounded_optpp_fd_newton_improves(self):
        # optpp_fd_newton converges only partly on this problem, bounded or not.
        study = _linear_calibration("optpp_fd_newton", slope_bounds=UNBOUNDED,
                                    intercept_bounds=UNBOUNDED)
        results = study.launch()
        self.assertTrue(results.success)
        self.assertAlmostEqual(results.outcome["best:m"], 2.0, delta=0.25)

    def test_mixed_bounded_and_unbounded_calibration_converges(self):
        for method in ["nl2sol", "rol"]:
            with self.subTest(method=method):
                study = _linear_calibration(method, slope_bounds=(0.0, np.inf))
                self._assert_line_found(study.launch())
        with open("matcal_rol_options.xml") as options_file:
            self.assertNotIn("Initial Radius", options_file.read())

    def test_rol_leaves_initial_radius_to_rol_when_unbounded(self):
        study = _linear_calibration("rol", slope_bounds=(0.0, np.inf))
        self.assertIsNone(study._rol_initial_radius())
        bounded_study = _linear_calibration("rol")
        self.assertEqual(bounded_study._rol_initial_radius(), 0.1)

    def test_bounds_step_size_type_rejects_unbounded_parameters(self):
        study = _linear_calibration(
            "nl2sol", lambda study: study.set_step_size_type("bounds"),
            slope_bounds=(0.0, np.inf))
        with self.assertRaisesRegex(ValueError, "unbounded parameters m"):
            study.launch()

    def test_bounds_step_size_type_allowed_for_bounded_parameters(self):
        study = _linear_calibration(
            "nl2sol", lambda study: study.set_step_size_type("bounds"))
        study._check_step_size_type_suits_bounds()
        self.assertEqual(study._get_step_size_type(), "bounds")

    def test_default_step_size_type_is_unset(self):
        study = _linear_calibration("nl2sol", slope_bounds=(0.0, np.inf))
        self.assertIsNone(study._get_step_size_type())
        study._check_step_size_type_suits_bounds()

    def test_optpp_cg_rejects_bounded_parameters(self):
        study = _linear_calibration("optpp_cg", slope_bounds=(0.0, np.inf),
                                    intercept_bounds=UNBOUNDED)
        with self.assertRaisesRegex(ValueError, "bounded parameters: m"):
            study.launch()


class TestGradientCalibrationStudyDefaultNl2sol(TestDakotaStudyBase.CommonTests):
    _study_class = GradientCalibrationStudy

    def setUp(self):
        super().setUp(__file__)

    def test_set_method(self):
        study = self._study_class(self.parameter_collection)
        with self.assertRaises(ValueError):
            study.set_method("not a method")
        study.set_method(_LeastSquaresMethodKeys.optpp_g_newton)

    def test_study_set_working_directory_initialize_with_restart(self):
        model = PythonModel(simple_func)
        data = convert_dictionary_to_data({"x":[1],"y":[1]})
        objective = CurveBasedInterpolatedObjective("x", "y")
        study = self._study_class(self.parameter_collection)
        study.set_working_directory("my_study_dir")
        study.add_evaluation_set(model, objective, data)
        study.set_restart_filename("my_restart.rst")
        results = study.launch()
        self.assertTrue(os.path.exists("my_study_dir/my_restart.rst"))

    def test_launch_reports_exit_information(self):
        results = _linear_calibration("nl2sol").launch()
        self.assertTrue(results.success)
        self.assertIn("CONVERGENCE", results.exit_message)
        self.assertTrue(os.path.exists("dakota_stdout.log"))

    def test_launch_outcome_matches_best_matcal_evaluation(self):
        results = _linear_calibration("nl2sol").launch()
        best_index = results.best_evaluation_index
        for name in ["m", "b"]:
            self.assertAlmostEqual(
                results.outcome[f"best:{name}"],
                results.parameter_history[name][best_index],
            )

    def test_launch_reports_iteration_limit(self):
        study = _linear_calibration(
            "optpp_q_newton", lambda study: study.set_max_iterations(1)
        )
        results = study.launch()
        self.assertFalse(results.success)
        self.assertIn("exceeds the specified limit", results.exit_message)


class TestGradientCalibrationStudyOptppGNewton(TestDakotaStudyBase.CommonTests):
    _study_class = GradientCalibrationStudy

    def setUp(self):
        super().setUp(__file__)

    def _set_study_specific_options(self, study):
        study.set_method(_LeastSquaresMethodKeys.optpp_g_newton)


class TestGradientCalibrationStudyNlssolSqp(TestDakotaStudyBase.CommonTests):
    _study_class = GradientCalibrationStudy

    def setUp(self):
        super().setUp(__file__)

    def _set_study_specific_options(self, study):
        study.set_method(_LeastSquaresMethodKeys.nlssol_sqp)


class TestGradientCalibrationStudyNpsolSqp(TestDakotaStudyBase.CommonTests):
    _study_class = GradientCalibrationStudy

    def setUp(self):
        super().setUp(__file__)

    def _set_study_specific_options(self, study):
        study.set_method(_GradientMethodKeys.npsol_sqp)


class TestGradientCalibrationStudyDotMmfd(TestDakotaStudyBase.CommonTests):
    _study_class = GradientCalibrationStudy

    def setUp(self):
        super().setUp(__file__)

    def _set_study_specific_options(self, study):
        study.set_method(_GradientMethodKeys.dot_mmfd)


class TestGradientCalibrationStudyDotSlp(TestDakotaStudyBase.CommonTests):
    _study_class = GradientCalibrationStudy

    def setUp(self):
        super().setUp(__file__)

    def _set_study_specific_options(self, study):
        study.set_method(_GradientMethodKeys.dot_slp)


class TestGradientCalibrationStudyDotSqp(TestDakotaStudyBase.CommonTests):
    _study_class = GradientCalibrationStudy

    def setUp(self):
        super().setUp(__file__)

    def _set_study_specific_options(self, study):
        study.set_method(_GradientMethodKeys.dot_sqp)


class TestGradientCalibrationStudyConminMfd(TestDakotaStudyBase.CommonTests):
    _study_class = GradientCalibrationStudy

    def setUp(self):
        super().setUp(__file__)

    def _set_study_specific_options(self, study):
        study.set_method(_GradientMethodKeys.conmin_mfd)


class TestGradientCalibrationStudyOptppQNewton(TestDakotaStudyBase.CommonTests):
    _study_class = GradientCalibrationStudy

    def setUp(self):
        super().setUp(__file__)

    def _set_study_specific_options(self, study):
        study.set_method(_GradientMethodKeys.optpp_q_newton)


class TestGradientCalibrationStudyOptppFdNewton(TestDakotaStudyBase.CommonTests):
    _study_class = GradientCalibrationStudy

    def setUp(self):
        super().setUp(__file__)

    def _set_study_specific_options(self, study):
        study.set_method(_GradientMethodKeys.optpp_fd_newton)

class TestMeshAdaptiveDakotaFile(TestCalibrationDakotaFile.CommonTests):
    _dakota_file_class = _MeshAdaptiveDakotaFile

    def setUp(self) -> None:
        super().setUp(__file__)
        self.df._set_method( self._dakota_file_class.valid_methods[-1])

    def test_init_valid_methods(self):

        for method in self._dakota_file_class.valid_methods:
            grad_file = self._dakota_file_class()
            grad_file._set_method(method)
            method_type_block = grad_file.get_method_type_block()

            self.assertEqual(method_type_block.name, method)
            resp = grad_file.get_response_block()
            self.assertTrue(isinstance(resp, NongradientResponseBlock))
    
    def test_set_variable_tolerance(self):
        var_tol = self.df.get_variable_tolerance()
        self.assertEqual(var_tol, None)
        self.df.set_variable_tolerance(1e-9)
        var_tol = self.df.get_variable_tolerance()
        self.assertEqual(var_tol, 1e-9)
        with self.assertRaises(TypeError):
            self.df.set_variable_tolerance("a")
        with self.assertRaises(ValueError):
            self.df.set_variable_tolerance(-1)
        with self.assertRaises(ValueError):
            self.df.set_variable_tolerance(1)

    def test_set_variable_neighborhood_search(self):
        val = self.df.get_variable_neighborhood_search()
        self.assertEqual(val, None)
        self.df.set_variable_neighborhood_search(1e-9)
        val = self.df.get_variable_neighborhood_search()
        self.assertEqual(val, 1e-9)
        with self.assertRaises(TypeError):
            self.df.set_variable_neighborhood_search("a")
        with self.assertRaises(ValueError):
            self.df.set_variable_neighborhood_search(-1)
        with self.assertRaises(ValueError):
            self.df.set_variable_neighborhood_search(1)


class TestMeshAdaptiveSearchCalibrationStudy(TestDakotaStudyBase.CommonTests):

    _study_class = MeshAdaptiveSearchCalibrationStudy

    def _set_study_specific_options(self, study):
        study.set_max_function_evaluations(4)

    def setUp(self):
        super().setUp(__file__)


class TestColinyNongradientDakotaFile(TestCalibrationDakotaFile.CommonTests):
    _dakota_file_class = _ColinyNongradientDakotaFile

    def setUp(self) -> None:
        super().setUp(__file__)
        self.df._set_method( self._dakota_file_class.valid_methods[-1])

    def test_init_valid_methods(self):

        for method in self._dakota_file_class.valid_methods:
            grad_file = self._dakota_file_class()
            grad_file._set_method(method)
            method_type_block = grad_file.get_method_type_block()

            self.assertEqual(method_type_block.name, method)
            resp = grad_file.get_response_block()
            self.assertTrue(isinstance(resp, NongradientResponseBlock))
    
    def test_set_solution_target(self):
        var_tol = self.df.get_solution_target()
        self.assertEqual(var_tol, None)
        self.df.set_solution_target(1e-9)
        var_tol = self.df.get_solution_target()
        self.assertEqual(var_tol, 1e-9)
        with self.assertRaises(TypeError):
            self.df.set_solution_target("a")


class TestSolisWetsCalibrationStudy(TestDakotaStudyBase.CommonTests):

    _study_class = SolisWetsCalibrationStudy

    def setUp(self):
        super().setUp(__file__)


class TestCobylaCalibrationStudy(TestDakotaStudyBase.CommonTests):

    _study_class = CobylaCalibrationStudy

    def setUp(self):
        super().setUp(__file__)


class TestPatternSearchDakotaFile(TestCalibrationDakotaFile.CommonTests):
    _dakota_file_class = _PatternSearchDakotaFile

    def setUp(self) -> None:
        super().setUp(__file__)
        self.df._set_method( self._dakota_file_class.valid_methods[-1])

    def test_init_valid_methods(self):

        for method in self._dakota_file_class.valid_methods:
            grad_file = self._dakota_file_class()
            grad_file._set_method(method)
            method_type_block = grad_file.get_method_type_block()

            self.assertEqual(method_type_block.name, method)
            resp = grad_file.get_response_block()
            self.assertTrue(isinstance(resp, NongradientResponseBlock))
    
    def test_set_exploratory_moves_not_set(self):
        exploratory_moves = self.df.get_exploratory_moves()
        self.assertEqual(exploratory_moves, None)

    def test_set_solution_target(self):
        self.df.set_solution_target(20)
        self.assertEqual(self.df.get_solution_target(), 20)

    def test_set_invalid_solution_target(self):
        with self.assertRaises(TypeError):
            self.df.set_solution_target("a")

    def test_set_exploratory_moves(self):
        self.df.set_exploratory_moves("adaptive_pattern")
        self.assertEqual(self.df.get_exploratory_moves(), "adaptive_pattern")

    def test_set_invalid_exploratory_moves(self):
        with self.assertRaises(TypeError):
            self.df.set_exploratory_moves(20)
        with self.assertRaises(ValueError):
            self.df.set_exploratory_moves("not_valid")

    def test_set_variable_tolerance(self):
        self.df.set_variable_tolerance(0.99)
        self.assertEqual(self.df.get_variable_tolerance(), 0.99)

    def test_set_invalid_variable_tolerance(self):
        with self.assertRaises(TypeError):
            self.df.set_variable_tolerance("a")
        with self.assertRaises(ValueError):
            self.df.set_variable_tolerance(1.1)


class TestPatternSearchCalibrationStudy(TestDakotaStudyBase.CommonTests):

    _study_class = PatternSearchCalibrationStudy

    def setUp(self):
        super().setUp(__file__)


class TestOptppPdsDakotaFile(TestCalibrationDakotaFile.CommonTests):
    _dakota_file_class = _OptppPdsDakotaFile

    def setUp(self) -> None:
        super().setUp(__file__)
        self.df._set_method( self._dakota_file_class.valid_methods[-1])

    def test_init_valid_methods(self):

        for method in self._dakota_file_class.valid_methods:
            grad_file = self._dakota_file_class()
            grad_file._set_method(method)
            method_type_block = grad_file.get_method_type_block()

            self.assertEqual(method_type_block.name, method)
            resp = grad_file.get_response_block()
            self.assertTrue(isinstance(resp, NongradientResponseBlock))
    
    def test_set_search_scheme_size(self):
        self.assertEqual( self.df.get_search_scheme_size(), 10)
        self.df.set_search_scheme_size(1)
        self.assertEqual( self.df.get_search_scheme_size(), 1)

    def test_set_invalid_search_scheme_size(self):
        with self.assertRaises(TypeError):
            self.df.set_search_scheme_size("a")
        with self.assertRaises(TypeError):
            self.df.set_search_scheme_size(1.1)
        with self.assertRaises(ValueError):
             self.df.set_search_scheme_size(-1)


class TestParallelDirectSearchCalibrationStudy(TestDakotaStudyBase.CommonTests):

    _study_class = ParallelDirectSearchCalibrationStudy

    def setUp(self):
        super().setUp(__file__)