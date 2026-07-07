import os

from matcal.core.data import convert_dictionary_to_data
from matcal.core.models import PythonModel
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


class TestGradientCalibrationStudyDefaultNl2sol(TestDakotaStudyBase.CommonTests):
    _study_class = GradientCalibrationStudy

    def setUp(self):
        super().setUp(__file__)

    def test_set_method(self):
        study = self._study_class(self.parameter_collection)
        with self.assertRaises(ValueError):
            study.set_method("not a method")
        study.set_method(_GradientMethodKeys.optpp_g_newton)

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


class TestGradientCalibrationStudyOptppGNewton(TestDakotaStudyBase.CommonTests):
    _study_class = GradientCalibrationStudy

    def setUp(self):
        super().setUp(__file__)

    def _set_study_specific_options(self, study):
        study.set_method(_GradientMethodKeys.optpp_g_newton)


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