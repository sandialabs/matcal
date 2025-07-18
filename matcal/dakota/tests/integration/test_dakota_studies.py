import numpy as np
import os

from matcal.core.data import DataCollection
from matcal.core.parameters import Parameter, ParameterCollection 
from matcal.core.tests.integration.test_study import CalibrationStudyBaseUnitTests

import matcal.dakota.local_calibration_studies as lcs
import matcal.dakota.global_calibration_studies as gcs
import matcal.dakota.sensitivity_studies as ss
import matcal.dakota.uncertainty_quantification_studies as uqs


class DakotaIntegrationCommonTests:
    def __init__():
        pass
    class DakotaCommonTests(CalibrationStudyBaseUnitTests.CalibrationCommonTests):
        def _assert_appropriate_error(self, subdir):
            is_zero_error = os.stat(os.path.join(subdir,"dakota.err")).st_size == 0
            is_harmless_error = self._confirm_harmless_error(os.path.join(subdir,"dakota.err"))
            self.assertTrue(is_zero_error or is_harmless_error)

        def _assert_output_created(self, subdir):
            self.assertTrue(os.path.exists(os.path.join(subdir,"dakota_tabular.dat")))
            self.assertTrue(os.path.exists(os.path.join(subdir,"dakota.out")))

        def _confirm_harmless_error(self, filename):
            error_found  = False
            with open(filename, 'r') as test_f:
                test_first_line = test_f.readline()
                for line in test_f.readlines():
                    if "error" in line.lower():
                        error_found=True
            okay_first_line = "Warning: failure in recovery of final values for locally recast optimization."
            if test_first_line == okay_first_line:
                return True and not error_found
            else:
                return not error_found
            

class TestSolisWetsCalibrationStudy(DakotaIntegrationCommonTests.DakotaCommonTests):

    _study_class = lcs.SolisWetsCalibrationStudy

    def setUp(self):
        super().setUp(__file__)

    def _set_study_specific_options(self, study):
        study.set_max_function_evaluations(2)


class TestPatternSearchCalibrationStudy(DakotaIntegrationCommonTests.DakotaCommonTests):

    _study_class = lcs.PatternSearchCalibrationStudy

    def setUp(self):
        super().setUp(__file__)

    def _set_study_specific_options(self, study):
        study.set_max_function_evaluations(2)


class TestParallelDirectSearchCalibrationStudy(DakotaIntegrationCommonTests.DakotaCommonTests):

    _study_class = lcs.ParallelDirectSearchCalibrationStudy

    def setUp(self):
        super().setUp(__file__)


class TestMeshAdaptiveSearchCalibrationStudy(DakotaIntegrationCommonTests.DakotaCommonTests):

    _study_class = lcs.MeshAdaptiveSearchCalibrationStudy

    def _set_study_specific_options(self, study):
        study.set_max_function_evaluations(1)

    def setUp(self):
        super().setUp(__file__)


class TestGradientCalibrationStudy(DakotaIntegrationCommonTests.DakotaCommonTests):
    _study_class = lcs.GradientCalibrationStudy

    def setUp(self):
        super().setUp(__file__)

    def test_launch_with_repeat_data_starting_value(self):
            study = self._study_class(self.param)
            dc = DataCollection("test", self.gold_results, self.gold_results, self.gold_results)
            study.add_evaluation_set(self.mock_model, self.objective, self.gold_results)
            study.set_core_limit(6)
            self._set_study_specific_options(study)
            study.set_parameters(self.param)
            results = study.launch()
            self.verify_results(results, self.param)


class TestGradientCalibrationStudyNpsolSqp(DakotaIntegrationCommonTests.DakotaCommonTests):
    _study_class = lcs.GradientCalibrationStudy

    def setUp(self):
        super().setUp(__file__)

    def _set_study_specific_options(self, study):
        study.set_method("npsol_sqp")
        study.set_max_iterations(3)
        
class TestGradientCalibrationStudyDotMmfd(DakotaIntegrationCommonTests.DakotaCommonTests):
    _study_class = lcs.GradientCalibrationStudy

    def setUp(self):
        super().setUp(__file__)

    def _set_study_specific_options(self, study):
        study.set_method("dot_mmfd")
    

class TestGradientCalibrationStudyDotSlp(DakotaIntegrationCommonTests.DakotaCommonTests):
    _study_class = lcs.GradientCalibrationStudy

    def setUp(self):
        super().setUp(__file__)

    def _set_study_specific_options(self, study):
        study.set_method("dot_slp")

class TestGradientCalibrationStudyDotSqp(DakotaIntegrationCommonTests.DakotaCommonTests):
    _study_class = lcs.GradientCalibrationStudy

    def setUp(self):
        super().setUp(__file__)

    def _set_study_specific_options(self, study):
        study.set_method("dot_sqp")


class TestGradientCalibrationStudyConminMfd(DakotaIntegrationCommonTests.DakotaCommonTests):
    _study_class = lcs.GradientCalibrationStudy

    def setUp(self):
        super().setUp(__file__)

    def _set_study_specific_options(self, study):
        study.set_method("conmin_mfd")


class TestGradientCalibrationStudyOptppQNewton(DakotaIntegrationCommonTests.DakotaCommonTests):
    _study_class = lcs.GradientCalibrationStudy

    def setUp(self):
        super().setUp(__file__)

    def _set_study_specific_options(self, study):
        study.set_method("optpp_q_newton")


class TestGradientCalibrationStudyOptppFdNewton(DakotaIntegrationCommonTests.DakotaCommonTests):
    _study_class = lcs.GradientCalibrationStudy

    def setUp(self):
        super().setUp(__file__)

    def _set_study_specific_options(self, study):
        study.set_method("optpp_fd_newton")


class TestGradientCalibrationStudyOptppGNewton(DakotaIntegrationCommonTests.DakotaCommonTests):
    _study_class = lcs.GradientCalibrationStudy

    def setUp(self):
        super().setUp(__file__)

    def _set_study_specific_options(self, study):
        study.set_method("optpp_g_newton")


class TestGradientCalibrationStudyNlssolSqp(DakotaIntegrationCommonTests.DakotaCommonTests):
    _study_class = lcs.GradientCalibrationStudy

    def setUp(self):
        super().setUp(__file__)

    def _set_study_specific_options(self, study):
        study.set_method("nlssol_sqp")


class TestCobylaCalibrationStudy(DakotaIntegrationCommonTests.DakotaCommonTests):

    _study_class = lcs.CobylaCalibrationStudy

    def setUp(self):
        super().setUp(__file__)

    def _set_study_specific_options(self, study):
        study.set_variable_tolerance(1e-9)
        

class TestAMBayesStudy(DakotaIntegrationCommonTests.DakotaCommonTests):
    _study_class = uqs.AdaptiveMetropolisBayesianCalibrationStudy

    def setUp(self):
        super().setUp(__file__)
    
    def _set_study_specific_options(self, study):
        study.set_proposal_covariance(1)
        study.set_number_of_samples(10)
        study.set_number_of_burnin_samples(5)
        study.set_random_seed(1)
        self.param = Parameter("a", 0, 10, distribution="uniform_uncertain")
        self.parameter_collection = ParameterCollection("Test", self.param)

    def verify_results(self, results, param, subdir=""):
        gold_result = {'parameters': {'a': 5.0},
                        'covariance': np.nan,
                        'mean': {'a': 4.136889e+00},
                        'stddev': {'a': 9.933130e-01},
                        'pearson': [1]}
        self.assertAlmostEqual(results.outcome["MAP:a"], gold_result["parameters"]["a"])
        self.assertAlmostEqual(results.outcome["mean:a"], gold_result["mean"]["a"], delta=1e-5)
        self.assertAlmostEqual(results.outcome["stddev:a"], gold_result["stddev"]["a"], delta=1e-5)
        self.assertAlmostEqual(results.outcome["pearson:a"], gold_result["pearson"][0], delta=1e-5)
        self.assertTrue(os.path.exists(os.path.join(subdir,"dakota_tabular.dat")))
        self.assertTrue(os.path.exists(os.path.join(subdir,"dakota.out")))


class SensitivityStudyBaseTests(object):
    class SensitivityCommonTests(DakotaIntegrationCommonTests.DakotaCommonTests):

        def setUp(self, filename):
            super().setUp(filename)

        def verify_results(self, results, param, subdir=""):
            if results:
                for value in results.outcome['pearson:a']:
                    self.assertTrue(np.isnan(value) or np.abs(value)<1e-14)
            else:
                self.assertEqual(results.outcome, {})
            self.assertTrue(os.path.exists(os.path.join(subdir,"dakota_tabular.dat")))
            self.assertTrue(os.path.exists(os.path.join(subdir,"dakota.out")))


class TestLhsSensitivityStudy(SensitivityStudyBaseTests.SensitivityCommonTests):
    _study_class = ss.LhsSensitivityStudy

    def setUp(self):
        super().setUp(__file__)

    def _set_study_specific_options(self, study):
        study.set_number_of_samples(5)
        study.set_random_seed(1)


class JegaCalibrationStudyTestBase(object):
    class JegaStudyCommonTests(DakotaIntegrationCommonTests.DakotaCommonTests):

        def _set_study_specific_options(self, study):
            study.set_population_size(2)
            study.set_max_iterations(1)

        def verify_results(self, results, param, subdir=""):
            self.assertTrue(os.path.exists(os.path.join(subdir,"dakota_tabular.dat")))
            self.assertTrue(os.path.exists(os.path.join(subdir,"dakota.out")))
            is_zero_error = os.stat(os.path.join(subdir,"dakota.err")).st_size == 0
            is_harmless_error = self._confirm_harmless_error(os.path.join(subdir,"dakota.err"))
            self.assertTrue(is_zero_error or is_harmless_error)


class TestMultiObjectiveGACalibrationStudy(JegaCalibrationStudyTestBase.JegaStudyCommonTests):

    _study_class = gcs.MultiObjectiveGACalibrationStudy

    def setUp(self):
        super().setUp(__file__)


class TestSingleObjectiveGACalibrationStudy(JegaCalibrationStudyTestBase.JegaStudyCommonTests):

    _study_class = gcs.SingleObjectiveGACalibrationStudy

    def setUp(self):
        super().setUp(__file__)
