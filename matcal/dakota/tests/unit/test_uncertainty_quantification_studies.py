import numpy as np

from matcal.core.parameters import Parameter, ParameterCollection

from matcal.dakota.tests.unit.test_dakota_studies_base import TestDakotaStudyBase
from matcal.dakota.tests.unit.test_input_file_writer import (TestDakotaFileBase, 
                                                             get_param_collection)
from matcal.dakota.uncertainty_quantification_studies import _DakotaQuesoFileAM, DramBayesianCalibrationStudy
from matcal.dakota.uncertainty_quantification_studies import AdaptiveMetropolisBayesianCalibrationStudy


class TestDakotaQuesoFile(TestDakotaFileBase.CommonTests):
    _dakota_file_class = _DakotaQuesoFileAM

    def setUp(self, file=__file__) -> None:
        super().setUp(file)
        self.df = self._dakota_file_class()
        self.pc = get_param_collection(Parameter.distributions.uniform_uncertain)
        self.df.set_proposal_covariance(0.1)

    def test_set_number_of_samples(self):
        with self.assertRaises(TypeError):
            self.df.set_number_of_samples("foo")
        with self.assertRaises(ValueError):
            self.df.set_number_of_samples(-1)
        self.assertEqual(self.df.get_number_of_samples(), 100)
        self.df.set_number_of_samples(10)
        self.assertEqual(self.df.get_number_of_samples(), 10)

    def test_set_number_of_burnin_samples(self):
        with self.assertRaises(TypeError):
            self.df.set_number_of_burnin_samples("foo")
        with self.assertRaises(ValueError):
            self.df.set_number_of_burnin_samples(-1)
        self.assertEqual(self.df.get_number_of_burnin_samples(), 10)
        self.df.set_number_of_burnin_samples(100)
        self.assertEqual(self.df.get_number_of_burnin_samples(), 100)

    def test_set_seed(self):
        self.assertEqual(None, self.df.get_seed())
        self.df.set_seed(100)
        self.assertEqual(100, self.df.get_seed())

        with self.assertRaises(TypeError):
            self.df.set_seed(0.1)
        with self.assertRaises(ValueError):
            self.df.set_seed(-1)

        self.df.set_random_seed(200)
        self.assertEqual(200, self.df.get_seed())    

    def test_set_proposal_covariance(self):
        df = self._dakota_file_class()

        with self.assertRaises(ValueError):
            df.set_proposal_covariance(-0.01)
        with self.assertRaises(ValueError):
            df.set_proposal_covariance(1.0, -0.01)

        self.assertIsNone(df.get_proposal_covariance())
        df.set_proposal_covariance("diagonal values 1.0 2.0")
        self.assertEqual(df.get_proposal_covariance(), "diagonal values 1.0 2.0")
        df.set_proposal_covariance(1.0, 0.01)
        self.assertEqual(df.get_proposal_covariance(), "diagonal values 1.0 0.01")
        df.set_proposal_covariance(1.0)
        self.assertEqual(df.get_proposal_covariance(), "diagonal values 1.0")


class TestAMBayesStudy(TestDakotaStudyBase.CommonTests):
    _study_class = AdaptiveMetropolisBayesianCalibrationStudy

    def setUp(self):
        super().setUp(__file__)

    def _set_study_specific_options(self, study):
        study.set_proposal_covariance(1.)
        study.set_number_of_samples(10)
        study.set_number_of_burnin_samples(5)
        study.set_random_seed(123)
        study.run_in_serial()
        self.param = Parameter("a", 0, 10, distribution="uniform_uncertain")
        self.parameter_collection = ParameterCollection("Test", self.param)

    def calibration_test_asserts(self, results):
        gold_result = {'parameters': {'a': 5.0},
                        'covariance': np.nan,
                        'mean': {'a': 4.136889e+00},
                        'stddev': {'a': 9.933130e-01},
                        'correlation': [-0.2609632490660545]}
        self.assertAlmostEqual(results["parameters"]["a"], gold_result["parameters"]["a"])
        self.assertAlmostEqual(results["mean"]["a"], gold_result["mean"]["a"], delta=1e-5)
        self.assertAlmostEqual(results["stddev"]["a"], gold_result["stddev"]["a"], delta=1e-5)
        self.assertAlmostEqual(results["correlation"][0], gold_result["correlation"][0], delta=1e-5)

# Because this is a function that wraps the DramClasses it behaves badly if self
# is passed. this wrapper removes the self. 
def TestWrapperForDramBayesianCalibrationStudy(self, *args, **kwdargs):
    return DramBayesianCalibrationStudy(*args, **kwdargs)


class TestDramBayesStudy(TestDakotaStudyBase.CommonTests):
    _study_class = TestWrapperForDramBayesianCalibrationStudy

    def setUp(self):
        super().setUp(__file__)

    def _set_study_specific_options(self, study):
        study.set_proposal_covariance(1.)
        study.set_number_of_samples(10)
        study.set_number_of_burnin_samples(5)
        study.set_random_seed(123)
        study.run_in_serial()
        self.param = Parameter("a", 0, 10, distribution="uniform_uncertain")
        self.parameter_collection = ParameterCollection("Test", self.param)

    def calibration_test_asserts(self, results):
        gold_result = {'parameters': {'a': 5.0},
                        'covariance': np.nan,
                        'mean': {'a': 4.136889e+00},
                        'stddev': {'a': 9.933130e-01},
                        'correlation': [-0.2609632490660545]}
        self.assertAlmostEqual(results["parameters"]["a"], gold_result["parameters"]["a"])
        self.assertAlmostEqual(results["mean"]["a"], gold_result["mean"]["a"], delta=1e-5)
        self.assertAlmostEqual(results["stddev"]["a"], gold_result["stddev"]["a"], delta=1e-5)
        self.assertAlmostEqual(results["correlation"][0], gold_result["correlation"][0], delta=1e-5)