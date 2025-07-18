from matcal.core.parameters import Parameter

from matcal.dakota.input_file_writer import (NongradientResidualsResponseBlock, 
                                             NongradientResponseBlock)
from matcal.dakota.sensitivity_studies import (LhsSensitivityStudy, _SampleMethod, 
                                               _DakotaSensitivityFile) 
from matcal.dakota.tests.unit.test_dakota_studies_base import TestDakotaStudyBase
from matcal.dakota.tests.unit.test_input_file_writer import (TestDakotaFileBase, 
                                                             get_param_collection)


class TesSensitivityDakotaFiles(TestDakotaFileBase.CommonTests):
    _dakota_file_class = _DakotaSensitivityFile

    def setUp(self, file=__file__) -> None:
        super().setUp(file)
        self.df = self._dakota_file_class()
        self.pc = get_param_collection(Parameter.distributions.uniform_uncertain)

    def test_set_number_of_samples(self):
        self.assertEqual(10, self.df.get_number_of_samples())
        self.df.set_number_of_samples(100)
        self.assertEqual(100, self.df.get_number_of_samples())

        with self.assertRaises(TypeError):
            self.df.set_number_of_samples(0.1)
        with self.assertRaises(ValueError):
            self.df.set_number_of_samples(-1)

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

    def test_switch_to_objective(self):
        resp = self.df.get_response_block()
        self.assertTrue(isinstance(resp, NongradientResidualsResponseBlock))
        self.df.use_overall_objective()
        resp = self.df.get_response_block()
        self.assertTrue(isinstance(resp, NongradientResponseBlock))


class TestLhsSensitivityStudy(TestDakotaStudyBase.CommonTests):

    _study_class = LhsSensitivityStudy

    def setUp(self):
        super().setUp(__file__)

    def test_sampling_sobol_indices(self):
        study = self._study_class(self.parameter_collection)
        method_type = study.get_method_type_block()
        sobol_study = (_SampleMethod.Keywords.variance_based_decomp in 
                       method_type.lines)
        self.assertFalse(sobol_study)
        study.make_sobol_index_study()
        sobol_study = (_SampleMethod.Keywords.variance_based_decomp in 
                       method_type.lines)
        self.assertTrue(sobol_study)



        