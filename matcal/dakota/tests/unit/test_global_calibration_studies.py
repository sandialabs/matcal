import os

from matcal.dakota.global_calibration_studies import (MultiObjectiveGACalibrationStudy, 
                                                      SingleObjectiveGACalibrationStudy, 
                                                      _JegaCalibrationDakotaFile)

from matcal.dakota.tests.unit.test_local_calibration_studies import TestCalibrationDakotaFile
from matcal.dakota.tests.unit.test_dakota_studies_base import TestDakotaStudyBase


class TestJegaCalibrationDakotaFiles(TestCalibrationDakotaFile.CommonTests):
    _dakota_file_class = _JegaCalibrationDakotaFile

    def setUp(self) -> None:
        super().setUp(__file__)
        self.df._set_method( self._dakota_file_class.valid_methods[-1])

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

    def test_set_population_size(self):
        self.assertEqual(50, self.df.get_population_size())
        self.df.set_population_size(10)
        self.assertEqual(10, self.df.get_population_size())
        with self.assertRaises(TypeError):
            self.df.set_population_size("str")
        with self.assertRaises(ValueError):
            self.df.set_population_size(-1)

    def test_set_mutation_type(self):
        self.assertEqual("offset_normal", self.df.get_mutation_type())
        self.df.set_mutation_type('bit_random')
        self.assertEqual('bit_random', self.df.get_mutation_type())
        with self.assertRaises(TypeError):
            self.df.set_mutation_type(1)
        with self.assertRaises(ValueError):
            self.df.set_mutation_type("str")

    def test_set_crossover_type(self):
        self.assertEqual('shuffle_random', self.df.get_crossover_type())
        self.df.set_crossover_type('multi_point_real')
        self.assertEqual('multi_point_real', self.df.get_crossover_type())
        with self.assertRaises(TypeError):
            self.df.set_crossover_type(1)
        with self.assertRaises(ValueError):
            self.df.set_crossover_type("str")

    def test_set_mutation_rate(self):
        self.assertEqual(0.2, self.df.get_mutation_rate())
        self.df.set_mutation_rate(0.7)
        self.assertEqual(0.7, self.df.get_mutation_rate())
        with self.assertRaises(TypeError):
            self.df.set_mutation_rate("str")
        with self.assertRaises(ValueError):
            self.df.set_mutation_rate(-1)

    def test_set_crossover_rate(self):
        self.assertEqual(0.7, self.df.get_crossover_rate())
        self.df.set_crossover_rate(0.2)
        self.assertEqual(0.2, self.df.get_crossover_rate())
        with self.assertRaises(TypeError):
            self.df.set_crossover_rate("str")
        with self.assertRaises(ValueError):
            self.df.set_mutation_rate(-1)


class JegaCalibrationStudyTestBase(object):
    def __init__():
        pass
    class JegaStudyCommonTests(TestDakotaStudyBase.CommonTests):

        def calibration_test_asserts(self, results):
            self.assertTrue(os.path.exists("dakota_tabular.dat"))
            with open("dakota_tabular.dat") as d_tab_f:
                text = d_tab_f.readlines()
                self.assertEqual(len(text), 3)
            self.assertTrue(os.path.exists("dakota.out"))
            self.assertTrue(os.stat("dakota.err").st_size == 0)

        def _set_study_specific_options(self, study):
            study.set_max_iterations(10)
            study.set_population_size(2)
            study.set_max_function_evaluations(10)


class TestMultiObjectiveGACalibrationStudy(JegaCalibrationStudyTestBase.JegaStudyCommonTests):

    _study_class = MultiObjectiveGACalibrationStudy

    def setUp(self):
        super().setUp(__file__)


class TestSingleObjectiveGACalibrationStudy(JegaCalibrationStudyTestBase.JegaStudyCommonTests):

    _study_class = SingleObjectiveGACalibrationStudy

    def setUp(self):
        super().setUp(__file__)
  



        