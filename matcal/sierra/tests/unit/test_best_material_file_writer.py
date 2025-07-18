from datetime import date

from matcal.core.utilities import get_username_from_environment
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest
from matcal.core.best_material_file_writer import DefaultResultsFileWriter, MatcalFileWriterFactory

from matcal.sierra.best_material_file_writer import BestApreproMaterialFileWriter



class BasicTestsForAdagioBestMaterialFileWriter(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)
        self._gold_and_input_file_folder_name = "gold_and_input_files"

    def test_basic_extraction_fake(self):
        parameters = {"a": 1, "b": -4, "c": 1000., "d": 1. / 3.}

        bmfw = BestApreproMaterialFileWriter(parameters)
        today = date.today()

        goal = "###################################\n"
        goal += "# Calibrated by: {}\n".format(get_username_from_environment())
        goal += "# Calibration Finish Date:\n"
        goal += "# Day: {} Month: {} Year: {}\n".format(today.day, today.month, today.year)
        goal += "# a = { a = 1.000000000000000E+00 }\n"
        goal += "# b = { b = -4.000000000000000E+00 }\n"
        goal += "# c = { c = 1.000000000000000E+03 }\n"
        goal += "# d = { d = 3.333333333333333E-01 }\n"
        goal += "###################################\n"

        self.assert_write_filename(bmfw, goal)

    def test_basic_extraction(self):
        parameters = {"YS": 80, "YRE": 5}

        bmfw = BestApreproMaterialFileWriter(parameters)
        today = date.today()

        goal = "###################################\n"
        goal += "# Calibrated by: {}\n".format(get_username_from_environment())
        goal += "# Calibration Finish Date:\n"
        goal += "# Day: {} Month: {} Year: {}\n".format(today.day, today.month, today.year)
        goal += "# YS = { YS = 8.000000000000000E+01 }\n"
        goal += "# YRE = { YRE = 5.000000000000000E+00 }\n"
        goal += "###################################\n"

        bmfw.write("examplefile.txt")
        self.assert_write_filename(bmfw, goal)

    def test_basic_extraction_parameters_key(self):
        parameters = {"YS": 80, "YRE": 5}

        bmfw = BestApreproMaterialFileWriter(parameters)
        today = date.today()

        goal = "###################################\n"
        goal += "# Calibrated by: {}\n".format(get_username_from_environment())
        goal += "# Calibration Finish Date:\n"
        goal += "# Day: {} Month: {} Year: {}\n".format(today.day, today.month, today.year)
        goal += "# YS = { YS = 8.000000000000000E+01 }\n"
        goal += "# YRE = { YRE = 5.000000000000000E+00 }\n"
        goal += "###################################\n"

        bmfw.write("examplefile.txt")
        self.assert_write_filename(bmfw, goal)

    def test_basic_extraction_parameters_key_list(self):
        parameters = {"YS": 80, "YRE": 5}

        bmfw = BestApreproMaterialFileWriter(parameters)
        today = date.today()

        goal = "###################################\n"
        goal += "# Calibrated by: {}\n".format(get_username_from_environment())
        goal += "# Calibration Finish Date:\n"
        goal += "# Day: {} Month: {} Year: {}\n".format(today.day, today.month, today.year)
        goal += "# YS = { YS = 8.000000000000000E+01 }\n"
        goal += "# YRE = { YRE = 5.000000000000000E+00 }\n"
        goal += "###################################\n"

        bmfw.write("examplefile.txt")
        self.assert_write_filename(bmfw, goal)

class ExtractionTestsForThermalMaterialFileWriter(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)
        self._gold_and_input_file_folder_name = "gold_and_input_files"


    def test_basic_extraction(self):
        parameters = {"k_xx": 1, "k_yy": 2}

        bmfw = BestApreproMaterialFileWriter(parameters)
        today = date.today()

        goal = "###################################\n"
        goal += "# Calibrated by: {}\n".format(get_username_from_environment())
        goal += "# Calibration Finish Date:\n"
        goal += "# Day: {} Month: {} Year: {}\n".format(today.day, today.month, today.year)
        goal += "# k_xx = { k_xx = 1.000000000000000E+00 }\n"
        goal += "# k_yy = { k_yy = 2.000000000000000E+00 }\n"
        goal += "###################################\n"

        bmfw.write("examplefile.txt")
        self.assert_write_filename(bmfw, goal)

class TestSierraBestFileWriterFactory(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)


    def test_matcals_default(self):
        fake_results = {}
        self.assertIsInstance(MatcalFileWriterFactory.create('not_a_key', fake_results), DefaultResultsFileWriter)

    def test_get_aria(self):
        fake_results = {}
        self.assertIsInstance(MatcalFileWriterFactory.create('aria', fake_results), BestApreproMaterialFileWriter)

    def test_get_adagio(self):
        fake_results = {}
        self.assertIsInstance(MatcalFileWriterFactory.create('adagio', fake_results), BestApreproMaterialFileWriter)