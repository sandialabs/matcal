from datetime import date


from matcal.core.best_material_file_writer import  BestFileWriterFactory, \
    DefaultResultsFileWriter, MatcalFileWriterFactory
from matcal.core.utilities import get_username_from_environment
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest


class BasicTestsForAdagioBestMaterialFileWriter(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)
        self._gold_and_input_file_folder_name = "gold_and_input_files"

    def test_default_extraction_parameters_key(self):
        parameters = {"parameters":{"YS": 80, "YRE": 5}}

        bmfw = DefaultResultsFileWriter(parameters)
        today = date.today()

        goal = "###################################\n"
        goal += "# Calibrated by: {}\n".format(get_username_from_environment())
        goal += "# Calibration Finish Date:\n"
        goal += "# Day: {} Month: {} Year: {}\n".format(today.day, today.month, today.year)
        goal += "parameters = {'YS': 80, 'YRE': 5}\n"
        goal += "###################################\n"

        bmfw.write("examplefile.txt")
        self.assert_write_filename(bmfw, goal)


class TestBestFileWriterFactory(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)
        self.fwf = BestFileWriterFactory()

    def test_get_default(self):
        fake_results = {}
        self.assertIsInstance(self.fwf.create('not_a_key', fake_results), DefaultResultsFileWriter)

    def test_matcals_default(self):
        fake_results = {}
        self.assertIsInstance(MatcalFileWriterFactory.create('not_a_key', fake_results), DefaultResultsFileWriter)
