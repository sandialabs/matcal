# test_preprocessors.py
import os

from matcal.core.state import SolitaryState

from matcal.sierra.tests.sierra_sm_models_for_tests import (
    UserDefinedSierraModelForTests,
)
from matcal.sierra.tests.utilities import write_empty_file

from model_tests_base import MatcalStandardModelUnitTestNewBase
from matcal.core.tests.unit.test_models import ModelTestBase
from matcal.full_field.data_importer import FieldSeriesData


#class TemperaturePreprocessorTests(MatcalStandardModelUnitTestNewBase.CommonTests):
#    """
#    Tests focused on temperature preprocessing logic (BC-data temperature vs state temperature,
#    clearing behavior, and coupled/adiabatic constraints).
#    """
#
#    # These tests are already present in MatcalStandardModelUnitTestNewBase.CommonTests.
#    # If you want them to *live* in this file instead, you can override by re-defining them
#    # here and removing them from model_tests_base.py. For now, we keep this file for
#    # any additional preprocessor-focused tests you may add.
#    pass
#

class UserDefinedSierraModelTests(
    ModelTestBase.CommonTests,
    UserDefinedSierraModelForTests,
):
    def setUp(self):
        super().setUp(__file__)

    def test_basic_init(self):
        model = self.init_model()
        self.assertEqual(model._input_filename, os.path.abspath(self._input_file))
        self.assertEqual(model._mesh_filename, os.path.abspath(self._mesh_file))
        self.assertEqual(model.executable, "adagio")

    def test_extra_files_needed_init(self):
        apr_files = ["fake_apr.inc", "fake_apr2.inc", "test_dir"]
        for f in apr_files:
            write_empty_file(f)
        write_empty_file(self._input_file)
        write_empty_file(self._mesh_file)

        model = self._model_class("aria", self._input_file, self._mesh_file, *apr_files)
        for apr_file in apr_files:
            self.assertIn(os.path.abspath(apr_file), model._additional_sources_to_copy)

        model._setup_state(SolitaryState(), build_mesh=False)

    def test_read_full_field_data(self):
        model = self.init_model()
        model.read_full_field_data("test.e")
        self.assertTrue(model._results_information.results_reader_object == FieldSeriesData)
        self.assertEqual(model._results_information.results_filename, "test.e")
