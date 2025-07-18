import os
from matcal.core.parameters import Parameter
import numpy as np

from matcal.core.reporter import plain_text_dictionary_report, \
    MatCalParameterReporterIdentifier
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest


class TestPlainTextReporter(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def test_return_empty_file_if_empty_dict(self):
        filename = "test.txt"
        test_dict = {}
        plain_text_dictionary_report(filename, test_dict)
        self.assertEqual(0, os.path.getsize(filename))

    def test_return_file_with_one_line_for_one_entry_dict(self):
        filename = "test.txt"
        test_dict = {'a':1}
        plain_text_dictionary_report(filename, test_dict)
        goal = "a=1\n"
        self.assert_file_equals_string(goal, filename)
    
    def test_return_file_with_one_line_for_one_entry_dict_various_types(self):
        filename = "test.txt"
        val_list = [0, -1, 1, 0., -1.01, 1e-16, 1e16, np.pi, Parameter]
        for test_val in val_list:
            test_dict = {'aLongerName':test_val}
            plain_text_dictionary_report(filename, test_dict)
            goal = f"aLongerName={test_val}\n"
            self.assert_file_equals_string(goal, filename)

    def test_multiline_dict_report(self):
        filename = 'test.txt'
        test_dict = {'a':1, 'bc':'cat', 'golf':'ocean', 'z':32}
        plain_text_dictionary_report(filename, test_dict)
        goal =  "a=1\n"
        goal += "bc=cat\n"
        goal += "golf=ocean\n"
        goal += "z=32\n"
        self.assert_file_equals_string(goal, filename)

    def test_matcal_parameter_reporter_identifier(self):
        from copy import deepcopy
        registry = deepcopy(MatCalParameterReporterIdentifier._registry)
        MatCalParameterReporterIdentifier._registry = {}
        param_reporter = MatCalParameterReporterIdentifier.identify()
        self.assertEqual(param_reporter, plain_text_dictionary_report)
        MatCalParameterReporterIdentifier._registry = registry