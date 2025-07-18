import numbers

from matcal.core.tests.MatcalUnitTest import MatcalUnitTest
from matcal.core.reporter import MatCalParameterReporterIdentifier

from matcal.sierra.aprepro_format import (format_aprepro_value, 
                                          write_aprepro_file_from_dict, 
                                          make_aprepro_string_from_name_val_pair, 
                                          parse_aprepro_variable_line)


class TestApreproFormating(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def test_format_aprepro_values(self):
        formatted_val = format_aprepro_value(1)
        self.assertIsInstance(formatted_val, numbers.Real)
        formatted_val = format_aprepro_value("1")
        self.assertEqual(formatted_val, "\'1\'")
        formatted_val = format_aprepro_value("\"1\"")
        self.assertEqual(formatted_val, "\'1\'")
        formatted_val = format_aprepro_value("\'1\"")
        self.assertEqual(formatted_val, "\'1\'")
        formatted_val = format_aprepro_value("\"1\"")
        self.assertEqual(formatted_val, "\'1\'")

    def test_make_aprepro_string_from_name_val_pair(self):
        name = "name"
        val = 1
        aprepro_str = make_aprepro_string_from_name_val_pair(name, val)
        self.assertEqual(f"# {name} = {{ {name} = {val:0.15E} }}\n", aprepro_str)

        val = "1"
        aprepro_str = make_aprepro_string_from_name_val_pair(name, val)
        goal = "# " + str(name) + " = { " + str(name) + " = \'" + str(val) + "\' }\n"
        self.assertEqual(goal, aprepro_str)


    def test_write_aprepro_file_from_dict(self):
        params = {"test":1, "test2":2}
        fn = "text.txt"
        write_aprepro_file_from_dict("text.txt", params)

        with open(fn, "r") as f:
                lines = f.readlines()
       
        self.assertEqual("{ECHO(OFF)}\n", lines[0])
        line1 = make_aprepro_string_from_name_val_pair("test", params["test"])
        self.assertEqual(line1, lines[1])
        line2 = make_aprepro_string_from_name_val_pair("test2", params["test2"])
        self.assertEqual(line2, lines[2])
        self.assertEqual("{ECHO(ON)}\n", lines[-1])
        
    def test_matcal_parameter_reporter_identifier(self):
        param_reporter = MatCalParameterReporterIdentifier.identify()
        self.assertEqual(param_reporter, write_aprepro_file_from_dict)


class ApreproParserTest(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)

    def test_parse_dense_line(self):
        line = "#m={m=5.000000000000000e+01}"
        name, value = parse_aprepro_variable_line(line)
        self.assertEqual(name, "m")
        self.assertAlmostEqual(value, 50)

    def test_parse_spaced_line(self):
        line = "# Value = { Value = -1}"
        name, value = parse_aprepro_variable_line(line)
        self.assertEqual(name, "Value")
        self.assertAlmostEqual(value, -1)
