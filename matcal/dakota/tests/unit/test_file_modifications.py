from matcal.core.file_modifications import (process_template_file, 
                                              use_jinja_preprocessor, 
                                              jinja2_processor, 
                                              MatCalTemplateFileProcessorIdentifier)
from matcal.dakota.file_modifications import (pyprepro_processor, 
                                              set_pyprepro_delimiters,
                                              pyprepro_delimiters)
                                              
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest


class TestPypreproDelimiters(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)
    
    def test_default_delimiters(self):
        self.assertEqual(pyprepro_delimiters.inline, "{ }")
        self.assertEqual(pyprepro_delimiters.code, "%")
        self.assertEqual(pyprepro_delimiters.code_block, "{% %}")
        set_pyprepro_delimiters()

    def test_set_pyprepro_delimiters(self):
        with self.assertRaises(TypeError):
            set_pyprepro_delimiters(None, "", "")
        with self.assertRaises(ValueError):
            set_pyprepro_delimiters("", None, "")
        with self.assertRaises(TypeError):
            set_pyprepro_delimiters("! !", None, "")
        with self.assertRaises(TypeError):
            set_pyprepro_delimiters("! !", "@ @", None)
        set_pyprepro_delimiters()

    def test_nonunique_delimiters(self):
        with self.assertRaises(ValueError):
            set_pyprepro_delimiters("! !", "! !")
        with self.assertRaises(ValueError):
            set_pyprepro_delimiters("! !", code="!")
        set_pyprepro_delimiters("! !", code="{!}")
        set_pyprepro_delimiters("{! !}", code="!")
        set_pyprepro_delimiters()

    def test_set_invalid_delimiters_more_than_one_space_inline(self):
        with self.assertRaises(ValueError):
            set_pyprepro_delimiters(inline="{  }", code_block="{% %}", code="%")

        with self.assertRaises(ValueError):
            set_pyprepro_delimiters(inline="{ } ", code_block="{% %}", code="%")

        with self.assertRaises(ValueError):
            set_pyprepro_delimiters(inline="{ } 1", code_block="{% %}", code="%")

    def test_set_invalid_delimiters_more_than_one_space_code_block(self):
        with self.assertRaises(ValueError):
            set_pyprepro_delimiters(inline="{ }", code_block="{%  %}", code="%")
        with self.assertRaises(ValueError):
            set_pyprepro_delimiters(inline="{ }", code_block="{% %} #", code="%")


class TestPypreproProcessor(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)
    
    def test_pyprepro_processor(self):
        contents = "a = {val1}\nb = {val2}\nc = {val1 + val2 * 5}"
        modified_contents = pyprepro_processor(contents, {"val1":1, "val2":2})
        goal = "a = 1\nb = 2\nc = 11"
        self.assertEqual(modified_contents, goal)


class TestMatCalTemplateFileProcessor(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def test_template_file_processor_identifier_default(self):
        template_processor = MatCalTemplateFileProcessorIdentifier.identify()
        self.assertEqual(template_processor, pyprepro_processor)
        use_jinja_preprocessor()
        template_processor = MatCalTemplateFileProcessorIdentifier.identify()
        self.assertEqual(template_processor, jinja2_processor)
        
    def test_process_template_file(self):
        contents = "a = {val1}\nb = {val2}\nc = {val1 + val2 * 5}"
        fname = "test.txt"
        with open(fname, "w") as f:
            f.write(contents)
        process_template_file(fname, {"val1":1, "val2":2})
        goal = "a = 1\nb = 2\nc = 11"
        self.assert_file_equals_string(goal, fname)

    def test_process_template_file_new_delimiter(self):
        contents = "a = !val1!\nb = !val2!\nc = !val1 + val2 * 5!"
        fname = "test.txt"
        with open(fname, "w") as f:
            f.write(contents)
        set_pyprepro_delimiters("! !")
        process_template_file(fname, {"val1":1, "val2":2})
        goal = "a = 1\nb = 2\nc = 11"
        self.assert_file_equals_string(goal, fname)
        set_pyprepro_delimiters()
