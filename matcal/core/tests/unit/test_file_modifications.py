from matcal.core.file_modifications import (jinja2_processor, 
                                            MatCalTemplateFileProcessorIdentifier, 
                                            process_template_file, jinja2_delimiters, 
                                            set_jinja_delimiters, use_jinja_preprocessor)
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest


class TestJinjaDelimiters(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)
    
    def test_default_delimiters(self):
        self.assertEqual(jinja2_delimiters.variable, "{{ }}")
        self.assertEqual(jinja2_delimiters.block, "{% %}")
        self.assertEqual(jinja2_delimiters.comment, "{# #}")
        self.assertEqual(jinja2_delimiters.line_statement_prefix, None)
        self.assertEqual(jinja2_delimiters.line_comment_prefix, None)

    def test_set_jinja_delimiters(self):
        with self.assertRaises(TypeError):
            set_jinja_delimiters(None, "", "")
        with self.assertRaises(ValueError):
            set_jinja_delimiters("", None, "")
        with self.assertRaises(TypeError):
            set_jinja_delimiters("! !", None, "")
        with self.assertRaises(TypeError):
            set_jinja_delimiters("! !", "@ @", None)
        with self.assertRaises(TypeError):
            set_jinja_delimiters("! !", "@ @", "* *", 1)
        with self.assertRaises(TypeError):
            set_jinja_delimiters("! !", "@ @", "* *", "1 1", 1)
        
        set_jinja_delimiters()


class TestJinja2Processor(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)
    
    def test_jinja2_processor(self):
        contents = "a = {{ val1 }}\nb = {{ val2 }}\nc = {{ val1 + val2 * 5}}"
        modified_contents = jinja2_processor(contents, {"val1":1, "val2":2})
        goal = "a = 1\nb = 2\nc = 11"
        self.assertEqual(modified_contents, goal)


class TestMatCalTemplateFileProcessor(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)
        use_jinja_preprocessor()

    def test_template_file_processor_identifier_default(self):
        template_processor = MatCalTemplateFileProcessorIdentifier.identify()
        self.assertEqual(template_processor, jinja2_processor)
    
    def test_process_template_file(self):
        contents = "a = {{ val1 }}\nb = {{ val2 }}\nc = {{ val1 + val2 * 5}}"
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
        set_jinja_delimiters("! !")
        process_template_file(fname, {"val1":1, "val2":2})
        goal = "a = 1\nb = 2\nc = 11"
        self.assert_file_equals_string(goal, fname)
        set_jinja_delimiters()
