
import os

from matcal.core.utilities import (_convert_list_of_files_to_abs_path_list, 
                                   _get_highest_version_subfolder, 
                                   is_text_file)

from matcal.core.tests.MatcalUnitTest import MatcalUnitTest


class TestUtilities(MatcalUnitTest):

    def setUp(self) -> None:
        super().setUp(__file__)

    def test_convert_list_of_files_to_abs_path_list(self):
        file_1 = "test.txt"
        file_2 = "test2.txt"
    
        folder = "sub_dir"
        os.mkdir(folder)
        files = [file_1, file_2]
        for file in files:
            with open(file, "w") as f:
               f.write("\n")
        files.append(folder)

        abs_path_list = _convert_list_of_files_to_abs_path_list(files)
        for idx, file in enumerate(files):
            self.assertEqual(abs_path_list[idx], os.path.abspath(file))

    def test_get_highest_version_subfolder(self):
        os.mkdir("test_1.2.3")
        os.mkdir("test_2.4.3")
        os.mkdir("test_6.3.100")
        os.mkdir("test_6.5.3")

        highest_version_folder = _get_highest_version_subfolder(os.path.abspath("."))

        self.assertEqual(highest_version_folder, os.path.abspath("test_6.5.3"))

    def test_is_text_file(self):
        text_fname = "test.txt"
        with open(text_fname, "w") as f:
            f.write("\n")
        self.assertTrue(is_text_file(text_fname))

        nontext_fname = "text.bin"
        with open(nontext_fname, "wb") as f:
            os.urandom(1024)
        self.assertFalse(is_text_file(nontext_fname))

        folder_name = "subfolder"
        os.mkdir(folder_name)
        self.assertFalse(is_text_file(folder_name))