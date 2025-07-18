import os

from matcal.core.tests.MatcalUnitTest import MatcalUnitTest

from matcal.sierra.material import Material
from matcal.sierra.tests.utilities import write_linear_elastic_material_file

class TestMaterial(MatcalUnitTest):

    def setUp(self) -> None:
        super().setUp(__file__)
        
        self.example_material_filename = write_linear_elastic_material_file()
        self.example_material = Material("304L", self.example_material_filename, "linear_elastic")

    def test_emptyName_willThrowValueError(self):
        self.assert_error_type(Material.InvalidNameError, Material, "", "", "")
        self.assert_error_type(Material.InvalidNameError, Material, None, "", "")
        self.assert_error_type(Material.InvalidNameError, Material, 1, "", "")

    def test_nonExistentFile_WillThrowFileNotFoundError(self):
        with self.assertRaises(FileNotFoundError):
            Material("example", "nonexistent.file", "")

    def test_initalizeWithModelString(self):
        Material("example", self.example_material_filename, "linear_elastic")

    def test_name(self):
        self.assertEqual("304L", self.example_material.name)

    def test_filename(self):
        self.assertEqual(os.path.join(os.getcwd(),self.example_material_filename),
                          self.example_material.filename)

    def test_model(self):
        self.assertEqual("linear_elastic", self.example_material.model)
