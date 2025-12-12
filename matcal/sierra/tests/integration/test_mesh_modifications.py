import os
import shutil

from matcal.core.tests.MatcalUnitTest import MatcalUnitTest

from matcal.sierra.mesh_modifications import (DecompMeshDecomposer, YadaMeshDecomposer,
                                               EpuMeshComposer)
from matcal.sierra.tests.utilities import TEST_SUPPORT_FILES_FOLDER

class MeshModifiers(MatcalUnitTest):

    def setUp(self) -> None:
        super().setUp(__file__)
        self._filename = "cube.g"
        source_file = os.path.join(TEST_SUPPORT_FILES_FOLDER, self._filename)
        shutil.copyfile(source_file, self._filename)

    def test_decomp_decomposer(self):
        decomposer = DecompMeshDecomposer()
        decomposer.decompose_mesh(self._filename, 2)
        self.assert_file_exists(self._filename+".2.0")
        self.assert_file_exists(self._filename+".2.1")

    def test_yada_decomposer(self):
        decomposer = YadaMeshDecomposer()
        decomposer.decompose_mesh(self._filename, 2)
        self.assert_file_exists(self._filename+".2.0")
        self.assert_file_exists(self._filename+".2.1")

    def test_epu_composer(self):
        decomposer = DecompMeshDecomposer()
        decomposer.decompose_mesh(self._filename, 2)
        os.remove(self._filename)
        composer = EpuMeshComposer()
        composer.compose_mesh(self._filename, 2)
        self.assert_file_exists(self._filename)