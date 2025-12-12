import os

from matcal.core.tests.MatcalUnitTest import MatcalUnitTest

from matcal.sierra.mesh_modifications import (DecompMeshDecomposer, YadaMeshDecomposer, 
                                              EpuMeshComposer)

class MeshModifiers(MatcalUnitTest):

    def setUp(self) -> None:
        super().setUp(__file__)

    def test_decomp_decomposer(self):
        decomposer = DecompMeshDecomposer()
        with self.assertRaises(RuntimeError):
            decomposer.decompose_mesh("not a file", 2)

    def test_yada_decomposer(self):
        decomposer = YadaMeshDecomposer()
        with self.assertRaises(RuntimeError):
            decomposer.decompose_mesh("not a file", 2)

    def test_epu_composer(self):
        composer = EpuMeshComposer()
        with self.assertRaises(RuntimeError):
            composer.compose_mesh("not a file", 2)
