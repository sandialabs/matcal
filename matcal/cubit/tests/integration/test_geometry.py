
from abc import ABC, abstractmethod
import os

from matcal.core.logger import matcal_print_message
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest

from matcal.cubit.geometry import (MaterialPointGeometry, RoundNotchedTensionGeometry, 
                                    RoundUniaxialTensionGeometry, 
                                    RectangularUniaxialTensionGeometry, SolidBarTorsionGeometry, 
                                    TopHatShearGeometry)
from matcal.sierra.tests.sierra_sm_models_for_tests import (RoundNotchedTensionModelForTests, 
    RectangularUniaxialTensionModelForTests, RoundUniaxialTensionModelForTests, 
    SolidBarTorsionModelForTests, TopHatShearModelForTests)


class TestGeometry(ABC):
    def __init__():
        pass
    
    class CommonTests(MatcalUnitTest):

        @property
        @abstractmethod
        def _geometry_class(self):
            """"""
        @property
        @abstractmethod
        def _params(self):
            """"""

        def setUp(self) -> None:
            super().setUp(__file__)

        def build_mesh_with_params(self, params):
            geo = self._geometry_class(
                "mesh.mesh",
                self._geometry_class.Parameters(**params)
            )
            return geo.create_mesh(self.build_dir)

        def build_mesh(self):
            return self.build_mesh_with_params(dict(self._params))

        def print_errors_for_mesh_build_failure(self, E):
            matcal_print_message("parameters failed with the following message:")
            matcal_print_message(E)
            matcal_print_message(self._params)

        def test_mesh_build(self):
            stdout, stderr, error_code = self.build_mesh()
            self.assertTrue(os.path.isfile(os.path.join(self.build_dir, "mesh.mesh")))

            self.assertEqual(error_code, 0)

        def test_mesh_build_composite_tet(self):
            params = dict(self._params)
            params["element_type"] = "tet10"
            stdout, stderr, error_code = self.build_mesh_with_params(params)
            self.assertTrue(os.path.isfile(os.path.join(self.build_dir, "mesh.mesh")))
            self.assertEqual(error_code, 0)


class MaterialPointGeometryIntegrationTests(TestGeometry.CommonTests):

    _geometry_class = MaterialPointGeometry
    _params = {}

    def test_mesh_build_composite_tet(self):
        """"""


class UniaxialStressTensionCommonTests(ABC):
    def __init__():
        pass
    
    class CommonTests(TestGeometry.CommonTests):
        def test_mesh_build_with_default_necking_region(self):
            params = dict(self._params)
            params.pop("necking_region", None)

            stdout, stderr, error_code = self.build_mesh_with_params(params)

            self.assertTrue(os.path.isfile(os.path.join(self.build_dir, "mesh.mesh")))
            self.assertEqual(error_code, 0)

        def test_mesh_build_with_user_specified_necking_region(self):
            params = dict(self._params)
            params["necking_region"] = 0.2

            stdout, stderr, error_code = self.build_mesh_with_params(params)

            self.assertTrue(os.path.isfile(os.path.join(self.build_dir, "mesh.mesh")))
            self.assertEqual(error_code, 0)


class RoundTensionGeometryIntegrationTests(UniaxialStressTensionCommonTests.CommonTests):
    _geometry_class = RoundUniaxialTensionGeometry
    _params = RoundUniaxialTensionModelForTests().geo_params


class RectangularTensionGeometryIntegrationTests(UniaxialStressTensionCommonTests.CommonTests):
    _geometry_class = RectangularUniaxialTensionGeometry
    _params = RectangularUniaxialTensionModelForTests().geo_params


class RoundNotchedTensionGeometryIntegrationTests(TestGeometry.CommonTests):
    _geometry_class = RoundNotchedTensionGeometry
    _params = RoundNotchedTensionModelForTests.geo_params

    def test_mesh_build_with_default_necking_region(self):
        params = dict(self._params)
        params.pop("necking_region", None)

        stdout, stderr, error_code = self.build_mesh_with_params(params)

        self.assertTrue(os.path.isfile(os.path.join(self.build_dir, "mesh.mesh")))
        self.assertEqual(error_code, 0)

    def test_mesh_build_with_user_specified_necking_region(self):
        params = dict(self._params)
        params["necking_region"] = 0.2

        stdout, stderr, error_code = self.build_mesh_with_params(params)

        self.assertTrue(os.path.isfile(os.path.join(self.build_dir, "mesh.mesh")))
        self.assertEqual(error_code, 0)

    def test_mesh_build_with_default_necking_region_when_notch_radius_equals_gauge_radius(self):
        params = dict(self._params)
        params.pop("necking_region", None)
        params["notch_radius"] = params["gauge_radius"]
        params["mesh_method"] = 2
        params["element_size"] = params["notch_gauge_radius"]/4.1

        stdout, stderr, error_code = self.build_mesh_with_params(params)

        self.assertTrue(os.path.isfile(os.path.join(self.build_dir, "mesh.mesh")))
        self.assertEqual(error_code, 0)


class SolidBarTorsionGeometryIntegrationTests(TestGeometry.CommonTests):
    _geometry_class = SolidBarTorsionGeometry
    _params = SolidBarTorsionModelForTests.geo_params


class TopHatGeometryIntegrationTests(TestGeometry.CommonTests):
    _geometry_class = TopHatShearGeometry
    _params = TopHatShearModelForTests.geo_params
