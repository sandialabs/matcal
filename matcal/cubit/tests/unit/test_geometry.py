import abc

from matcal.core.tests.MatcalUnitTest import MatcalUnitTest

from matcal.cubit.geometry import (MaterialPointGeometry, RoundUniaxialTensionGeometry, 
    GeometryParameters, RectangularUniaxialTensionGeometry, 
    RoundNotchedTensionGeometry, SolidBarTorsionGeometry, TopHatShearGeometry)

from matcal.sierra.tests.sierra_sm_models_for_tests import (
    RoundUniaxialTensionModelForTests, 
    RectangularUniaxialTensionModelForTests, RoundNotchedTensionModelForTests, 
    SolidBarTorsionModelForTests, TopHatShearModelForTests,
    UniaxialLoadingMaterialPointModelForTests)


class TestUniaxialLoadingMaterialPointGeometry(MatcalUnitTest):
    _geometry_class = MaterialPointGeometry
    _params = UniaxialLoadingMaterialPointModelForTests().geo_params

    def setUp(self) -> None:
        self.params = self._geometry_class.Parameters(**self._params)
        super().setUp(__file__)

    def test_create_mesh(self):
        geo = self._geometry_class("test_mesh.g", self.params)
        stdout, stderr, return_code = geo.create_mesh()
        self.assert_file_exists("test_mesh.g")
        self.assertEqual(return_code, 0)


class TestUniaxialTensionLoadingGeometry(abc.ABC):

    @abc.abstractmethod
    def _geometry_class(self):
        """"""

    def __init__():
        pass

    class CommonTests(MatcalUnitTest):
    
        def test_uniaxial_tension_extensometer_vs_gauge_length(self):
            with self.assertRaises(GeometryParameters.ValueError):
                self.params["extensometer_length"] = self._params["gauge_length"]*1.01
            
        def test_uniaxial_tension_total_vs_gauge_length(self):           
            with self.assertRaises(GeometryParameters.ValueError):
                self.params["total_length"]  = self._params["gauge_length"]*0.99
            self.params["total_length"]  = self._params["total_length"]

        def test_uniaxial_tension_necking_region_length(self):           
            with self.assertRaises(GeometryParameters.ValueError):
                self.params["necking_region"]  = (self._params["extensometer_length"]-\
                    2*self._params["element_size"])/self._params["extensometer_length"]
            with self.assertRaises(GeometryParameters.ValueError):
                self.params["necking_region"]  = (2*self._params["element_size"])/self._params["extensometer_length"]
            self.params["necking_region"]  = self._params["necking_region"]
   
        def test_access_params(self):
            for key, value in self._params.items():
                self.assertEqual( value,  self.params[key])
            self.params["extensometer_length"] = 1.0*0.0254
            self.assertEqual(self.params["extensometer_length"], 1.0*0.0254)


class UniaxialStressTensionGeometryUnitTest(TestUniaxialTensionLoadingGeometry):

    def __init__():
        pass

    class CommonTests(TestUniaxialTensionLoadingGeometry.CommonTests):

        def test_uniaxial_stress_taper_param(self):
            with self.assertRaises(GeometryParameters.ValueError):
                self.params["taper"]  = (self.params["grip_width"]-self.params["gauge_width"])

        def test_uniaxial_stress_element_size_mesh_method_1_2_3(self):
            with self.assertRaises(GeometryParameters.ValueError):
                self.params["element_size"]  = self.params["gauge_width"]/3.9


class RoundUniaxialStressTensionGeometryUnitTest(UniaxialStressTensionGeometryUnitTest.CommonTests):

    _geometry_class = RoundUniaxialTensionGeometry
    _params = RoundUniaxialTensionModelForTests().geo_params

    def setUp(self) -> None:
        super().setUp(__file__)
        self.params = self._geometry_class.Parameters(**self._params)

    def test_round_tension_gauge_vs_grip_radius(self):
        with self.assertRaises(GeometryParameters.ValueError):
            self.params["gauge_radius"]  = self._params["grip_radius"]*1.01
    
    def test_round_tension_mesh_method_2_element_size(self):
        with self.assertRaises(GeometryParameters.ValueError):
            self.params["element_size"] = self._params["gauge_radius"]/3.5
            self.params["mesh_method"]  = 2
 
    def test_round_tension_mesh_method_4_element_size(self):
        with self.assertRaises(GeometryParameters.ValueError):
            self.params["element_size"] = self._params["gauge_radius"]/8.5
            self.params["mesh_method"]  = 4
 
    def test_round_tension_mesh_method_5_element_size(self):
        with self.assertRaises(GeometryParameters.ValueError):
            self.params["element_size"] = self._params["gauge_radius"]/23.5
            self.params["mesh_method"]  = 5
 

class SolidBarTorsionGeometryUnitTest(RoundUniaxialStressTensionGeometryUnitTest):
    _geometry_class = SolidBarTorsionGeometry  
    _params = SolidBarTorsionModelForTests.geo_params


    def test_access_params(self):
        for key, value in self._params.items():
            self.assertEqual( value,  self.params[key])
        self.params["extensometer_length"] = 0.2*0.0254
        self.assertEqual(self.params["extensometer_length"], 0.2*0.0254)

    def test_appended_cmds(self):
        self._params = self._geometry_class.Parameters(**self._params)
        geo = self._geometry_class("solid_torsion.g", self._params)
        cmds = geo._raw_cmds
        self.assertIn("Volume all copy reflect x", cmds)
        self.assertIn("Volume all copy reflect z", cmds)
        
        
class RectangularUniaxialTensionLoadingGeometryUnitTest(UniaxialStressTensionGeometryUnitTest.CommonTests):
    
    _geometry_class = RectangularUniaxialTensionGeometry
    _params = RectangularUniaxialTensionModelForTests().geo_params

    def setUp(self) -> None:

        self.params = self._geometry_class.Parameters(**self._params)

        super().setUp(__file__)

    def test_rectangular_tension_gauge_vs_grip_width(self):
        with self.assertRaises(GeometryParameters.ValueError):
            self.params["gauge_width"]  = self._params["grip_width"]*1.01

    def test_rectangular_mesh_method_4_element_size(self):
        with self.assertRaises(GeometryParameters.ValueError):
            self.params["element_size"] = self._params["thickness"]/2.5
            self.params["mesh_method"]  = 4
        
    def test_rectangular_mesh_method_5_element_size(self):
        with self.assertRaises(GeometryParameters.ValueError):
            self.params["element_size"] = self._params["thickness"]/8.5
            self.params["mesh_method"]  = 5


class RoundNotchedTensionGeometryUnitTest(TestUniaxialTensionLoadingGeometry.CommonTests):
    _geometry_class = RoundNotchedTensionGeometry  
    _params = RoundNotchedTensionModelForTests.geo_params

    def setUp(self) -> None:
        self.params = self._geometry_class.Parameters(**self._params)
        super().setUp(__file__)

    def test_round_notch_tension_gauge_vs_grip_radius(self):
        with self.assertRaises(GeometryParameters.ValueError):
            self.params["gauge_radius"]  = self._params["grip_radius"]*1.01
    
    def test_round_notch_tension_notch_gauge_vs_gauge_radius(self):
        with self.assertRaises(GeometryParameters.ValueError):
            self.params["notch_gauge_radius"]  = self._params["gauge_radius"]*1.01

    def test_round_notch_tension_notch_height_vs_extensometer_length(self):
        with self.assertRaises(GeometryParameters.ValueError):
            self.params["notch_radius"]  = 1.1

    def test_round_notch_tension_mesh_method_2_element_size(self):
        with self.assertRaises(GeometryParameters.ValueError):
            self.params["element_size"] = self._params["gauge_radius"]/3.5
            self.params["mesh_method"]  = 2

    def test_round_notch_tension_mesh_method_4_element_size(self):
        with self.assertRaises(GeometryParameters.ValueError):
            self.params["element_size"] = self._params["gauge_radius"]/8.5
            self.params["mesh_method"]  = 4
        
    def test_round_notch_tension_mesh_method_5_element_size(self):
        with self.assertRaises(GeometryParameters.ValueError):
            self.params["element_size"] = self._params["gauge_radius"]/23.5
            self.params["mesh_method"]  = 5


class TopHatGeometryUnitTest(MatcalUnitTest):
    _geometry_class = TopHatShearGeometry  
    _params = TopHatShearModelForTests.geo_params

    def setUp(self) -> None:
        self.params = self._geometry_class.Parameters(**self._params)
        super().setUp(__file__)

    def test_top_hat_base_height(self):
        with self.assertRaises(GeometryParameters.ValueError):
            self.params["base_height"]  = (self._params["total_height"] + 
                                          2*self._params["external_radius"])*1.01
        with self.assertRaises(GeometryParameters.ValueError):
            self.params["base_height"]  = (self._params["hole_height"] + 
                                          self._params["base_bottom_height"])*0.99

    def test_top_hat_base_width(self):
        with self.assertRaises(GeometryParameters.ValueError):
            self.params["base_width"]  = (self._params["top_width"] + \
                                     2*self._params["element_size"]*3**self._params["numsplits"])*0.99

    def test_top_hat_trapezoid_top_params(self):
        with self.assertRaises(GeometryParameters.ValueError):
            self.params["trapezoid_angle"]  = 0.0
        with self.assertRaises(GeometryParameters.ValueError):
            self.params["trapezoid_angle"]  = 50
        
        with self.assertRaises(GeometryParameters.ValueError):
            self.params["trapezoid_angle"]  = 49
            self.params["base_width"]  =  self._params["base_width"]*0.01
        
    def test_top_hat_localization_region_params(self):
        with self.assertRaises(GeometryParameters.ValueError):
            self.params["localization_region_scale"]  = -3

    def test_top_hat_hole_params(self):
        with self.assertRaises(GeometryParameters.ValueError):
            self.params["lower_radius_center_width"]  = (self._params["base_width"] - \
                                            2*self._params["element_size"]*3**self._params["numsplits"])*1.01

        with self.assertRaises(GeometryParameters.ValueError):
            self.params["lower_radius_center_width"] =  self._params["lower_radius_center_width"]*0.01

    def test_top_hat_mesh_params(self):
        with self.assertRaises(GeometryParameters.ValueError):
            self.params["numsplits"] = -1

        with self.assertRaises(GeometryParameters.ValueError):
            self.params["numsplits"] = 3
        
        with self.assertRaises(GeometryParameters.ValueError):
            self.params["numsplits"] = 2
            self.params["element_size"] = self._params["base_bottom_height"]
 



