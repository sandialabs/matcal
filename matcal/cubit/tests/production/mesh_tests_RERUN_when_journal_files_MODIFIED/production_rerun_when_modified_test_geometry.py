
from matcal.core.logger import matcal_print_message
import numpy as np
import numbers
import os

from matcal.cubit.geometry import  (RoundNotchedTensionGeometry, RoundUniaxialTensionGeometry, 
     RectangularUniaxialTensionGeometry, TopHatShearGeometry)
from matcal.sierra.tests.sierra_sm_models_for_tests import (RoundUniaxialTensionModelForTests, 
     RoundNotchedTensionModelForTests, RectangularUniaxialTensionModelForTests, 
     TopHatShearModelForTests)
from matcal.cubit.tests.integration.test_geometry import TestGeometry

class ProductionGeometryTestsBase:
    def __init__():
        pass
    class CommonTests(TestGeometry.CommonTestsBase):

        def attempt_to_build_mesh(self, count, built_count, total_expected_attempts):
            try:
                self.build_mesh()
                built_count+=1
                self.assertTrue(os.path.isfile(os.path.join(self.build_dir, "mesh.mesh")))
                matcal_print_message(f"Attempted {count} of {total_expected_attempts}, {built_count} had valid parameters and were successfully built")
            except self._geometry_class.Parameters.ValueError as E:
                self.print_errors_for_mesh_build_failure(E)

            return built_count


class RoundUniaxialTensionLoadingGeometryProductionTests(ProductionGeometryTestsBase.CommonTests):
    _geometry_class = RoundUniaxialTensionGeometry
    _params = RoundUniaxialTensionModelForTests().geo_params

    def test_param_combos(self):
        for key in self._params.keys():
            if isinstance(self._params[key], numbers.Real):
                self._params[key] *= 1/0.0254

        total_lengths = np.linspace(2, 4, 4)
        gauge_lengths = np.linspace(1.5,2.5, 4)
        gauge_radii = np.linspace(0.125, 0.75, 4)
        grip_radii = np.linspace(0.5, 2, 4)
        fillet_radii = np.linspace(0.1, 2, 4)
        necking_regions = np.linspace(0.01, 1, 4)

        mesh_methods = [2,3,4,5]
        count=0
        built_count=0
        total_expected_attempts = 4**6*4*4
        for total_length in total_lengths:
            for gauge_length in gauge_lengths:
                for gauge_radius in gauge_radii:
                    for grip_radius in grip_radii:
                        for fillet_radius in fillet_radii: 
                            for necking_region in necking_regions:
                                for mesh_method in mesh_methods:
                                    element_sizes = np.logspace(np.log10(gauge_radius/2), np.log10(gauge_radius/36), 4)
                                    for element_size in element_sizes:
                                        count+=1
                                        self._params["total_length"] = total_length
                                        self._params["gauge_length"] = gauge_length
                                        self._params["gauge_radius"] = gauge_radius
                                        self._params["grip_radius"] = grip_radius
                                        self._params["fillet_radius"] = fillet_radius
                                        self._params["mesh_method"] = mesh_method
                                        self._params["element_size"] = element_size
                                        self._params["necking_region"] = necking_region
                                        built_count = self.attempt_to_build_mesh(count, built_count, total_expected_attempts)
                                        

class RectangularUniaxialTensionLoadingGeometryProductionTests(ProductionGeometryTestsBase.CommonTests):
    _geometry_class = RectangularUniaxialTensionGeometry
    _params = RectangularUniaxialTensionModelForTests().geo_params

    def test_param_combos(self):
        for key in self._params.keys():
            if isinstance(self._params[key], numbers.Real):
                self._params[key] *= 1/0.0254


        total_lengths = np.linspace(2, 4, 4)
        gauge_lengths = np.linspace(1.5,2.5, 4)
        gauge_widths = np.linspace(0.125, 0.75, 4)*2
        grip_widths = np.linspace(0.5, 2, 4)*2
        fillet_radii = np.linspace(0.1, 2, 4)
        necking_regions = np.linspace(0.01, 1, 4)

        mesh_methods = [3,4,5]
        count=0
        total_expected_attempts = 4**7*4
        built_count = 0
        for total_length in total_lengths:
            for gauge_length in gauge_lengths:
                for gauge_width in gauge_widths:
                    for grip_width in grip_widths:
                        for fillet_radius in fillet_radii: 
                            for necking_region in necking_regions:
                                for mesh_method in mesh_methods:
                                    element_sizes = np.logspace(np.log10(gauge_width/2), np.log10(gauge_width/36), 4)
                                    for element_size in element_sizes:
                                        count+=1

                                        self._params["total_length"] = total_length
                                        self._params["gauge_length"] = gauge_length
                                        self._params["gauge_width"] = gauge_width
                                        self._params["grip_width"] = grip_width
                                        self._params["fillet_radius"] = fillet_radius
                                        self._params["mesh_method"] = mesh_method
                                        self._params["element_size"] = element_size
                                        self._params["necking_region"] = necking_region
                                        built_count = self.attempt_to_build_mesh(count, built_count, total_expected_attempts)


class RoundNotchedTensionGeometryProductionTests(ProductionGeometryTestsBase.CommonTests):
    _geometry_class = RoundNotchedTensionGeometry
    _params = RoundNotchedTensionModelForTests.geo_params

    def test_param_combos(self):
        total_lengths = np.linspace(2, 4, 3)
        gauge_lengths = np.linspace(1.5,2.5, 3)
        notch_radii = np.logspace(np.log10(0.039), np.log10(.390), 3)
        notch_gauge_radii = np.linspace(0.0625, 0.5, 3)
        gauge_radii = np.linspace(0.125, 0.75, 3)
        grip_radii = np.linspace(0.5, 2, 3)
        fillet_radii = np.linspace(0.1, 2, 3)
        necking_regions = np.linspace(0.01, 1, 3)
        mesh_methods = [2,3,4,5]
        count=0
        built_count=0
        total_expected_attempts = 3**8*4*4

        for key in self._params.keys():
            if isinstance(self._params[key], numbers.Real):
                self._params[key] *= 1/0.0254


        for total_length in total_lengths:
            for gauge_length in gauge_lengths:
                for notch_radius in notch_radii:
                    for notch_gauge_radius in notch_gauge_radii:
                        for gauge_radius in gauge_radii:
                            for grip_radius in grip_radii:
                                for fillet_radius in fillet_radii: 
                                    for necking_region in necking_regions:
                                        for mesh_method in mesh_methods:
                                            element_sizes = np.logspace(np.log10(gauge_radius/2), np.log10(gauge_radius/36), 4)
                                            for element_size in element_sizes:
                                                count+=1
                                                self._params["total_length"] = total_length
                                                self._params["gauge_length"] = gauge_length
                                                self._params["gauge_radius"] = gauge_radius
                                                self._params["grip_radius"] = grip_radius
                                                self._params["fillet_radius"] = fillet_radius
                                                self._params["mesh_method"] = mesh_method
                                                self._params["element_size"] = element_size
                                                self._params["notch_radius"] = notch_radius
                                                self._params["notch_gauge_radius"] = notch_gauge_radius
                                                self._params["necking_region"] = necking_region
                                                built_count = self.attempt_to_build_mesh(count, built_count, total_expected_attempts)

class TopHatGeometryProductionTests(ProductionGeometryTestsBase.CommonTests):
    _geometry_class = TopHatShearGeometry
    _params = TopHatShearModelForTests.geo_params

    def test_param_combos(self):
        trap_angles = np.linspace(0.01, 49.9, 3)
        total_heights = np.linspace(0.5, 1.5, 3)
        base_heights = np.linspace(0.25, 1.125, 3)
        ext_int_radii = np.linspace(0.01, 0.5, 3)
        top_widths = np.linspace(0.125, 1.5, 3)*2
        lower_center_widths = np.linspace(0.125, 1.5, 3)*2
        base_bottom_heights = np.linspace(0.1, 1.5, 3)
        hole_heights = np.linspace(0.01, 1.5, 3)
        base_widths = np.linspace(0.2, 1.5, 3)*2
        localization_region_scales = np.linspace(-2, 2, 3)

        for key in self._params.keys():
            if isinstance(self._params[key], numbers.Real):
                self._params[key] *= 1/0.0254

        count=0
        built_count=0
        total_expected_attempts = 3**10*2*4
        self._params["thickness"] = 0.1
        for trap_angle in trap_angles:
            for total_height in total_heights:
                for base_height in base_heights:
                    for ext_int_radius in ext_int_radii:
                        for top_width in top_widths:
                            for lower_center_width in lower_center_widths:
                                for base_bottom_height in base_bottom_heights: 
                                    for hole_height in hole_heights:
                                        for base_width in base_widths:
                                            for local_region_scale in localization_region_scales:
                                                for numsplit in [0,2]:
                                                    shear_band_thickness = base_height-(hole_height+base_bottom_height)
                                                    element_sizes = np.logspace(np.log10(shear_band_thickness/2), np.log10(shear_band_thickness/28), 4)
                                                    for element_size in element_sizes:
                                                        count+=1
                                                        self._params["trapezoid_angle"] = trap_angle
                                                        self._params["total_height"] = total_height
                                                        self._params["base_height"] = base_height
                                                        self._params["external_radius"] = ext_int_radius
                                                        self._params["internal_radius"] = ext_int_radius
                                                        self._params["top_width"] = top_width
                                                        self._params["lower_radius_center_width"] = lower_center_width
                                                        self._params["base_bottom_height"] = base_bottom_height
                                                        self._params["element_size"] = element_size
                                                        self._params["hole_height"] = hole_height
                                                        self._params["base_width"] = base_width
                                                        self._params["numsplits"] = numsplit
                                                        built_count = self.attempt_to_build_mesh(count, built_count, total_expected_attempts)

