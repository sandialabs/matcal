import numpy as np

from matcal.core.data import convert_dictionary_to_data
from matcal.core.data_analysis import determine_line_intersection, \
determine_slope_and_intercept, determine_pt2_offset_yield
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest

class TestPt2PercentOffsetYield(MatcalUnitTest):
    
    def setUp(self):
        super().setUp(__file__)
    
    def test_determine_slope_and_intercept(self):
        x = np.array([0, 4])
        y = np.e*x-np.pi

        data = convert_dictionary_to_data({"x":x, "y":y})
        m,b = determine_slope_and_intercept(data, 0, "x", "y")
        self.assertAlmostEqual(m, np.e)
        self.assertAlmostEqual(b, -np.pi)

    def test_determine_line_intersection(self):
        intersect = determine_line_intersection(0,0, 1, -1)
        self.assertEqual(intersect, (1,0))

        intersect = determine_line_intersection(-1,0, 1, -1)
        self.assertEqual(intersect, (0.5,-0.5))

        intersect = determine_line_intersection(-2,0, 1, -1)
        self.assertAlmostEqual(intersect, (1.0/3,-2.0/3))

    def _get_simple_data(self):
        emod = 1e3
        hmod = 1e1
        transition = 1e2
        strain = np.linspace(0,0.5, 100)
        stress = emod*strain
        transition_strain = transition/emod
        plasticity = stress > transition
        stress[plasticity]  = transition + hmod*(strain[plasticity]
                                                 -transition_strain)
        stress_strain_data = convert_dictionary_to_data({
            "engineering_stress":stress,
            "engineering_strain":strain})
        return stress_strain_data, emod
    
    def test_determine_pt2_offset_yield(self):
        stress_strain_data, emod = self._get_simple_data()
        pt2_offset_yield = determine_pt2_offset_yield(stress_strain_data, emod)
        self.assertAlmostEqual(10.1/99, pt2_offset_yield[0])
        self.assertAlmostEqual((10.1/99-0.002)*emod, pt2_offset_yield[1])
    
    def test_determine_pt2_offset_yield_different_field_names(self):
        stress_strain_data, emod = self._get_simple_data()
        strain_field = "strain"
        stress_field = "stress"
        
        stress_strain_data.rename_field("engineering_strain", strain_field)
        stress_strain_data.rename_field("engineering_stress", stress_field)
        
        pt2_offset_yield = determine_pt2_offset_yield(stress_strain_data, emod,
                                                      strain_field=strain_field, 
                                                      stress_field=stress_field)
        self.assertAlmostEqual(10.1/99, pt2_offset_yield[0])
        self.assertAlmostEqual((10.1/99-0.002)*emod, pt2_offset_yield[1])
    

    def test_determine_pt2_offset_yield_plot(self):
        import matplotlib.pyplot as plt
        plt.close('all')
        stress_strain_data, emod = self._get_simple_data()
        pt2_offset_yield = determine_pt2_offset_yield(stress_strain_data, emod, 
                                                      plot=True, 
                                                      blocking_plot=False)
        self.assertEqual(len(plt.get_fignums()), 1)
        plt.close('all')