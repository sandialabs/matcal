from matcal.full_field.TwoDimensionalFieldGrid import GridAxis
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest

class TestGridAxis(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)
        self.a = GridAxis(3, 5, 10)

    def test_return_lower_bound(self):
        self.assertEqual(self.a.lower_bound, 3)

    def test_return_upper_bound(self):
        self.assertEqual(self.a.upper_bound, 5)

    def test_raise_error_for_inverted_bounds(self):
        with self.assertRaises(GridAxis.GridAxisImproperBoundsError):
            GridAxis(1, -1, 10)

    def test_return_number_of_nodes(self):
        self.assertEqual(self.a.node_count, 10)

    def test_raise_error_for_invalid_node_count(self):
        with self.assertRaises(GridAxis.GridAxisInvalidNodeCount):
            GridAxis(0, 1, 1)
            GridAxis(0, 1, 0)
            GridAxis(0, 1, -2)
            GridAxis(0, 1, 5.3)

    def test_return_number_of_intervals(self):
        self.assertEqual(self.a.interval_count, 9)
