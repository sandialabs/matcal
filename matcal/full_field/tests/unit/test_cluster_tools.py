from matcal.core.tests import MatcalUnitTest
from matcal.full_field.cluster_tools import MedianSplit
import numpy as np

class TestMedianSplit(MatcalUnitTest.MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def test_init(self):
        split_axis = 0
        ms = MedianSplit(split_axis)

    def test_cluster_two_groups_x(self):
        points = np.array([[-10, 1],[-11, -1],[10, 1],[11, -1]])
        split_axis = 0
        ms = MedianSplit(split_axis)
        cluster = ms.fit_predict(points)
        goal = [0, 0, 1, 1]
        self.assert_close_arrays(goal, cluster)

    def test_cluster_two_groups_y(self):
        points = np.array([[-10, 1],[-11, -1],[10, 1],[11, -1]])
        split_axis = 1
        ms = MedianSplit(split_axis)
        cluster = ms.fit_predict(points)
        goal = [1, 0, 1, 0]
        self.assert_close_arrays(goal, cluster)

    def test_fit_then_predict_x(self):
        ref_points = np.array([[-10, 1],[-11, -1],[10, 1],[11, -1]])
        split_axis = 0
        ms = MedianSplit(split_axis)
        ms.fit_predict(ref_points)
        test_points = np.array([[1,1],[2,1],[-20, -5],[20, 5]])
        cluster = ms.predict(test_points)
        goal = [1, 1, 0, 1]
        self.assert_close_arrays(goal, cluster)
    
    def test_fit_then_predict_y(self):
        ref_points = np.array([[-10, 1],[-11, -1],[10, 1],[11, -1]])
        split_axis = 1
        ms = MedianSplit(split_axis)
        ms.fit_predict(ref_points)
        test_points = np.array([[1,1],[2,1],[-20, -5],[20, 5]])
        cluster = ms.predict(test_points)
        goal = [1, 1, 0, 1]
        self.assert_close_arrays(goal, cluster)
    
    def test_non_zero_median_fit_predict(self):
        offset = np.array([1, 12])
        ref_points = np.array([[-10, 1],[-11, -1],[10, 1],[11, -1]]) + offset
        split_axis = 1
        ms = MedianSplit(split_axis)
        ms.fit_predict(ref_points)
        test_points = np.array([[1,1],[2,1],[-20, -5],[20, 5]]) + offset
        cluster = ms.predict(test_points)
        goal = [1, 1, 0, 1]
        self.assert_close_arrays(goal, cluster)