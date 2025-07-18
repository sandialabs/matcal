from matcal.core.tests.MatcalUnitTest import MatcalUnitTest
from matcal.full_field.NodeData import NodeData
import numpy as np


class TestNodeData(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)
        self.node_data = NodeData()

    def test_add_and_return_one_data_array(self):
        self.node_data.add_node_data("field_name", np.ones(5))
        self.assertTrue(np.allclose(np.ones(5), self.node_data.get_node_data('field_name')))

    def test_add_twice_and_return_set(self):
        self.node_data.add_node_data("X", np.ones(3))
        self.node_data.add_node_data("Y", np.ones(3) * 3)
        goal = np.ones([2, 3])
        goal[1, :] = goal[1, :] * 3
        self.assertTrue(np.allclose(goal, self.node_data.get_node_data("X", "Y")))

    def test_produce_error_when_no_fieldnames_are_passed(self):
        self.node_data.add_node_data("X", np.ones(3))
        self.assert_error_type(NodeData.MissingFieldError, self.node_data.get_node_data)

    def test_return_dict(self):
        self.node_data.add_node_data("X", np.ones(3))
        self.node_data.add_node_data("Y", np.ones(3) * 3)
        full_dict = self.node_data.get_full_data()
        goal_dict = {'X': np.ones(3), 'Y': np.ones(3)*3}
        self.assertEqual(len(full_dict.field_names), 2)
        for key, value in goal_dict.items():
            self.assertIn(key, full_dict.field_names)
            self.assert_close_arrays(value, full_dict[key])