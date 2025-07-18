from matcal.core.tests import MatcalUnitTest
import numpy as np
from matcal.full_field.trees import TreeNode, VoidParentNode, BianaryNode, cluster_from_tree, get_nodes_at_depth, initialize_cluster_tree, make_cluster_results_from_tree

class CommonTreeTests:

    class BaseTests(MatcalUnitTest.MatcalUnitTest):

        def setUp(self):
            super().setUp(__file__)
            
        def NodeType(self):
            pass

        @property
        def n_children(self):
            pass

        def test_defaults(self):
            node = self.NodeType()
            self.assertIsNone(node.depth)
            self.assertFalse(node.is_leaf)
            self.assertEqual(node.children, [])
            self.assertIsNone(node.parent)
            self.assertIsNone(node.population)
            self.assertIsNone(node.side_index)

        def test_assign_root(self):
            node = self.NodeType()
            node.assign_as_root()
            self.assertEqual(node.depth, 0)
            self.assertIsInstance(node.parent, VoidParentNode)
            self.assertEqual(node.side_index, 0)

        def test_assign_leaf(self):
            node = self.NodeType()
            self.assertFalse(node.is_leaf)
            node.assign_as_leaf()
            self.assertTrue(node.is_leaf)

        def test_set_index(self):
            node = self.NodeType()
            self.assertIsNone(node.side_index)
            new_index = 23
            node.set_side_index(new_index)
            self.assertEqual(node.side_index, new_index)

        def test_set_index_fails_if_already_assigned(self):
            node = self.NodeType()
            my_index = 11
            node.set_side_index(my_index)
            with self.assertRaises(self.NodeType.DoubleIndexAssignmentError):
                node.set_side_index(my_index)

        def test_set_index_fails_if_index_is_non_int(self):
            node = self.NodeType()
            bad_ids = [1.23, 'a', 1e4, 3/4]
            for id in bad_ids:
                with self.assertRaises(self.NodeType.NonIntegerIndexError):
                    node.set_side_index(id)

        def test_set_parent(self):
            root = self.NodeType()
            root.assign_as_root()
            node = self.NodeType()
            node.set_parent(root)
            self.assertEqual(node.depth, 1)
            self.assertEqual(node.parent.side_index, root.side_index)

        def test_set_parent_line(self):
            root = self.NodeType()
            root.assign_as_root()
            node1 = self.NodeType()
            node1.set_parent(root)
            node2 = self.NodeType()
            node2.set_parent(node1)
            self.assertEqual(node2.depth, 2)
            self.assertEqual(node2.parent.parent.side_index, root.side_index)

        def test_set_parent_fork(self):
            root = self.NodeType()
            root.assign_as_root()
            node1 = self.NodeType()
            node1.set_parent(root)
            node2 = self.NodeType()
            node2.set_parent(root)
            self.assertEqual(node1.depth, 1)
            self.assertEqual(node1.parent.side_index, root.side_index)
            self.assertEqual(node2.depth, 1)
            self.assertEqual(node2.parent.side_index, root.side_index)
            
        def test_fail_if_has_parent_and_new_set(self):
            root = self.NodeType()
            root.assign_as_root()
            node1 = self.NodeType()
            node1.set_parent(root)
            with self.assertRaises(self.NodeType.ExistingParentError):
                node1.set_parent(root)
        
        def test_fail_if_root_assigned_parent(self):
            root = self.NodeType()
            root.assign_as_root()
            node = self.NodeType()
            with self.assertRaises(self.NodeType.ExistingParentError):
                root.set_parent(node)
            
        def test_fail_if_self_assigned_as_parent(self):
            node = self.NodeType()
            with self.assertRaises(self.NodeType.IsParentError):
                node.set_parent(node)
        
        def test_set_population(self):
            node = self.NodeType()
            pop = list(range(5))
            node.set_population(pop)
            self.assert_close_arrays(pop, node.population)

        def test_set_population_second_time_overwrites(self):
            node = self.NodeType()
            pop = list(range(5))
            pop2 = list(range(10))
            node.set_population(pop)
            node.set_population(pop2)
            self.assert_close_arrays(pop2, node.population)

        def test_set_children_retrieve_correct_children(self):
            n_nodes = self.n_children
            root = self._assign_generic_children(n_nodes)
            for goal_idx, child in enumerate(root.children):
                self.assertEqual(len(child.population), 1)
                self.assertEqual(child.population[0], goal_idx)

        def _assign_generic_children(self, n_nodes):
            root = self.NodeType()
            root.assign_as_root()
            nodes = []
            for i in range(n_nodes):
                new_node = self.NodeType()
                new_node.set_population([i])
                nodes.append(new_node)
            root.set_children(*nodes)
            return root
        
        def test_set_children_all_have_correct_parent(self):
            root = self._assign_generic_children(self.n_children)
            for child in root.children:
                self.assertEqual(child.parent, root)
                self.assertEqual(child.depth, 1)

        def test_set_children_all_have_a_side_index_based_on_oder_of_entry(self):
            n_children = self.n_children
            root = self._assign_generic_children(n_children)
            for goal, child in enumerate(root.children):
                self.assertEqual(child.side_index, goal)

        def test_fail_if_node_is_leaf_and_children_are_added(self):
            leaf_node = self.NodeType()
            leaf_node.assign_as_leaf()
            nodes = []
            for i in range(self.n_children):
                nodes.append(self.NodeType())
            with self.assertRaises(self.NodeType.ChildrenOfLeafError):
                leaf_node.set_children(*nodes)

        def test_fail_set_children_if_children_exist(self):
            root = self._assign_generic_children(self.n_children)
            nodes = []
            for i in range(self.n_children):
                nodes.append(self.NodeType())
            with self.assertRaises(self.NodeType.ExistingChildrenError):
                root.set_children(*nodes)

class TestTreeNode(CommonTreeTests.BaseTests):

    NodeType = TreeNode
    @property
    def n_children(self):
        return 10

class TestBinaryNode(CommonTreeTests.BaseTests):
     
    NodeType = BianaryNode
    @property
    def n_children(self):
        return 2
    
    def test_set_two_children(self):
        root = self.NodeType()
        root.assign_as_root()
        node1 = self.NodeType()
        node2 = self.NodeType()
        root.set_children(node1, node2)
        self.assertEqual(root.children[0], node1)
        self.assertEqual(root.children[1], node2)

    def test_fail_if_more_than_2_children(self):
        root = self.NodeType()
        root.assign_as_root()
        children = []
        for i in range(3):
            children.append(self.NodeType())
        with self.assertRaises(TypeError):
            root.set_children(*children)

    def test_fail_if_1_child(self):
        root = self.NodeType()
        root.assign_as_root()
        children = []
        for i in range(1):
            children.append(self.NodeType())
        with self.assertRaises(TypeError):
            root.set_children(*children)

class TestVoidParentNode(MatcalUnitTest.MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def test_init(self):
        node = VoidParentNode()

# class TestKMeansBinaryNode(TestBinaryNode):

#     NodeType = KMeansBinaryNode
#     @property
#     def n_children(self):
#         return 2
    
#     def test_preclustered_1d_fit_and_cluster(self):
#         root = self.NodeType()
#         root.assign_as_root()
#         points = np.array([-11, -10, -9, 9, 10, 11]).reshape(-1, 1)
#         clusters = root.fit_and_classify(points)
#         # cant predict which is 0 and 1, therefore check for a category
#         if clusters[0] == 0:
#             goal  = [0,0,0,1,1,1]
#         else:
#             goal = [1,1,1,0,0,0]
#         self.assert_close_arrays(clusters, goal)

#     def test_preclustered_2d_fit_and_cluster(self):
#         root = self.NodeType()
#         root.assign_as_root()
#         points = np.array([[-11, -20], [-10, -21], [-9,-19], [9, 20], [10,19], [11, 21]])
#         clusters = root.fit_and_classify(points)
#         # cant predict which is 0 and 1, therefore check for a category
#         if clusters[0] == 0:
#             goal  = [0,0,0,1,1,1]
#         else:
#             goal = [1,1,1,0,0,0]
#         self.assert_close_arrays(clusters, goal)

#     def test_preclustered_2d_cluster_after_fit(self):
#         root = self.NodeType()
#         root.assign_as_root()
#         points = np.array([[-11, -20], [-10, -21], [-9,-19], [9, 20], [10,19], [11, 21]])
#         root.fit_and_classify(points)
#         test_points = np.array([[-22, -20],[8, 15],[13, 25]])
#         test_clusters = root.classify(test_points)
#         # cant predict which is 0 and 1, therefore check for a category
#         if test_clusters[0] == 0:
#             goal  = [0,1,1]
#         else:
#             goal = [1,0,0]
#         self.assert_close_arrays(test_clusters, goal)

class Test_initialize_cluster_tree(MatcalUnitTest.MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def test_root_has_all_points(self):
        points = np.array([-25, -15, 15, 25]).reshape(-1,1)
        minimum_cluster_size = 1
        max_depth = 5
        cluster_tree, results = initialize_cluster_tree(points, minimum_cluster_size, max_depth)
        goal = [0, 1, 2, 3]
        self.assert_close_arrays(results.clusters[0], goal)
        self.assert_close_arrays(cluster_tree.population, goal)
        self.assertEqual(results.side_index[0], 0)
        self.assertEqual(results.depth[0], 0)

    def test_depth_1_has_size_two_population(self):
        points = np.array([-25, -15, 15, 25]).reshape(-1,1)
        minimum_cluster_size = 1
        max_depth = 5
        cluster_tree, results = initialize_cluster_tree(points, minimum_cluster_size, max_depth)
        d1_children = cluster_tree.children
        if 0 in d1_children[0].population:
            c1_goal = [0, 1]
            c2_goal = [2, 3]
        else:
            c1_goal = [2, 3]
            c2_goal = [0, 1]
        self.assertIn(c1_goal, d1_children[0].population)
        self.assertIn(c2_goal, d1_children[1].population)

    def test_depth_2_has_size_one_population(self):
        points = np.array([-25, -15, 15, 25]).reshape(-1,1)
        minimum_cluster_size = 1
        max_depth = 5
        cluster_tree, results = initialize_cluster_tree(points, minimum_cluster_size, max_depth)
        d1_children = get_nodes_at_depth(1, cluster_tree)
        d2_children = get_nodes_at_depth(2, cluster_tree)
        for child in d2_children:
            self.assertEqual(len(child.population), 1)
        if 0 in d1_children[0].population:
            c1_goal = [0, 1]
            c2_goal = [2, 3]
        else:
            c1_goal = [2, 3]
            c2_goal = [0, 1]
        c1_children = d2_children[:2]
        c2_children = d2_children[2:4]
        child_groups = [c1_children, c2_children]
        goals = [c1_goal, c2_goal]
        for cg, goal in zip(child_groups, goals):
            self.assertFalse(np.allclose(cg[0].population, cg[1].population))
            for child in cg:
                self.assertIn(child.population, goal)

    def test_stop_at_max_depth(self):
        locations = np.linspace(0, 10, 100).reshape(-1,1)
        min_size = 1
        max_depth = 3
        cluster_tree, results = initialize_cluster_tree(locations, min_size, max_depth)
        d4_children = get_nodes_at_depth(4, cluster_tree)
        self.assertEqual(len(d4_children), 0)
        d3_children = get_nodes_at_depth(3, cluster_tree)
        self.assertEqual(len(d3_children), 8)

    def test_stop_at_bin_size(self):
        locations = np.linspace(0, 10, 4*8).reshape(-1,1)
        min_size = 4
        max_depth = 100
        cluster_tree, results = initialize_cluster_tree(locations, min_size, max_depth)
        d4_children = get_nodes_at_depth(4, cluster_tree)
        self.assertEqual(len(d4_children), 0)
        d3_children = get_nodes_at_depth(3, cluster_tree)
        # becuase of some random initialization, I can not be sure how it will splt so I need a softer check
        self.assertTrue(len(d3_children) > 4) 

    def test_get_correct_clustering_results_from_function(self):
        max_depth = 3
        n_children = 2
        pop_size = n_children**max_depth
        population = []
        for i in range(pop_size):
            population.append(i)
        population = np.array(population)
        tree = _make_tree_pop_list(max_depth, n_children, population, BianaryNode)
        results = make_cluster_results_from_tree(tree)
        goal_depth = [0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3]
        goal_sides = [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        goal_pop = [[0,1,2,3,4,5,6,7], [0, 1, 2, 3], [4, 5, 6, 7], [0, 1], [2, 3], [4, 5], [6, 7], [0], [1], [2], [3], [4], [5], [6], [7]]
        self.assert_close_arrays(goal_depth, results.depth)
        self.assert_close_arrays(goal_sides, results.side_index)
        for pop_i, sub_pop in enumerate(goal_pop):
            res_pop = results.clusters[pop_i]
            self.assert_close_arrays(sub_pop, res_pop)

class Test_cluster_from_tree(MatcalUnitTest.MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def test_cluster_same_points_gives_same_results(self):
        points = np.array([-25, -15, 15, 25]).reshape(-1,1)
        minimum_cluster_size = 1
        max_depth = 5
        cluster_tree, results = initialize_cluster_tree(points, minimum_cluster_size, max_depth)
        new_results = cluster_from_tree(points, cluster_tree, minimum_cluster_size)
        self.assert_close_arrays(results.depth, new_results.depth)
        self.assert_close_arrays(results.side_index, new_results.side_index)
        for old_pop, new_pop in zip(results.clusters, new_results.clusters):
            self.assert_close_arrays(old_pop, new_pop)


        

def _make_tree(max_depth, n_children, node_type):
    root = node_type()
    root.assign_as_root()
    root.set_population(-1)
    current_nodes = [root]
    cnt = 1
    for i_depth in range(1, max_depth+1):
        next_nodes =[]
        for my_node in current_nodes:
            new_nodes = []
            for i_ch in range(n_children):
                new_node_i = node_type()
                new_node_i.set_population(cnt)
                cnt += 1
                new_nodes.append(new_node_i)
            my_node.set_children(*new_nodes)
            next_nodes += new_nodes
        current_nodes = next_nodes
    return root

def _make_tree_pop_list(max_depth, n_children, pop_list,node_type):
    root = node_type()
    root.assign_as_root()
    root.set_population(pop_list)
    current_nodes = [root]
    n_pop = len(pop_list)
    for i_depth in range(1, max_depth+1):
        next_nodes =[]
        for my_node in current_nodes:
            new_nodes = []
            div_size = len(my_node.population) // (n_children)
            for i_ch in range(n_children):
                new_node_i = node_type()
                new_node_i.set_population(my_node.population[div_size*i_ch:(i_ch+1)*div_size])
                new_nodes.append(new_node_i)
            my_node.set_children(*new_nodes)
            next_nodes += new_nodes
        current_nodes = next_nodes
    return root

class Test_get_children_at_depth(MatcalUnitTest.MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def test_get_depth_0(self):
        root = _make_tree(3, 2, BianaryNode)
        nodes = get_nodes_at_depth(0, root)
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].population, -1)

    def test_get_depth_1(self):
        root = _make_tree(3, 2, BianaryNode)
        nodes = get_nodes_at_depth(1, root)
        self.assertEqual(len(nodes), 2)
        self.assertAlmostEqual(nodes[0].population, 1)
        self.assertAlmostEqual(nodes[1].population, 2)

    def test_get_depth_2(self):
        root = _make_tree(3, 2, BianaryNode)
        nodes = get_nodes_at_depth(2, root)
        goal_node_count = 4
        self.assertEqual(len(nodes), goal_node_count)
        offset = 3
        for i in range(goal_node_count):
            self.assertAlmostEqual(nodes[i].population, offset + i)
    
    def test_get_depth_3_TreeNode(self):
        root = _make_tree(3, 3, TreeNode)
        nodes = get_nodes_at_depth(3, root)
        goal_node_count = 27
        self.assertEqual(len(nodes), goal_node_count)
        offset = 13
        for i in range(goal_node_count):
            self.assertAlmostEqual(nodes[i].population, offset + i)