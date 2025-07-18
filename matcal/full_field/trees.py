from abc import ABC, abstractmethod
from matcal.full_field import cluster_tools
# from sklearn.cluster import cluster
import numpy as np
import matplotlib.pyplot as plt

class VoidParentNode:
    pass

class TreeNode:

    def __init__(self, space_dim=None):
        self._children = []
        self._parent = None
        self._population = None
        self._side_index = None
        self._is_leaf = False
        self._child_classifier = None
        self._depth = None
        self._space_dim = space_dim


    @property
    def depth(self):
        return self._depth

    @property
    def children(self):
        return self._children

    @property
    def parent(self):
        return self._parent

    @property
    def population(self):
        return self._population

    @property
    def is_leaf(self):
        return self._is_leaf

    @property
    def side_index(self):
        return self._side_index

    def assign_as_root(self):
        self._depth = 0
        self._parent = VoidParentNode()
        self.set_side_index(0)

    def assign_as_leaf(self):
        self._is_leaf = True

    def set_parent(self, parent):
        self._confirm_valid_parent_assignment(parent)
        self._parent = parent
        self._depth = parent.depth + 1

    def set_side_index(self, index):
        self._confirm_valid_index_assignment(index)
        self._side_index = index

    def set_population(self, population):
        self._population = population

    def set_children(self, *children):
        self._confirm_valid_state_for_children()
        for side_index, child in enumerate(children):
            child.set_parent(self)
            child.set_side_index(side_index)
        self._children = children

    def _confirm_valid_state_for_children(self):
        if self._is_leaf:
            raise self.ChildrenOfLeafError()
        if self.children != []:
            raise self.ExistingChildrenError()

    def _confirm_valid_parent_assignment(self, parent):
        if self == parent:
            raise self.IsParentError()
        if self._parent != None:
            raise self.ExistingParentError()  

    def _confirm_valid_index_assignment(self, index):
        if self._side_index != None:
            raise self.DoubleIndexAssignmentError(self._side_index, index)
        if not isinstance(index, int):
            raise self.NonIntegerIndexError(index)

    class ChildrenOfLeafError(RuntimeError):
        def __init__(self):
            message = "Cannot add children to a leaf node"
            super().__init__(message)

    class DoubleIndexAssignmentError(RuntimeError):
        def __init__(self, old_index, new_index):
            message = f"\n  Attempting to assign an index to a node that has already been assigned"
            message += f"\n  Current index: {old_index}"
            message += f"\n  New index: {new_index}"
            super().__init__(message)

    class NonIntegerIndexError(RuntimeError):
        def __init__(self, index):
            message = f"\n  Attempted to set a node index with a non-integer value."
            message += f"\n  Passed: {index}"
            message += f'\n  Passed Type: {type(index)}'
            super().__init__(message)

    class ExistingParentError(RuntimeError):
        pass

    class ExistingChildrenError(RuntimeError):
        pass

    class IsParentError(RuntimeError):
        pass


class BianaryNode(TreeNode):

    def set_children(self, child1, child2):
        return super().set_children(*[child1, child2])


class ClassifyingBinaryNodeBase(BianaryNode):

    def __init__(self, space_dim=None):
        super().__init__(space_dim)
        self._child_classifier = None

    def fit_and_classify(self, points):
        cluster = self._child_classifier.fit_predict(points)
        return cluster

    def classify(self, points):
        if self._is_leaf:
            raise self.ChildrenOfLeafError()
        return self._child_classifier.predict(points)

# class KMeansBinaryNode(ClassifyingBinaryNodeBase):
#     def __init__(self, current_depth=None):
#         super().__init__()
#         self._child_classifier = cluster.KMeans(2)

class MedianSplitTwoDimBinaryNode(ClassifyingBinaryNodeBase):
    def __init__(self, space_dim):
        super().__init__(space_dim)
        self._child_classifier = None

    def set_parent(self, parent):
        super().set_parent(parent)
        self._init_classifier()

    def assign_as_root(self):
        super().assign_as_root()
        self._init_classifier()

    def _init_classifier(self):
        self._child_classifier = cluster_tools.MedianSplit(self.depth%self._space_dim)





def get_nodes_at_depth(depth, tree_root):
    current_depth = tree_root.depth
    current_nodes = [tree_root]
    while current_depth < depth:
        new_nodes = []
        for node in current_nodes:
            new_nodes += node.children
        current_depth += 1
        current_nodes = new_nodes
    return current_nodes


class ClusterResults:

    def __init__(self):
        self.clusters = [] # n_clusters x points_in_cluster
        self.side_index = [] # n_clusters
        self.depth = [] # n_clusters 



def initialize_cluster_tree(locations:np.array, minimum_cluster_size:int, max_depth:int, node_class=MedianSplitTwoDimBinaryNode):
    n_children = 2
    space_dim = np.shape(locations)[1]
    root = _init_root(locations, node_class, space_dim)
    done = False
    

    current_depth_nodes = [root]
    while not done:
        new_children = []
        for current_node in current_depth_nodes:
            current_nodes_children = []   
            current_population = current_node.population
            if _is_at_max_depth(max_depth, current_node) or _at_or_below_min_size(minimum_cluster_size, current_population):
                current_node.assign_as_leaf()
                continue
            
            new_clusters = current_node.fit_and_classify(locations[current_population])
            is_safe_split, population_split = _split_population(new_clusters, current_population, minimum_cluster_size, n_children)
            if not is_safe_split:
                current_node.assign_as_leaf
                continue
            
            for side_index in range(n_children):
                child = _init_new_child(node_class, population_split[side_index], space_dim)
                current_nodes_children.append(child)

            current_node.set_children(*current_nodes_children)    
            new_children += current_nodes_children
        if len(new_children) < 1:
            done = True
        else:
            current_depth_nodes = new_children

    result = make_cluster_results_from_tree(root)
    return root, result

def _at_or_below_min_size(minimum_cluster_size, current_population):
    return len(current_population) <= minimum_cluster_size

def _is_at_max_depth(max_depth, current_node):
    return current_node.depth >= max_depth

def _split_population(new_clusters, current_population, minimum_cluster_size, n_children):
    is_safe = True
    new_populations = []
    for side_index in range(n_children):
        population_downselect = np.argwhere(new_clusters==side_index).flatten()
        if len(population_downselect) < minimum_cluster_size:
            is_safe = False
        new_populations.append(current_population[population_downselect])
    return is_safe, new_populations

def _init_new_child(node_class, population, space_dim):
    child = node_class(space_dim)
    child.set_population(population)
    return child

def _init_root(locations, node_class, space_dim):
    root = node_class(space_dim)
    root.assign_as_root()
    n_points = locations.shape[0]
    root.set_population(np.array(list(range(n_points))))
    return root

def make_cluster_results_from_tree(tree):
    results = ClusterResults()
    current_depth = 0
    current_nodes = get_nodes_at_depth(current_depth, tree)
    while len(current_nodes) > 0:
        for node in current_nodes:
            results.clusters.append(np.array(node.population, dtype=int))
            results.depth.append(node.depth)
            results.side_index.append(node.side_index)
        current_depth += 1
        current_nodes = get_nodes_at_depth(current_depth, tree)
    results.depth = np.array(results.depth, dtype=int)
    results.side_index = np.array(results.side_index, dtype=int)
    return results


class PoorlySupportedDomainError(RuntimeError):

    def __init__(self, population_split, min_size):
        message = "\n  Cannot split domain to align with reference decomposition and maintain minimum cluster size"
        size_1 = len(population_split[0])
        size_2 = len(population_split[1])
        message += f"\n minimum size: {min_size}"
        message += f"\n split sizes: {size_1} , {size_2}"

        if size_1 < 20 and size_2 < 20:
            message += f"\n\n  Split 0 Population:\n {population_split[0]}"
            message += f"\n\n  Split 1 Population:\n {population_split[1]}"
        super().__init__(message)

def cluster_from_tree(locations, reference_tree, minimum_cluster_size, node_type=BianaryNode):
    space_dim = np.shape(locations)[1]
    root = _init_root(locations, node_type, space_dim)
    done = False
    n_children = 2
    ref_nodes = [reference_tree]
    cur_nodes = [root]
    it = 0
    while not done:
        it += 1
        new_nodes = []
        new_ref_nodes = []
        for node, ref_node in zip(cur_nodes, ref_nodes):
            current_nodes_children = []
            if ref_node.is_leaf:
                node.assign_as_leaf()
                continue
            node_locations = locations[node.population]
            clusters = ref_node.classify(node_locations)
            is_safe_split, population_split = _split_population(clusters, node.population, minimum_cluster_size, n_children)
            if not is_safe_split:
                raise PoorlySupportedDomainError(population_split, minimum_cluster_size)
            
            for side_index in range(n_children):
                child = _init_new_child(node_type, population_split[side_index], space_dim)
                current_nodes_children.append(child)

            node.set_children(*current_nodes_children)    
            new_nodes += current_nodes_children
            new_ref_nodes += ref_node.children
        if len(new_ref_nodes) < 1:
            done = True
        else:
            cur_nodes = new_nodes
            ref_nodes = new_ref_nodes

    result = make_cluster_results_from_tree(root)
    return result

