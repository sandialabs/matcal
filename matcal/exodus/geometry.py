from abc import ABC, abstractmethod
from collections import OrderedDict
import numpy as np

from matcal.exodus.library_importer import create_exodus_class_instance
from matcal.cubit.geometry import GeometryParameters


class ExodusGeometryCreatorBase(ABC):
    _exo_indexing = 1
    class Parameters(GeometryParameters):
        _required_geometry_parameters = ['thickness', 'reference_mesh_grid']

        def _check_parameters(self):
            pass
        
        def _set_derived_parameters(self):
            pass

    def __init__(self, mesh_filename, geometry_parameters,block_name="block_main"):
        self._mesh_filename = mesh_filename
        self._reference_mesh_grid = geometry_parameters["reference_mesh_grid"]
        self._thickness = geometry_parameters["thickness"]
        self._block_name = block_name

    @abstractmethod
    def _node_set_names_ids(self):
        """"""

    @abstractmethod
    def _element_type(self):
        """"""

    @abstractmethod
    def _update_connectivity(self, connectivity):
        """"""

    @abstractmethod
    def _make_nodes_for_exo_mesh(self):
        """"""

    @abstractmethod
    def _create_node_set_node_arrays(self):
        """"""

    @abstractmethod
    def _get_number_of_nodes(self):
        """"""

    def _build_topology(self, mesh_filename):
        number_of_nodes = self._get_number_of_nodes()
        exo_obj = create_exodus_class_instance(mesh_filename, mode='w', 
                                               title="VFMStressMesh", array_type='numpy', numDims=3,
                                               numNodes=number_of_nodes, 
                                               numElems=self._reference_mesh_grid.cell_count,
                                               numBlocks=1, numSideSets=0, 
                                               numNodeSets=len(self._node_set_names_ids.keys()))
        exo_connectivity, exo_node_positions, nodes_per_element = self._generate_topology_information()
        exo_obj.put_coords(exo_node_positions[:, 0], exo_node_positions[:, 1], exo_node_positions[:, 2])
        self._assign_connectivity_information(exo_connectivity, exo_obj, nodes_per_element)
        return exo_obj

    def _generate_topology_information(self):
        exo_node_positions = self._make_nodes_for_exo_mesh()
        three_dimensional_connectivity = self._update_connectivity(self._reference_mesh_grid.cell_connectivity)
        exo_connectivity, nodes_per_element = self._make_exodus_connectivity(three_dimensional_connectivity)
        return exo_connectivity, exo_node_positions, nodes_per_element

    def _map_connectivity(self, connectivity):
        gap_map = np.ones(self._reference_mesh_grid.node_count,dtype=int) * -1
        gap_map[self._reference_mesh_grid.node_list] = NodeRelabeler().relabel(
            self._reference_mesh_grid.node_list)
        mapped_con = np.array([gap_map[connectivity[:, 0]], gap_map[connectivity[:, 1]], gap_map[connectivity[:, 2]],
                               gap_map[connectivity[:, 3]]]).T
        return mapped_con

    def _assign_connectivity_information(self, exo_connectivity, exo_obj, nodes_per_element):
        block_number = 1
        number_of_attributes = 0
        number_of_elements = self._reference_mesh_grid.cell_count
        exo_obj.put_elem_blk_info(block_number, self._element_type(), number_of_elements,
                                  nodes_per_element, number_of_attributes)
        exo_obj.put_elem_connectivity(block_number, exo_connectivity)

    def _make_exodus_connectivity(self, twoD_connectivity):
        exodus_standard_factor = int(1)
        nodes_per_element = np.shape(twoD_connectivity)[1]
        return twoD_connectivity.flatten() + exodus_standard_factor, nodes_per_element

    def create_mesh(self, template_dir=None):
        exo_obj = self._build_topology(self._mesh_filename)
        self._label_sidesets(exo_obj)
        exo_obj.close()

    def _label_sidesets(self, exo_obj):
        exo_obj = self._assign_node_sets(exo_obj)
        exo_obj.put_elem_blk_name(1, self._block_name)
    
    def _assign_node_sets(self, exo_obj):
        node_set_arrays = self._create_node_set_node_arrays()
        node_set_info_sets = zip(self._node_set_names_ids.keys(),
                                 self._node_set_names_ids.values(), 
                                 node_set_arrays)
        for node_set_name, node_set_id, node_set_array in node_set_info_sets:
            node_set_id = self._node_set_names_ids[node_set_name]
            exo_obj = self._insert_node_set(exo_obj, node_set_id, node_set_name, node_set_array)
        return exo_obj

    def _insert_node_set(self, exo_obj, node_set_id, node_set_name, node_array):
        exo_obj.put_node_set_params(node_set_id, len(node_array))
        exo_obj.put_node_set(node_set_id, node_array.tolist())
        exo_obj.put_node_set_name(node_set_id, node_set_name)
        return exo_obj

    def _make_node_sheet(self, z_displacement):
        return np.concatenate((self._reference_mesh_grid.node_positions,
                               z_displacement * np.ones([len(self._reference_mesh_grid.node_positions), 1])),
                              axis=1)

 
class ExodusHexGeometryCreator(ExodusGeometryCreatorBase):

    _node_set_names_ids = OrderedDict([("front_node_set",100), ("back_node_set", 200), ('fixed_z_node_set',300)])

    def _make_nodes_for_exo_mesh(self):
        front = self._make_node_sheet(0)
        back = self._make_node_sheet(-self._thickness)
        return np.concatenate((front, back), axis=0)

    def _update_connectivity(self, connectivity):
        two_dim_node_count = self._reference_mesh_grid.node_count
        mapped_con = self._map_connectivity(connectivity)
        new_connectivity = np.array([mapped_con[:, 0], mapped_con[:, 1], mapped_con[:, 1] + two_dim_node_count,
                                     mapped_con[:, 0] + two_dim_node_count, mapped_con[:, 3], mapped_con[:, 2],
                                     mapped_con[:, 2] + two_dim_node_count, mapped_con[:, 3] + two_dim_node_count],
                                    dtype=int)
        return new_connectivity.T

    def _create_node_set_node_arrays(self):
        node_sets = []
        n_sheets = 2
        for sheet_i in range(n_sheets):
            node_sets.append(np.array(NodeRelabeler().relabel(
            self._reference_mesh_grid.node_list),dtype=int)+self._exo_indexing + 
            self._reference_mesh_grid.node_count * sheet_i)
        node_sets.append(self._create_fixed_z_node_set())
        return node_sets

    def _create_fixed_z_node_set(self):
        node_list = self._reference_mesh_grid.node_list
        nodes_below_threshold = self._find_left_bottom_corner_node(node_list)
        relabed_nodes = (NodeRelabeler().relabel(node_list)[nodes_below_threshold]
                         +self._exo_indexing)
        rear_nodes = relabed_nodes + len(node_list)
        return rear_nodes

    def _find_left_bottom_corner_node(self, used_nodes):
        target = self._reference_mesh_grid
        bottom_y = np.min(target.get_grid_corners()[:, 1])
        top_y = np.max(target.get_grid_corners()[:, 1])
        height_fraction = 1e-4
        y_threshold = (top_y - bottom_y) * height_fraction + bottom_y

        width_fraction = height_fraction
        left_x = np.min(target.get_grid_corners()[:, 0])
        right_x = np.max(target.get_grid_corners()[:, 0])
        
        x_left_threshold = +(right_x-left_x)*width_fraction + left_x
        
        y_positions = target.node_positions[used_nodes, 1]
        x_positions = target.node_positions[used_nodes, 0]

        bottom_nodes = y_positions < y_threshold
        left_nodes = x_positions < x_left_threshold
        nodes_to_select = ((bottom_nodes) & (left_nodes))
                
        selected_node_indices = np.array(np.argwhere(nodes_to_select)).flatten()
        return selected_node_indices

    def _get_number_of_nodes(self):
        return 2 * self._reference_mesh_grid.node_count

    def _element_type(self):
        return "HEX8"


class ExodusShellGeometryCreator(ExodusGeometryCreatorBase):

    _node_set_names_ids = OrderedDict([("surface_node_set",100)])

    def _update_connectivity(self, connectivity):
        return self._map_connectivity(connectivity)

    def _make_nodes_for_exo_mesh(self):
        return self._make_node_sheet(0)

    def _create_node_set_node_arrays(self):
        node_sets = []
        node_sets.append(np.array(NodeRelabeler().relabel(
            self._reference_mesh_grid.node_list),dtype=int)+self._exo_indexing)
        return node_sets

    def _get_number_of_nodes(self):
        return self._reference_mesh_grid.node_count

    def _element_type(self):
        return "SHELL4"


class NodeRelabeler:

    def __init__(self):
        self._expected_diff = 1

    def relabel(self, nodes_array):
        sorted_index, working_array = self._sort_array(nodes_array)
        working_array = self._generate_node_offsets(working_array)
        working_array = self._return_array_to_original_order(sorted_index, working_array)
        return working_array

    def _return_array_to_original_order(self, sorted_index, working_array):
        unsorted_index = np.argsort(sorted_index)
        working_array = working_array[unsorted_index]
        return working_array

    def _generate_node_offsets(self, sorted_working_array):
        index_diff = np.diff(sorted_working_array)
        offset = np.zeros(np.shape(sorted_working_array), dtype=int)
        skip_first_entry = 1
        for diff_idx, diff_value in enumerate(index_diff):
            apply_offset = (diff_value > self._expected_diff)
            offset[diff_idx + skip_first_entry:] += (diff_value - self._expected_diff) * apply_offset
        offset = self._set_first_entry_to_zeros(offset, sorted_working_array)
        sorted_working_array -= offset
        return sorted_working_array

    def _set_first_entry_to_zeros(self, offset, sorted_working_array):
        if len(sorted_working_array) > 0:
            offset += sorted_working_array[0]
        return offset

    def _sort_array(self, nodes_array):
        working_array = np.array(nodes_array, dtype=int)
        sorted_index = np.argsort(working_array)
        working_array = working_array[sorted_index]
        return sorted_index, working_array
