from abc import ABC, abstractmethod
from matcal.core.serializer_wrapper import _format_serial

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spar_linalg

from matcal.core.logger import initialize_matcal_logger
logger = initialize_matcal_logger(__name__)


class MeshSkeleton:
    """
    MeshSkeleton serves as a light container to store point cloud or mesh information in a form agnostic to its source. 
    """
    def __init__(self, points=None, connectivity=None):
        self.spatial_coords = points
        self.connectivity = connectivity
        self.subset_name = None
        self.surfaces = {}

    def add_node_sets(self, **name_node_sets):
        for key, value in name_node_sets.items():
            name_node_sets[key] = np.array(value)
        self.surfaces.update(name_node_sets)

    def serialize(self):
        serial = {}
        serial['spatial_coords'] = _format_serial(self.spatial_coords)
        serial['connectivity'] = _format_serial(self.connectivity)
        serial['subset_name'] = self.subset_name
        
        serial['surfaces'] = {}
        for s_name, s_array in self.surfaces.items():
            serial['surfaces'][s_name] = _format_serial(s_array)
        return serial

class FieldGridBase(ABC):

    def __init__(self):
        self._space_dim = 2
        self._node_positions = None
        self._cell_connectivity = None

    @abstractmethod
    def get_containing_cell(self, point_coordinates):
        """"""

    @abstractmethod
    def get_cell_node_locations(self, cell_index):
        """"""

    @abstractmethod
    def get_cell_nodes(self, cell_index):
        """"""

    @property
    @abstractmethod
    def node_count(self):
        """"""

    @property
    def node_list(self):
        return np.array(list(range(self.node_count)), dtype=int)

    @property
    @abstractmethod
    def cell_count(self):
        """"""

    @property
    def cell_list(self):
        return np.array(list(range(self.cell_count)), dtype=int)

    @property
    @abstractmethod
    def node_positions(self):
        """"""

    @property
    @abstractmethod
    def cell_connectivity(self):
        """"""

    @property
    @abstractmethod
    def cell_sizes(self):
        """"""

    @abstractmethod
    def get_cell_areas(self):
        """"""

    @abstractmethod
    def get_cell_centers(self):
        """"""

    def get_grid_corners(self):
        corners = np.zeros([4, 2])
        maxes = np.max(self._node_positions, axis=0)
        mins = np.min(self._node_positions, axis=0)
        corners[0, :] = [mins[0], mins[1]]
        corners[1, :] = [maxes[0], mins[1]]
        corners[2, :] = [maxes[0], maxes[1]]
        corners[3, :] = [mins[0], maxes[1]]
        
        return corners


class GridAxis:
    class GridAxisImproperBoundsError(RuntimeError):
        def __init__(self, lower, upper, *args):
            message = "\nBad Bounds:\nLower: {}\nUpper: {}".format(lower, upper)
            super().__init__(message, *args)

    class GridAxisInvalidNodeCount(RuntimeError):
        def __init__(self, node_count, *args):
            message = "\nnode_count must be of type int, and be greater than 1.\n" \
                      "passed: {} which is of type {}".format(node_count, type(node_count))
            super().__init__(message, *args)

    def __init__(self, lower_bound, upper_bound, node_count):
        self._check_bounds(lower_bound, upper_bound)
        self._check_node_count(node_count)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.node_count = node_count
        self.interval_count = self._generate_interval_count()

    def __str__(self):
        return f"Lower: {self.lower_bound}  Upper: {self.upper_bound}  Nodes: {self.node_count}"

    def _check_bounds(self, lower, upper):
        if lower >= upper:
            raise self.GridAxisImproperBoundsError(lower, upper)

    def _check_node_count(self, node_count):
        if node_count < 1 or not isinstance(node_count, int):
            raise self.GridAxisInvalidNodeCount(node_count)

    def _generate_interval_count(self):
        return self.node_count - 1
    
def _get_surface_nodes(mesh_skeleton, surface_name):
    try:
        s_nodes = mesh_skeleton.surfaces[surface_name]
    except KeyError:
        message = f"Bad Surface Lookup\nAttempted to find: {surface_name}\nPossible Surfaces: {mesh_skeleton.surfaces.keys()}"
        raise RuntimeError(message)

    return s_nodes


def _get_node_coordinate_mapping(new_skeleton, old_skeleton, surface_name=None):
    old_coords = old_skeleton.spatial_coords
    if surface_name:
        new_nodes = _get_surface_nodes(new_skeleton, surface_name)
        new_coords = new_skeleton.spatial_coords[new_nodes,:]
    else:
        new_coords = new_skeleton.spatial_coords
        new_nodes = np.arange(0,new_coords.shape[0])
    new_to_old = _generate_node_map(old_coords, new_coords)
    return new_nodes[new_to_old]

def _generate_node_map(old_coords, new_coords):
    mapper = NewToOldRemapper()
    new_to_old =  mapper(new_coords, old_coords)
    return new_to_old


class NewToOldRemapper:
    
    class DifferentPointSetError(RuntimeError):
    
        def __init__(self, n_old, n_new):
            message = "Remapper Detected different numbers of points passed to it."
            message += f"\n Old: {n_old}   New: {n_new}"
            super().__init__(message)

    def __init__(self):
        pass

    def __call__(self, new_points, old_points):
        n_pts = self._count_points(old_points)
        self._confrim_equal_points(new_points, n_pts)
        new_to_old_map = np.zeros(n_pts, dtype=int)
        for point_id in range(n_pts):
            close_x = self._find_close(old_points, new_points, point_id, 0)
            close_y = self._find_close(old_points, new_points, point_id, 1)
            new_id = np.argwhere(np.multiply(close_x, close_y))[:,0]
            self._confirm_id(new_id)
            new_to_old_map[point_id] = new_id
        return new_to_old_map

    def _confrim_equal_points(self, new_points, n_pts):
        n_pts_new = self._count_points(new_points)
        if not n_pts == n_pts_new:
            raise self.DifferentPointSetError(n_pts, n_pts_new)

    def _count_points(self, points):
        return points.shape[0]

    def _confirm_id(self, id_array):
        if len(id_array) > 1:
            raise self.NonuniqueMappingError()
        if len(id_array) == 0:
            raise self.NoMapFoundError()
    
    class NonuniqueMappingError(RuntimeError):
        def __init__(self):
            message = "There exists more than one point that sufficiently maps to a reference point"
            super().__init__(message)
    
    class NoMapFoundError(RuntimeError):
        def __init__(self):
            message = "Cannot find a corresponding point in the new domain that aligns with a reference point"
            super().__init__(message)

    def _find_close(self, old_points, new_points, point_id, dim):
        x = old_points[point_id, dim]
        return np.isclose(new_points[:, dim], x, 1e-8)


class TwoDimensionalFieldGrid(FieldGridBase):

    def __init__(self, x_axis, y_axis):
        super().__init__()
        self._node_positions = self._create_nodes(x_axis, y_axis)
        self._cell_connectivity = self._create_cells(x_axis, y_axis)
        self._cell_sizes = self._calculate_cell_sizes()
        self._lower_left_corner = self._assign_lower_left_corner()
        self._intervals = [x_axis.interval_count, y_axis.interval_count]

    def get_containing_cell(self, point_coordinates):
        number_of_points = self._get_number_of_points(point_coordinates)
        point_coordinates_two_dimensional = np.reshape(point_coordinates, [number_of_points, 2])
        cell_indices = np.zeros([number_of_points, 2])
        for i in range(2):
            cell_indices[:, i] = self._get_axis_index(point_coordinates_two_dimensional[:, i],
                                                      self._lower_left_corner[i],
                                                      self._cell_sizes[i],
                                                      self._intervals[i])

        containing_cells = np.rint(cell_indices[:, 0] + cell_indices[:, 1] * self._intervals[0])
        return containing_cells.astype(int)

    def _get_number_of_points(self, point_coordinates):
        shape = np.shape(point_coordinates)
        if len(shape) > 1:
            return shape[0]
        else:
            return 1

    def get_cell_nodes(self, cell_index):
        return self._cell_connectivity[cell_index, :]

    def get_cell_node_locations(self, cell_index):
        nodes = self.get_cell_nodes(cell_index)
        locations = self._node_positions[nodes, :]
        return locations

    def get_cell_centers(self):
        centers = np.zeros([self.cell_count, self._space_dim])
        for cell_index in range(self.cell_count):
            node_positions = self.get_cell_node_locations(cell_index)
            centers[cell_index, :] = np.average(node_positions, axis=0)
        return centers

    def get_cell_areas(self):
        return np.ones(self.cell_count) * self._cell_sizes[0] * self._cell_sizes[1]

    @property
    def node_count(self):
        return np.shape(self._node_positions)[0]

    @property
    def cell_count(self):
        return np.shape(self._cell_connectivity)[0]

    @property
    def node_positions(self):
        return self._node_positions

    @property
    def cell_connectivity(self):
        return self._cell_connectivity

    @property
    def cell_sizes(self):
        size_array = np.ones([3, self.cell_count])
        average = 0
        for i in range(self._space_dim):
            size_array[i, :] = np.ones(self.cell_count) * self._cell_sizes[i]
            average += self._cell_sizes[i] / self._space_dim
        size_array[2, :] = np.ones(self.cell_count) * average
        return size_array

    def _create_nodes(self, x_axis, y_axis):
        x_range = self._create_axis(x_axis)
        y_range = self._create_axis(y_axis)
        return self._make_nodes_from_ranges(x_range, y_range)

    def _create_axis(self, axis):
        return np.linspace(axis.lower_bound, axis.upper_bound, axis.node_count)

    def _make_nodes_from_ranges(self, x_range, y_range):
        gridx, gridy = np.meshgrid(x_range, y_range)
        return np.array([gridx.flatten(), gridy.flatten()]).T

    def _create_cells(self, x_axis, y_axis):
        total_cells = x_axis.interval_count * y_axis.interval_count
        cells = np.zeros([total_cells, 4], dtype=int)
        for i in range(total_cells):
            cells[i] = self._create_connections(i, x_axis)
        return cells

    def _create_connections(self, cell_idx, x_axis):
        ll_idx = self._calculate_lower_left_index(cell_idx, x_axis)
        return self._compose_cell(ll_idx, x_axis)

    def _calculate_lower_left_index(self, cell_idx, x_axis):
        offset = int(np.floor(cell_idx / x_axis.interval_count))
        idx = offset + cell_idx
        return idx

    def _compose_cell(self, lower_left_index, x_axis):
        return np.array([lower_left_index,
                         lower_left_index + 1,
                         lower_left_index + x_axis.node_count + 1,
                         lower_left_index + x_axis.node_count],
                        dtype=int)

    def _calculate_cell_sizes(self):
        cell = self.get_cell_nodes(0)
        displacement = self._node_positions[cell[2]] - self._node_positions[cell[0]]
        return displacement

    def _assign_lower_left_corner(self):
        return self._node_positions[0]

    def _get_axis_index(self, location, lower_bound, delta, interval_count):
        axis_index = np.floor((location - lower_bound) / delta)
        max_index = interval_count - 1
        axis_index = self._check_bounds(axis_index, max_index)
        return axis_index

    def _check_bounds(self, axis_index, max_index):
        if np.any(axis_index > max_index):
            logger.warning("When binning point, point found outside of grid (Values greater than max).\n"
                           "Point will be pulled to nearest cell")
            axis_index[np.where(axis_index > max_index)] = max_index
        if np.any(axis_index < 0):
            logger.warning("When binning point, point found outside of grid (Values less than min).\n"
                           "Point will be pulled to nearest cell")
            axis_index[np.where(axis_index < 0)] = 0
        return axis_index


def _elementwise_node_position(node_id, node_set, cell_connectivity, node_positions):
    node_locs = node_positions[cell_connectivity[:, node_set[node_id]], :]
    return node_locs


def _make_vector_set(node_set_list, cell_connectivity, node_positions):
    vectors = []
    for node_set in node_set_list:
        end_node = _elementwise_node_position(0, node_set, cell_connectivity, node_positions)
        start_node = _elementwise_node_position(1, node_set, cell_connectivity, node_positions)
        vectors.append(end_node - start_node)
    return vectors


def _vector_area(vector_set):
    return np.cross(vector_set[0], vector_set[1])


class MeshSkeletonTwoDimensionalMesh(FieldGridBase):

    def __init__(self, mesh_skeleton):
        self._node_positions = self._get_x_and_y(mesh_skeleton.spatial_coords)
        self._cell_connectivity =  mesh_skeleton.connectivity
        self._cell_areas = self._calculate_cell_areas()
        self._cell_centers = self._calculate_cell_centers()
        self._longest_length = self._calculate_longest_length()
        self._element_bins = self._initialize_bins()
        self._containing_elements = None

    def get_containing_cell(self, point_coordinates):
        n_points = np.shape(point_coordinates)[0]
        if self._can_reuse_element_lookup(n_points):
            return self._containing_elements
        else:
            self._containing_elements = -1 * np.ones(n_points, dtype=int)
            for point_index in range(n_points):
                current_location = point_coordinates[point_index, :]
                containing_ele = self._element_bins.find_containing_element(current_location)
                if containing_ele < 0:
                    logger.info(f"WARNING: cloud point {point_index} @ ({current_location}) falls outside of computational mesh")
                                               
                self._containing_elements[point_index] = containing_ele
        return self._containing_elements

    def get_cell_node_locations(self, cell_index):
        nodes = self.get_cell_nodes(cell_index)
        locations = self._node_positions[nodes, :]
        return locations

    def get_cell_nodes(self, cell_index):
        return self._cell_connectivity[cell_index]

    @property
    def node_count(self):
        return np.shape(self._node_positions)[0]

    @property
    def cell_count(self):
        return np.shape(self._cell_connectivity)[0]

    @property
    def node_positions(self):
        return self._node_positions

    @property
    def cell_connectivity(self):
        return self._cell_connectivity

    def get_cell_areas(self):
        return self._cell_areas

    @property
    def cell_sizes(self):
        pass

    def get_cell_centers(self):
        return self._cell_centers

    def _get_x_and_y(self, points):
        if points.shape[1] == 3:
            return points[:,:2]
        else:
            return points

    def _can_reuse_element_lookup(self, n_points):
        return self._containing_elements is not None and n_points == np.size(self._containing_elements)

    def _calculate_cell_areas(self):
        first_vector_set = _make_vector_set([[2, 1], [0, 1]], self._cell_connectivity, self._node_positions)
        second_vector_set = _make_vector_set([[0, 3], [2, 3]], self._cell_connectivity, self._node_positions)
        area = _vector_area(first_vector_set) / 2
        area += _vector_area(second_vector_set) / 2
        return area

    def _calculate_cell_centers(self):
        element_position_vectors = self._node_positions[self._cell_connectivity, :]
        cell_centers = np.average(element_position_vectors, axis=1)
        return cell_centers

    def _calculate_longest_length(self):
        all_edges_and_diagonals = _make_vector_set([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3]],
                                                   self._cell_connectivity, self._node_positions)
        edge_and_diagonal_lengths = np.linalg.norm(all_edges_and_diagonals, axis=2)
        return np.max(edge_and_diagonal_lengths)

    def _calculate_bin_count_and_sizes(self):
        corners = self.get_grid_corners()
        box_lengths = np.array([corners[2][0] - corners[0][0], corners[2][1] - corners[0][1]])
        number_of_bins = np.floor(box_lengths / self._longest_length).astype(int)
        small_ids = np.argwhere(number_of_bins < 1)
        number_of_bins[small_ids] = 1
        bin_lengths = np.divide(box_lengths, number_of_bins)
        return number_of_bins, bin_lengths

    def _initialize_bins(self):
        number_of_bins, bin_lengths = self._calculate_bin_count_and_sizes()
        lower_left_corner = self.get_grid_corners()[0]
        element_bins = ElementBins(number_of_bins[0], number_of_bins[1], bin_lengths[0], bin_lengths[1],
                                   lower_left_corner[0], lower_left_corner[1])
        element_bins.bin(self._cell_centers, self._cell_connectivity, self._node_positions)
        return element_bins


def auto_generate_two_dimensional_field_grid(number_x_nodes, number_y_nodes, field_data):
    range_x, range_y = _get_buffered_axis_range(field_data)
    axis_x, axis_y = _generate_axis_space(range_x, range_y, number_x_nodes, number_y_nodes)
    return TwoDimensionalFieldGrid(axis_x, axis_y)


def _generate_axis_space(range_x, range_y, number_x_nodes, number_y_nodes):
    axis_x = GridAxis(range_x[0], range_x[1], number_x_nodes)
    axis_y = GridAxis(range_y[0], range_y[1], number_y_nodes)
    return axis_x, axis_y


def _get_buffered_axis_range(field_data):
    range_x, range_y = _get_two_dimensional_field_range(field_data)
    range_x = _buffer_range(range_x)
    range_y = _buffer_range(range_y)
    return range_x, range_y


def _buffer_range(axis_range):
    axis_range[0] = _apply_lower_buffer(axis_range[0])
    axis_range[1] = _apply_upper_buffer(axis_range[1])
    return axis_range


def _apply_lower_buffer(coordinate, range_buffer=1e-10):
    if coordinate < 0:
        return coordinate * (1 + range_buffer)
    else:
        return coordinate * (1 - range_buffer)


def _apply_upper_buffer(coordinate, range_buffer=1e-10):
    if coordinate < 0:
        return coordinate * (1 - range_buffer)
    else:
        return coordinate * (1 + range_buffer)


def _get_two_dimensional_field_range(field_data):
    range_x = _get_field_range(0, field_data)
    range_y = _get_field_range(1, field_data)
    return range_x, range_y


def _get_field_range(dimension_index, field_data):
    upper_bound = np.max(field_data.spatial_coords[:, dimension_index])
    lower_bound = np.min(field_data.spatial_coords[:, dimension_index])
    upper_bound = np.max(field_data.spatial_coords[:, dimension_index])
    lower_bound = np.min(field_data.spatial_coords[:, dimension_index])
    return np.array([lower_bound, upper_bound])


class ElementBins:
    class RebinningError(RuntimeError):

        def __init__(self):
            message = "Binning structure has been previously binned, can not bin again"
            super().__init__(message)

    def __init__(self, x_bin_count, y_bin_count, x_bin_length, y_bin_length, lower_left_x, lower_left_y):
        self._buffer_size = 25
        self._bin_shape = np.array([x_bin_count, y_bin_count], dtype=int)
        self._total_bins = x_bin_count * y_bin_count
        self._bin_lengths = np.array([x_bin_length, y_bin_length])
        self._lower_left_corner = np.array([lower_left_x, lower_left_y])
        self._element_bins = -1 * np.ones([self._total_bins, self._buffer_size], dtype=int)
        self._elements_in_bin = np.zeros(self._total_bins, dtype=int)
        self._element_lookups = None

    def bin(self, element_center_array, element_connectivity, node_positions):
        self._check_if_rebinning()
        bin_ids = self._get_bin_ids(element_center_array)
        self._put_elements_in_bins(bin_ids)
        self._create_element_lookups(element_connectivity, node_positions)

    def get_elements(self, bin_id):
        return self._element_bins[bin_id, 0:self._elements_in_bin[bin_id]]

    def find_containing_element(self, location):
        bin_id = self._get_bin_ids(np.reshape(location, (1, 2)))[0]
        row = self._bin_shape[0]
        relevant_off_sets = [0, 1, row + 1, row, row - 1, -1, -1 - row, -row, 1 - row]
        element_in_bin_index = None
        for offset in relevant_off_sets:
            offset_bin = bin_id + offset
            if offset_bin < 0 or offset_bin >= self._total_bins:
                continue
            element_in_bin_index = self._element_lookups[offset_bin].find_containing_element(location)
            if element_in_bin_index is not None:
                break
        if element_in_bin_index is not None:
            element_id = int(self._element_bins[offset_bin, element_in_bin_index])
        else:
            element_id = -1
        return element_id

    class ElementNotFoundError(RuntimeError):
        def __init__(self, location):
            message = f"No Element found for point at location:\n {location}"
            super().__init__(self, message)

    def _create_element_lookups(self, element_connectivity, node_positions):
        self._element_lookups = []
        for bin_id in range(self._total_bins):
            if self._elements_in_bin[bin_id] < 1:
                self._element_lookups.append(NullContainingElementIdentifier())
            else:
                self._element_lookups.append(ContainingElementIdentifier(element_connectivity[self.get_elements(
                    bin_id)], node_positions))

    def _check_if_rebinning(self):
        if self._element_lookups is not None:
            raise self.RebinningError()

    def _put_elements_in_bins(self, bin_ids):
        for ele_idx, bin_id in enumerate(bin_ids):
            pos = self._elements_in_bin[bin_id]
            self._element_bins[bin_id, pos] = ele_idx
            self._elements_in_bin[bin_id] += 1

    def _get_bin_ids(self, element_center_array):
        steps = np.floor(np.divide(element_center_array - self._lower_left_corner, self._bin_lengths)).astype(int)
        steps = self._move_far_edge_inside(steps)
        bin_ids = self._steps_to_bin_id(steps).astype(int)
        return bin_ids

    def _move_far_edge_inside(self, steps):
        n_lookup = steps.shape[0]
        for point_idx in range(n_lookup):
            for pos_index in range(2):
                correction = 0
                if steps[point_idx, pos_index] == self._bin_shape[pos_index]:
                    correction = -1
                steps[point_idx, pos_index] += correction
        return steps

    def _steps_to_bin_id(self, steps):
        return steps[:, 0] + self._bin_shape[0] * steps[:, 1]


class ElementIdentifierBase(ABC):

    @abstractmethod
    def find_containing_element(self, location):
        """"""


class ContainingElementIdentifier(ElementIdentifierBase):
    """
    Containing Element identifier identifies if an point is inside a convex quadralateral by breaking the quadralaterial into 
    two triangles. Then the identifier checks to see if the point is within either of the two triangles. The triangle check is 
    done by creating a local basis using the edges of the quadralateral, and finding the location of the point in question in the 
    local basis. Because this basis is scaled to the lengths of the edges, if the coordinates are greater than zero, and sum to less
    than one, then the point is within the triange. 
    """

    def __init__(self, pruned_connectivity, node_locations):
        self._pruned_connectivity = pruned_connectivity
        self._n_elements = len(self._pruned_connectivity)
        self._node_positions = node_locations
        self._sets_per_element = 4
        self._basis_matrix = self._make_coordinate_matrix()

    def find_containing_element(self, location):
        coordinates = self._find_coordinates(location)
        for i_ele in range(self._n_elements):
            for i_tri in range(2):
                offset = i_ele * self._sets_per_element + i_tri * 2
                mapped_cords = coordinates[offset:offset + 2]
                if self._is_in_triangle(mapped_cords):
                    return i_ele
        return None

    def _is_in_triangle(self, coordinate):
        boundary_eps = 1e-12
        if np.min(coordinate) >= -boundary_eps and coordinate[0] + coordinate[1] <= 1. + boundary_eps:
            return True
        else:
            return False

    def _calculate_distance_vector(self, point_location):
        node_locs = self._node_positions[self._pruned_connectivity[:, [0, 2]], :]
        return np.array(-1 * (node_locs - point_location))

    def _find_coordinates(self, location):
        distance = self._calculate_distance_vector(location)
        residual = self._stamp_distance(distance)
        return spar_linalg.spsolve(self._basis_matrix, residual)

    def _stamp_distance(self, distance):
        residual = np.zeros(self._sets_per_element * self._n_elements)
        for element_index in range(self._n_elements):
            element_offset = self._sets_per_element * element_index
            residual[element_offset:element_offset + self._sets_per_element] = distance[element_index, :, :].flatten()
        return residual

    def _make_coordinate_matrix(self):
        basis_vectors = _make_vector_set([[1, 0], [3, 0], [1, 2], [3, 2]], self._pruned_connectivity,
                                         self._node_positions)
        return self._fill_matrix(self._n_elements, basis_vectors)

    def _fill_matrix(self, n_elements, basis_vectors):
        matrix = sparse.lil_matrix((self._sets_per_element * n_elements, self._sets_per_element * n_elements))
        for element_index in range(n_elements):
            ele_offset = element_index * self._sets_per_element
            for triangle_index in range(2):
                tri_offset = triangle_index * 2
                total_offset = ele_offset + tri_offset
                tri_matrix = np.array(
                    [basis_vectors[tri_offset][element_index], basis_vectors[tri_offset + 1][element_index]]).T
                for i in range(2):
                    for j in range(2):
                        matrix[total_offset + i, total_offset + j] = tri_matrix[i, j]
        return matrix.tocsr()


class NullContainingElementIdentifier(ElementIdentifierBase):

    def find_containing_element(self, location):
        return None
