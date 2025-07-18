from abc import ABC, abstractmethod
from collections import OrderedDict
import numpy as np
from scipy import sparse
from scipy.optimize import root
from typing import Callable

from matcal.core.utilities import _time_interpolate
from matcal.core.utilities import check_item_is_correct_type
from matcal.full_field.data import FieldData, convert_dictionary_to_field_data
from matcal.full_field.data_importer import FieldSeriesData
from matcal.full_field.data_exporter import MatCalFieldDataExporterIdentifier 
from matcal.full_field.NodeData import NodeData
from matcal.full_field.shapefunctions import TwoDim4NodeBilinearShapeFunction

from matcal.core.logger import initialize_matcal_logger
logger = initialize_matcal_logger(__name__)

class FieldProjectorBase(ABC):

    @abstractmethod
    def _shape_function(self):
        """"""

    class InvalidFieldnameError(RuntimeError):
        def __init__(self, fieldname, possible_fieldnames):
            message = "Field '{}' is not available for projection.\n" \
                      "Possible fields:\n{}".format(fieldname, possible_fieldnames)
            super().__init__(self, message)

    def __init__(self, source_data, target_domain):
        self._source_data = source_data
        self._target_domain = target_domain
        self._node_data = NodeData()
        self._local_element_data = _local_element_systems(self._target_domain.cell_count, self._shape_function)

    @abstractmethod
    def project_field(self, fieldname):
        """"""

    @abstractmethod
    def get_results_data(self):
        """"""

class _local_element_systems:
    _contained_point_buffer = 200

    def __init__(self, number_of_elements, shape_function):
        self.element_residual = np.zeros([number_of_elements, self._contained_point_buffer])
        self.element_node_values = np.zeros([number_of_elements, self._contained_point_buffer,
                                             shape_function.number_of_functions])
        self.contained_points = np.zeros([number_of_elements, self._contained_point_buffer], dtype=int)
        self.contained_point_count = np.zeros(number_of_elements, dtype=int)

    @property
    def buffer_size(self):
        return self._contained_point_buffer

    @property
    def number_of_points(self):
        return np.sum(self.contained_point_count)

    def reset_local_system(self):
        self.contained_point_count = self.contained_point_count * 0
        self.element_residual = self.element_residual * 0


class _InterpolationMatrixGenerator:

    def __init__(self, shape_function):
        self._shape_function = shape_function
        
    def generate(self, grid, cloud_points):
        num_grid_points, num_cloud_points, num_elem = self._calc_counts(grid, cloud_points)
        M = sparse.lil_matrix((num_cloud_points, num_grid_points))
        cloud_to_matrix_map = -np.ones(num_cloud_points, dtype=int)
        local_element_data = self._generate_local_element_data(num_elem, self._shape_function, grid, cloud_points)
        for element_index in range(num_elem):
            grid_point_indices = grid.cell_connectivity[element_index, :]
            num_contained_points = local_element_data.contained_point_count[element_index]
            for contained_point_index in range(num_contained_points):
                reference_cloud_index = local_element_data.contained_points[element_index, contained_point_index]
                M[reference_cloud_index, grid_point_indices] = local_element_data.element_node_values[element_index,
                                                          contained_point_index, :]
        interp_mat = sparse.csr_matrix(M)
        e_col = 1-np.ones(interp_mat.shape[0],dtype=bool)*interp_mat.astype(bool)
        if np.any(e_col) > 0:
            logger.info("WARNING: Unsupported nodes in matrix. Use of the current mesh may cause problems for VFM calibrations")
        return interp_mat

    def _calc_counts(self, grid, cloud_points):
        num_grid_points = grid.node_count
        num_elem = grid.cell_count
        num_cloud_points = np.shape(cloud_points)[0]
        return num_grid_points,num_cloud_points,num_elem

    def _generate_local_element_data(self, num_elem, shape_function, grid, cloud_points):
        local_elem_data = _local_element_systems(num_elem, shape_function)
        cloud_points_to_element_map = grid.get_containing_cell(cloud_points)
        for point_index, element_index in enumerate(cloud_points_to_element_map):
            if element_index < 0:
                continue
            shape_function_values = self._caculate_shape_function_values(element_index, grid, cloud_points, point_index)
            local_elem_data = self._add_shapefunction_data_to_local_element_data(local_elem_data, shape_function_values, element_index, point_index)
        return local_elem_data
    
    def _caculate_shape_function_values(self, element_index, grid, cloud_points, point_index):
        mapper = _LabToParametricSpaceMapper(cloud_points[point_index, :],
                                            grid.get_cell_node_locations(element_index))
        parametric_coordinates = mapper.calculate_parametric_location()
        shape_function_values = self._shape_function.values(parametric_coordinates)
        return shape_function_values

    def _add_shapefunction_data_to_local_element_data(self, local_element_data, array_values, element_index, point_index):
        equation_point_index = self._get_equation_point_index(element_index, local_element_data)
        local_element_data.element_node_values[element_index, equation_point_index, :] = array_values
        local_element_data.contained_points[element_index, equation_point_index] = point_index
        self._increment_equation_point_index(element_index, local_element_data)
        return local_element_data

    def _increment_equation_point_index(self, element_index, local_element_data):
        local_element_data.contained_point_count[element_index] += 1

    def _get_equation_point_index(self, element_index, local_element_data):
        equation_index = local_element_data.contained_point_count[element_index]
        return equation_index


class _TwoDimensionalFieldProjector(FieldProjectorBase):
    _shape_function = TwoDim4NodeBilinearShapeFunction()

    def __init__(self, source_data, target_domain):
        super().__init__(source_data, target_domain)
        self._matrix = None
        self._cloud_map = None

    def project_field(self, fieldname, frame_index):
        self._projection_error_checking(fieldname)
        if self._matrix_is_empty():
            self._create_projection_matrix()
        self._node_data = NodeData()
        self._project(fieldname, frame_index)
        return self.get_results_data()

    def get_results_data(self):
        return self._node_data.get_full_data()

    def reset(self, new_source_data=None):
        if (new_source_data is not None):
            self._source_data = new_source_data
            self._create_projection_matrix()
        self._node_data = NodeData()
        return self

    def _projection_error_checking(self, field_name):
        if field_name not in self._source_data.field_names:
            raise self.InvalidFieldnameError(field_name, self._source_data.field_names)

    def _matrix_is_empty(self):
        return self._matrix == None

    def _create_projection_matrix(self):
        mat_generator = _InterpolationMatrixGenerator(self._shape_function)
        self._matrix = mat_generator.generate(self._target_domain, self._source_data.spatial_coords) 

    def _project(self, fieldname, index):
        residual = self._source_data[fieldname][index]
        field_values = self._solve(residual)
        self._node_data.add_node_data(fieldname, np.array(field_values))

    def _solve(self, residual):
        result = sparse.linalg.lsqr(self._matrix, residual.T,atol=1e-12, btol=1e-12)
        return result[0]

class _NodeMapper:
    def __init__(self, buffer_size):
        self._gtol = -np.ones(buffer_size, dtype=int)
        self._ltog = -np.ones(buffer_size, dtype=int)
        self._item_count = 0

    def append(self, value):
        if int(value) not in self._gtol:
            self._gtol[self._item_count] = value
            self._ltog[value] = self._item_count
            self._item_count += 1

    def getGlobalToLocal(self, idx):
        return self._gtol[idx]

    def getLocalToGlobal(self, idx):
        g = self._ltog[idx]
        if np.all(g < 0):
            raise RuntimeError("Indexing Eliminated Node")
        return g

    @property
    def size(self):
        return self._item_count

class _LabToParametricSpaceMapper:
    _shape_function = TwoDim4NodeBilinearShapeFunction()

    def __init__(self, location_array, element_location_array):
        self._goal_location = location_array
        self._cell_locations = element_location_array

    def calculate_parametric_location(self):
        initial_guess = np.zeros([1, 2])
        results = root(self._calculate_residual, initial_guess, jac=self._calculate_tangent)
        return results.x.reshape((1,2))

    def _calculate_residual(self, parametric_location):
        shape_function_values = self._shape_function.values(parametric_location.reshape([1,2]))
        cell_location = np.dot(shape_function_values, self._cell_locations)
        residual = cell_location - self._goal_location
        return residual[0]

    def _calculate_tangent(self, parametric_location):
        shape_function_grad = self._shape_function.gradients(parametric_location.reshape([1,2]))
        sf_x = shape_function_grad[0,0,:]
        sf_y = shape_function_grad[0,1,:]
        dpdx = np.dot(sf_x, self._cell_locations)
        dpdy = np.dot(sf_y, self._cell_locations)
        tangent = -1 * np.array([[dpdx[0],dpdy[0]], [dpdx[1], dpdy[1]]])
        return tangent

class _TwoDimensionalFieldInterpolator:
    _shape_function = TwoDim4NodeBilinearShapeFunction()

    def __init__(self, grid_geometry, cloud_points):
        self._grid = grid_geometry
        self._cloud_points = cloud_points
        self._matrix = self._generate_interpolation_matrix()
        

    def interpolate(self, grid_data):
        num_fields = self._parse_number_of_fields(grid_data)
        result = self._matrix.dot(grid_data)
        return result.reshape(-1,num_fields)
    
    @property
    def number_of_nodes_required(self):
        return self._matrix.shape[1]

    def _parse_number_of_fields(self, grid_data):
        if grid_data.ndim > 1:
            return grid_data.shape[1]
        else:
            return 1

    def _generate_interpolation_matrix(self):
        M_i = self._make_interpolation_matrix_from_transposed_projection_matrix()
        return M_i
    
    def _make_interpolation_matrix_from_transposed_projection_matrix(self):
        mat_gen = _InterpolationMatrixGenerator(self._shape_function)
        proj_Mat = mat_gen.generate(self._grid, self._cloud_points)
        return proj_Mat


class NonIntegerPolynomialOrderError(RuntimeError):
    def __init__(self, value):
        message = f"Polynomial Order must be of type int, passed {type(value)}"
        super().__init__(message)

class SmallSearchRadiusError(RuntimeError):

    def __init__(self, mult):
        message = f"Search Radius multiplier should be greater than 1.05, passed {mult}"
        super().__init__(message)

class BadPolynomialOrderError(RuntimeError):

    def __init__(self, order):
        message = f"Polynomial order should be >= 1 and < 11, passed {order}"
        super().__init__(message)

class BadSearchTypeError(RuntimeError):

    def __init__(self, mult):
        message = f"Search radius multiplier needs to be of type double or int, passed {type(mult)}"
        super().__init__(message)

def _check_gmls_parameters(polynomial_order, epsilon_multiplier):
        if not isinstance(polynomial_order, int):
            raise NonIntegerPolynomialOrderError(polynomial_order)
        if polynomial_order < 1 or polynomial_order > 10:
            raise BadPolynomialOrderError(polynomial_order)
        
        if not isinstance(epsilon_multiplier, (float, int)):
            raise BadSearchTypeError(epsilon_multiplier)
        if epsilon_multiplier < 1.05:
            raise SmallSearchRadiusError(epsilon_multiplier)


class MeshlessMapperGMLS:
    """
    Class that performs meshless mapping.
    """
    count = 0
    default_polynomial_order=1 #: Default value for polynomial order.
    default_epsilon_multiplier=2.75 #: Default value for search radius multiplier
    def __init__(self, target_coords, source_coords, polynomial_order=default_polynomial_order, 
                 epsilon_multiplier=default_epsilon_multiplier, number_of_batches=2):
        _check_gmls_parameters(polynomial_order, epsilon_multiplier)
        self._increment_total_instances_count()
        n_dim = target_coords.shape[1]
        self._n_source_points = source_coords.shape[0]
        self._kokkos_parser = None
        self._gmls = None
        self._helper = None
        self._initialize_gmls_tool(target_coords, source_coords, 
                                   polynomial_order, epsilon_multiplier, 
                                   number_of_batches, n_dim)



    def _initialize_gmls_tool(self, target_coords, source_coords, 
                              polynomial_order, epsilon_multiplier, 
                              number_of_batches, n_dim):
        import pycompadre

        try:
            self._kokkos_parser = pycompadre.KokkosParser()
            self._gmls = pycompadre.GMLS(polynomial_order, n_dim)
            self._helper = pycompadre.ParticleHelper(self._gmls)
        except:
            self.finish()
            raise self.InitializeError()
        try:
            self._helper.generateKDTree(source_coords)
            self._helper.generateNeighborListsFromKNNSearchAndSet(target_coords, 
                                                                  polynomial_order, 
                                                                  n_dim,
                                                                  epsilon_multiplier)
        except (Exception, RuntimeError):
            self.finish()
            raise self.NeighborDectectionError()
        try:
            self._gmls.addTargets(pycompadre.TargetOperation.ScalarPointEvaluation)
            self._gmls.generateAlphas(number_of_batches, keep_coefficients=False)
        except Exception:
            self.finish()
            raise self.AlphaGenerationError()

    def finish(self):
        try:
            del self._gmls
        except:
            pass
        try:
            del self._helper
        except:
            pass
        if self._one_instance_remains():
            try:
                del self._kokkos_parser
            except:
                pass
        self._decrement_total_instances_count()

    def _one_instance_remains(self):
        return __class__.count == 1

    def _increment_total_instances_count(self):
        __class__.count += 1

    def _decrement_total_instances_count(self):
        __class__.count -=1

    def map(self, source_value):
        import pycompadre
        n_points = source_value.shape[0]
        if n_points != self._n_source_points:
            self.finish()
            raise self.IncorrectLengthError(self._n_source_points, n_points)
        try:
            map_values = self._helper.applyStencil(source_value, 
                                                   pycompadre.TargetOperation.ScalarPointEvaluation)
        except Exception:
            raise self.MappingError()
        return map_values
    
    class NeighborDectectionError(RuntimeError):

        def __init__(self):
            message = "Nearest Neighbor Detection Failure:  Examine point clouds to ensure sufficient support"
            super().__init__(message)

    class InitializeError(RuntimeError):

        def __init__(self):
            message = 'Something went wrong with GMLS initialization'
            super().__init__(message)
    
    class AlphaGenerationError(RuntimeError):
        pass

    class MappingError(RuntimeError):
        pass

    class IncorrectLengthError(RuntimeError):
        def __init__(self, source_point_size, passed_point_size):
            message = f"Initialized point count({source_point_size}) is not equal to the length of the passed value array({passed_point_size})"
            super().__init__(message)


def meshless_remapping(field_data, fields_to_map, target_points,
                       polynomial_order=MeshlessMapperGMLS.default_polynomial_order, 
                       search_radius_multiplier=MeshlessMapperGMLS.default_epsilon_multiplier,
                       target_time=None, time_field=None):
    """
    Stand alone function for performing meshless interpolation between two point clouds.
    This function uses generalized moving least squares(GMLS) to perform local interpolations. 
    GMLS tools provided by pycompadre.

    This function is intended for interpolation, but has limited ability to do extrapolation. 
    If extrapolation is expected, use lower order(<=2) polynomials to reduce 
    errant edge effects of higher order polynomials. 
    For default values of the search_radius_multiplier and polynomial_order 
    see :class:`~matcal.full_field.field_mappers.MeshlessMapperGMLS`

    :param field_data: FieldData object that contains the data to be mapped to a new point cloud. 
    :type field_data: :class:`~matcal.full_field.data.FieldData`

    :param fields_to_map: List of field names to be mapped from the provided field data to the target_points.
    :type fields_to_map: list(str)

    :param target_points: Two-dimensional array containing the points to be interpolated on.
        each column holding the coordinate, and each row representing a new point. Currently only tested for 
        two-dimensional interpolation. Other dimensions may work as well, but should be used with caution. 
    :type target_points: ArrayLike

    :param polynomial_order: The order of polynomial to use for interpolation/extrapolation.  
    :type polynomial_order: int

    :param search_radius_multiplier: multiplier used to gather additional interpolation points once the minimum radius for 
        a given polynomial order is reached. Higher values will include more points and in-general have a greater smoothing effect. 
        Recommended values for this parameter are between 1.5 and 3. 
    :type search_radius_multiplier: float  
    """
    mapping_tool = MeshlessMapperGMLS(target_points, field_data.spatial_coords,
                                       polynomial_order, search_radius_multiplier)
    mapped_data = {}

    for field in fields_to_map:
        cloud_field = field_data[field]
        n_time = cloud_field.shape[0]
        mesh_field = np.zeros([n_time, target_points.shape[0]])
        for i_time in range(n_time):
            mesh_field[i_time, :] = mapping_tool.map(cloud_field[i_time,:])
        mapped_data[field] = mesh_field
    for field in field_data.field_names:
        current_data  = field_data[field]
        if _is_a_global_field(field, current_data):
            mapped_data[field] = current_data

    mapped_field_data = convert_dictionary_to_field_data(mapped_data)
    mapped_field_data.set_spatial_coords(target_points)
    mapping_tool.finish()
    if _has_information_for_time_interp(target_time, time_field):
        mapped_field_data = _map_in_time(target_time, time_field, mapped_field_data)
    
    return mapped_field_data


def _has_information_for_time_interp(target_time, time_field):
    return isinstance(time_field, str) and len(target_time) > 0


def _map_in_time(target_time, time_field, mapped_field_data):
    time_mapped_data = {}
    source_time = mapped_field_data[time_field]
    if source_time.ndim > 1:
        raise BadTimeDataShape(time_field, source_time.ndim)
    for field in mapped_field_data.field_names:
        source_data = mapped_field_data[field]
        target_data = _time_interpolate(target_time, source_time, source_data)
        time_mapped_data[field] = target_data
    time_mapped_data = convert_dictionary_to_field_data(time_mapped_data)
    time_mapped_data.set_spatial_coords(mapped_field_data.spatial_coords)
    return time_mapped_data


def _is_a_global_field(field, current_data):
    is_global = current_data.ndim == 1 
    return is_global


class FullFieldCalculator:
    """
    A class for generating spatially dependent measurements between 
    different sets of full-field data.
    """
    
    def __init__(self, reference_data, independent_field_vals, 
                 independent_field_name="time", position_names=['X', 'Y'], 
                 add_global_variables=True)->None:
        """
        :param reference_data: Passed as either a path to a full-field data file or
            a MatCal :class:`~matcal.full_field.data.FieldData` object. This set the reference data file.
        
        :param independent_field_vals: An array of times or other time-like values to be used as the 
            interpolation points in time for calculations.

        :param independent_field_name: The name of a field to be used as
            the time interpolation axis. 
            Any monotonic global field can be used. For example applied displacement can be
            used for uniaxial tension data. This field must exist on all data sets used.
        :type time_field_name: str

        :param position_names: A list of the different position names contained in any data sets used. 
            The number of names passed will change the dimensionality of the problem. (Passing ['x', 'y'] 
            will create a 2D problem, while passing ['x', 'y', 'z'] will create a 3D problem.) 
            This value defaults to ['X', 'Y'].
        :type position_names: list(str)
        
        :param add_global_variables: copy global variables from all data sources to 
            the new field data class.
        :type add_global_variables: bool
        """
        self._check_position_names_input(position_names)
        self._position_names = position_names

        check_item_is_correct_type(independent_field_vals, (list, np.ndarray), 
                                   "FullFieldCalculator", "independent_field_vals", 
                                   TypeError)
        self._independent_field_vals = independent_field_vals

        self._data_sets = OrderedDict()
        self._spatial_calculations = OrderedDict()

        check_item_is_correct_type(independent_field_name, str, "FullFieldCalculator", 
                                   "independent_field_name", TypeError)
        self._independent_field_name = independent_field_name

        check_item_is_correct_type(reference_data, (str, FieldData), 
                                   "FullFieldCalculator", "reference_data", 
                                   TypeError)        
        self._original_reference_data = reference_data
        self._reference_data = self._process_data_input(reference_data, 
                                                        self._position_names)
        self._mapping_polynomial_order = MeshlessMapperGMLS.default_polynomial_order
        self._mapping_search_radius_multiplier = MeshlessMapperGMLS.default_epsilon_multiplier
        
        check_item_is_correct_type(add_global_variables, bool, "FullFieldCalculator", 
                                   "global_variables", TypeError)
        self._add_global_variables = add_global_variables
    def _check_position_names_input(self, position_names):
        check_item_is_correct_type(position_names, list, "FullFieldCalculator", 
                                   "position_names", TypeError)
        for pos_name in position_names:
            check_item_is_correct_type(pos_name, str, "FullFieldCalculator", 
                                   "position_name", TypeError)

    def set_mapping_parameters(self, polynomial_order, search_radius_multiplier):
        """
        Set the mapping parameters for the PyCompadre GMLS mapping algorithm. 
        See :func:`~matcal.full_field.field_mappers.meshless_remapping` for more 
        information on the mapping parameters.
        """
        import numbers
        check_item_is_correct_type(polynomial_order, numbers.Integral, 
                                   "FullFieldCalculator.set_mapping_parameters", 
                                   "polynomial_order", TypeError)
        check_item_is_correct_type(search_radius_multiplier, numbers.Real, 
                                   "FullFieldCalculator.set_mapping_parameters", 
                                   "search_radius_multiplier", TypeError)
        self._mapping_polynomial_order=polynomial_order
        self._mapping_search_radius_multiplier=search_radius_multiplier

    def calculate_and_export(self, export_filename:str, file_type=None)->None:
        """
        Perform calculations on the reference data, and the added data sets. Loops over all added 
        data files and calculations, added with :add_data: and: add_spatial_calculation:, respectively.
        The results are exported to an external data file defined by the user.

        Exported data includes all fields used in calculations. This data will appear as 
        "<field_name>_ref" or "<field_name>_interp_<data_name>" with the former indicating values
        from the reference data and the later indicating the interpolated values from the additional 
        data to the reference locations. 

        The calculated fields will be named "<function_name>_<data_name>". 

        :param export_filename: filename to store the calculated data to. 
        :type export_filename: str

        :param file_type: By default MatCal will select the exporter and file type 
            for export based on the extension of export_filename. However, the 
            file type can be manually specified here. The "json" file type 
            is supported in MatCal core; 
            however, other file types may be available in other modules. 
        :type file_type: str
        """
        check_item_is_correct_type(export_filename, str, 
                                   "FullFieldCalculator.calculate_and_export",
                                   "export_filename", TypeError)
        if file_type is not None:
            check_item_is_correct_type(file_type, str, 
                                       "FullFieldCalculator.calculate_and_export",
                                       "file_type", TypeError)
        
        self._precalculate_check()
        fields_to_map = self._assemble_fields_to_map()
        dict_to_export = self._initialize_export_dict(fields_to_map)
        for current_data_name, data_input in self._data_sets.items():
            current_data = self._process_data_input(data_input,
                                                    self._position_names)
            if self._add_global_variables:
                dict_to_export.update(self._get_interp_global_fields(current_data, 
                                                                     current_data_name))

            mapped_data_dict = self._get_mapped_reference_data(current_data, current_data_name, 
                                                          fields_to_map)
            spatial_calc_data_dict = self._perform_spatial_calculations(mapped_data_dict, 
                                                                        current_data_name)
            dict_to_export.update(mapped_data_dict)
            dict_to_export.update(spatial_calc_data_dict)
        if self._add_global_variables:
            dict_to_export.update(self._get_interp_global_fields(self._reference_data, 
                                                                     "ref"))
        data_to_export = self._prepare_export_data(dict_to_export)
        self._export_data(export_filename, data_to_export, file_type)

    def _assemble_fields_to_map(self):
        fields_to_map = []
        for function, field_names in self._spatial_calculations.values():
            for field_name in field_names:
                if field_name not in fields_to_map:
                    fields_to_map.append(field_name)
        return fields_to_map
    
    def _initialize_export_dict(self, fields_to_map):
        export_dict = {}
        for field in fields_to_map:
            ref_field_name = self._get_mapped_field_name(field, "ref")
            export_dict[ref_field_name] = self._reference_data[field]
        return export_dict

    def _process_data_input(self, ref_data, position_names):
        data = None
        data = self._parse_data(ref_data, position_names)
        current_time_series = data[self._independent_field_name]
        if current_time_series.ndim > 1:
            raise BadTimeDataShape(self._independent_field_name, current_time_series.ndim)
        processed_data = _map_in_time(self._independent_field_vals, self._independent_field_name, 
                            data)
        processed_data._graph = data._graph
        return processed_data

    def _get_mapped_reference_data(self, current_data, 
                                       current_data_name, fields_to_map):
        mapped_data = meshless_remapping(current_data, fields_to_map,
                                        self._reference_data.spatial_coords, 
                                        self._mapping_polynomial_order,
                                        self._mapping_search_radius_multiplier,
                                        self._independent_field_vals,
                                        self._independent_field_name)
        mapped_data_dict = {}
        for field in fields_to_map:
            cur_field_name = self._get_mapped_field_name(field, current_data_name)
            mapped_data_dict[cur_field_name] = mapped_data[field]
        return mapped_data_dict

    def _get_mapped_field_name(self, field_name, current_data_name):
        return f"{field_name}_interp_{current_data_name}"

    def _perform_spatial_calculations(self, mapped_data_dict, 
                                                      current_data_name):
        dict_with_spatial_calc_fields = {}
        for calc_name, (calc_function, calc_fields) in self._spatial_calculations.items():
            for calc_field in calc_fields:
                new_calc_field_name = self._get_calculated_field_name(calc_name, 
                                                        current_data_name, 
                                                        calc_field)
                mapped_calc_field_name = self._get_mapped_field_name(calc_field, 
                                                            current_data_name)
                calc_value = calc_function(self._reference_data[calc_field], 
                                        mapped_data_dict[mapped_calc_field_name], 
                                        self._reference_data.spatial_coords, 
                                        self._independent_field_vals)
                dict_with_spatial_calc_fields[new_calc_field_name] = calc_value
        return dict_with_spatial_calc_fields
    
    def _get_calculated_field_name(self, calc_name, current_data_name, field):
        return f"{calc_name}_{current_data_name}_{field}"

    def _get_interp_global_fields(self, data, cur_data_name):
        global_fields_dict = {}
        for field in data.field_names:
            if len(np.shape(data[field])) <= 1:
                interped_global_field = _time_interpolate(self._independent_field_vals, 
                                  data[self._independent_field_name], 
                                  data[field])
                updated_field_name = self._get_mapped_field_name(field, cur_data_name)
                global_fields_dict[updated_field_name] = interped_global_field
        return global_fields_dict

    def _prepare_export_data(self, dict_to_export):
        dict_to_export[self._independent_field_name] = self._independent_field_vals
        data_to_export = convert_dictionary_to_field_data(dict_to_export)
        data_to_export._graph = self._reference_data._graph
        return data_to_export

    def _export_data(self, export_filename, data_to_export, file_type):
        exporter = self._get_exporter(export_filename, file_type)
        fields_to_export = data_to_export.field_names
        exporter(target_filename=export_filename, data_to_export=data_to_export, 
                 fields=fields_to_export, 
                 reference_source_mesh=self._original_reference_data, 
                 independent_field=self._independent_field_name)        

    def _get_exporter(self, export_filename, file_type):
        if file_type is None:
            file_type = export_filename.split(".")[-1]
        return MatCalFieldDataExporterIdentifier.identify(file_type)

    def add_spatial_calculation(self, calculation_name:str, 
                                calculation_function:Callable, *field_names)->None:
        """
        Add a new calculation to be preformed on the full-field data. 

        :param calculation_name: Name that will serve as the base for referencing the results of the
            caculations performed on the various full-field data sets. This name must be unique.
        :type calculation_name: str

        :param calculation_function: Function used for performing calculations on the full-field data.
            Funcations expect a call signature of (reference_field[n_time, n_points],
            current_field[n_time, n_points], point_locations[n_points, n_dim], time[n_time]).
        :type calculation_function: Callable

        :param field_names: fields to pass into the calculation function. These field must exist 
            on both the reference and additional data sets.
        :type field_names: list(str)
        """
        self._check_measurement_inputs(calculation_name, calculation_function,
                                        field_names, self._spatial_calculations)
        self._spatial_calculations[calculation_name] = [calculation_function, field_names]


    def add_data(self, data_name, field_data):
        """
        Add full field data to be compared to the reference data field.

        :param data_name: a name that will be used by reference this data set. This name must be unique. 
        :type data_name: str

        :param field_data: passed as either a path to a full-field data file or
          a MatCal :class:`~matcal.full_field.data.FieldData` object. This will add a new
          data set to used in the calculations. 
        """
        self._check_data_inputs(data_name, field_data)
        self._data_sets[data_name] = field_data

    def get_calculation_functions(self)->dict:
        """
        Return a dictionary of all the calculation functions
        """
        all_measurements = {}
        all_measurements['spatial'] = self._spatial_calculations
        return all_measurements

    def get_data(self)->dict:
        """
        Return a dictionary of all the additional datasets
        """
        return self._data_sets


    def _parse_data(self, input_data, position_names):
        if isinstance(input_data, str):
            data = FieldSeriesData(input_data, position_names=position_names)
        elif isinstance(input_data, FieldData):
            data = input_data
        return data

    def _precalculate_check(self):
        if not self._has_measurements():
            raise NoMeasurementError()
        n_data_sets = len(self._data_sets)
        n_required = 1
        if n_data_sets < n_required:
            raise InsuffichentFieldDataSetsError(n_data_sets, n_required)

    def _check_measurement_inputs(self, new_name:str, new_function:Callable,
                                   new_fields:str, measurement_record:dict)->None:
        check_item_is_correct_type(new_name, str, 
                                   "FullFieldCalculator.add_spatial_calculation", 
                                   "new measurement function name", TypeError)
        check_item_is_correct_type(new_function, Callable, 
                                   "FullFieldCalculator.add_spatial_calculation", 
                                   "calculation function", TypeError)
        check_item_is_correct_type(new_fields, tuple, 
                                   "FullFieldCalculator.add_spatial_calculation", 
                                   "calculation fields", TypeError)
        for field in new_fields:
            check_item_is_correct_type(field, str, 
                                   "FullFieldCalculator.add_spatial_calculation", 
                                   "a calculation field", TypeError)
        
        if new_name in list(measurement_record.keys()):
            old_function, old_fields = measurement_record[new_name]
            raise SameMeasurementNameError(new_name, old_function, new_function, 
                                           old_fields, new_fields)

    def _check_data_inputs(self, new_name, new_data):
        check_item_is_correct_type(new_name, str, "FullFieldCalculator.add_data", 
                                   "added data name", TypeError)
        check_item_is_correct_type(new_data, (str, FieldData), 
                                   "FullFieldCalculator.add_data", 
                                   "added data", TypeError)
        
        if new_name in list(self._data_sets.keys()):
            raise SameDataNameError(new_name)
        
    def _has_measurements(self):
        return len(self._spatial_calculations) > 0


class BadTimeDataShape(RuntimeError):

    def __init__(self, field_name, field_dim):
        message = f"time field name {field_name} is of dimension {field_dim}, must be 1D."
        super().__init__(message)


class SameMeasurementNameError(RuntimeError):

    def __init__(self, name, old_function, new_function, old_field, new_field):
        message = "Measurements must have unique names for full-field statistics."
        message += f"\nAttempted to add redundant measurements for name: {name}"
        message += f"\nOriginal field and measurement: {old_field}\n{old_function}"
        message += f"\nAttempted add field and measurement: {new_field}\n{new_function}\n"
        super().__init__(message)


class SameDataNameError(RuntimeError):

    def __init__(self, name):
        message = "Data must have unique names for full-field statistics."
        message += f"\nAttempted to add redundant data for name: {name}"
        super().__init__(message)


class InsuffichentFieldDataSetsError(RuntimeError):

    def __init__(self, n_data_sets, n_required_sets):
        message = f"Require {n_required_sets} for statistics calculation. Currently have {n_data_sets}"
        super().__init__(message)


class NoMeasurementError(RuntimeError):

    def __init__(self):
        message = "No measurements added to generate statistics please add them with:"
        message += "    add_spatial_measurement(name, function, field_name)"
        super().__init__(message)
