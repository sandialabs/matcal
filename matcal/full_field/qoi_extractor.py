"""
This module contains all classes related to 
data QoI extractors. Most are user facing classes that can 
be added to objectives, however, there a few that 
are not intended for users.
"""
from matcal.core.utilities import _time_interpolate
from matcal.full_field.data import FieldData
import numpy as np
from abc import abstractmethod

from matcal.core.data import Data, convert_dictionary_to_data
from matcal.core.logger import initialize_matcal_logger
from matcal.core.qoi_extractor import QoIExtractorBase

from matcal.full_field.field_mappers import MeshlessMapperGMLS, \
     _TwoDimensionalFieldInterpolator, meshless_remapping
from matcal.full_field.TwoDimensionalFieldGrid import MeshSkeleton, \
     MeshSkeletonTwoDimensionalMesh, _get_node_coordinate_mapping



logger = initialize_matcal_logger(__name__)

    
class MeshlessSpaceInterpolatorExtractor(QoIExtractorBase):

    class NodeMapFlagError(RuntimeError):
        pass

    def __init__(self, source_coords:np.array, target_coords:np.array, 
                 time_field:str, poly_order:int=1, eps:float=2.75):
        self._time_field =time_field
        self._space_interpolator = None
        self._init_params = (target_coords, source_coords, poly_order, eps)
        self._n_interp_points = target_coords.shape[0]

    @property
    def required_experimental_data_fields(self):
        return [self._time_field]


    def _init_mapper(self):
        if self._space_interpolator is None:
            self._space_interpolator = MeshlessMapperGMLS(*self._init_params)

    def clean_up(self):
        self._space_interpolator.finish()

    def calculate(self, working_data, reference_data, fields):
        self._init_mapper()
        qoi = {}
        n_time = len(working_data[self._time_field])
        for field in fields:
            field_data = working_data[field]
            interp_data = np.zeros([n_time, self._n_interp_points])
            for time_index in range(n_time):
                interp_vars_at_time = self._space_interpolator.map(field_data[time_index, :])
                interp_data[time_index, :] = interp_vars_at_time
            qoi[field] = interp_data.flatten()
        data_qoi = convert_dictionary_to_data(qoi)
        return data_qoi

    # @property
    # def number_of_nodes_required(self):
    #     return self._space_interpolator.number_of_nodes_required

    def _extract_data(self, working_data, field, working_nodes):
        data = working_data[field].T
        return data[working_nodes,:]


class FieldTimeInterpolatorExtractor(QoIExtractorBase):

    class NodeMapFlagError(RuntimeError):
        pass

    def __init__(self, mesh_skeleton, time_field):
        self._old_grid = mesh_skeleton
        self._time_field = time_field
        self._node_map = None

    @property
    def required_experimental_data_fields(self):
        return [self._time_field]

    def calculate(self, working_data, reference_data, fields):
        qoi = {}

        working_nodes = _get_node_coordinate_mapping(working_data.skeleton, 
                                                     self._old_grid, self._old_grid.subset_name)
        reference_time = reference_data[self._time_field]
        working_time = working_data[self._time_field]
        
        for field in fields:
            data_to_interp = self._extract_data(working_data, field, working_nodes)
            qoi[field] = _time_interpolate(reference_time, working_time, data_to_interp).flatten()
        data_qoi = convert_dictionary_to_data(qoi)
        return data_qoi

    # @property
    # def number_of_nodes_required(self):
    #     return self._space_interpolator.number_of_nodes_required

    def _extract_data(self, working_data, field, working_nodes):
        data = working_data[field]
        return data[:, working_nodes]


def _default_velocity_function(points, field_data):
    h = np.max(field_data.spatial_coords[:,1])-np.min(field_data.spatial_coords[:,1])
    y_offset = h/2+np.min(field_data.spatial_coords[:,1])
    centered_points = points - y_offset
    velocity = np.zeros((points.shape[0],2))
    velocity[:,0] = np.cos(np.pi*centered_points[:,1]/h)
    velocity[:,1] = (2*centered_points[:,1]+h)/(2*h)
    return velocity


class ExternalVirtualPowerExtractor(QoIExtractorBase):
    """
    VFM QoI extractor automatically applied to VFM calibrations and not intended for users.
    """

    def __init__(self, time_field, load_field, velocity_function=_default_velocity_function):
        self._time_field = time_field
        self._load_field = load_field
        self._velocity_function = velocity_function
        super().__init__()

    @property
    def required_experimental_data_fields(self) -> list:
        return []

    def calculate(self, working_data, reference_data, fields):
        velocity = self._get_y_velocity(reference_data)
        results_dict = {}
        results_dict[self._time_field] = working_data[self._time_field]
        results_dict['virtual_power'] = velocity * working_data[self._load_field]
        data = convert_dictionary_to_data(results_dict)
        data.set_state(working_data.state)
        return data

    def _get_y_velocity(self, field_data):
        velocity = self._calculate_virtual_velocity_at_load_location(field_data)[:,1]
        return velocity

    def _calculate_virtual_velocity_at_load_location(self, field_data):
        load_location = self._get_load_location(field_data)
        virtual_velocity = self._velocity_function(load_location, field_data)
        return virtual_velocity

    def _get_load_location(self, field_data):
        max_x = np.max(field_data.spatial_coords[:,0])
        min_x = np.min(field_data.spatial_coords[:,0])
        max_y = np.max(field_data.spatial_coords[:,1])

        return np.array([(min_x+max_x)/2, max_y]).reshape([1,2])


def _default_velocity_gradient_function(points, field_data):
    h = np.max(field_data.spatial_coords[:,1])-np.min(field_data.spatial_coords[:,1])
    y_offset = h/2+np.min(field_data.spatial_coords[:,1])
    centered_points = points - y_offset
    velocity_gradient = np.zeros((points.shape[0],2,2))
    velocity_gradient[:,0, 0] = 0
    velocity_gradient[:,0, 1] = -np.pi/h*np.sin(np.pi*centered_points[:,1]/h)
    velocity_gradient[:,1, 0] = 0
    velocity_gradient[:,1, 1] = 1/(h)
    return velocity_gradient


class InternalVirtualPowerExtractor(QoIExtractorBase):
    """
    VFM QoI extractor automatically applied to VFM calibrations and not intended for users.
    """
    def __init__(self, time_field, _velocity_gradient_function=_default_velocity_gradient_function):
        self._time_field = time_field
        self._velocity_gradient_function = _velocity_gradient_function
        self._stress_fields = ['first_pk_stress_xx', 'first_pk_stress_yy', 'first_pk_stress_xy',
                               'first_pk_stress_yx']

    @property
    def required_experimental_data_fields(self) -> list:
        return [self._time_field]

    def _calculate_volume_scaled_velocity_gradient(self, working_data):
        if "volume" in working_data.field_names:
            volumes = working_data[0]["volume"]
        elif "element_thickness" in working_data.field_names and "element_area" in working_data.field_names:
            volumes = working_data[0]["element_thickness"]*working_data[0]["element_area"]

        centroids = np.array([working_data[0]["centroid_x"], working_data[0]["centroid_y"]]).T
        velocity_gradients = self._velocity_gradient_function(centroids, working_data)
        return  np.einsum('i,ijk->ijk', volumes, velocity_gradients) 

    def calculate(self, working_data, reference_data, fields):
        block_index = 1
        n_times = len(reference_data[self._time_field])
        n_cells = np.shape(working_data[0][self._stress_fields[0]])[0]
        internal_power = []
        for timestep in range(len(working_data[self._time_field])):
            virtual_velocity_gradient = self._calculate_volume_scaled_velocity_gradient(working_data)        
            stress_array = self._form_stress_array(working_data[timestep], n_cells)
            internal_power.append(np.einsum('ijk,ijk', stress_array, virtual_velocity_gradient))
        interp_internal_power = self._interpolate_evaluation_data_to_projection_data(reference_data[self._time_field],  
                                                                                     working_data['time'], 
                                                                                     internal_power)

        data = convert_dictionary_to_data({'virtual_power':interp_internal_power, 
                                           self._time_field:reference_data[self._time_field]})
        data.set_state(working_data.state)
        
        return data

    def _interpolate_evaluation_data_to_projection_data(self, reference_time, 
                                                        working_time, internal_power ):
        return np.interp(reference_time, working_time, internal_power)
        
    def _form_stress_array(self, data, n_cells):
        stress = np.zeros([n_cells, 2, 2])
        stress[:, 0, 0] = data[ self._stress_fields[0]]
        stress[:, 0, 1] = data[ self._stress_fields[2]]
        stress[:, 1, 0] = data[ self._stress_fields[3]]
        stress[:, 1, 1] = data[ self._stress_fields[1]]
        return stress

class HWDPolynomialSimulationSurfaceExtractorBASE(QoIExtractorBase):

    @abstractmethod
    def hwd_init(poly_order, depth):
        """"""
        
    @abstractmethod
    def _get_field_weights(working_data, is_exp, 
                           point_cloud_working, time_interp_working_data):
        """"""

    def __init__(self, inital_points:MeshSkeleton, depth:int, 
                 poly_order:int, time_field:str):
        self._time_field = time_field
        self._mesh_extraction_surface = None
        self._initial_skeleton = inital_points
        self._is_colocated= False
        self._R_ref = None
        self._hwd = None
        self.hwd_init(poly_order, depth, inital_points.spatial_coords[:,:2])

    @property
    def required_experimental_data_fields(self):
        return [self._time_field]

    def calculate(self, working_data:FieldData, reference_data:FieldData, 
                  field_names:list, is_exp:bool=False)->Data:
        weights = {}
        point_cloud_working = None
        for name in field_names:
            time_ref = reference_data[self._time_field]
            time_work = working_data[self._time_field]
            field_data_work = self._parse_field_data(working_data, name, is_exp)
            time_interp_working_data = _time_interpolate(time_ref, time_work, field_data_work)
            weights[name] = self._get_field_weights(working_data, is_exp, 
                                                    point_cloud_working, 
                                                    time_interp_working_data, name)
        weights["weight_id"] = np.arange(len(weights[field_names[-1]]))
        qoi_data = convert_dictionary_to_data(weights)
        qoi_data.set_state(working_data.state)
        return qoi_data

    def _parse_point_cloud(self, working_data, is_exp):
        point_cloud_working = working_data.skeleton.spatial_coords[:,:2].T
        point_cloud_working = self._down_select(working_data, point_cloud_working, is_exp)
        return point_cloud_working.T

    def _down_select(self, working_data, target_information, is_exp):
        if self._should_down_select(working_data):
            data_dimension = target_information.ndim
            node_map = self._map_nodes_if_needed(working_data, is_exp)
            if data_dimension == 1:
                target_information = target_information[node_map]
            elif data_dimension == 2:
                target_information = target_information[:, node_map]
            else:
                raise RuntimeError("Invalid Dimension of Data")
        return target_information

    def _map_nodes_if_needed(self, working_data, is_exp):
        if not is_exp and self._is_colocated:
            node_map = _get_node_coordinate_mapping(working_data.skeleton, 
                                                    self._initial_skeleton, 
                                                    self._mesh_extraction_surface)
        else:
            node_map = working_data.surfaces[self._mesh_extraction_surface]
        return node_map

    def _parse_field_data(self, working_data, name, is_exp):
        field_data_work = working_data[name]
        field_data_work = self._down_select(working_data, field_data_work, is_exp)
        return field_data_work

    def _should_down_select(self, working_data):
        has_surface_selected = self._mesh_extraction_surface is not None
        surfaces_are_defined = len(working_data.surfaces) > 0
        return has_surface_selected and surfaces_are_defined

    def extract_cloud_from_mesh(self, surface_name):
        self._mesh_extraction_surface = surface_name

class HWDPolynomialSimulationSurfaceExtractor(HWDPolynomialSimulationSurfaceExtractorBASE):

    def hwd_init(self, poly_order, depth, coords):
        import matcal.full_field.hwd as hwd
        self._hwd = hwd.ReducedTwoDPolynomialHWD(poly_order, depth)
        fake_data = np.ones([coords.shape[0], 1])
        self._hwd.map_data(fake_data, coords)
        self._R_ref = self._hwd.get_basis()[1]

    def calculate(self, working_data:FieldData, referecne_data:FieldData, 
                  field_names:list, is_exp:bool= False)->Data:
        return super().calculate(working_data, referecne_data, 
                                 field_names, is_exp)
  
    def _get_field_weights(self, working_data, is_exp, point_cloud_working, 
                           time_interp_working_data, field):
        if point_cloud_working is None:
            point_cloud_working = self._parse_point_cloud(working_data, is_exp)
            field_weights = self._hwd.map_data(time_interp_working_data.T, 
                                               point_cloud_working, 
                                               self._R_ref).T.flatten()
        elif point_cloud_working is not None:
            field_weights = self._hwd.map_data(time_interp_working_data.T,
                                                R_ref=self._R_ref).T.flatten()
        else:
            raise RuntimeError("Bad switch")

        return field_weights


# class HWDColocatingPolynomialSimulationSurfaceExtractor(HWDPolynomialSimulationSurfaceExtractorBASE):

#     def hwd_init(self, poly_order, depth, coords):
#         self._poly = poly_order
#         self._depth = depth 
#         self._thresh = 1e-3
#         self._hwd = {}
#         self._max_weights = 0
#         self._n_weights = {}

#     def __init__(self, *args, **kargs):
#         super().__init__(*args, **kargs)
#         self._is_colocated = True

#     def calculate(self, working_data:FieldData, referecne_data:FieldData, field_names:list, is_exp:bool=False)->Data:
#         return super().calculate(working_data, referecne_data, field_names, is_exp=is_exp)

#     def _get_field_weights(self, working_data, is_exp, point_clould_working, time_interp_working_data, field):
#         n_time = time_interp_working_data.shape[0]
#         field_weights = np.zeros(self._max_weights * n_time)
#         field_weights[:n_time * self._n_weights[field]] = self._hwd[field].map_data(time_interp_working_data.T).T.flatten()
#         field_weights = self._hwd[field].map_data(time_interp_working_data.T).T.flatten()

#         return field_weights

#     def hwd_complete(self, points, data, fields):
#         import hwd.hwd as hwd
#         for field in fields:
#             last_time_data = data[field][-1, :]
#             self._hwd[field] = hwd.ReducedTwoDPolynomialROHWD(self._poly, self._depth, self._thresh)
#             self._hwd[field] = hwd.ReducedTwoDPolynomialHWD(self._poly, self._depth)

#             self._hwd[field].build_compressed_space(points, last_time_data)
#             self._n_weights[field] = self._hwd[field]._Q.shape[1]
#             self._max_weights = np.max([self._max_weights, self._n_weights[field]])


class HWDExperimentSurfaceExtractor(QoIExtractorBase):

    def __init__(self, sim_hwd_extractor):
        self._sim_extractor = sim_hwd_extractor

    @property
    def required_experimental_data_fields(self):
        return [self._sim_extractor._time_field]

    def calculate(self, working_data, reference_data, fields):
        return self._sim_extractor.calculate(working_data, working_data,fields)
    

class HWDColocatingExperimentSurfaceExtractor(QoIExtractorBase):

    def __init__(self, sim_hwd_extractor, experiment_coords, simulation_coords, 
                 interp_poly_order=1, interp_search_mult=2.75):
        self._sim_extractor = sim_hwd_extractor
        self._sim_coords = simulation_coords
        self._exp_coords = experiment_coords
        self._interp_poly_order = interp_poly_order
        self._interp_search_mult = interp_search_mult
        self._is_colocated = True

    @property
    def required_experimental_data_fields(self):
        return [self._sim_extractor._time_field]

    def calculate(self, working_data, reference_data, fields):
        mapped_field_data = meshless_remapping(working_data, fields, 
                                               self._sim_coords, 
                                               self._interp_poly_order, 
                                               self._interp_search_mult)
        mapped_field_data.set_spatial_coords(np.hstack((self._sim_coords, 
                                                        np.zeros((len(self._sim_coords[:,0]),1)))))
        #self._sim_extractor.hwd_complete(self._sim_coords, mapped_field_data, fields)
        qois = self._sim_extractor.calculate(mapped_field_data, 
                                             mapped_field_data, 
                                             fields, is_exp=True)
        del mapped_field_data
        return qois
    

class FlattenFieldDataExtractor(QoIExtractorBase):

    def calculate(self, working_data, reference_data, fields):
        qoi = {}
        for field in fields:
            qoi[field] = working_data[field].flatten()
        return convert_dictionary_to_data(qoi)
    @property
    def required_experimental_data_fields(self) -> list:
        return []
    

class FieldInterpolatorExtractor(QoIExtractorBase):

    class NodeMapFlagError(RuntimeError):
        pass

    def __init__(self, mesh_skeleton, cloud_points, time_field):
        grid_geometry = MeshSkeletonTwoDimensionalMesh(mesh_skeleton)
        self._space_interpolator = _TwoDimensionalFieldInterpolator(grid_geometry, cloud_points)
        self._old_grid = mesh_skeleton
        self._time_field = time_field
        self._node_map = None

    @property
    def required_experimental_data_fields(self):
        return [self._time_field]

    def calculate(self, working_data, reference_data, fields):
        qoi = {}

        working_nodes = _get_node_coordinate_mapping(working_data.skeleton, 
                                                     self._old_grid, 
                                                     self._old_grid.subset_name)
        reference_time = reference_data[self._time_field]
        working_time = working_data[self._time_field]
        
        for field in fields:
            data_to_interp = self._extract_data(working_data, field, working_nodes)
            space_interp_data = self._space_interpolator.interpolate(data_to_interp)

            qoi[field] = _time_interpolate(reference_time, working_time, 
                                           space_interp_data.T).flatten()
        data_qoi = convert_dictionary_to_data(qoi)
        return data_qoi

    @property
    def number_of_nodes_required(self):
         return self._space_interpolator.number_of_nodes_required

    def _extract_data(self, working_data, field, working_nodes):
        data = working_data[field].T
        return data[working_nodes,:]
    
