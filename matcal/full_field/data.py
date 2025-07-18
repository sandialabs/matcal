"""
The data module contains classes and functions for converting 
data into the structure that MatCal requires for studies.
"""
from collections import OrderedDict

from matcal.core.object_factory import ObjectCreator
from matcal.core.simulators import MatCalDataReaderFactory
import numpy as np
from matcal.core.data import Data, _check_dictionary_data, _create_array_from_dict
from matcal.core.state import SolitaryState
from matcal.full_field.TwoDimensionalFieldGrid import MeshSkeleton

from matcal.core.logger import initialize_matcal_logger
logger = initialize_matcal_logger(__name__)

class FieldData(Data):
    """
    Extension of the :class:`~matcal.core.data.Data` that incorporates the ability to store
    nodal positions and connectivity. 
    """
    
    def __new__(cls, data, state=SolitaryState(), name=None):
        obj = super().__new__(cls, data, state, name)
        obj._graph = MeshSkeleton()
        return obj

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
        self._graph = getattr(obj, '_graph', None)

    @property
    def spatial_coords(self):
        """
        :return: the spatial coordinates of the two dimensional point cloud/mesh nodes
            for which the FieldData object stores data on. A two-dimensional 
            array is returned where the columns correspond to two-dimensions 
            of interest (name 'X' and 'Y' by default) and the rows are 
            the values for each point.
        """
        return self._graph.spatial_coords

    @property
    def num_dimensions(self):
        """
        :return: get the number of spatial dimensions the coordinates are in. 
        """
        if self._graph.spatial_coords is None:
            err_msg = "Dimensions not defined, no spatial coordinates set."
            raise RuntimeError(err_msg)
        num_dim = self._graph.spatial_coords.shape[1]
        return num_dim
        

    @property
    def num_nodes(self):
        """
        :return: Get the number of nodes. Will return 0 if the coordinates are not
            defined.
        """
        if self._graph.spatial_coords is None:
            n_nodes = 0
        else:
            n_nodes = self._graph.spatial_coords.shape[0]
        return n_nodes


    @property
    def num_elements(self):
        """
        :return: Get the number of elements. Will return 0 if the connectivity is
            not defined.
        """
        if self._graph.connectivity is None:
            n_nodes = 0
        else:
            n_nodes = self._graph.connectivity.shape[0]
        return n_nodes

    @property
    def connectivity(self):
        """
        :return: the nodal connectivity for the data if generated from 
            a finite element mesh. None is data is for a point cloud.
        """
        return self._graph.connectivity

    @property
    def surfaces(self):
        """
        :return: a dictionary containing groups of point or node indices. 
            The key is the name of the group ( a node set name for meshes)
            and the values is an array of indices that can be used to 
            access this set of point/node data from all of the field data.
        """
        return self._graph.surfaces

    @property
    def skeleton(self):
        """
        Development property, not intended for users.
        """
        return self._graph

    def set_spatial_coords(self, coords):
        """
        Set the spatial coordinates for the field data nodes/point positions.
        A two-dimensional array is expected with the number of 
        point rows and two columns. The columns corresponds to the 
        position names passed to the :func:`~matcal.full_field.data_importer.FieldSeriesData`
        function.
        """

        self._graph.spatial_coords = coords

    def set_connectivity(self, connectivity):
        """
        Development method, not intended for users.
        """
        self._graph.connectivity = connectivity

    def add_node_sets(self, **surfaces):
        """
        Development method, not intended for users.
        """
        self._graph.add_node_sets(**surfaces)

def convert_dictionary_to_field_data(dict_data, coordinate_names=[], connectivity_name=None, node_set_name=None):
    """
    Takes a dictionary and attempts to create a
    MatCal :class:`~matcal.full_field.data.FieldData` object.
    The keys for the dictionary are expected to be 
    strings for the field names and the values 
    are expected to be valid numeric or string data. 

    :param dict_data: a dictionary with field names as keys and 
       the data values as the dictionary values.
    :type dict_data: dict or OrderedDict

    :param coordinate_names: a list of the names of the coordinates used in the data, in the conventional x, y, z ordering. 
    :type coordinate_names: list(str)

    :param connectivity_name: the name of the dictionary key containing the connectivity information
    :type connectivity_name: str

    :return: a Data object with the default state :class:`~matcal.core.state.SolitaryState`. 
    :rtype: :class:`~matcal.full_field.data.FieldData`
    """
    _check_dictionary_data(dict_data)
    mesh_skeleton = _extract_mesh_skeleton(dict_data, coordinate_names, connectivity_name, node_set_name)
    no_mesh_data_dict = _prune_mesh_information(dict_data, coordinate_names + [connectivity_name, node_set_name])
    data = FieldData(_create_array_from_dict(no_mesh_data_dict))
    data.set_spatial_coords(mesh_skeleton.spatial_coords)
    data.set_connectivity(mesh_skeleton.connectivity)
    data.add_node_sets(**mesh_skeleton.surfaces)
    return data

def _extract_mesh_skeleton(dict_data, coordinate_names, connectivity_name, node_set_name):
    mesh_skeleton = MeshSkeleton()
    if len(coordinate_names) > 0:
        locations = []
        for loc_key in coordinate_names:
            locations.append(dict_data[loc_key])
        mesh_skeleton.spatial_coords = np.array(locations).T
    if connectivity_name is not None:
        mesh_skeleton.connectivity = np.array(dict_data[connectivity_name])
    if node_set_name is not None:
        surf_dict =dict_data[node_set_name]
        mesh_skeleton.add_node_sets(**surf_dict)
    return mesh_skeleton

def _prune_mesh_information(dict_data, mesh_variable_names):
    no_mesh_data_dict = OrderedDict()
    for key, value in dict_data.items():
        if key in mesh_variable_names:
            continue
        no_mesh_data_dict[key] = value
    return no_mesh_data_dict


class _FieldDataReaderCreator(ObjectCreator):

    class CustomConverter:

        def __init__(self, coordinate_names):
            self._coord_names = coordinate_names

        def __call__(self, information):
            return convert_dictionary_to_field_data(information, self._coord_names)

    def __call__(self, *args, **kwargs):      
        converter = self.CustomConverter(args[0])
        return converter


MatCalDataReaderFactory.register_creator(True, _FieldDataReaderCreator())