"""
The classes and functions in this module are intended 
to import data into MatCal from external sources for use 
in MatCal studies.
"""

import os
import numpy as np
from abc import ABC, abstractmethod

from matcal.core.object_factory import BasicIdentifier, ObjectCreator, SpecificObjectFactory
from matcal.core.data_importer import FileData

from matcal.full_field.TwoDimensionalFieldGrid import (MeshSkeleton,
                                            MeshSkeletonTwoDimensionalMesh)
from matcal.full_field.data import FieldData, convert_dictionary_to_field_data

from matcal.core.data import Data, convert_dictionary_to_data 
from matcal.core.logger import initialize_matcal_logger
from matcal.core.serializer_wrapper import json_serializer
from matcal.core.state import SolitaryState
from matcal.core.utilities import get_current_time_string


logger = initialize_matcal_logger(__name__)


def FieldSeriesData(global_filename, series_directory="./", 
                    position_names = ['X','Y'], state=SolitaryState(), 
                    file_type=None, n_cores=1):
    """
    A function used to import a MatCal :class:`~matcal.core.data.Data` object 
    from series field data. The user needs to use
    this function to load experimental data from a file or series of files into MatCal

    :param global_filename: the name of the file or primary file to be loaded.
    :type filename: str

    :param series_directory: the name of the directory where all files are located.
        Defaults to current working directory.
    :type series_directory: str

    :param position_names: optional names of the fields that store point/nodal 
        coordinates. Defaults to ['X', 'Y'].
    :type position_names: list(str)

    :param state: optional state to be assigned to the data being imported
    :type state: :class:`~matcal.core.state.State`

    :param file_type: optional file type passed by the user. MatCal will attempt 
        to guess the file type based on the
        file prefix. MatCal recognizes "csv", "e" file types.
    :type file_type: str

    :param n_cores: the number of cores to be used to load the data. This is 
        only active when reading file data from separate 
        files such as DIC data saved as CSV files.

    :return: a populated :class:`~matcal.full_field.data.FieldData` object.
    """
    _check_filename_type(global_filename)
    _check_series_directory(series_directory)
    _check_position_names(position_names)
    _check_n_cores(n_cores)
    file_type = _get_file_type(global_filename, file_type)

    return _import_field_data(global_filename, series_directory, 
                              position_names, state, file_type, n_cores)

def _import_field_data(global_filename, series_directory="./", 
                       position_names = ['X','Y'], state=SolitaryState(), 
                       file_type=None, n_cores=1):

    try:
        field_parser = MatCalFieldDataFactory.create(file_type, global_filename, 
                                                     series_directory, n_cores=n_cores)
    except KeyError:
        err_str = (f"Data file \"{global_filename}\" of type \"{file_type}\" " +
            "is not a supported file type. MatCal supports the following data types:"
            +f"\n{list(MatCalFieldDataFactory.keys())}")
        raise RuntimeError(err_str)
   
    _log_with_time(global_filename, "Start: Parsing Field Series Data")
    series_array = _create_series_data_array(field_parser, position_names)
    series_data = FieldData(series_array)
    series_data.set_state(state)
    series_data = _create_position_data(series_data, field_parser, position_names)
    try:
        connectivity = field_parser._get_connectivity()
        series_data.set_connectivity(connectivity)
    except AttributeError:
        logger.debug(f"Not importing mesh connectivity for file {global_filename}. "+
                     "Not a mesh file format.")
    field_parser.close()
    _log_with_time(global_filename, "Done: Parsing Field Series Data")
    return series_data


def _log_with_time(global_filename, message):
    current_time = get_current_time_string()
    logger.info(f"{message}({current_time}):  {global_filename}")


def _get_file_type(filename, file_type):
    if file_type is None:
        file_type = filename.split(".")[-1]
    _check_file_type_is_string(file_type)
    file_type = file_type.lower()

    return file_type


def _check_file_type_is_string(file_type):
    try:
        assert isinstance(file_type, str)
    except AssertionError:
        raise TypeError("The file type passed to a data importer "
                                          "must be a string. Received "
                                      "variable of type {}".format(type(file_type)))


def _check_filename_type(filename):
    try:
        assert isinstance(filename, str)
    except AssertionError:
        raise TypeError("The filename passed to a data importer "
                                          "must be a string. Received "
                                      "variable of type '{}'".format(type(filename)))


def _check_series_directory(dirname):
    if not isinstance(dirname, str):
        raise TypeError("The parameter 'series_directory' passed "
                                          " to FieldSeriesData must be of type 'string'."
                                      f" Received variable of type '{type(dirname)}'.")


def _check_n_cores(n_cores):
    from numbers import Integral
    is_int = isinstance(n_cores, Integral)
    greater_than_1 = False
    if is_int:
        greater_than_1 = n_cores >= 1

    if not is_int or not greater_than_1:
        raise ValueError("The parameter 'n_cores' passed to FieldSeriesData must"
                                        f" be an integer greater than 0. Received '{n_cores}'.")


def _check_position_names(position_names):
    if not isinstance(position_names, (list, tuple)):
        raise TypeError("The parameter 'position_names' passed to "
                                          "FieldSeriesData must be of type 'list' or 'tuple'."
                                      f" Received variable of type '{type(position_names)}'.")
    
    for idx, name in enumerate(position_names):
        if not isinstance(name, str):
            raise TypeError("The parameter 'position_names' passed "
                                              "to FieldSeriesData must"
                                f" contain only strings. Received variable of type '{type(name)}'"
                                 f" in position {idx} of the 'position_names'.")

class FieldDataParserBase(ABC):

    def __init__(self, data_file, series_directory, n_cores=1):
        self._data_file = self._confirm_path_exists_and_return(data_file)
        self._series_directory = self._confirm_path_exists_and_return(series_directory)
        self._state = None
        #self._coord_names = ['X','Y','Z']
        self._n_cores = n_cores

    class FieldDataDataSeriesMissingPathObject(RuntimeError):
        def __init__(self, filename, *args):
            super().__init__("File not found: {}".format(filename), *args)

    class FieldDataDataSeriesBadFrameIndex(RuntimeError):
        def __init__(self, frame_index, frame_limit):
            super().__init__(f"Bad frame index: {frame_index} \n"
                             f" Total Number of Frames: {frame_limit}.")

    def __str__(self):
        return self._data_file

    @property
    def filename(self):
        return self._data_file

    @property
    def state(self):
        return self._state

    @property
    def number_of_cores(self):
        return self._n_cores
    
    def set_state(self, state):
        """
        Sets the optional state value for the data.

        :param state: The state for this particular data set.
        :type state: :class:`~matcal.core.state.State`
        """
        self._state = state

    @property
    @abstractmethod
    def number_of_frames(self) -> int:
        """"""
    
    @property
    @abstractmethod
    def number_of_nodes(self) -> int:
        """"""

    @property
    @abstractmethod
    def number_of_elements(self) -> int:
        """"""

    @property
    @abstractmethod
    def global_field_names(self) -> list:
        """"""

    @property
    @abstractmethod
    def node_field_names(self) -> list:
        """"""

    @property
    @abstractmethod
    def element_field_names(self) -> list:
        """"""

    @abstractmethod
    def get_frame(self, frame_index):
        """
        returns a Data instance
        """

    @abstractmethod
    def get_global_data(self):
        """
        returns a Data instance
        """

    @abstractmethod
    def get_surfaces(self) -> dict:
        """
        Returns a dict of the surfaces names to their corresponding nodes, 0 indexed
        """
    
    @abstractmethod
    def _files_in_parallel(self, filename) -> bool:
        """
        """

    def _confirm_path_exists_and_return(self, path_object):
        if not os.path.exists(path_object) and not self._files_in_parallel(path_object):
            raise self.FieldDataDataSeriesMissingPathObject(path_object)
        return path_object

    def close(self):
        """"""

    def _get_connectivity(self):
        return None


class _JSONFullFieldParser(FieldDataParserBase):

    def __init__(self, json_filename, series_directory='./', n_cores=1):
        super().__init__(json_filename, series_directory, n_cores=1)
        self._data = _import_full_field_data_from_json(self._data_file)
        self._global_names, self._node_names = self._parse_field_names()
        self._showed_element_warning = False

    @property
    def number_of_frames(self) -> int:
        return self._data.length
    
    @property
    def number_of_nodes(self) -> int:
        return self._data.skeleton.spatial_coords.shape[0]

    @property
    def number_of_elements(self) -> int:
        return len(self._data.skeleton.connectivity)

    @property
    def global_field_names(self) -> list:
        return self._global_names

    @property
    def node_field_names(self) -> list:
        return self._node_names

    @property
    def element_field_names(self) -> list:
        if not self._showed_element_warning:
            logger.warning("JSON parser currently does not support element data import.")
        self._showed_element_warning = True
        return []

    def get_frame(self, frame_index):
        out_dict = {}
        for n_name in self._node_names:
            out_dict[n_name] = self._data[n_name][frame_index,:]
        position_names = ['X', 'Y', 'Z']
        for pos_idx in range(self._data.skeleton.spatial_coords.shape[1]):
            out_dict[position_names[pos_idx]] = self._data.skeleton.spatial_coords[:,pos_idx]
        out_data = convert_dictionary_to_data(out_dict)
        out_data.set_state(self._data.state)
        return out_data

    def get_global_data(self):
        out_dict = {}
        for g_name in self._global_names:
            out_dict[g_name] = self._data[g_name]
        out_data = convert_dictionary_to_data(out_dict)
        out_data.set_state(self._data.state)
        return out_data

    def get_surfaces(self) -> dict:
        return self._data.skeleton.surfaces

    def _files_in_parallel(self, filename)->bool:
        return False
    
    def _parse_field_names(self):
        names = self._data.field_names
        field_names = []
        global_names = []
        for name in names:
            if self._data[name].ndim > 1: 
                field_names.append(name)
            else:
                global_names.append(name)
        return global_names, field_names

    def _get_connectivity(self):
        return self._data.skeleton.connectivity


class CSVFieldDataSeriesParser(FieldDataParserBase):
    """
    Class used to import a series of field data from file sources. The file must 
    contain a field called "file", which
    lists the filenames for the field data files.

    :param global_data_filename: path to a csv file containing the series 
        filenames_list, and respective global variables.
    :type filename: str

    :param series_directory: path to directory containing the field 
        data snapshots described in the global data file.
    :type filename: str
    """

    def __init__(self, global_data_file, series_directory, n_cores=1, file_type=None):
        super().__init__(global_data_file, series_directory, n_cores)
        self._state = SolitaryState()
        self._number_of_frames = None
        self._field_data_file_list = None
        self._global_fields = None
        self._number_of_nodes = None
        self._file_type = file_type
        self._n_cores=n_cores
        self._setUp()

    @property
    def number_of_frames(self):
        """
        Get the number of frames present in the data series.
        Frames align with the different time steps, if applicable.
        """
        return self._number_of_frames

    @property
    def number_of_nodes(self):
        return self._number_of_nodes

    @property
    def number_of_elements(self):
        return 0

    @property
    def global_field_names(self):
        """
        get a list of the imported global field names

        :return: the field names
        :rtype: list
        """
        gf_names = list(self._global_fields.field_names)
        return gf_names

    @property
    def state(self):
        """
        :return: The physical state of the data corresponding to the experimental conditions.
        :rtype: :class:`~matcal.core.state.State`
        """
        return self._state

    @property
    def node_field_names(self):
        """
        get a list of the imported node field names

        :return: the field names
        :rtype: list
        """
        return list(self.get_frame(0).keys())

    @property
    def element_field_names(self):
        return []

    def get_frame(self, frame_index):
        """
        Return an instance of the appropriate field data object.

        :param frame_index: index of frame data desired. (0 indexed)
        :type frame_index: int

        :return: frame data
        :rtype: :class:`matcal.full_field.data.FieldData`
        """
        self._check_index(frame_index)
        frame_name = os.path.join(self._series_directory,self._field_data_file_list[frame_index])
        return FileData(frame_name, file_type=self._file_type)

    def get_global_data(self):
        """
        Return all global field data as MatCal Data class.

        :rtype:  :class:`~matcal.core.data.Data`
        """
        return self._global_fields

    def get_surfaces(self) -> dict:
        message = "Surface information not currently collected from csv data."
        logger.info(message)
        return {}

    def _check_index(self, frame_index):
        if frame_index is None or frame_index >= self._number_of_frames or frame_index < 0 or not \
                isinstance(frame_index, int):
            raise self.FieldDataDataSeriesBadFrameIndex(frame_index, self.number_of_frames)

    def _setUp(self):
        global_data = self._parse_global_data_file()
        self._assign_field_data_files(global_data)
        self._assign_number_of_frames()
        self._assign_global_fields(global_data)
        self._number_of_nodes = self.get_frame(0).length

    def _parse_global_data_file(self):
        return FileData(self._data_file, import_strings=True)

    def _assign_field_data_files(self, global_data):
        self._field_data_file_list = list(np.atleast_1d(global_data['file_']))

    def _assign_number_of_frames(self):
        self._number_of_frames = len(self._field_data_file_list)

    def _extract_key(self, global_data, key):
        return global_data[key]

    def _assign_global_fields(self, global_data):
        global_data_fields = list(global_data.field_names)
        global_data_fields.remove("file_")
        self._global_fields = global_data[global_data_fields]

    def _files_in_parallel(self, filename):
        return False


def _is_csv(filename):
    extension  = filename.split('.')[-1]
    return extension == 'csv'


def _get_number_of_points_and_frames(field_parser):
    n_times = field_parser.number_of_frames
    n_points = field_parser.number_of_nodes
    return n_points, n_times, field_parser.number_of_elements


def _create_series_data_array(field_parser, position_names):
    n_points, n_times, n_ele = _get_number_of_points_and_frames(field_parser)
    global_keys, node_keys, element_keys = _get_field_parser_info(field_parser)
    
    ignore_keys = ['file'] + position_names
    data_list = _add_global_data_type(global_keys, ignore_keys)
    data_list = _add_space_data_type(data_list, node_keys, ignore_keys, n_points)
    data_list = _add_space_data_type(data_list, element_keys, ignore_keys, n_ele)

    data = np.zeros(n_times, dtype=data_list)
    logger.info(f"{field_parser.filename}: Reading Global Data")
    for gkey in global_keys:
        data[gkey] = field_parser.get_global_data()[gkey]

    if field_parser.number_of_cores > 1:
        _read_data_in_parallel(field_parser, data, ignore_keys)
    else:
        _read_data_in_serial(field_parser, data, ignore_keys)

    return Data(data)


def _read_data_in_parallel(field_parser, data, ignore_keys):
    n_times = field_parser.number_of_frames
    from concurrent.futures import ProcessPoolExecutor
    futures = []
    with ProcessPoolExecutor(max_workers=field_parser.number_of_cores) as executor:
        for time_index in range(n_times):    
            futures.append(executor.submit(_get_frame_data, field_parser, time_index))

    for time_index, future in enumerate(futures):
        _add_frame_data_to_data(future.result(), data, time_index, ignore_keys, field_parser)


def _read_data_in_serial(field_parser, data, ignore_keys):
    n_times = field_parser.number_of_frames
    for time_index in range(n_times):
        frame_data = _get_frame_data(field_parser, time_index)
        _add_frame_data_to_data(frame_data, data, time_index, ignore_keys, field_parser)


def _get_field_parser_info(field_parser):
    global_field_names = field_parser.global_field_names
    node_field_names = field_parser.node_field_names
    ele_field_names = field_parser.element_field_names
    return global_field_names, node_field_names, ele_field_names


def _get_frame_data(field_parser, time_index):
    _log_frame_import(field_parser, field_parser.number_of_frames, time_index)
    frame_data = field_parser.get_frame(time_index)
    return frame_data


def _add_frame_data_to_data(frame_data, data, time_index, ignore_keys, field_parser):
    _log_frame_processing(field_parser, field_parser.number_of_frames, time_index)

    global_keys, node_keys, element_keys = _get_field_parser_info(field_parser)

    for skey in node_keys:
        if skey in ignore_keys:
            continue
        node_data = frame_data[skey]
        data[skey][time_index] = node_data
    for ekey in element_keys:
        if ekey in ignore_keys:
            continue
        data[ekey][time_index] = frame_data[ekey]


def _log_frame_processing(field_parser, n_times, time_index):
    if _output_store_data(time_index, n_times, 5):
        logger.info(f"{field_parser.filename}: Processing Frame {time_index}") 


def _log_frame_import(field_parser, n_times, time_index):
    if _output_store_data(time_index, n_times, 5):
        logger.info(f"{field_parser.filename}: Reading Frame {time_index}")


def _output_store_data(current_index, max_index, max_out):
    if max_index < max_out:
        return True
    freq = max_index // max_out
    if current_index%freq == 0:
        return True
    else:
        return False    


def _add_global_data_type(global_keys, ignore_keys):
    data_list = []
    for gkey in global_keys:
        if gkey in ignore_keys:
            continue
        data_list.append((gkey, np.double))
    return data_list


def _add_space_data_type(data_list, space_keys, ignore_keys, n_space):
    for skey in space_keys:
        if skey in ignore_keys:
            continue
        data_list.append((skey, np.double, (n_space,)))
    return data_list


def _create_position_data(series_data, parser, position_names):
    x = []
    frame = parser.get_frame(0)
    for pos_name in position_names:
        x.append(frame[pos_name])
    series_data.set_spatial_coords(np.array(x).T)
    series_data.add_node_sets(**parser.get_surfaces())
    return series_data


class _FieldDataImporterSelector(SpecificObjectFactory):
    pass


class _CSVFieldDataImporterCreator(ObjectCreator):

    def __call__(self, *args, **kwargs):
        return CSVFieldDataSeriesParser(*args, **kwargs)


class _JSONFiledDataImporterCreator(ObjectCreator):
    def __call__(self, *args, **kwargs):
        return _JSONFullFieldParser(*args, **kwargs)


MatCalFieldDataFactory = _FieldDataImporterSelector()
MatCalFieldDataFactory.register_creator('csv', _CSVFieldDataImporterCreator())
MatCalFieldDataFactory.register_creator('json', _JSONFiledDataImporterCreator())


class MeshFileScraperSelector(BasicIdentifier):
    
    def identify(self, mesh_filename:str):
        extension = self._extract_extension(mesh_filename)
        return super().identify(extension)
    
    def _extract_extension(self, mesh_filename:str)->str:
        return mesh_filename.split('.')[-1]


def _json_mesh_skeleton_scraper(filename:str, subset_name:str=None):
    with open(filename, 'r') as f:
        mesh_dict = json_serializer.load(f)
    mesh_skele = _convert_ff_dict_to_mesh_skeleton(subset_name, mesh_dict)
    return mesh_skele


def _convert_ff_dict_to_mesh_skeleton(subset_name, mesh_dict):
    if subset_name == None:
        mesh_skele = _full_json_import(mesh_dict)
    else:
        mesh_skele = _surface_json_import(subset_name, mesh_dict)
    return mesh_skele


def _surface_json_import(subset_name, mesh_dict):
    node_ids = mesh_dict['surfaces'][subset_name]
    mesh_skele = MeshSkeleton(np.array(mesh_dict['spatial_coords'])[node_ids,:])
    mesh_skele.subset_name = subset_name
    return mesh_skele


def _full_json_import(mesh_dict):
    mesh_skele = MeshSkeleton(np.array(mesh_dict['spatial_coords']), 
                              np.array(mesh_dict['connectivity']))
    mesh_skele.subset_name = mesh_dict['subset_name']
    surfaces = {}
    for name, node_list in mesh_dict['surfaces'].items():
        surfaces[name] = np.array(node_list)
    mesh_skele.add_node_sets(**surfaces)
    return mesh_skele


MatCalMeshFileScraperSelector = MeshFileScraperSelector()
MatCalMeshFileScraperSelector.register('json', _json_mesh_skeleton_scraper)


def mesh_file_to_skeleton(mesh_filename:str, subset_name:str=None)->MeshSkeleton:
    """
    This will load a mesh file and return a data structure containing the
    mesh cloud points, connectivity and side set information. 
    """
    scraper = MatCalMeshFileScraperSelector.identify(mesh_filename)
    return scraper(mesh_filename, subset_name)


class ImportedTwoDimensionalMesh(MeshSkeletonTwoDimensionalMesh):
  def __init__(self, mesh_filename):
        mesh_skeleton = mesh_file_to_skeleton(mesh_filename)
        super().__init__(mesh_skeleton)


def _import_full_field_data_from_json(source_filename:str):
    new_dict = None
    with open(source_filename, 'r') as f:
        new_dict = json_serializer.load(f)
    skeleton = _convert_ff_dict_to_mesh_skeleton(None, new_dict)
    cleaned_data_dict = _remove_skeleton_fields(new_dict)

    new_data = convert_dictionary_to_field_data(cleaned_data_dict)
    new_data._graph = skeleton
    return new_data


def _remove_skeleton_fields(data_dict):
    fields = ['spatial_coords', 'connectivity', 'subset_name', 'surfaces']
    for field in fields:
        data_dict.pop(field)
    return data_dict