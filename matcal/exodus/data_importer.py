from collections import OrderedDict
from glob import glob
import numpy as np
import os

from matcal.core.data import convert_dictionary_to_data
from matcal.core.state import SolitaryState
from matcal.exodus.library_importer import create_exodus_class_instance
from matcal.exodus.mesh_modifications import extract_exodus_mesh, _extract_exodus_surfaces
from matcal.full_field.data_importer import (FieldDataParserBase, 
                                             MatCalMeshFileScraperSelector, 
                                             MatCalFieldDataFactory, mesh_file_to_skeleton)
from matcal.core.mesh_modifications import get_mesh_composer


MatCalMeshFileScraperSelector.register("g", extract_exodus_mesh)
MatCalMeshFileScraperSelector.register("e", extract_exodus_mesh)
MatCalMeshFileScraperSelector.register("exo", extract_exodus_mesh)


class ExodusFieldDataSeriesImporter(FieldDataParserBase):

    def __init__(self, exodus_filename, series_directory='./', n_cores=1):
        super().__init__(exodus_filename, series_directory, n_cores=n_cores)
        self._state = SolitaryState()
        self._exodus_handle = None
        self._number_of_frames = None
        self._number_of_blocks = None
        self._number_of_nodes = None
        self._number_of_elements = None
        self._global_fields = OrderedDict()
        self._block_id_map = None
        self._coord_names = ['X', 'Y', 'Z']
        self._setUp()

    @property 
    def number_of_frames(self):
        return self._number_of_frames

    @property
    def number_of_nodes(self):
        return self._number_of_nodes

    @property
    def number_of_elements(self):
        return self._number_of_elements

    @property
    def global_field_names(self):
        return self._global_fields.keys()

    @property
    def node_field_names(self):
        return self._exodus_handle.get_node_variable_names()

    @property
    def element_field_names(self):
        return self._exodus_handle.get_element_variable_names()

    def get_frame(self, frame_index):
        self._check_index(frame_index)
        return self._create_data_frame_from_exodus_timestep(frame_index)

    def get_global_data(self):
        return self._global_fields

    def get_surfaces(self):
        exodus_offset = 1
        return _extract_exodus_surfaces(self._exodus_handle, exodus_offset)

    def get_values_for_all_time(self, block_index, field_name):
        values = np.zeros(self._number_of_frames)
        for i in range(self._number_of_frames):
            time_index = i + 1
            block_value = self._exodus_handle.get_element_variable_values(block_index, 
                                                                          field_name, time_index)
            values[i] = block_value
        return values

    def get_all_element_values_for_all_time(self, block_index, field_name):
        number_of_elements = self._get_number_of_elements(block_index, field_name)
        values = np.zeros([number_of_elements, self._number_of_frames])
        for i in range(self._number_of_frames):
            time_index = i + 1
            block_value = self._exodus_handle.get_element_variable_values(block_index, 
                                                                          field_name, time_index)
            values[:, i] = block_value
        return values

    def close(self):
        self._exodus_handle.close()
        
    @property
    def number_of_blocks(self):
        return self._number_of_blocks
    
    def _files_in_parallel(self, filename):
        parallel_mesh_files = glob(filename+".*")
        return len(parallel_mesh_files) > 0

    def _get_number_of_elements(self, block_index, field_name):
        number_of_elements = len(self._exodus_handle.get_element_variable_values(block_index,
                                                                                  field_name, 1))
        return number_of_elements

    def _setUp(self):
        if not os.path.isfile(self._data_file):
            self._compose_parallel_files()
        self._exodus_handle = create_exodus_class_instance(self._data_file, array_type='numpy')
        self._number_of_frames = len(self._exodus_handle.get_times())
        self._number_of_blocks = self._exodus_handle.num_blks()
        self._number_of_nodes = self._exodus_handle.num_nodes()
        self._number_of_elements = self._exodus_handle.num_elems()
        self._block_id_map = self._exodus_handle.get_elem_blk_ids()
        self._build_global_fields()

    def _compose_parallel_files(self):
        mesh_composer_class = get_mesh_composer(self._data_file)
        mesh_composer = mesh_composer_class()
        filename, file_dir = self._divide_file_and_path()
        n_splits = len(glob(self._data_file+".*"))
        mesh_composer.compose_mesh(filename, n_splits, file_dir)

    def _divide_file_and_path(self):
        split_filename = os.path.split(self._data_file)
        filename = split_filename[-1]
        if len(split_filename) > 1:
            file_dir = os.path.join(*split_filename[:-1])
        else:
            file_dir = "."
        return filename,file_dir

    def _build_global_fields(self):
        keys = self._exodus_handle.get_global_variable_names()
        for key in keys:
            self._global_fields[key] = self._exodus_handle.get_global_variable_values(key)
        self._global_fields["time"] = self._exodus_handle.get_times()
        self._global_fields = convert_dictionary_to_data(self._global_fields)

    def _create_data_frame_from_exodus_timestep(self, frame_index):
        element_variable_names = self._exodus_handle.get_element_variable_names()
        node_variable_names = self._exodus_handle.get_node_variable_names()
        frame_data_dict = OrderedDict()
        for variable_name in element_variable_names:
            extracted_data = self._extract_first_block_element_frame_data(frame_index, 
                variable_name)
            frame_data_dict[variable_name] = extracted_data
        for variable_name in node_variable_names:
            extracted_data = self._extract_first_block_node_frame_data(frame_index, variable_name)
            frame_data_dict[variable_name] = extracted_data
        coords = self._exodus_handle.get_coords()
        for x_i, coord_name in enumerate(self._coord_names):
            frame_data_dict[coord_name] = coords[x_i]
        return frame_data_dict

    def _extract_first_block_element_frame_data(self, frame_index, variable_name):
        timestep_index = self._convert_frame_index_to_exodus_timestep(frame_index)
        value_collection = []
        for block_index in range(len(self._block_id_map)):
            block_id = self._convert_index_to_exodus_block_id(block_index)
            extracted_value = self._exodus_handle.get_element_variable_values(block_id,
                variable_name, timestep_index)
            value_collection.append(extracted_value)
        value_collection  = np.concatenate(value_collection).flatten()
        variable_values = value_collection
        return variable_values

    def _extract_first_block_node_frame_data(self, frame_index, variable_name):
        timestep_index = self._convert_frame_index_to_exodus_timestep(frame_index)
        extracted_value = self._exodus_handle.get_node_variable_values(variable_name,
            timestep_index)
        variable_values = extracted_value
        return variable_values

    def _convert_index_to_exodus_block_id(self, block_index):
        return self._block_id_map[block_index]

    def _convert_frame_index_to_exodus_timestep(self, frame_index):
        return frame_index + 1

    def _check_index(self, frame_index):
        if frame_index is None or frame_index >= self._number_of_frames or frame_index < 0 or not \
                isinstance(frame_index, int):
            raise self.FieldDataDataSeriesBadFrameIndex(frame_index, self.number_of_frames)

    def _get_connectivity(self):
        mesh_skeleton = mesh_file_to_skeleton(self._data_file)
        return mesh_skeleton.connectivity+1 #add 1 to account for exodus indexing at 1
    

MatCalFieldDataFactory.register_creator("e", ExodusFieldDataSeriesImporter)
MatCalFieldDataFactory.register_creator("g", ExodusFieldDataSeriesImporter)
MatCalFieldDataFactory.register_creator("exo", ExodusFieldDataSeriesImporter)