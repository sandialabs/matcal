"""
The models module includes an interpolator class for interpolating 
full field data to meshes.
"""
from abc import abstractmethod

from matcal.full_field.data_importer import mesh_file_to_skeleton
from matcal.full_field.field_mappers import meshless_remapping

from matcal.core.models import ModelPreprocessorBase
from matcal.core.logger import initialize_matcal_logger
logger = initialize_matcal_logger(__name__)


class MeshFieldInterpPreprocessorBase(ModelPreprocessorBase):

    def process(self, template_dir, mesh_filename, field_data, 
                fields_to_map, polynomial_order, 
                search_radius_multiplier):
        target_skeleton = mesh_file_to_skeleton(mesh_filename)

        mapped_field_data = self._map_data(field_data, fields_to_map, target_skeleton, 
                                           polynomial_order, search_radius_multiplier)
        self._assign_field_information_to_mesh(mapped_field_data, fields_to_map, mesh_filename)

    def _map_data(self, field_data, fields_to_map, target_skeleton, 
                  polynomial_order, search_radius_multiplier):
        tartget_points = target_skeleton.spatial_coords[:,:2]
        field_data_to_map = field_data
        if field_data_to_map.spatial_coords.shape[1] > 2:
            logger.warning("Forcing data and mesh to be 2D for VFM interpolation.")
            field_data_to_map.set_spatial_coords(field_data_to_map.spatial_coords[:,:2])
        mapped_field_data = meshless_remapping(field_data_to_map, fields_to_map, tartget_points, 
                                               polynomial_order, search_radius_multiplier)
        return mapped_field_data

    @abstractmethod
    def _assign_field_information_to_mesh(self, field_data, fields_to_add, state_mesh_filename):
        """"""

