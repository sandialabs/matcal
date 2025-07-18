
import numpy as np
import os

from matcal.core.constants import TIME_KEY

from matcal.exodus.library_importer import create_exodus_class_instance

from matcal.full_field.data import convert_dictionary_to_field_data, FieldData
from matcal.full_field.models import MeshFieldInterpPreprocessorBase
from matcal.full_field.TwoDimensionalFieldGrid import MeshSkeleton


_exo_face_lookup = np.array([[1, 2, 6, 5],
                            [2, 3, 7, 6],
                            [3, 4, 8, 7],
                            [1, 5, 8, 4],
                            [1, 4, 3, 2],
                            [5, 6, 7, 8]]) - 1


def extract_exodus_mesh(filename, surface_name=None):
    if surface_name==None:
        mesh_data = _extract_full_exodus_mesh(filename)
    else:
        mesh_data = _extract_subset_exodus_mesh(filename, surface_name)
    return mesh_data


def copy_mesh_and_store_data(source_mesh, target_mesh, data, 
                                  field_data_names, mode='a'):
    exo_target = _get_exo_target(source_mesh, target_mesh)
    exo_target = store_information_on_mesh(exo_target, data, field_data_names)
    exo_target.close()


def _get_exo_target(source_mesh, target_mesh):
    if isinstance(source_mesh, str):
        exo_source = create_exodus_class_instance(source_mesh, mode='r', array_type='numpy')
        exo_target = exo_source.copy(target_mesh)
        exo_source.close()
    elif isinstance(source_mesh, FieldData):
        exo_target = _make_exo_space_time_from_field_data(source_mesh, target_mesh)
    else:
        msg = f"Cannot make exodus file with source of type {type(source_mesh)}"
        msg += f"must be a file path of type str or a FieldData instance."
        raise TypeError(msg)
    
    return exo_target


def _make_exo_space_time_from_field_data(field_data, target_filename):
    only_block_id = 1
    kwargs = {'mode':'w',
              'array_type':'numpy', 
              'title': f'Modified: {field_data.name}',
              "numDims": field_data.num_dimensions,
              "numNodes": field_data.num_nodes, 
              "numElems": field_data.num_elements,
              "numBlocks": 1}
    target = create_exodus_class_instance(target_filename, **kwargs)
    x = field_data.spatial_coords[:,0]
    y = field_data.spatial_coords[:,1]
    if field_data.num_dimensions > 2:
        z = field_data.spatial_coords[:,2]
    else:
        z = np.zeros_like(x)
    target.put_coords(x, y, z)
    n_node_per_ele = len(field_data.connectivity[0])
    elem_type = _element_lookup(field_data.num_dimensions, n_node_per_ele)
    target.put_elem_blk_info(only_block_id, elem_type, field_data.num_elements,
                             n_node_per_ele, 0)
    target.put_elem_connectivity(only_block_id, field_data.connectivity.flatten())
    return target


def _element_lookup(n_dim, n_node_per_ele):
    ele_type = None
    if n_dim == 2:
        if n_node_per_ele == 3:
            ele_type = "TRI3"
        elif n_node_per_ele == 4:
            ele_type = "QUAD4"
    elif n_dim == 3:
        if n_node_per_ele == 4:
            ele_type = "TET4"
        elif n_node_per_ele == 8:
            ele_type = "HEX8"
    if ele_type == None:
        msg = f"No element type found for NDim: {n_dim} with {n_node_per_ele} nodes per element."
        raise RuntimeError(msg)
    return ele_type
    

def store_information_on_mesh(exo_obj, field_data, fields_to_add):
    nodal_fields, global_fields = _get_variable_nodal_and_global_field_names(field_data,
                                                                             fields_to_add)
    exo_obj = _add_nodal_fields_to_mesh(exo_obj, nodal_fields)
    exo_obj = _add_global_fields_to_mesh(exo_obj, global_fields)
    for time_index, time in enumerate(field_data[TIME_KEY]):
        exo_time_index = time_index + 1
        exo_obj.put_time(exo_time_index, time)
        for field_name in nodal_fields:
            node_data = field_data[field_name][time_index, :]
            exo_obj.put_node_variable_values(field_name, exo_time_index, node_data)
        for field_name in global_fields:
                global_data = field_data[field_name][time_index]
                exo_obj.put_global_variable_value(field_name, exo_time_index, global_data)
    return exo_obj


def _get_variable_nodal_and_global_field_names(field_data, fields_to_add):
    global_fields = []
    nodal_fields = []
    for field_name in fields_to_add:
        if field_data[field_name].ndim > 1:
            nodal_fields.append(field_name)
        elif field_name != TIME_KEY:
            global_fields.append(field_name)
    return nodal_fields, global_fields


def _add_global_fields_to_mesh(exo_obj, fields_to_add):
    num_gvars_orig = exo_obj.get_global_variable_number()
    num_gvar_new = num_gvars_orig + len(fields_to_add)
    exo_obj.set_global_variable_number(num_gvar_new)
    exodus_offset = 1
    for new_field_index, field in enumerate(fields_to_add):
        field_index = num_gvars_orig + new_field_index + exodus_offset
        exo_obj.put_global_variable_name(field, field_index)
    return exo_obj


def _add_nodal_fields_to_mesh(exo_obj, fields_to_add):
    exo_obj.set_node_variable_number(len(fields_to_add))
    for index, field_name in enumerate(fields_to_add):
        exo_obj.put_node_variable_name(field_name, index + 1)
    return exo_obj


class ExodusHex8MeshExploder():
    def __init__(self, exodus_filename, **kwargs):
        self._exodus_filename = exodus_filename

    def boom(self):
        original_mesh = _open_exodus_mesh_for_read(self._exodus_filename)
        exploded_mesh = self._open_exploded_mesh_object(original_mesh)

        exploded_mesh, new_to_old_node_index_map = self._initialize_exploded_mesh(original_mesh, 
                                                                                  exploded_mesh)
        exploded_mesh = self._add_time_and_node_vars_to_exploded_mesh(original_mesh, 
                                                                      exploded_mesh, 
                                                                      new_to_old_node_index_map)
        exploded_mesh = self._add_node_sets(original_mesh, 
                                            exploded_mesh, new_to_old_node_index_map)
        exploded_mesh.close()
        original_mesh.close()
        os.rename(self._get_exploded_mesh_filename(), self._exodus_filename)

    def _open_exploded_mesh_object(self, original_mesh):
        numNodes=self._get_exploded_mesh_num_nodes(original_mesh) 
        exploded_mesh = create_exodus_class_instance(self._get_exploded_mesh_filename(),
                                                     mode='w', title="exploded_mesh", 
                                                     array_type='numpy', numDims=3,
                                                     numNodes=numNodes,
                                                     numElems=original_mesh.num_elems(),
                                                     numBlocks=original_mesh.num_blks(), 
                                                     numSideSets=0,
                                                     numNodeSets=original_mesh.num_node_sets())
       
        return exploded_mesh

    def _get_exploded_mesh_num_nodes(self, orig_mesh):
        num_nodes = 0
        for block in orig_mesh.get_elem_blk_ids():
            connectivity, num_eles, nodes_per_ele = orig_mesh.get_elem_connectivity(block)
            num_nodes += num_eles*nodes_per_ele
        return num_nodes

    def _initialize_exploded_mesh(self, orig_mesh, exploded_mesh):
        orig_coords = orig_mesh.get_coords()
        new_coords = [[], [], []]
        new_to_old_node_index_map = []
        total_nodes_added_to_connectivity = 0 
        for block in orig_mesh.get_elem_blk_ids():
            block_connectivity, num_eles_in_blk, nodes_per_ele = orig_mesh.get_elem_connectivity(block)
            new_block_connectivity = []
            for element_index in range(orig_mesh.num_elems_in_blk(block)):
                this_cell_node_indices = block_connectivity[(element_index*8):(element_index+1)*8]
                new_coords = self._get_new_element_nodes(orig_coords, new_coords, 
                                                         this_cell_node_indices)
                new_block_info = self._get_new_block_info(element_index, 
                                                          total_nodes_added_to_connectivity, 
                                                          new_block_connectivity, 
                                                          new_to_old_node_index_map, 
                                                          this_cell_node_indices)
                new_block_connectivity = new_block_info[0]
                new_to_old_node_index_map = new_block_info[1]
            total_nodes_added_to_connectivity += len(new_block_connectivity)
            exploded_mesh = self._initialize_block(orig_mesh, exploded_mesh, block, nodes_per_ele, 
                                                   num_eles_in_blk, new_block_connectivity)
        exploded_mesh.put_coords(new_coords[0], new_coords[1], new_coords[2])
        exploded_mesh = self._intialize_node_set_names(orig_mesh, exploded_mesh)
        return exploded_mesh, new_to_old_node_index_map

    def _get_new_element_nodes(self, orig_coords, new_coords, this_cell_node_indices):
            new_coords[0] += list(orig_coords[0][this_cell_node_indices-1])
            new_coords[1] += list(orig_coords[1][this_cell_node_indices-1])
            new_coords[2] += list(orig_coords[2][this_cell_node_indices-1])
            return new_coords

    def _get_new_block_info(self, element_index, total_nodes_added_to_connectivity, 
                                  new_block_connectivity, new_to_old_node_index_map, 
                                  this_cell_node_indices):
        first_new_connect_val = (element_index*8)+1+total_nodes_added_to_connectivity
        last_new_connect_val =  (element_index+1)*8+1+total_nodes_added_to_connectivity
        new_block_connectivity += list(range(first_new_connect_val,last_new_connect_val))
        new_to_old_node_index_map += list(this_cell_node_indices-1)
        return new_block_connectivity, new_to_old_node_index_map

    def _initialize_block(self, orig_mesh, exploded_mesh, block, nodes_per_elem, 
                          num_eles_in_blk, connectivity):
        element_type = orig_mesh.elem_type(block)
        exploded_mesh.put_elem_blk_info(block, element_type, num_eles_in_blk, 
                                        nodes_per_elem, 0)
        exploded_mesh.put_elem_blk_name(block, orig_mesh.get_elem_blk_name(block))
        exploded_mesh.put_elem_connectivity(block, connectivity)
        return exploded_mesh

    def _intialize_node_set_names(self, orig_mesh, exploded_mesh):
        node_variable_names = orig_mesh.get_node_variable_names()
        exploded_mesh.set_node_variable_number(len(node_variable_names))
        for node_variable_index, node_variable_name in enumerate(node_variable_names):
                exploded_mesh.put_node_variable_name(node_variable_name, node_variable_index+1)
        return exploded_mesh

    def _add_time_and_node_vars_to_exploded_mesh(self, orig_mesh, exploded_mesh, 
                                                 new_to_old_node_index_map):
        node_variable_names = exploded_mesh.get_node_variable_names()
        for time_step, time in enumerate(orig_mesh.get_times()):
            time_step += 1
            exploded_mesh.put_time(time_step, time)
            for node_variable_name in node_variable_names:
                node_variable_values = orig_mesh.get_node_variable_values(node_variable_name, 
                                                                          time_step)
                new_mesh_node_variable_values = node_variable_values[new_to_old_node_index_map]
                exploded_mesh.put_node_variable_values(node_variable_name, time_step, 
                                                       new_mesh_node_variable_values)

        return exploded_mesh

    def _add_node_sets(self, orig_mesh, exploded_mesh, new_to_old_node_index_map):
        node_set_ids = orig_mesh.get_node_set_ids()
        for node_set_id in node_set_ids:
            nset_name = orig_mesh.get_node_set_name(node_set_id)
            nset_nodes = orig_mesh.get_node_set_nodes(node_set_id)-1
            new_node_set_node_ids = []
            for nset_node in nset_nodes:
                new_indices =  np.where(new_to_old_node_index_map==nset_node)[0]
                new_indices += 1
                new_node_set_node_ids+= new_indices.tolist()
            exploded_mesh.put_set_params('EX_NODE_SET', node_set_id, len(new_node_set_node_ids))
            exploded_mesh.put_node_set(node_set_id, new_node_set_node_ids)
            exploded_mesh.put_node_set_name(node_set_id, nset_name)
            
        return exploded_mesh

    def _get_exploded_mesh_filename(self):
        mesh_path = os.path.dirname(self._exodus_filename)
        mesh_filename_basename = os.path.basename(self._exodus_filename)
        exploded_file_name = mesh_filename_basename.split(".")[0]
        exploded_file_name += "_exploded."+mesh_filename_basename.split(".")[-1]
        return os.path.join(mesh_path, exploded_file_name)
        
        
def _extract_full_exodus_mesh(filename):
    mesh_object = _open_exodus_mesh_for_read(filename)
    mesh_data = _get_exodus_object_mesh_skeleton(mesh_object)
    mesh_object.close()
    return mesh_data


def _get_exodus_object_mesh_skeleton(exo_obj):
    x, y, z, exodus_offset = _extract_all_but_connectivity(exo_obj)
    connectivity = _extract_full_connectivity(exo_obj, exodus_offset)
    mesh_skeleton = MeshSkeleton(np.array([x, y, z]).T, connectivity)
    mesh_skeleton.add_node_sets(**_extract_exodus_surfaces(exo_obj, exodus_offset))
    return mesh_skeleton


def _extract_full_connectivity(mesh_object, exodus_offset):
    block_id_list = mesh_object.get_elem_blk_ids()
    connectivity_collection = []
    for block_id in block_id_list:
        one_d_connectivity, num_eles, nodes_per_ele = mesh_object.get_elem_connectivity(block_id)
        connectivity = one_d_connectivity.reshape(num_eles, nodes_per_ele) - exodus_offset
        if num_eles > 0:
            connectivity_collection.append(connectivity)
    final_connectivity = np.concatenate(connectivity_collection)
    return final_connectivity


def _extract_subset_exodus_mesh(filename, surface_name):
    mesh_object = _open_exodus_mesh_for_read(filename)
    x, y, z, exodus_offset = _extract_all_but_connectivity(mesh_object)
    sideset_id = _get_surface_id_from_surface_name(surface_name, mesh_object)
    uniqe_node_list = _get_unique_side_nodes(mesh_object, sideset_id,exodus_offset)
    connectivity = _get_connectivity(mesh_object, exodus_offset, sideset_id, 
                                     uniqe_node_list)
  
    mesh_data = MeshSkeleton()
    mesh_data.spatial_coords = np.array([x[uniqe_node_list], y[uniqe_node_list],
                                          z[uniqe_node_list]]).T
    mesh_data.connectivity = connectivity
    mesh_data.node_map = uniqe_node_list
    mesh_data.subset_name = surface_name
    mesh_object.close()
    return mesh_data

def _get_connectivity(mesh_object, exodus_offset, sideset_id, uniqe_node_list):
    exo_connectivity = _extract_full_connectivity(mesh_object, exodus_offset)
    sideset_elems, sideset_faces = mesh_object.get_side_set(sideset_id)
    sideset_elems -= exodus_offset
    sideset_faces -= exodus_offset
    sideset_connectivity = np.zeros([len(sideset_elems), 4], dtype=int)
    for row_id, (elem_id, face_id) in enumerate(zip(sideset_elems, sideset_faces)):
        sideset_connectivity[row_id,:] = exo_connectivity[elem_id,_exo_face_lookup[face_id]]
    connectivity = np.zeros_like(sideset_connectivity)
    for new_node_id, old_node_id in enumerate(uniqe_node_list):
        connectivity[sideset_connectivity==old_node_id] = new_node_id
    return connectivity


def _get_unique_side_nodes(mesh_object, sideset_id, exodus_offset):
    nodes_per_ele, nodes_in_ele = mesh_object.get_side_set_node_list(sideset_id)
    unique_node_list = np.unique(nodes_in_ele.flatten()) - exodus_offset
    return unique_node_list.astype(int)


def _extract_exodus_surfaces(mesh_object, exodus_offset):
    surface_nodes = {}
    # exodus errors if you call the names directly, and there aren't any
    side_set_ids = mesh_object.get_side_set_ids()
    if len(side_set_ids) > 0:
        sideset_names = mesh_object.get_side_set_names()
        for name in sideset_names:
            sideset_id = _get_surface_id_from_surface_name(name, mesh_object)
            unique_node_list = _get_unique_side_nodes(mesh_object, sideset_id, 
                                                      exodus_offset)
            surface_nodes[name] = unique_node_list
    return surface_nodes


def _extract_all_but_connectivity(mesh_object):
    coords = mesh_object.get_coords()
    x = coords[0]
    y = coords[1]
    z = coords[2]
    exodus_offset = 1
    return x,y,z,exodus_offset


def _open_exodus_mesh_for_read(filename):
    mesh_object = create_exodus_class_instance(filename, mode='r', array_type='numpy')
    return mesh_object


def _get_surface_id_from_surface_name(surface_name, mesh_object):
    ss_names = mesh_object.get_side_set_names()
    ss_ids = mesh_object.get_side_set_ids()
    if isinstance(surface_name, int):
        if surface_name not in ss_ids:
            raise SurfaceNotFoundError(ss_ids, surface_name)
        return surface_name
    elif isinstance(surface_name, str):
        found_index = -1
        for name_i, ss_name in enumerate(ss_names):
            if ss_name.lower() == surface_name.lower():
                found_index = name_i
                break
        if found_index < 0:
            raise SurfaceNotFoundError(ss_names, surface_name)
        return ss_ids[found_index]
    else:
        raise TypeError("The surface id must be a string or integer. " + 
                        f"received a type \"{type(surface_name)}\".")


class SurfaceNotFoundError(RuntimeError):

    def __init__(self, side_set_names, target_surface):
        message = (f"Target Surface({target_surface}) not found. " +
                   f"Available surfaces:\n{side_set_names}")
        super().__init__(message)


class _ExodusFieldInterpPreprocessor(MeshFieldInterpPreprocessorBase):

    def _assign_field_information_to_mesh(self, field_data, fields_to_add, 
                                          state_mesh_filename):
        exo_obj = create_exodus_class_instance(state_mesh_filename, mode='a', 
                                               array_type='numpy')
        front_and_back_field_data = self._double_field_data(field_data, fields_to_add)
        exo_obj = store_information_on_mesh(exo_obj, front_and_back_field_data, 
                                            fields_to_add)
        exo_obj.close()
    
    def _double_field_data(self, data, field_names):
        n_points_on_face = len(data[0][field_names[0]])
        out = {}
        out[TIME_KEY] = data[TIME_KEY]
        for f_name in field_names:
            out[f_name] = np.zeros([len(data[TIME_KEY]), 2*n_points_on_face])
            for time_idx in range(len(out[TIME_KEY])):
                node_data = data[time_idx][f_name]
                out[f_name][time_idx, :n_points_on_face] = node_data
                out[f_name][time_idx, n_points_on_face:] = node_data
        return convert_dictionary_to_field_data(out)
    
    
