import os

from matcal.core.object_factory import BasicIdentifier



def get_mesh_operator(mesh_file_basename, identifier, identifier_name, operator_str):
    mesh_file_extension = os.path.splitext(mesh_file_basename)[-1].strip(".")
    try:
        mesh_operator = identifier.identify(mesh_file_extension)
    except KeyError:
        err_str = (f"A mesh {operator_str} class has not been registered in " +
            f"the \"{identifier_name}\" for mesh file " +
            f"\"{mesh_file_basename}\" with extension \"{mesh_file_extension}\". "+
            "Import the identifier from "+
            "\'matcal.core.mesh_modifications\' and assign  " +
            f"the appropriate mesh {operator_str} class for file types of interest.")
        raise RuntimeError(err_str)
    return mesh_operator


matcal_mesh_composer_identifier = BasicIdentifier()


def get_mesh_composer(mesh_file_basename):
    return get_mesh_operator(mesh_file_basename, matcal_mesh_composer_identifier, 
                             "matcal_mesh_composer_identifier", "composer")


matcal_mesh_decomposer_identifier = BasicIdentifier()


def get_mesh_decomposer(mesh_file_basename):
    return get_mesh_operator(mesh_file_basename, matcal_mesh_decomposer_identifier, 
                             "matcal_mesh_decomposer_identifier", "decomposer")