import os

from matcal.exodus.library_importer import (create_exodus_class_instance)


test_support_files_dir = os.path.join(os.path.dirname(__file__), "test_support_files")

def _open_mesh( mesh_filename, open_mode="r"):
    mesh = create_exodus_class_instance(mesh_filename, mode=open_mode, array_type='numpy')
    return mesh
