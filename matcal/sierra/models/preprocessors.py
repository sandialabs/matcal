"""
Preprocessors and low-level file/mesh preparation utilities for SIERRA models.

This module is intended for internal use by MatCal SIERRA model implementations.
"""

from glob import glob
import os
import shutil

from matcal.core.constants import DESIGN_PARAMETER_FILE, STATE_PARAMETER_FILE
from matcal.core.logger import initialize_matcal_logger
from matcal.core.models import (
    ModelPreprocessorBase,
    _copy_file_or_directory_to_target_directory,
    _get_mesh_template_folder,
)
from matcal.core.mesh_modifications import get_mesh_decomposer


logger = initialize_matcal_logger(__name__)


def add_aprepro_to_input(filename: str, message: str) -> None:
    """
    Prepend a line to a SIERRA input deck file (used to inject aprepro includes).

    :param filename: Path to SIERRA input deck to modify in place.
    :param message: Line/string to prepend.
    """
    temp_file = filename + ".temp"
    with open(filename, "r") as f_read:
        with open(temp_file, "w") as f_write:
            f_write.write(message)
            for line in f_read:
                f_write.write(line)
    shutil.copy(temp_file, filename)
    os.remove(temp_file)


class AddApreproParamFileLinesPreprocessor(ModelPreprocessorBase):
    """
    Prepends aprepro include lines for MatCal parameter/state files.

    Adds (in order) the includes:
      - {include(design_parameters.i)}
      - {include(state_parameters.i)}
    """

    def __init__(self):
        self.param_aprepro_include = f"{{include({DESIGN_PARAMETER_FILE})}}\n"
        self.state_aprepro_include = f"{{include({STATE_PARAMETER_FILE})}}\n"

    def process(self, template_dir, input_filename):
        input_filename = os.path.basename(input_filename)
        input_file = f"{template_dir}/{input_filename}"
        add_aprepro_to_input(input_file, self.param_aprepro_include)
        add_aprepro_to_input(input_file, self.state_aprepro_include)


class DecomposeAndCopyMeshPreprocessor(ModelPreprocessorBase):
    """
    Mesh decomposition/copy helper.

    - Copies (or moves) the source mesh into the template mesh folder.
    - If running in parallel (n_cores > 1), decomposes the mesh and symlinks
      decomposed pieces into the state template dir.
    - If serial, copies the mesh into the template mesh folder and symlinks into
      the state template dir.
    """

    def process(self, computing_info, template_dir, mesh_filename, delete_source_mesh=False):
        mesh_decomposer_class = get_mesh_decomposer(mesh_filename)
        mesh_decomposer = mesh_decomposer_class()

        logger.info(f'\t\tPreparing mesh "{os.path.split(mesh_filename)[-1]}"')
        n_cores = computing_info.number_of_cores

        mesh_files_template_folder = _get_mesh_template_folder(template_dir)
        template_mesh_filename = os.path.join(
            mesh_files_template_folder, 
            os.path.basename(mesh_filename)
        )

        logger.debug(f"\t\tThe path to the mesh is:\n{template_mesh_filename}\n")

        if delete_source_mesh:
            shutil.move(mesh_filename, template_mesh_filename)
            mesh_filename = template_mesh_filename

        if n_cores > 1:
            mesh_decomposer.decompose_mesh(
                os.path.abspath(mesh_filename),
                n_cores,
                mesh_files_template_folder,
            )
            # remove the undecomposed mesh from the mesh folder, SIERRA will use pieces
            if os.path.exists(template_mesh_filename):
                os.remove(template_mesh_filename)
        else:
            _copy_file_or_directory_to_target_directory(mesh_files_template_folder, mesh_filename)

        # Symlink all mesh-folder files into the template_dir (state working dir)
        for file in glob(mesh_files_template_folder + os.path.sep + "*"):
            src_file = os.path.abspath(file)
            dest_file = os.path.join(template_dir, os.path.basename(file))
            if src_file != dest_file:
                os.symlink(src_file, dest_file)

        logger.info("\t\tMesh ready")