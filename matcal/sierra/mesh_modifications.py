import os
import shutil
import glob

from matcal.core.computing_platforms import local_computer
from matcal.core.external_executable import matcal_external_executable_factory
from matcal.core.logger import initialize_matcal_logger
from matcal.core.models import MeshComposer, MeshDecomposer


logger = initialize_matcal_logger(__name__)


class DecompMeshDecomposer(MeshDecomposer):
    def __init__(self, ):
        super().__init__()
        self._modules_to_load = ['sierra']

    def _build_commands(self, mesh_file, number_of_cores):
        self._commands = ["decomp", "--processors", str(number_of_cores),
                          "--rcb", mesh_file]

    def decompose_mesh(self, mesh_file, number_of_cores, output_directory='.',
                       computer=local_computer):
        self._build_commands(mesh_file, number_of_cores)
        orig_dir = os.getcwd()
        os.chdir(output_directory)
        mesh_decompose_runner = matcal_external_executable_factory.create(self._commands, 
                                                                       self._modules_to_load, 
                                                                       computer)
        stdout, stderr, return_code = mesh_decompose_runner.run()
        check_if_mesh_operation_failed(return_code, mesh_file, stderr, operation="decomposition")
        os.chdir(orig_dir)


def check_if_mesh_operation_failed(return_code, mesh_file, stderr, operation):
    if return_code != 0:
        raise RuntimeError(f"Mesh {operation} failed for mesh '{mesh_file}'. Exiting.\n"
                           f"The following errors were returned from the executable:\n{stderr}")


class YadaMeshDecomposer(MeshDecomposer):
    def __init__(self):
        super().__init__()
        self._fastspread_commands = None
        self._mesh_basename = None
        self._modules_to_load = ['sierra']

    def _build_commands(self, mesh_file, number_of_cores):
        self._commands = ["yada", mesh_file, str(number_of_cores),
                          "-nomech", "-nodis"]

    def _build_fastspread_commands(self, mesh_file):
        self._mesh_basename = mesh_file.split('.g')[0]
        self._fastspread_commands = ["fastspread", self._mesh_basename]

    def _move_rename_files(self):
        decomp_files = sorted(glob.glob('./1/*.par*'))
        for decomp_file in decomp_files:
            decomp_name = decomp_file.split('/')[-1]
            decomp_num = decomp_name.split('.par.')[-1]
            new_name = self._mesh_basename + '.g.' + decomp_num
            shutil.move(decomp_file, new_name)

    def _cleanup(self):
        shutil.rmtree('./1')
        os.remove(self._mesh_basename + '.nem')

    def decompose_mesh(self, mesh_file, number_of_cores, output_directory='.', 
                       computer=local_computer):
        self._build_commands(mesh_file, number_of_cores)
        self._build_fastspread_commands(mesh_file)
        orig_dir = os.getcwd()
        os.chdir(output_directory)
        mesh_decompose_runner = matcal_external_executable_factory.create(self._commands, 
                                                                       self._modules_to_load, 
                                                                       computer)
        stdout, stderr, return_code = mesh_decompose_runner.run()
        check_if_mesh_operation_failed(return_code, mesh_file, stderr, operation="decomposition")

        fastspread_runner = matcal_external_executable_factory.create(self._fastspread_commands, 
                                                                   self._modules_to_load, 
                                                                   computer)

        stdout, stderr, return_code = fastspread_runner.run()
        check_if_mesh_operation_failed(return_code, mesh_file, stderr, operation="decomposition")

        self._move_rename_files()
        self._cleanup()

        os.chdir(orig_dir)


class EpuMeshComposer(MeshComposer):

    def __init__(self):
        self._modules_to_load = ['sierra']
    
    def _build_commands(self, mesh_file, number_of_cores):
        split_filename = mesh_file.split('.')
        extension = split_filename[-1]
        base = '.'.join(split_filename[:-1])
        commands  = ["epu", "-extension", extension, "-processor_count", 
                     str(number_of_cores), base]
        return commands

    def compose_mesh(self, mesh_file, number_of_cores, mesh_directory=".", 
                     computer=local_computer):
        orig_dir = os.getcwd()
        os.chdir(mesh_directory)
        commands = self._build_commands(mesh_file, number_of_cores)
        runner = matcal_external_executable_factory.create(commands, 
                                                        self._modules_to_load, 
                                                        computer)
        stdour, stderr, return_code = runner.run()
        check_if_mesh_operation_failed(return_code, mesh_file, stderr, operation="composition")

        os.chdir(orig_dir)