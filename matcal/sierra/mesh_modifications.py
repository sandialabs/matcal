import os
import shutil
import glob

from matcal.core.models import MeshComposer, MeshDecomposer
from matcal.core.external_executable import MatCalExternalExecutableFactory
from matcal.core.computing_platforms import local_computer


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
        mesh_decompose_runner = MatCalExternalExecutableFactory.create(self._commands, 
                                                                       self._modules_to_load, 
                                                                       computer)
        stdout, stderr, return_code = mesh_decompose_runner.run()
        os.chdir(orig_dir)


class YadaMeshDecomposer(MeshDecomposer):
    def __init__(self):
        super().__init__()
        self._fastspread_commands = None
        self._mesh_basename = None
        self._modules_to_load = ['sierra']

    def _build_commands(self, mesh_file, number_of_cores):
        self._commands = ["yada", mesh_file, str(number_of_cores),
                          "-nomech", "-nodis", "&>", "yada.out"]

    def _build_fastspread_commands(self, mesh_file):
        self._mesh_basename = mesh_file.split('.g')[0]
        self._fastspread_commands = ["fastspread", self._mesh_basename]

    def _move_rename_files(self):
        decomp_files = sorted(glob.glob('./1/*.par*'))
        for decomp_file in decomp_files:
            decomp_name = decomp_file.split('/')[-1]
            decomp_num = decomp_name.split('.par.')[0]
            new_name = self._mesh_basename + '.g.' + decomp_num
            shutil.rename(decomp_file, new_name)

    def _cleanup(self):
        shutil.rmtree('./1')
        os.remove(self._mesh_basename + '.nem')

    def decompose_mesh(self, mesh_file, number_of_cores, output_directory='.', 
                       computer=local_computer):
        self._build_commands(mesh_file, number_of_cores)
        self._build_fastspread_commands(mesh_file)
        orig_dir = os.getcwd()
        os.chdir(output_directory)
        mesh_decompose_runner = MatCalExternalExecutableFactory.create(self._commands, 
                                                                       self._modules_to_load, 
                                                                       computer)
        stdout, stderr, return_code = mesh_decompose_runner.run()

        fastspread_runner = MatCalExternalExecutableFactory.create(self._fastspread_commands, 
                                                                   self._modules_to_load, 
                                                                   computer)

        stdout, stderr, return_code = fastspread_runner.run()
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
        runner = MatCalExternalExecutableFactory.create(commands, 
                                                        self._modules_to_load, 
                                                        computer)
        stdour, stderr, return_code = runner.run()
        os.chdir(orig_dir)