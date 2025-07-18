import os

from matcal.core.external_executable import LocalExternalExecutable
from matcal.core.object_factory import BasicIdentifier





MatCalCubitExecutablePathIdentifier = BasicIdentifier()


def get_cubit_path(version='release'):
    try:
        path = MatCalCubitExecutablePathIdentifier.identify(version)
    except KeyError:
        err_str = ("A path to the release cubit executable has not been registered in " +
            "the \'MatCalCubitExecutablePathIdentifier\'. Import the identifier from "+
            "\'matcal.cubit.cubit_runner\' and assign  " +
            "the appropriate path for \'release\' in a file to setup paths")
        raise RuntimeError(err_str)
    return path


class CubitExternalExecutable(LocalExternalExecutable):

    def __init__(self, cubit_cmds, modules_to_load=[], cubit_version='release',
                 working_directory=None):
        cubit_path=get_cubit_path(cubit_version)
        cmds = self._get_updated_commands(cubit_cmds, cubit_path)
        super().__init__(cmds, modules_to_load, working_directory=working_directory)

    def _get_updated_commands(self, cubit_cmds, cubit_path):
        cmds = ([os.path.join(cubit_path, "cubit"), '-nojournal', 
               '-nobanner', '-noecho', '-batch', '-nographics', '-nogui'] + 
               cubit_cmds)
        return cmds
   
    def run(self):
        results = super().run()
        errors  = self.cubit_has_errors(results)
        if errors:
            return_code = 1
            with open("cubit.err", "w") as f:
                f.write(results[0])
                f.write(results[1])
        else:
            return_code = 0

        return results[0], results[1], return_code

    def cubit_has_errors(self, runner_results):
        if "error" in runner_results[0].lower() or "error" in runner_results[1].lower():
            return True
        else:
            return False

