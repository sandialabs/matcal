import os
import subprocess

import matcal.core.constants as matcal_constants
from matcal.core.external_executable import attempt_to_execute
from matcal.core.object_factory import IdentifierByTestFunction, BasicIdentifier
from matcal.core.external_executable import MatCalExecutableEnvironmentSetupFunctionIdentifier

def raise_no_module_paths_added():
    err_str = ("Paths to the module commands must be specified. "+""
        "Two module commands are supported. A path for the 'lmod' command and "+
        "a path for the 'modulecmd'. To add paths for these commands, import " +
        "the 'matcal_module_command_identifier' from "+
        "\'matcal.core.linux modules\' and assign  " +
        "the appropriate path for \'lmod\' and/or \'modulecmd\' in a file to setup " +
        "site specific options and paths.")
    raise RuntimeError(err_str)

matcal_module_command_identifier = BasicIdentifier(raise_no_module_paths_added)


def get_lmod_modules_command_path():
    return matcal_module_command_identifier.identify('lmod')


def get_modulecmd_modules_command_path():
    return matcal_module_command_identifier.identify('modulecmd')



def modulecmd_command_exists():
    return os.path.exists(get_modulecmd_modules_command_path())


def lmod_command_exists():
    return os.path.exists(get_lmod_modules_command_path())


def default_modules_command_exists():
    return lmod_command_exists() or modulecmd_command_exists()


def default_modules_command_does_not_exist():
    try:
        lmod_exists = lmod_command_exists()
    except RuntimeError:
        lmod_exists = False

    try:
        modulecmd_exists = modulecmd_command_exists()
    except RuntimeError:
        modulecmd_exists = False

    return not (lmod_exists or modulecmd_exists)


# MatCal will default to lmod command unless the modulecmd exists for the user on 
# the current machine
MatCalLinuxModulesPathFunctionIdentifier = IdentifierByTestFunction(get_lmod_modules_command_path)
MatCalLinuxModulesPathFunctionIdentifier.register(modulecmd_command_exists, 
                                                  get_modulecmd_modules_command_path)


def check_for_module_command_error(module_cmds, error_string):
   test_for_libtcl_error(error_string)
   if "error" in error_string.lower():
        error_msg = f"Module commands '{module_cmds}' failed with error:\n{error_string}"
        raise RuntimeError(error_msg)


def test_for_libtcl_error(message:str):
    bad_library_name = "libtcl8.5.so"
    if bad_library_name in message:
        raise RuntimeError(f"{bad_library_name} library Error")


def run_module_commands(*cmds):
    cmds = list(cmds)
    (output, error) = subprocess.Popen(cmds, shell=False,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE).communicate()
    exec(output)
    output = output.decode()
    error = error.decode()
    return output, error


def issue_module_commands(*args):
    if type(args[0]) == type([]):
        args = args[0]
    args = list(args)
    module_command_identifier = MatCalLinuxModulesPathFunctionIdentifier.identify()
    module_cmd = [module_command_identifier()]
    all_commands = module_cmd + ["python"] + args
    output, error = run_module_commands(*all_commands)
    check_for_module_command_error(args, error)
    return output, error


def get_all_loaded_modules():
    loaded_modules = []
    out, err = issue_module_commands("list")
    returned_string = err   
    returned_string = returned_string.replace("(default)", "")
    if len(returned_string.split(")")) > 1:
        modules = returned_string.split(")")[1:]
        for module_str in modules:
            module_str = module_str.strip().split(" ")[0].strip()
            loaded_modules.append(module_str)
    return loaded_modules


def setup_environment_with_modules(modules_to_load):
        issue_module_commands("purge")
        out = None
        err = None
        for module_to_load in modules_to_load:
            out, err = issue_module_commands('load', module_to_load)
            if len(err) > 0:
                return out, err
        return []


def module_command_executer(modules:list):
    use_shell = False
    max_load_attempts = 4
    attempt_to_execute(setup_environment_with_modules, max_load_attempts,
                       matcal_constants.MODULE_PAUSE_TIME, modules)
    return [], use_shell


def module_command_writer(modules:list):
    use_shell = True
    load_base_string = "module load"
    load_string = "module purge;"
    for module in modules:
        load_string += f"{load_base_string} {module};"
    return load_string, use_shell


MatCalExecutableEnvironmentSetupFunctionIdentifier.register(default_modules_command_exists, 
                                                            module_command_executer)

def raise_no_test_module_error():
    raise RuntimeError("No valid linux module for testing has been added. "+
                       f"In your site setup files, register a valid module " +
                       f"to load for testing. Import \"MatCalTestModuleIdentifier\" "+ 
                       "from matcal.core.linux_modules and set the default test module to "+
                       "load string with "+
                       "\"MatCalTestModuleIdentifier.set_default('module_name')\"'." )
                



MatCalTestModuleIdentifier = BasicIdentifier(raise_no_test_module_error)