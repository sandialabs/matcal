import os, shutil, numbers

from matcal.core.constants import DESIGN_PARAMETER_FILE
from matcal.core.parameters import Parameter, ParameterCollection
from matcal.core.state import SolitaryState
from matcal.core.utilities import get_current_files_path

from matcal.sierra.models import UserDefinedSierraModel


TEST_SUPPORT_FILES_FOLDER = os.path.join(get_current_files_path(__file__), "test_support_files")
GENERATED_TEST_DATA_FOLDER = os.path.join(get_current_files_path(__file__), 
                                          "delete_reused_test_generated_files")
if not os.path.exists(GENERATED_TEST_DATA_FOLDER): 
    os.mkdir(GENERATED_TEST_DATA_FOLDER)


def write_empty_file(filename):
    with open(filename, "w") as f:
        f.write("\n")


def write_string_to_file(string, filename):
    with open(filename, "w") as f:
        f.write(string)


def replace_string_in_file(filename, old_string, new_string):
    input_lines = read_file_lines(filename)
    for index, line in enumerate(input_lines):
        if old_string in line:
            line = line.replace(old_string, new_string)
            input_lines[index] = line
    write_strings_to_file(filename, *input_lines)


def write_strings_to_file(filename, *strings):
    with open(filename, "w") as f:
        for string in strings:
            f.write(string.strip("\n"))
            f.write("\n")


def write_design_param_file():
    with open(DESIGN_PARAMETER_FILE, "w") as f:
        f.write("# elastic_modulus = {elastic_modulus = 200e9}\n")


def write_linear_elastic_material_file():
    file_str = """
    begin property specification for material matcal_test
        density = {density}                                                                          
        begin parameters for model linear_elastic
            youngs modulus = {elastic_modulus}                                                       
            poissons ratio = {nu}
        end
    end 
"""
    filename = "matcal_test_material_file.inc"
    write_string_to_file(file_str, filename)    
    return filename


def write_j2_plasticity_material_file():
    file_str = """
begin property specification for material matcal_test
   density = {density}
   begin parameters for model j2_plasticity
    youngs modulus = {elastic_modulus}
    poissons ratio = {nu}
    yield stress = {yield_stress}

    hardening model   =  decoupled_flow_stress

    isotropic hardening model = voce
    hardening modulus = {A}
    exponential coefficient = {b}

    yield rate multiplier = power_law_breakdown
    yield rate coefficient = 1000
    yield rate exponent = 8

    {if(coupling!="uncoupled")}

      thermal softening model = {coupling}
      beta_tq                 = {beta_tq}
      specific heat           = {specific_heat}
    {endif}
   end
   begin parameters for model linear_elastic
    youngs modulus    = {elastic_modulus}
    poissons ratio    = {nu}
   end
end
"""
    filename = "matcal_test_material_file.inc"
    write_string_to_file(file_str, filename)    
    return filename    

def run_cubit_with_commands(cmds):
    j_file = "j_file.jou"
    with open(j_file, "w") as file:
        for cmd in cmds:
            file.write(f"{cmd}\n")
    run_cubit_with_journal_file(j_file)

def run_cubit_with_journal_file(journal_file):
    from matcal.cubit.cubit_runner import CubitExternalExecutable
    cubit_runner = CubitExternalExecutable([journal_file])
    results = cubit_runner.run()  
    return results

FINE_COMPLEX_MESH_NAME = "fine_complex_test.g"
COARSE_COMPLEX_MESH_NAME = "coarse_complex_test.g"


def make_complex_mesh_for_tests():

    cubit_str = f"""
reset
brick x 6 y 3 z 0.1
webcut volume 1 with cylinder radius 1 axis z center 1.5 0 0
webcut volume 1 with cylinder radius 1 axis z center -1.5 0 0
del vol 2 3
webcut volume 1 with cylinder radius 1 axis z center 0 1.8 0
webcut volume 1 with cylinder radius 1 axis z center 0 -1.8 0
del vol 4 5
vol all size 0.075
mesh vol all
block 1 add surf 29
export mesh "{FINE_COMPLEX_MESH_NAME}"
del mesh vol all prop
vol all size 0.125
mesh vol all
export mesh "{COARSE_COMPLEX_MESH_NAME}"
"""
    run_cubit_with_commands(cubit_str.split("\n"))


def make_simple_mesh_for_tests():
    mesh_name = "test_mesh.g"
    cubit_str = f"""
reset
brick x 1 y 2 z 0.5
webcut vol all with zplane

curve with length == 1 interval 3
curve with length == 0.25 interval 1

vol all size 0.5
mesh vol all
export mesh "{mesh_name}"
"""
    run_cubit_with_commands(cubit_str.split("\n"))
    return mesh_name


def read_file_lines(filename):
    if not os.path.exists(filename):
        raise RuntimeError(f"File {filename} does not exist.")
    with open(filename, "r") as f:
        lines = f.readlines()
    return lines


def make_mesh_from_journal_file(journal_file):
    run_cubit_with_journal_file(journal_file)


def run_model(model, run_dir=None, state=SolitaryState(), 
               **params):
   
    pc = ParameterCollection("test")
    for name, val in params.items():
        if isinstance(val, numbers.Real):
            pc.add(Parameter(name, 0.9*val, 1.1*val, val))
    
    if run_dir is not None:
        run_dir = os.path.abspath(run_dir)
        if not os.path.exists(run_dir):
            raise RuntimeError(f"The directory {run_dir} does not exist")
    else:
        run_dir = os.getcwd()
    results = model.run(state, pc, run_dir)
    return model.get_target_dir_name(state), results


def make_user_defined_model(exec, input, mesh, results_filename, *additional_files, cores=8, constants={}):
    model = UserDefinedSierraModel(exec, input, mesh, *additional_files)
    model.set_number_of_cores(cores)
    model.add_constants(**constants)
    if results_filename.split(".")[-1] == "e":
        model.read_full_field_data(results_filename)
    else:
        model.set_results_filename(results_filename)
    return model

def make_mesh_from_string_or_journal(mesh_filename, mesh_journal=None, mesh_str=None):
    init_dir = os.getcwd()
    dest_dir = os.path.dirname(mesh_filename)
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
    os.chdir(dest_dir)
    if mesh_journal is not None and not os.path.exists(mesh_filename):
        make_mesh_from_journal_file(mesh_journal)
    if mesh_str is not None and not os.path.exists(mesh_filename):
        if isinstance(mesh_str, str):
            mesh_str = mesh_str.split("\n")
        run_cubit_with_commands(mesh_str)
    os.chdir(init_dir)


def create_goal_user_model_simulation_results(input, mesh_filename, target_results_filename,  
                                *additional_files, mesh_journal=None, mesh_str=None,
                                  run_dir=".", exec="adagio", 
                                  constants = {}, cores=8, 
                                  state=SolitaryState(), **params):
    if not os.path.exists(run_dir):
            os.mkdir(run_dir)
    for filename in additional_files:
        shutil.copy(filename, run_dir)
    make_mesh_from_string_or_journal(mesh_filename, 
                                     mesh_journal=mesh_journal, mesh_str=mesh_str)
    results_basename = os.path.basename(target_results_filename)
    model = make_user_defined_model(exec, input, mesh_filename, results_basename, *additional_files, 
                                        cores=cores, constants=constants)
    create_goal_model_simulation_results(model, target_results_filename,
                                        run_dir=run_dir, constants=constants,
                                        cores=cores, state=state, **params)
        

def create_goal_model_simulation_results(model, target_results_filename, 
                                         run_dir=".", state=SolitaryState(), **params):
    if not os.path.exists(run_dir):
            os.mkdir(run_dir)
    results_basename = os.path.basename(target_results_filename)
    if not os.path.exists(target_results_filename):
        target_dir, results = run_model(model, run_dir=run_dir, state=state, **params)
        completed_results_file = os.path.join(run_dir, target_dir, results_basename)
        shutil.move(completed_results_file, target_results_filename)

