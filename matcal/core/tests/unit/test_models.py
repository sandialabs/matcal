from abc import ABC, abstractmethod
from glob import glob
from matcal.core.objective import CurveBasedInterpolatedObjective
from matcal.core.parameter_studies import ParameterStudy
import numpy as np
import os
from scipy.stats import qmc

from matcal.core.computing_platforms import local_computer, RemoteComputingPlatform
from matcal.core.constants import (MATCAL_MESH_TEMPLATE_DIRECTORY, 
                                   MATCAL_TEMPLATE_DIRECTORY)
from matcal.core.data import convert_dictionary_to_data, Data
from matcal.core.file_modifications import use_jinja_preprocessor
from matcal.core.models import (_ComputerControllerComponentBase,
                                AdditionalFileCopyPreprocessor, 
                                InputFileCopyPreprocessor, MatCalSurrogateModel, 
                                PythonModel, _DefaultComputeInformation, 
                                _ResultsInformation, _copy_file_or_directory_to_target_directory, 
                                _create_template_folder, _get_mesh_template_folder, 
                                UserExecutableModel)
from matcal.core.parameters import Parameter, ParameterCollection
from matcal.core.serializer_wrapper import _format_serial
from matcal.core.simulators import PythonSimulator, SimulatorResults, ExecutableSimulator
from matcal.core.state import SolitaryState, State
from matcal.core.study_base import StudyResults
from matcal.core.surrogates import SurrogateGenerator
from matcal.core.tests.integration.test_surrogates import _setup_initial_surrogate_generator
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest


class ModelForTestsBase(ABC):
    
    @property
    @abstractmethod
    def _model_class(self):
        """"""

    @property        
    @abstractmethod
    def _simulator_class(self):
        """"""

    @abstractmethod
    def init_model(self):
       """"""


class ModelTestBase(object):
    def __init__():
            pass

    class CommonTests(MatcalUnitTest):

        def setUp(self, filename):
            super().setUp(filename)

        def test_set_results_reader(self):
            model = self.init_model()
            model._set_results_reader_object(int)
            self.assertEqual(model._results_information.results_reader_object, int)

        def test_set_computer(self):
            model = self.init_model()
            model._set_computing_platform(local_computer)
            self.assertEqual(model.computer, local_computer)

            with self.assertRaises(_ComputerControllerComponentBase.InvalidComputerSpecified):
                model._set_computing_platform("not a comp.")

        def test_set_queue_id(self):
            model = self.init_model()

            model.run_in_queue("mywcid", 1, True)
            self.assertEqual(model.queue_id, "mywcid")

            with self.assertRaises(_ComputerControllerComponentBase.InvalidQueueIDSpecified):
                model.run_in_queue(1, 1, True)

        def test_set_time_limit(self):
            model = self.init_model()

            model.run_in_queue("id", 0.5, True)
            self.assertEqual(model.time_limit_seconds, 0.5*60*60)
            model.run_in_queue("id",1, True)
            self.assertEqual(model.time_limit_seconds, 1*60*60)

            with self.assertRaises(_ComputerControllerComponentBase.InvalidTimeLimitSpecified):
                model.run_in_queue("id",-0.1, True)
                model.run_in_queue("id","a", True)

        def test_default_time_limit(self):
            model = self.init_model()

            self.assertIsNone(model.time_limit_seconds)

        def test_add_environment_module(self):
            model = self.init_model()

            with self.assertRaises(TypeError):
                model.add_environment_module(1)

            model.add_environment_module("sierra/master")
            model.add_environment_module("dakota")
            model.add_environment_module("anaconda3")

            self.assertListEqual(model.modules_to_load, ["sierra/master", "dakota", "anaconda3"])

        def test_default_continue_on_failure(self):
            model = self.init_model()

            self.assertTrue(model.cause_failure_if_simulation_fails())

        def test_enable_calibration_to_continue_if_simulation_fails(self):
            model = self.init_model()

            model.continue_when_simulation_fails()
            self.assertFalse(model.cause_failure_if_simulation_fails())

        def test_raise_error_if_passed_default_values_are_different_lengths(self):
            model = self.init_model()
            with self.assertRaises(RuntimeError):
                model.continue_when_simulation_fails(x=[1, 2, 3], y=[2])

        def test_raise_error_if_passed_non_numeric(self):
            model = self.init_model()
            with self.assertRaises(RuntimeError):
                model.continue_when_simulation_fails(x=[1, 2, 3], y=[2, 'a', 3])

        def test_set_number_of_cores(self):
            model = self.init_model()

            self.assertEqual(model.number_of_cores, 1)
            model.set_number_of_cores(10)
            self.assertEqual(model.number_of_cores, 10)
            self.assertEqual(model.number_of_local_cores, 10)
            model._set_computing_platform(RemoteComputingPlatform(None,
                                                                  None,
                                                                  None,
                                                                  None))
            self.assertEqual(model.number_of_cores, 10)
            self.assertEqual(model.number_of_local_cores, 1)

        def test_throw_error_of_n_cores_less_than_one(self):
            model = self.init_model()
            with self.assertRaises(_ComputerControllerComponentBase.InvalidCoreUseValueError):
                model.set_number_of_cores(0)
            with self.assertRaises(_ComputerControllerComponentBase.InvalidCoreUseValueError):
                model.set_number_of_cores(-2)

        def test_default_name(self):
            model = self.init_model()
            id = model._id_number
            self.assertEqual(model.name, f'{model.model_type}_{id}')
            model = self.init_model()
            self.assertEqual(model.name, f'{model.model_type}_{id+1}')

        def test_set_name_return_name(self):
            model = self.init_model()
            my_name = "this is my name"
            model.set_name(my_name)
            self.assertEqual(model.name, my_name)

        def test_simulator_class_is_python(self):
            model = self.init_model()
            self.assertEqual(model._simulator_class, self._simulator_class)

        def test_set_number_of_cores_to_passed_value(self):
            model = self.init_model()
            n_cores = 4
            model.set_number_of_cores(n_cores)
            self.assertEqual(model.number_of_cores, n_cores)

        def test_add_constant(self):
            model = self.init_model()
            model.reset_constants()
            model.add_constants(a=1, b=2)
            goal = {'a':1, 'b':2}
            self.assertEqual(model._stateless_user_variables, goal)
            model.add_constants(element="SD")
            goal['element'] = "SD"
            self.assertEqual(model._stateless_user_variables, {'a':1, 'b':2, 'element':"SD"})
            with self.assertRaises(TypeError):
                model.add_constants(a=(1, 2))

        def test_add_constant_to_state_stateless(self):
            model = self.init_model()
            model.reset_constants()
            model.add_constants(a=1, b=2)
            model.add_constants(element="SD")
            self.assertEqual(model.get_model_constants(), {'a':1, 'b':2, 'element':"SD"})

        def test_add_state_constant(self):
            model = self.init_model()
            state1=State('state1', c=4,d="D")
            state2=State('state2', c=6,d="d")
            model.reset_constants()

            model.add_state_constants(state1, a=1, b=2)
            model.add_state_constants(state2, a=3, b=4)

            self.assertEqual(model._state_user_variables[state1], {'a':1, 'b':2})
            self.assertEqual(model._state_user_variables[state2], {'a':3, 'b':4})
            model.add_state_constants(state1, element="SD")
            model.add_state_constants(state2, element="TL")
            self.assertEqual(model._state_user_variables[state1], {'a':1, 'b':2, 'element':"SD"})
            self.assertEqual(model._state_user_variables[state2], {'a':3, 'b':4, 'element':"TL"})

            with self.assertRaises(TypeError):
                model.add_state_constants(state1, a=(1, 2))

        def test_add_constant_to_state_stateless_and_state(self):
            model = self.init_model()
            model.reset_constants()
            state1 = State("state1", d=1)
            state2 = State("state2", d=2)

            model.add_constants(a=1, b=2)
            model.add_constants(element="SD")

            model.add_state_constants(state1, a=3, c=4)
            model.add_state_constants(state2, b=3, d=3, c=5)

            self.assertEqual(model.get_model_constants(state1), {'a':3, 'b':2, 'element':"SD", 'c':4})
            self.assertEqual(model.get_model_constants(state2), {'a':1, 'b':3, 'd':3, 'element':"SD", 'c':5})

        def test_reset_constants(self):
            model = PythonModel(linear_python_model_more_constants)
            model.reset_constants()
            state1 = State("state1", d=1)
            state2 = State("state2", d=2)

            model.add_constants(a=1, b=2)
            model.add_constants(element="SD")

            model.add_state_constants(state1, a=3, c=4)
            model.add_state_constants(state2, b=3, d=3, c=5)
            model.reset_constants()
            self.assertEqual({}, model._state_user_variables)
            self.assertEqual({}, model._stateless_user_variables)

            
            self.assertEqual(model.get_model_constants(state1), {})
            self.assertEqual(model.get_model_constants(state2), {})


class TestComputeInformation(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def test_read_defaults(self):
        info = _DefaultComputeInformation(None)
        self.assertEqual(info.computer, local_computer)
        self.assertEqual(info.number_of_cores, 1)
        self.assertIsNone(info.time_limit_seconds)
        self.assertIsNone(info.queue_id)
        self.assertIsNone(info.modules_to_load)
        self.assertTrue(info.fail_on_simulation_failure)

    def test_read_executable(self):
        info = _DefaultComputeInformation("my_exe")
        self.assertEqual(info.executable, "my_exe")

    
class TestResultsInformation(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def test_defaults(self):
        info = _ResultsInformation()
        self.assertTrue(info.results_filename, "results.csv")
    
    def test_read_simple_csv(self):
        csv_name = "simple.csv"
        self._make_csv(csv_name)
        info = _ResultsInformation()
        data = info.read(csv_name)
        goal = {"X": [0, 1, 2], "Y": [0, 2, 4]}
        for key, goal_value in goal.items():
            self.assert_close_arrays(data[key], goal_value)
    
    def _make_csv(self, filename):
        with open(filename, 'w') as f:
            f.write("X,Y\n")
            for i in range(3):
                line = f"{i},{2*i}\n"
                f.write(line)

def linear_python_model(slope, intercept):
    time_max = 10
    num_time_steps = 100
    time = np.linspace(0, time_max, num_time_steps)
    values = slope * time + intercept
    return {'time': time, "Y": values}

def linear_python_model_constants(slope, intercept, a, b, element):
    time_max = 10
    num_time_steps = 100
    time = np.linspace(0, time_max, num_time_steps)
    values = slope * time + intercept
    return {'time': time, "Y": values}

def linear_python_model_more_constants(slope, intercept, a, b, c, d, element):
    time_max = 10
    num_time_steps = 100
    time = np.linspace(0, time_max, num_time_steps)
    values = slope * time + intercept
    return {'time': time, "Y": values, "d":np.ones(num_time_steps)*d}

class _simple_params:
    
    def __init__(self):
        self._names = []
        self._lower = []
        self._upper = []
        
        
    def add(self, name, lower, upper):
        self._names.append(name)
        self._lower.append(lower)
        self._upper.append(upper)

    @property
    def lower(self):
        return self._lower
    
    @property
    def upper(self):
        return self._upper
    
    @property
    def names(self):
        return self._names
    
    @property
    def count(self):
        return len(self._names)
    
    def __getitem__(self, name):
        idx = self._names.index(name)
        return self._lower[idx], self._upper[idx]
    
    
def _generate_mock_eval_hist(params, samples):
        pc = ParameterCollection('test')
        params_hist = {}
        for idx, name in enumerate(params.names):
            param_spread = samples[:,idx]
            low, high = params[name]
            pc.add(Parameter(name, low, high))
            params_hist[name] = _format_serial(param_spread)
        n_samples = samples.shape[0]
        eval_ids = _format_serial(np.arange(n_samples))
        objs = _format_serial(np.random.uniform(0, 100, n_samples))
        return params_hist, pc    
    
    
def build_surrogate_from_python_function(py_func, interp_locations, probes, indep_var, 
                                         params, n_samples, decomp_var):
    results_file = "py_results"
    sur_file = "my_surrogate"
    lhs = qmc.LatinHypercube(params.count)
    samples = qmc.scale(lhs.random(n_samples), params.lower, params.upper)
    study_results = StudyResults()
    params_hist, param_collect = _generate_mock_eval_hist(params, samples)
    _generate_parameter_evaluations(py_func, samples, params.names, n_samples, results_file)
    sur_gen = SurrogateGenerator(s2s_info, decomp_var=decomp_var)
    return sur_gen.generate(sur_file)


def remove_old_surrogate_files():
    results_file = "py_results"
    command = f"rm -f {results_file}*.json" 
    os.system(command)


def _generate_parameter_evaluations(function, samples, param_order,
                                        n_samples, filename):
    for sample_idx in range(n_samples):
        export_name = filename+str(sample_idx)+".json"
        params = {}
        for p_i, p_name in enumerate(param_order):
            params[p_name] = samples[sample_idx, p_i]
        results = convert_dictionary_to_data(function(**params))


class PythonModelForTests(ModelForTestsBase):
    _model_class = PythonModel
    _simulator_class = PythonSimulator

    def init_model(self):
        return self._model_class(linear_python_model)


def build_basic():
    probes = ['Y']
    indep = 'time'
    interp_loc = np.linspace(0, 10, 100)
    params = _simple_params()
    params.add('slope', -1, 3)
    params.add('intercept', -1, 3)
    p_names = ['slope', 'intercept']
    p_low = [-1, -1]
    p_high = [3, 3]
    n_samples = 10
    sur_gen = _setup_initial_surrogate_generator(n_samples, p_names, p_low, 
                                                 p_high, indep, linear_python_model)
    
    sur_gen.set_surrogate_details(interpolation_locations=interp_loc)
    sur_gen.set_PCA_details(None, 1e-2)
    surrogate = sur_gen.generate('basic')
    
    return surrogate

def return_dict(**params):
    return {"Y":np.zeros(10)}

class SurrogateModelForTests(ModelForTestsBase):
    _model_class = MatCalSurrogateModel
    _simulator_class = PythonSimulator

    def init_model(self, build=False):
        if build:
            BASIC_SURROGATE = build_basic()
        else:
            BASIC_SURROGATE = return_dict
        return self._model_class(BASIC_SURROGATE)
    



class TestPythonModel(ModelTestBase.CommonTests, PythonModelForTests):

    def setUp(self):
        super().setUp(__file__)
        self.py_model_file = os.path.join(self.get_current_files_path(__file__),  
                              "test_reference", "python_function_importer", 
                              "linear_python_model.py")

    def test_input_file_is_None(self):
        model = self.init_model()
        self.assertIsNone(model.input_file)

    def test_results_filename_is_None(self):
        model = self.init_model()
        self.assertIsNone(model.results_filename)

    def test_results_filetype_is_None(self):
        model = self.init_model()
        self.assertIsNone(model.results_file_type)

    def test_model_type_is_python(self):
        model = self.init_model()
        self.assertEqual(model.model_type, 'python')

    def test_set_executable(self):
        model = self.init_model()
        init_exec = model.executable
        self.assertEqual(init_exec, "python")
        model.set_executable("test")
        self.assertEqual(model.executable, "test")
    
    def test_run_in_queue(self):
        model = self.init_model()
        model.run_in_queue("queue_id", 1)
        self.assertEqual("queue_id", model.queue_id)
        self.assertEqual(1*60*60, model.time_limit_seconds)
    
    def test_add_env_module(self):
        model = self.init_model()
        env_modules = model.modules_to_load
        self.assertEqual(env_modules, None)
        model.add_environment_module("test")
        env_modules = model.modules_to_load
        self.assertEqual(env_modules, ["test"])
        model.add_environment_module("test2")
        self.assertEqual(env_modules, ["test", "test2"])
        with self.assertRaises(TypeError):
            model.add_environment_module(1)
        
    def test_return_python_function(self):
        py_fun = PythonModel(linear_python_model).python_function
        vars = {'slope': 2, 'intercept': 1}
        values = py_fun(**vars)

        values_goal = 2 * np.linspace(0, 10, 100) + 1.
        values_delta = values['Y'] - values_goal
        self.assertAlmostEqual(np.linalg.norm(values_delta), 0, delta=1e-10)

    def test_python_model_results_reader(self):
        from matcal.core.serializer_wrapper import matcal_save
        from matcal.core.models import _python_model_results_reader
        py_fun = PythonModel(linear_python_model).python_function
        vars = {'slope': 2, 'intercept': 1}
        values = py_fun(**vars)
        matcal_save("test.joblib", values)
        read_res = _python_model_results_reader("test.joblib")
        self.assert_close_arrays(values['Y'], read_res['Y'])

    def test_return_python_function_from_file(self):
        py_fun = PythonModel("linear_python_model", self.py_model_file).python_function
        vars = {'slope': 2, 'intercept': 1}
        values = py_fun(**vars)

        values_goal = 2 * np.linspace(0, 10, 100) + 1.
        values_delta = values['Y'] - values_goal
        self.assertAlmostEqual(np.linalg.norm(values_delta), 0, delta=1e-10)

    def test_build_python_simulator(self):
        state = SolitaryState()
        model = PythonModel(linear_python_model)
        slope = Parameter("slope", 0, 4, 2)
        intercept = Parameter("intercept", 0, 2, 1)
        
        results = model.run(state, ParameterCollection("test", slope, intercept))

        values_goal = 2 * np.linspace(0, 10, 100) + 1.
        values_delta = results.results_data['Y'] - values_goal
        self.assertAlmostEqual(np.linalg.norm(values_delta), 0, delta=1e-10)

    def test_add_constant_to_state_stateless(self):
        model = PythonModel(linear_python_model_constants)
        model.add_constants(a=1, b=2)
        model.add_constants(element="SD")
        self.assertEqual(model.get_model_constants(), {'a':1, 'b':2, 'element':"SD"})
        slope = Parameter('slope', 0, 2)
        intercept = Parameter('intercept', 0, 2)
        
        results = model.run(SolitaryState(), ParameterCollection('pc', slope, intercept))
        self.assertEqual(results.results_data.state.params, {})

    def test_add_constant_to_state_stateless_and_state(self):
        model = PythonModel(linear_python_model_more_constants)
        state1 = State("state1", d=1)
        state2 = State("state2", d=2)

        model.add_constants(a=1, b=2)
        model.add_constants(element="SD")

        model.add_state_constants(state1, a=3, c=4)
        model.add_state_constants(state2, b=3, d=3, c=5)

        self.assertEqual(model.get_model_constants(state1), {'a':3, 'b':2, 'element':"SD", 'c':4})
        self.assertEqual(model.get_model_constants(state2), {'a':1, 'b':3, 'd':3, 'element':"SD", 'c':5})
        
        slope = Parameter('slope', 0, 2)
        intercept = Parameter('intercept', 0, 2)
        
        results = model.run(state1, ParameterCollection('pc', slope, intercept))
        self.assertEqual(results.results_data.state.params, {'d':1 })

        results = model.run(state2, ParameterCollection('pc', slope, intercept))
        self.assertEqual(results.results_data.state.params, {"d":2})

        d = Parameter("d", 0, 20, 10)
        results = model.run(state2, ParameterCollection('pc', slope, intercept, d))
        self.assertEqual(results.results_data.state.params, {"d":2})
        self.assertEqual(results.results_data["d"][0], 10)

    def test_simulation_fails_half_the_time_complete_the_study(self):
        def fail_if_above_0(**parameters):
            import numpy as np
            n = 3
            if parameters['switch'] < 0:
                return {'time':[0,1,2], 'a':np.ones(n)}
            else:
                raise RuntimeError('bad val')
        
        model = PythonModel(fail_if_above_0)
        model.continue_when_simulation_fails()
        obj = CurveBasedInterpolatedObjective('time', 'a')
        state = State("custom_state_name_no_args")
        data = convert_dictionary_to_data({'time':[0,1,2], 'a':np.ones(3)})
        data.set_state(state)
        p = Parameter('switch', -1, 1)
        study = ParameterStudy(p)
        study.add_evaluation_set(model, obj, data)
        study.add_parameter_evaluation(switch=.1)
        study.add_parameter_evaluation(switch=-.1)
        study.add_parameter_evaluation(switch=.1)
        study.add_parameter_evaluation(switch=-.1)
        study.run_in_serial()
        results = study.launch()
        sim_hist = results.simulation_history[model.name][data.state]
        failure_default = {'time':np.linspace(-1, 1, 20), 'a':np.linspace(-1, 1, 20)}
        for i, model_results in enumerate(sim_hist):
            if i%2 != 0:
                goal = data
            else:
                goal = failure_default
            for name in ['time', 'a']:
                self.assert_close_arrays(goal[name], model_results[name])


class TestMatCalSurrogateModel(ModelTestBase.CommonTests, SurrogateModelForTests):

    def setUp(self):
        super().setUp(__file__)

    def test_model_type_is_python(self):
        model = self.init_model(build=False)
        self.assertEqual(model.model_type, 'matcal_surrogate')

    def test_return_python_function(self):
        model = self.init_model(build=True) 
        py_fun = model.python_function
        self.assertTrue(callable(py_fun))
        vars = {'slope': 2, 'intercept': 1}
        values = py_fun(**vars)
        self.assertTrue("Y" in values)
        self.assertTrue(len(values["Y"] == 100))

    def test_build_run_python_simulator(self):
        state = SolitaryState()
        model = self.init_model(build=True) 
        slope = Parameter("slope", 0, 4, 2)
        intercept = Parameter("intercept", 0, 2, 1)
        results = model.run(state, ParameterCollection("test", slope, intercept))
        self.assertIsInstance(results, SimulatorResults)
        
        
class TestModelFunctions(MatcalUnitTest):
    
    def setUp(self):
        super().setUp(__file__)
    
    def test_make_template_folders(self):
        cwd = os.getcwd()
        dir_path = f"{cwd}/first/second"
        _create_template_folder(dir_path)
        self.assertTrue(os.path.exists(dir_path))
        
    def test_get_mesh_template_dir_does_not_exist(self):
        current_dir = '.'
        mesh_template = _get_mesh_template_folder(current_dir)
        goal = f"{MATCAL_MESH_TEMPLATE_DIRECTORY}/{current_dir}"
        self.assertEqual(mesh_template, goal)
        
    def test_get_mesh_template_dir_template_exists(self):
        current_dir = f"./{MATCAL_TEMPLATE_DIRECTORY}/Stuff"
        mesh_template= _get_mesh_template_folder(current_dir)
        goal = f"./{MATCAL_MESH_TEMPLATE_DIRECTORY}/Stuff"
        self.assertEqual(mesh_template, goal)
        
    def test_copy_file_to_directory(self):
        filename = "test.txt"
        with open(filename, 'w') as f:
            f.write('X')
        target_dir = "target"
        os.mkdir(target_dir)
        _copy_file_or_directory_to_target_directory(target_dir, filename)
        self.assert_file_exists(f"{target_dir}/{filename}")
        
    def test_copy_dir_to_directory(self):
        source_dir = "sourece"
        os.mkdir(source_dir)
        filename = f"{source_dir}/test.txt"
        with open(filename, 'w') as f:
            f.write('X')
        target_dir = "target"
        os.mkdir(target_dir)
        _copy_file_or_directory_to_target_directory(target_dir, source_dir)
        self.assert_file_exists(f"{target_dir}/{filename}")
        
class ModelProcessorTest(MatcalUnitTest):
    
    def setUp(self):
        super().setUp(__file__)
        
    def test_copy_files_passed_though_keyword_args(self):
        afcp = AdditionalFileCopyPreprocessor()
        target_dir = "target"
        os.mkdir(target_dir)
        file_list = ['test.txt', 'good.txt', 'another_file.txt']
        self._make_mock_files(file_list)
        computing_info = None
        afcp.process(target_dir, additional_files=file_list)
        for filename in file_list:
            goal_path = f"{target_dir}/{filename}"
            self.assert_file_exists(goal_path)

    def _make_mock_files(self, file_list):
        for filename in file_list:
            with open(filename, 'w') as f:
                f.write("stuff")
        
    def test_input_file_copy_process(self):
        ifcp = InputFileCopyPreprocessor()
        target_dir = "target"
        os.mkdir(target_dir)
        file_list = ['my_input.i']
        self._make_mock_files(file_list)
        ifcp.process(target_dir, input_filename=file_list[0])
        self.assert_file_exists(f"{target_dir}/{file_list[0]}")


class UserExecutableModelForTests(ModelForTestsBase):
    _model_class = UserExecutableModel
    _simulator_class = ExecutableSimulator

    def init_model(self, *args):
        return self._model_class("ls", "-ltah", *args, results_filename="results.csv")


class TestUserExecutableModel(ModelTestBase.CommonTests, UserExecutableModelForTests):

    def setUp(self):
        super().setUp(__file__)
        use_jinja_preprocessor()

    def test_init(self):
        mod = self.init_model("extra_arg")
        self.assertEqual(mod.executable, "ls")
        self.assertEqual(mod.results_filename, "results.csv")
        self.assertEqual(mod._arguments, ["-ltah", "extra_arg"])
        with self.assertRaises(TypeError):
            self._model_class(1, "test.csv")
        with self.assertRaises(TypeError):
            self._model_class("ls", 1, "test.csv")
        with self.assertRaises(TypeError):  
            self._model_class("ls", "1", 1) 

    def test_add_necessary_files(self):
        add_files = ["fake_apr.inc", "fake_apr2.inc", 
                     'fake_input.i', "fake_geo.g"]
        for file in add_files:
            if file.split(".")[-1] == "g":
                with open(file, "wb") as f:
                    f.write(os.urandom(1024))
            else:
                with open(file, "w") as f:
                    f.write("{{ var }}\n")
                    f.write("{{ design_param }}\n")

        os.mkdir("test_dir")
        with open("test_dir/extra_file.txt", "w") as f:
            f.write("{{ var }}\n")
            f.write("{{ design_param }}\n")

        with open("results.csv", "w") as f:
            f.write("displacement, load\n")
            f.write("0, 0\n")
            f.write("1, 1\n")

        model = self.init_model()
        model.add_necessary_files("test_dir", "results.csv", *add_files)
        goal_files = []
        for file in add_files+["test_dir"]:
            goal_files.append(os.path.abspath(file))
            self.assertTrue(goal_files[-1] in model._additional_sources_to_copy)
        goal_files.append(os.path.abspath("results.csv"))

        for file in model._additional_sources_to_copy:
            self.assertTrue(file in goal_files)

    def test_run_and_check_files_modified(self):
        use_jinja_preprocessor()
        add_files = ["fake_apr.inc", "fake_apr2.inc", 
                     'fake_input.i', "fake_geo.g"]
        for file in add_files:
            if file.split(".")[-1] == "g":
                with open(file, "wb") as f:
                    f.write(os.urandom(1024))
            else:
                with open(file, "w") as f:
                    f.write("{{ var }}\n")
                    f.write("{{ design_param }}\n")

        os.mkdir("test_dir")
        with open("test_dir/extra_file.txt", "w") as f:
            f.write("{{ var }}\n")
            f.write("{{ design_param }}\n")

        with open("results.csv", "w") as f:
            f.write("displacement, load\n")
            f.write("0, 0\n")
            f.write("1, 1\n")

        model = self._model_class('ls', results_filename="results.csv")
        model.add_necessary_files("test_dir", "results.csv", *add_files)

        state = State("test", var=1)
        param = Parameter("design_param", 0, 2, 2)
        pc = ParameterCollection("test", param)
        run_dir = model.get_target_dir_name(state)
        model.run(state, pc)
        dir_files = glob(os.path.join(run_dir, "*.*"))
        for file in add_files:
            self.assertTrue(os.path.join(run_dir, file) in dir_files)
            if ".g" not in file:
                with open(os.path.join(run_dir, file), "r") as f:
                    lines = f.readlines()
                self.assertEqual(lines[0], "1\n")
                self.assertEqual(lines[1], "2.0")
    
    def test_build_simulator(self):
        mod = self.init_model("extra_arg")
        mod.add_constants(a=1)
        state = State("test", b=2)
        mod.add_state_constants(state, c=3)
        sim = mod.build_simulator(state)

        self.assertIsInstance(sim, ExecutableSimulator)
        self.assertEqual(sim._commands, ["ls", "-ltah", "extra_arg"])
        self.assertEqual(sim._model_constants, {"a":1, "c":3})

    def test_continue_on_failure_bad_exec(self):
        with open("results.csv", "w") as f:
            f.write("displacement, load\n")
            f.write("0, 0\n")
            f.write("1, 1\n")
        model = self._model_class("ls", "bad_option", results_filename="results.csv")
        model.add_necessary_files("results.csv")
        pc = ParameterCollection("null", Parameter("null", 0, 1))
        with self.assertRaises(RuntimeError):
            res = model.run(SolitaryState(), pc)
        model.continue_when_simulation_fails()
        res = model.run(SolitaryState(), pc)
        self.assertIsInstance(res.results_data, Data)
        self.assertEqual(res.stdout, "")
        self.assertEqual(res.return_code, 2)
        self.assertTrue(len(res.stderr) > 0)

    def test_continue_on_failure_bad_command(self):
        with open("results.csv", "w") as f:
            f.write("displacement, load\n")
            f.write("0, 0\n")
            f.write("1, 1\n")
        model = self._model_class("bad", "bad_option", results_filename="results.csv")
        model.add_necessary_files("results.csv")
        pc = ParameterCollection("null", Parameter("null", 0, 1))
        with self.assertRaises(RuntimeError):
            res = model.run(SolitaryState(), pc)
        model.continue_when_simulation_fails()
        res = model.run(SolitaryState(), pc)
        self.assertIsInstance(res.results_data, Data)

    def test_continue_on_failure_no_file_write_clean_error_code(self):
        with open("results.csv", "w") as f:
            f.write("displacement, load\n")
            f.write("0, 0\n")
            f.write("1, 1\n")
        model = self._model_class("ls", results_filename="no_file.csv")
        pc = ParameterCollection("null", Parameter("null", 0, 1))
        with self.assertRaises(FileNotFoundError):
            res = model.run(SolitaryState(), pc)
        model.continue_when_simulation_fails()
        res = model.run(SolitaryState(), pc)
        self.assertIsNone(res.results_data, Data)
        self.assertTrue(len(res.stdout) > 0)
        self.assertEqual(res.return_code, 0)
        self.assertEqual(res.stderr, "")
