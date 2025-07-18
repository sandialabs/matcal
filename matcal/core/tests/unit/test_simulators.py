
import numpy as np
import os
import shutil
import sys

from matcal.core.constants import MATCAL_WORKDIR_STR
from matcal.core.data import DataCollection, convert_dictionary_to_data
from matcal.core.models import PythonModel
from matcal.core.simulators import PythonSimulator, SimulatorFailureResults, SimulatorResults
from matcal.core.state import SolitaryState, State
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest
from matcal.core.tests.utilities_for_tests import MockExecutableModel


def py_sim_function(**variables):
    time = np.linspace(0, 1, 10)
    value = np.ones(10) * variables['m']
    return {'time': time, 'values': value}


def _set_directory_structure(model_name):
    sys.path.append(os.getcwd())
    path_to_copy = os.path.join(os.path.dirname(__file__), 
                                "test_reference", "simulator", 
                                f"{MATCAL_WORKDIR_STR}.1")
    shutil.copytree(path_to_copy, f"{MATCAL_WORKDIR_STR}.1")
    os.chdir(f"{MATCAL_WORKDIR_STR}.1")
    os.rename("python_0", model_name)


class TestSimulator(MatcalUnitTest):

    def setUp(self) -> None:
        super().setUp(__file__)
        data_dict = {"x":np.linspace(0,1,20), "y":np.linspace(0,1,20)**2}
        self.data = convert_dictionary_to_data(data_dict)
        filename = "mock_data"
        self.dat_filename = filename+".dat"
        self.csv_filename = filename+".csv"
        
        np.savetxt(self.dat_filename, self.data, delimiter=",",  header="x, y")
        np.savetxt(self.csv_filename, self.data, delimiter=",", header="x, y")

        self.dat_mock_model = MockExecutableModel(self.dat_filename)
        self.dat_mock_model.set_results_filename(os.path.abspath(self.dat_filename), 
                                                 file_type="csv")
        self.csv_mock_model = MockExecutableModel(self.csv_filename)
        
        self.csv_sim = self.csv_mock_model.build_simulator(SolitaryState())
        self.dat_sim = self.dat_mock_model.build_simulator(SolitaryState())

    def test_results_filename(self):
        self.assertEqual(self.csv_sim.results_filename, os.path.abspath(self.csv_filename))
        self.assertEqual(self.dat_sim.results_filename, os.path.abspath(self.dat_filename))

    def test_run_and_check_results(self):
        _set_directory_structure(self.csv_mock_model.name)
        params = {}
        sim_results = self.csv_sim.run(params)
        self.assert_close_arrays(sim_results.results_data, self.data)

        sim_results = self.dat_sim.run(self.dat_mock_model.name)
        self.assert_close_arrays(sim_results.results_data, self.data)


class TestPythonSimulator(MatcalUnitTest):

    class PythonModelSpy(PythonModel):

        def spy_preprocessors(self):
            return self._preprocessors

        def spy_simulator(self):
            return self._simulator_class

        def spy_id(self):
            return self._id_number

        def spy_n_cores(self):
            return self._simulation_information.number_of_cores

        def simulation_information(self):
            return self._simulation_information

    def setUp(self) -> None:
        super().setUp(__file__)
        self.py_model = self.PythonModelSpy(py_sim_function)
        self.py_sim = PythonSimulator(self.py_model.name, 
                                      self.py_model.simulation_information, 
                                      self.py_model._results_information,
                                      SolitaryState(),  self.py_model)

    def test_name_creation(self):
        self.assertEqual(self.py_sim.model_name, 'python_{}'.format(int(self.py_model._id_number)))

    def test_results_filename(self):
        self.assertEqual(self.py_sim.results_filename, None)

    def test_run_and_check_results(self):
        py_sim_results = self.py_sim.run({"m":50})
        results = py_sim_results.results_data
        results_goal = py_sim_function(m=50)
        results_delta = results['values'] - results_goal['values']
        self.assertEqual(py_sim_results.return_code, None)
        self.assertAlmostEqual(np.linalg.norm(results_delta), 0, delta=1e-10)


class TestSimulatorResults(MatcalUnitTest):
    def setUp(self) -> None:
        super().setUp(__file__)
        self.stdout = "stdout"
        self.stderr = "stderr"
        self.return_code = 1
        data_dict1 = {"x":np.linspace(0,1,100), "y":np.linspace(0,1,100)**2}
        data_dict2 = {"x":np.linspace(0,1,100), "y":2*np.linspace(0,1,100)**2}
        self.results_data = DataCollection("sim test", convert_dictionary_to_data(data_dict1), 
            convert_dictionary_to_data(data_dict2))

    def test_simulator_results(self):
        sim_results = SimulatorResults(self.results_data, self.stdout, self.stderr, 
                                       self.return_code)
        self.assertTrue(self.results_data, sim_results.results_data)
        self.assertEqual(self.stdout, sim_results.stdout)
        self.assertEqual(self.stderr, sim_results.stderr)
        self.assertEqual(self.return_code, sim_results.return_code)
        

class TestSimulatorFailureResults(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def test_can_return_std_notice(self):
        stdout = "out"
        stderr = "err"
        return_code = 1
        state = State("test")
        sfr = SimulatorFailureResults(stdout, stderr, return_code, state)
        self.assertEqual(sfr.stdout, stdout)
        self.assertEqual(sfr.stderr, stderr)
        self.assertEqual(sfr.return_code, return_code)
        self.assertIsNone(sfr.results_data)
        self.assertEqual(sfr.state, state)
