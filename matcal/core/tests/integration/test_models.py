"""
Integration tests for MatCal models.
Tests UserExecutableModel and MatCalSurrogateModel with full workflows.
"""
import matcal as mc
import numpy as np
import unittest

from matcal.core.external_executable import matcal_executable_environment_setup_function_identifier
from matcal.core.file_modifications import use_jinja_preprocessor
from matcal.core.models import MatCalSurrogateModel, UserExecutableModel
from matcal.core.parameters import Parameter, ParameterCollection
from matcal.core.state import State, SolitaryState
from matcal.core.surrogates import SurrogateGenerator
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest
from matcal.core.tests.unit.test_adaptive_surrogates import HAS_PYAPPROX


python_model_string = """
m = {{ m }}
b = {{ b }}
exponent = {{ exponent }}

import numpy as np
x = np.linspace(0,2,100)
y = m*x**exponent+b
np.savetxt("model_res.csv", np.array([x,y]).T, header = "x, y", comments="", delimiter=',')
"""


def linear_model(slope, intercept, **kwargs):
    """Simple linear model for testing."""
    time = np.linspace(0, 10, 100)
    y = slope * time + intercept
    return {"time": time, "y": y}


def quadratic_model(a, b, c, **kwargs):
    """Quadratic model for testing."""
    x = np.linspace(0, 5, 50)
    y = a * x**2 + b * x + c
    return {"x": x, "y": y}


class TestUserExecutbleModels(MatcalUnitTest):

  def setUp(self) -> None:
    super().setUp(__file__)
    matcal_executable_environment_setup_function_identifier._registry={}
    
  def test_user_exec_model_python(self):
    with open("python_model.py", "w") as f:
        f.write(python_model_string)
    use_jinja_preprocessor()
    mod = UserExecutableModel("python3", "python_model.py", results_filename="model_res.csv")
    mod.add_necessary_files("python_model.py")
    state = State("linear", exponent=1)
    m = Parameter("m", 0, 10, 2.5)
    b = Parameter("b", 0, 10, 5)
    params = ParameterCollection("test", m,b)
    gold = np.linspace(0, 2, 100)*2.5+5

    results = mod.run(state, params)
    self.assert_close_arrays(results.results_data["y"], gold)

    state_2 = State("quadratic", exponent=2)
    results = mod.run(state_2, params)
    gold2 = np.linspace(0, 2, 100)**2*2.5+5

    self.assert_close_arrays(results.results_data["y"], gold2)

    mod.add_constants(exponent=3)
    results = mod.run(state_2, params)
    gold3 = np.linspace(0, 2, 100)**3*2.5+5
    self.assert_close_arrays(results.results_data["y"], gold3)
    
    mod.add_state_constants(state, exponent=4)
    results = mod.run(state, params)
    gold4 = np.linspace(0, 2, 100)**4*2.5+5
    self.assert_close_arrays(results.results_data["y"], gold4)

    exponent_param = Parameter("exponent", 0, 10)
    params.add(exponent_param)
    results = mod.run(state, params)
    gold5 = np.linspace(0, 2, 100)**5*2.5+5
    self.assert_close_arrays(results.results_data["y"], gold5)



class TestMatCalSurrogateModelIntegration(MatcalUnitTest):
    """Integration tests for MatCalSurrogateModel with real surrogates."""
    
    def setUp(self):
        super().setUp(__file__)
    
    def _build_linear_surrogate(self):
        """Build a surrogate from the linear model."""
        from scipy.stats import qmc
        
        # Generate training data
        n_samples = 20
        param_names = ["slope", "intercept"]
        bounds = np.array([[-1, 3], [-1, 3]])
        
        sampler = qmc.LatinHypercube(d=2, seed=42)
        samples = qmc.scale(sampler.random(n=n_samples), bounds[:, 0], bounds[:, 1])
        
        # Create study results for surrogate generation
        from matcal.core.study_base import StudyResults
        from matcal.core.data import convert_dictionary_to_data
        
        results = StudyResults()
        
        # Generate parameter history
        param_hist = {}
        for param_name in param_names:
            param_hist[param_name] = []
        
        # Generate simulation results
        sim_results = []
        for i in range(n_samples):
            params = {param_names[j]: samples[i, j] for j in range(len(param_names))}
            model_output = linear_model(**params)
            sim_results.append(convert_dictionary_to_data(model_output))
            
            for param_name in param_names:
                param_hist[param_name].append(params[param_name])
        
        # Package into StudyResults format
        results._parameter_history = param_hist
        from matcal.core.data import DataCollection
        results._simulation_history = {"model": DataCollection("test")}
        for sim_result in sim_results:
            results._simulation_history["model"].add(sim_result)
        
        # Build surrogate
        sur_gen = SurrogateGenerator(results, "model")
        sur_gen.set_surrogate_details(
            interpolation_field="time",
            interpolation_locations=np.linspace(0, 10, 100)
        )
        sur_gen.set_PCA_details(
            decomposition_variable=None,
            reconstruction_tolerance=1e-3
        )
        
        surrogate = sur_gen.generate("linear_surrogate")
        return surrogate
    
    def test_surrogate_model_basic_run(self):
        """Test basic run of MatCalSurrogateModel with a real surrogate."""
        surrogate = self._build_linear_surrogate()
        
        model = MatCalSurrogateModel(surrogate)
        state = SolitaryState()
        
        slope = Parameter("slope", -1, 3, 2.0)
        intercept = Parameter("intercept", -1, 3, 1.0)
        pc = ParameterCollection("test", slope, intercept)
        
        results = model.run(state, pc)
        
        # Verify results structure
        self.assertIn("y", results.results_data)
        self.assertIn("time", results.results_data)
        self.assertEqual(len(results.results_data["y"]), 100)
        
        # Values should be reasonable for slope=2, intercept=1
        y_values = results.results_data["y"]
        time_values = results.results_data["time"]
        
        # Check endpoints approximately
        # At time=0: y should be close to intercept (1.0)
        # At time=10: y should be close to slope*10 + intercept (21.0)
        self.assertTrue(abs(y_values[0] - 1.0) < 2.0, 
                       f"y(0) = {y_values[0]}, expected ~1.0")
        self.assertTrue(abs(y_values[-1] - 21.0) < 5.0,
                       f"y(10) = {y_values[-1]}, expected ~21.0")
    
    def test_surrogate_model_with_state_parameters(self):
        """Test MatCalSurrogateModel with state parameters."""
        surrogate = self._build_linear_surrogate()
        
        model = MatCalSurrogateModel(surrogate)
        state = State("custom_state", extra_param=5.0)
        
        slope = Parameter("slope", -1, 3, 1.0)
        intercept = Parameter("intercept", -1, 3, 0.5)
        pc = ParameterCollection("test", slope, intercept)
        
        # Should run without error even though surrogate doesn't use extra_param
        results = model.run(state, pc)
        
        self.assertIn("y", results.results_data)
    
    def test_surrogate_model_with_model_constants(self):
        """Test MatCalSurrogateModel with model constants."""
        surrogate = self._build_linear_surrogate()
        
        model = MatCalSurrogateModel(surrogate)
        model.add_constants(unused_constant=10.0)
        
        state = SolitaryState()
        slope = Parameter("slope", -1, 3, 1.5)
        intercept = Parameter("intercept", -1, 3, 0.0)
        pc = ParameterCollection("test", slope, intercept)
        
        # Should run without error
        results = model.run(state, pc)
        
        self.assertIn("y", results.results_data)
    
    def test_surrogate_model_multiple_evaluations(self):
        """Test multiple evaluations with different parameter values."""
        surrogate = self._build_linear_surrogate()
        
        model = MatCalSurrogateModel(surrogate)
        state = SolitaryState()
        
        # First evaluation
        slope1 = Parameter("slope", -1, 3, 0.5)
        intercept1 = Parameter("intercept", -1, 3, 0.0)
        pc1 = ParameterCollection("test1", slope1, intercept1)
        
        results1 = model.run(state, pc1)
        y1 = results1.results_data["y"]
        
        # Second evaluation with different parameters
        slope2 = Parameter("slope", -1, 3, 2.5)
        intercept2 = Parameter("intercept", -1, 3, 1.0)
        pc2 = ParameterCollection("test2", slope2, intercept2)
        
        results2 = model.run(state, pc2)
        y2 = results2.results_data["y"]
        
        # Results should be different
        self.assertFalse(np.allclose(y1, y2))
        
        # Second result should have larger values (higher slope and intercept)
        self.assertTrue(np.mean(y2) > np.mean(y1))


@unittest.skipIf(
    not HAS_PYAPPROX,
    "pyapprox not installed – skipping adaptive surrogate tests"
)
class TestMatCalSurrogateModelWithAdaptive(MatcalUnitTest):
    """Integration tests for MatCalSurrogateModel with AdaptiveSurrogate."""
    
    def setUp(self):
        super().setUp(__file__)
    
    def test_adaptive_surrogate_voronoi(self):
        """Test MatCalSurrogateModel with VoronoiAdaptiveSurrogateStudy."""
        # Define parameters
        a = Parameter("a", 0, 5, 2.0)
        b = Parameter("b", 0, 5, 1.0)
        c = Parameter("c", 0, 5, 0.5)
        
        # Create adaptive study
        model_func = mc.PythonModel(quadratic_model)
        
        study = mc.VoronoiAdaptiveSurrogateStudy(a, b, c)
        study.set_independent_variable("x", np.linspace(0, 5, 50))
        study.set_target_field_name("y")
        study.set_number_of_test_samples(10)
        study.set_initial_sample_count(15)
        study.set_max_iterations(3)  # Keep it small for testing
        study.set_test_group_random_seed(42)
        
        study.add_evaluation_set(model_func)
        study.launch()
        
        # Get the adaptive surrogate
        adaptive_surrogate = study.surrogate
        
        # Wrap in MatCalSurrogateModel
        surrogate_model = MatCalSurrogateModel(adaptive_surrogate)
        
        state = SolitaryState()
        a_param = Parameter("a", 0, 5, 3.0)
        b_param = Parameter("b", 0, 5, 2.0)
        c_param = Parameter("c", 0, 5, 1.0)
        pc = ParameterCollection("test", a_param, b_param, c_param)
        
        results = surrogate_model.run(state, pc)
        
        # Verify results
        self.assertIn("y", results.results_data)
        self.assertIn("x", results.results_data)
        self.assertEqual(len(results.results_data["y"]), 50)
        
        # Values should be reasonable for quadratic function
        y_values = results.results_data["y"]
        self.assertTrue(np.all(np.isfinite(y_values)))
    
    def test_adaptive_surrogate_sparse_grid(self):
        """Test MatCalSurrogateModel with SparseGridAdaptiveSurrogateStudy."""
        # Define parameters
        slope = Parameter("slope", 0, 4, 2.0)
        intercept = Parameter("intercept", 0, 3, 1.0)
        
        # Create adaptive study
        model_func = mc.PythonModel(linear_model)
        
        study = mc.SparseGridAdaptiveSurrogateStudy(slope, intercept)
        study.set_independent_variable("time", np.linspace(0, 10, 100))
        study.set_target_field_name("y")
        study.set_number_of_test_samples(10)
        study.set_error_stopping_criteria(rmse=1e-3, max_error=1e-2)
        study.set_test_group_random_seed(42)
        
        study.add_evaluation_set(model_func)
        study.launch()
        
        # Get the adaptive surrogate
        adaptive_surrogate = study.surrogate
        
        # Wrap in MatCalSurrogateModel
        surrogate_model = MatCalSurrogateModel(adaptive_surrogate)
        
        state = SolitaryState()
        slope_param = Parameter("slope", 0, 4, 2.5)
        intercept_param = Parameter("intercept", 0, 3, 0.5)
        pc = ParameterCollection("test", slope_param, intercept_param)
        
        results = surrogate_model.run(state, pc)
        
        # Verify results
        self.assertIn("y", results.results_data)
        self.assertEqual(len(results.results_data["y"]), 100)
        
        # Check accuracy - for linear model, surrogate should be very accurate
        y_values = results.results_data["y"]
        time_values = results.results_data["time"]
        expected = 2.5 * time_values + 0.5
        
        # Should be close since linear functions are easy to approximate
        rmse = np.sqrt(np.mean((y_values - expected)**2))
        self.assertLess(rmse, 0.5, f"RMSE = {rmse}, too large for linear function")
    
    def test_adaptive_surrogate_with_state_and_constants(self):
        """Test AdaptiveSurrogate in model with state params and constants."""
        # Build adaptive surrogate
        a = Parameter("a", 0, 5)
        b = Parameter("b", 0, 5)
        c = Parameter("c", 0, 5)
        
        model_func = mc.PythonModel(quadratic_model)
        
        study = mc.VoronoiAdaptiveSurrogateStudy(a, b, c)
        study.set_independent_variable("x", np.linspace(0, 5, 50))
        study.set_target_field_name("y")
        study.set_number_of_test_samples(10)
        study.set_initial_sample_count(15)
        study.set_max_iterations(3)
        study.set_test_group_random_seed(123)
        
        study.add_evaluation_set(model_func)
        study.launch()
        
        # Wrap in model
        surrogate_model = MatCalSurrogateModel(study.surrogate)
        surrogate_model.add_constants(unused_const=10.0)
        
        state = State("test_state", state_var=5.0)
        
        a_param = Parameter("a", 0, 5, 2.0)
        b_param = Parameter("b", 0, 5, 1.0)
        c_param = Parameter("c", 0, 5, 0.5)
        pc = ParameterCollection("test", a_param, b_param, c_param)
        
        # Should run without error
        results = surrogate_model.run(state, pc)
        
        self.assertIn("y", results.results_data)


if __name__ == '__main__':
    unittest.main()
