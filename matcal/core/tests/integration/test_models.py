from abc import ABC, abstractmethod
import numpy as np

from matcal.core.external_executable import MatCalExecutableEnvironmentSetupFunctionIdentifier
from matcal.core.file_modifications import use_jinja_preprocessor
from matcal.core.parameters import Parameter, ParameterCollection
from matcal.core.models import UserExecutableModel
from matcal.core.state import State
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest


python_model_string = """
m = {{ m }}
b = {{ b }}
exponent = {{ exponent }}

import numpy as np
x = np.linspace(0,2,100)
y = m*x**exponent+b
np.savetxt("model_res.csv", np.array([x,y]).T, header = "x, y", comments="", delimiter=',')
"""


class TestUserExecutbleModels(MatcalUnitTest):

  def setUp(self) -> None:
    super().setUp(__file__)
    MatCalExecutableEnvironmentSetupFunctionIdentifier._registry={}
    
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
