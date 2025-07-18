import numpy as np
import glob

from matcal.core.data import convert_dictionary_to_data
from matcal.core.plotting import make_standard_plots
from matcal.core.models import PythonModel
from matcal.core.objective import CurveBasedInterpolatedObjective, Objective
from matcal.core.parameters import Parameter
from matcal.core.qoi_extractor import MaxExtractor
from matcal.core.parameter_studies import ParameterStudy
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest


class TestMakeStandardPlots(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)
    
    def test_different_sim_exp_qois_two_objective(self):
        x = np.linspace(0,1,10)
        data_dict = {"x":x, "y":x**2+1, "z":2.1**x + .9}
        data = convert_dictionary_to_data(data_dict)
        def model(a,b):
            import numpy as np
            x = np.linspace(0,1,10)

            y = x**a+b
            z = a**x+b
            return {"x":x, "y":y, "z":z}

        pymodel = PythonModel(model)

        param_a = Parameter("a",0, 4, 1)
        param_b = Parameter("b", -4, 4, 0)

        obj = CurveBasedInterpolatedObjective("x", "y")
        obj2 = Objective("y")
        obj2.set_qoi_extractors(MaxExtractor("z"))

        study = ParameterStudy(param_a, param_b)
        study.add_evaluation_set(pymodel, obj, data)
        study.add_evaluation_set(pymodel, obj2, data)
        study.set_core_limit(6)
        study.add_parameter_evaluation(a=2,b=-1)
        study.add_parameter_evaluation(a=3,b=-2)
        study.launch()
        make_standard_plots("x", plot_model_objectives=True, show=False)
        glob_search = "user_plots/*.pdf"
        plot_files = glob.glob(glob_search)
        self.assertEqual(len(plot_files), 7)
        
    def test_different_sim_exp_qois_one_obj(self):
        x = np.linspace(0,1,10)
        data_dict = {"x":x, "y":x**2+1, "z":2.1**x + .9}
        data = convert_dictionary_to_data(data_dict)
        def model(a,b):
            import numpy as np
            x = np.linspace(0,1,10)

            y = x**a+b
            z = a**x+b
            return {"x":x, "y":y, "z":z}

        pymodel = PythonModel(model)

        param_a = Parameter("a",0, 4, 1)
        param_b = Parameter("b", -4, 4, 0)

        obj = CurveBasedInterpolatedObjective("x", "y")

        study = ParameterStudy(param_a, param_b)
        study.add_evaluation_set(pymodel, obj, data)
        study.set_core_limit(6)
        study.add_parameter_evaluation(a=2,b=-1)
        study.add_parameter_evaluation(a=3,b=-2)
        study.launch()
        make_standard_plots("x", show=False)
        glob_search = "user_plots/*.pdf"
        plot_files = glob.glob(glob_search)
        self.assertEqual(len(plot_files), 3)
