
import os
import numpy as np

from matcal.core.data_importer import FileData
from matcal.core.data import DataCollection
from matcal.core.evaluation_set import StudyEvaluationSet
from matcal.core.objective import ObjectiveSet, ObjectiveCollection, CurveBasedInterpolatedObjective
from matcal.core.parameters import Parameter, ParameterCollection
from matcal.core.state import State, StateCollection
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest

from matcal.sierra.material import Material
from matcal.sierra.models import UniaxialLoadingMaterialPointModel
from matcal.sierra.tests.platform_options import MatCalTestPlatformOptionsFunctionIdentifier
from matcal.sierra.tests.utilities import write_linear_elastic_material_file


SET_PLATFORM_OPTIONS = MatCalTestPlatformOptionsFunctionIdentifier.identify()


class TestStudyEvaluationSet(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)
        self.param = Parameter("a", 0, 10)
        self.parameter_collection = ParameterCollection("Test", self.param)

        self.state_1 = State("state 1", x=1, y="hello")
        self.state_2 = State("state 2")
        self.state_3 = State("state 3")

        self.state_collection = StateCollection('states', self.state_1, self.state_2, self.state_3)

        self.data_mat1 = np.array([[0, 1, 1.5, 3, 4], [1, 3, 4, 7, 9]]).T
        np.savetxt("data1.csv", self.data_mat1, header="displacement, load", comments="", delimiter=",")

        self.data_mat2 = np.array([[0, 1, 1.5, 3], [0, 2, 3, 6]]).T
        np.savetxt("data2.csv", self.data_mat2, header="displacement, load", comments="", delimiter=",")

        self.data_stress_strain = np.array([[0, 0.01, 0.05, 0.1, 0.2, 0.5, 0.6 ,0.7], [0, 100,105, 110, 120, 130, 125, 90]]).T
        np.savetxt("data_stress_strain.csv", self.data_mat2, header="engineering_strain, engineering_stress", comments="", delimiter=",")

        self.data2_stress_strain = np.array([[0, 0.01, 0.05, 0.1, 0.2, 0.5, 0.6 ,0.7], [0, 100,103, 113, 119, 127, 120, 80]]).T
        np.savetxt("data2_stress_strain.csv", self.data_mat2, header="engineering_strain, engineering_stress", comments="", delimiter=",")


        self.data1 = FileData("data1.csv")
        self.data1.set_state(self.state_1)
        self.data2 = FileData("data1.csv")
        self.data2.set_state(self.state_2)
        self.data3 = FileData("data2.csv")
        self.data3.set_state(self.state_3)

        self.data_stress_strain = FileData("data_stress_strain.csv")
        self.data_stress_strain.set_state(self.state_1)
        self.data2_stress_strain = FileData("data2_stress_strain.csv")
        self.data2_stress_strain.set_state(self.state_2)
        self.data3_stress_strain = FileData("data2_stress_strain.csv")
        self.data3_stress_strain.set_state(self.state_3)

        self.data_collection_stress_strain = DataCollection("Test", self.data_stress_strain, self.data2_stress_strain, self.data3_stress_strain)

        self.obj_stress_strain = CurveBasedInterpolatedObjective("engineering_strain", "engineering_stress")
        self.obj_stress_strain_col = ObjectiveCollection("test3", self.obj_stress_strain)

        self.data_collection = DataCollection("Test", self.data1, self.data2)
        self.data_collection2 = DataCollection("CondTest", self.data1, self.data3)

        self.obj = CurveBasedInterpolatedObjective("displacement", "load")
        self.objective_collection = ObjectiveCollection("test1", self.obj)

        self.obj2 = CurveBasedInterpolatedObjective("displacement", "load")
        self.objective_collection2 = ObjectiveCollection("test2", self.obj2)

        self.obj3 = CurveBasedInterpolatedObjective("displacement", "load")
        self.objective_collection3 = ObjectiveCollection("test3", self.obj3)

        self.results_mat = np.array([[0, 1, 2, 3, 4, 5], [0, -1, -2, -3, -4, -5]]).T
        np.savetxt("results_file.csv", self.results_mat, header="displacement, load", comments="", delimiter=",")

        self.objective_set_stress_strain = ObjectiveSet(self.obj_stress_strain_col, self.data_collection_stress_strain,
                                           self.data_collection_stress_strain.states)

        self.objective_set = ObjectiveSet(self.objective_collection, self.data_collection,
                                          self.data_collection.states)
        self.objective_set2 = ObjectiveSet(self.objective_collection2, self.data_collection2,
                                           self.data_collection2.states)

        self.objective_set3 = ObjectiveSet(self.objective_collection3, self.data_collection,
                                           self.data_collection.states)

    def test_prepare_model_and_simulators(self):
        material_file = write_linear_elastic_material_file()
        material_example = Material("matcal_test", material_file, "linear_elastic")
        model = UniaxialLoadingMaterialPointModel(material_example)
        model.add_boundary_condition_data(self.data_collection_stress_strain)
        model.add_constants(nu=0.3, elastic_modulus=10e6, density=0.0025)
        if not "SLURM_JOB_ID" in os.environ.keys() or not "SLURM_JOBID" in os.environ.keys():
            SET_PLATFORM_OPTIONS(model)

        eval_set = StudyEvaluationSet(model, self.objective_set)

        parameters = {}

        eval_set.prepare_model_and_simulators()
        for state, sim in eval_set.simulators.items():
            sim_results = sim.run_check_syntax(parameters)
            self.assertRegex(sim_results.stdout, "Executing: aprepro")
            self.assertRegex(sim_results.stdout, "Executing sierra")