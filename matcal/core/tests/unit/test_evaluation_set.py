import os
import numpy as np

from matcal.core.tests.MatcalUnitTest import MatcalUnitTest
from matcal.core.evaluation_set import StudyEvaluationSet
from matcal.core.objective import ObjectiveSet, ObjectiveCollection, CurveBasedInterpolatedObjective
from matcal.core.constants import  STATE_PARAMETER_FILE
from matcal.core.parameters import Parameter, ParameterCollection
from matcal.core.state import State, StateCollection
from matcal.core.utilities import matcal_name_format
from matcal.core.computing_platforms import RemoteComputingPlatform
from matcal.core.data_importer import FileData
from matcal.core.data import DataCollection
from matcal.core.reporter import MatCalParameterReporterIdentifier
from matcal.core.tests.utilities_for_tests import MockExecutableModel

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
        self.model = MockExecutableModel("results_file.csv")

        self.objective_set_stress_strain = ObjectiveSet(self.obj_stress_strain_col, self.data_collection_stress_strain,
                                           self.data_collection_stress_strain.states)

        self.objective_set = ObjectiveSet(self.objective_collection, self.data_collection,
                                          self.data_collection.states)
        self.objective_set2 = ObjectiveSet(self.objective_collection2, self.data_collection2,
                                           self.data_collection2.states)

        self.objective_set3 = ObjectiveSet(self.objective_collection3, self.data_collection,
                                           self.data_collection.states)

    def test_init(self):
        StudyEvaluationSet(self.model, self.objective_set)

    def test_make_directories_and_files(self):
        MatCalParameterReporterIdentifier._registry = {}

        eval_set = StudyEvaluationSet(self.model, self.objective_set)
        eval_set.prepare_model_and_simulators()

        for state_name in eval_set.states.keys():
            dir = os.path.join(matcal_name_format(eval_set.model.name), matcal_name_format(state_name))
            self.assertTrue(os.path.exists(dir))
            self.assertTrue(os.path.exists(os.path.join(dir, STATE_PARAMETER_FILE)))

        dir = os.path.join(matcal_name_format(eval_set.model.name), matcal_name_format(self.state_1.name))
        gold_lines = ["x=1\n", "y=hello\n"]
        with open(os.path.join(dir, STATE_PARAMETER_FILE), "r") as f:
            for goldline, line in zip(gold_lines, f.readlines()):
                self.assertEqual(goldline, line)

    def test_make_directories_and_files_in_dir(self):
        MatCalParameterReporterIdentifier._registry = {}

        eval_set = StudyEvaluationSet(self.model, self.objective_set)
        eval_set.prepare_model_and_simulators("test_dir")

        for state_name in eval_set.states.keys():
            dir = os.path.join("test_dir", matcal_name_format(eval_set.model.name), matcal_name_format(state_name))
            self.assertTrue(os.path.exists(dir))
            self.assertTrue(os.path.exists(os.path.join(dir, STATE_PARAMETER_FILE)))

        dir = os.path.join("test_dir", matcal_name_format(eval_set.model.name), matcal_name_format(self.state_1.name))
        gold_lines = ["x=1\n", "y=hello\n"]
        with open(os.path.join(dir, STATE_PARAMETER_FILE), "r") as f:
            for goldline, line in zip(gold_lines, f.readlines()):
                self.assertEqual(goldline, line)

    def test_active_states_multiple_objective_sets(self):
        eval_set = StudyEvaluationSet(self.model, self.objective_set)
        eval_set.add_objective_set(self.objective_set2)
        eval_set.add_objective_set(self.objective_set3)

        self.assertTrue(self.state_1 in list(eval_set.states.values()))
        self.assertTrue(self.state_2 in list(eval_set.states.values()))
        self.assertTrue(self.state_3 in list(eval_set.states.values()))

    def test_make_directories_and_files_multiple_objective_sets(self):
        eval_set = StudyEvaluationSet(self.model, self.objective_set)
        eval_set.add_objective_set(self.objective_set2)
        eval_set.add_objective_set(self.objective_set3)

        eval_set.prepare_model_and_simulators()

        for state_name in eval_set.states.keys():
            dir = os.path.join(matcal_name_format(eval_set.model.name), matcal_name_format(state_name))
            self.assertTrue(os.path.exists(dir))
            self.assertTrue(os.path.exists(os.path.join(dir, STATE_PARAMETER_FILE)))

    def test_get_max_cores_required_one_state(self):
        self.model.set_number_of_cores(2)
        self.state_collection.pop('state_1')
        obj_set = ObjectiveSet(self.objective_collection, self.data_collection, StateCollection('test', self.data_collection.states["state_1"] ))
        eval_set = StudyEvaluationSet(self.model, obj_set)

        self.assertEqual(eval_set.get_cores_required(), 2)

    def test_get_max_cores_required_two_states(self):
        self.model.set_number_of_cores(2)

        eval_set = StudyEvaluationSet(self.model, self.objective_set)

        self.assertEqual(eval_set.get_cores_required(), 4)

    
    def test_get_max_cores_required_two_states_remote_compute(self):
        self.model.set_number_of_cores(2)
        self.model._set_computing_platform(RemoteComputingPlatform(None, 
                                                                   None, 
                                                                   None,
                                                                   None))

        eval_set = StudyEvaluationSet(self.model, self.objective_set)

        self.assertEqual(eval_set.get_cores_required(), 2)

    def test_residual_vector_length_and_number_of_objectives(self):
        eval_set = StudyEvaluationSet(self.model, self.objective_set)
        eval_set.add_objective_set(self.objective_set2)
        eval_set.add_objective_set(self.objective_set3)

        self.assertEqual(eval_set.residual_vector_length, 29)
        self.assertEqual(eval_set.number_of_objectives, 3)

    def test_add_objective_set_twice_and_error(self):
        eval_set = StudyEvaluationSet(self.model, self.objective_set)
        with self.assertRaises(StudyEvaluationSet.InputError):
            eval_set.add_objective_set(self.objective_set)

    def test_get_all_objective_names(self):
        eval_set = StudyEvaluationSet(self.model, self.objective_set)
        eval_set.add_objective_set(self.objective_set2)
        eval_set.add_objective_set(self.objective_set3)

        gold_names = [self.obj.name, self.obj2.name, self.obj3.name]

        objective_names = eval_set.get_objective_names()
        self.assertEqual(len(objective_names), 3)
        for name, gold_name in zip(objective_names, gold_names):
            self.assertTrue(isinstance(name, str))
            self.assertEqual(name, gold_name)
        list_set = list(set(objective_names))
        self.assertTrue(len(list_set)==3)

    def test_get_all_objective_names_user_named(self):
        self.obj.set_name("user name 1")
        self.obj2.set_name("user name 2")
        self.obj3.set_name("user name 3")
        
        eval_set = StudyEvaluationSet(self.model, self.objective_set)
        eval_set.add_objective_set(self.objective_set2)
        eval_set.add_objective_set(self.objective_set3)

        gold_names = ["user name 1", "user name 2", "user name 3"]

        objective_names = eval_set.get_objective_names()
        self.assertEqual(len(objective_names), 3)
        for name, gold_name in zip(objective_names, gold_names):
            self.assertTrue(isinstance(name, str))
            self.assertEqual(name, gold_name)
        list_set = list(set(objective_names))
        self.assertTrue(len(list_set)==3)

    def test_get_all_objective_names_user_named_repeated(self):
        self.obj.set_name("user name 1")
        self.obj2.set_name("user name 1")
        
        eval_set = StudyEvaluationSet(self.model, self.objective_set)
        with self.assertRaises(StudyEvaluationSet.InputError):
            eval_set.add_objective_set(self.objective_set2)
        
    def test_zero_objectives(self):
        empty_objective_collection = ObjectiveCollection("my collection")
        empty_obj_set = ObjectiveSet(empty_objective_collection, self.data_collection,
                     self.data_collection.states)
        with self.assertRaises(StudyEvaluationSet.ZeroObjectiveError):
            StudyEvaluationSet(self.model, empty_obj_set)

    def test_repeated_objective(self):
        eval_set = StudyEvaluationSet(self.model, self.objective_set)
        self.obj.set_name("test obj name")
        obj_col = ObjectiveCollection("test", self.obj)
        objective_set = ObjectiveSet(obj_col, self.data_collection,
                                          self.data_collection.states)
        with self.assertRaises(StudyEvaluationSet.InputError):
            eval_set.add_objective_set(objective_set)

    def test_get_total_cores(self):
        self.model.set_number_of_cores(5)

        eval_set = StudyEvaluationSet(self.model, self.objective_set)
        eval_set.add_objective_set(self.objective_set2)        
        eval_set.add_objective_set(self.objective_set3)

        self.assertEqual(eval_set.get_cores_required(), 15)

