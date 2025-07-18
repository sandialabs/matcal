import glob
import os
import unittest 
from matcal.core.serializer_wrapper import matcal_load
import numpy as np
from abc import abstractmethod
from collections import OrderedDict

from matcal.core.constants import ( MATCAL_WORKDIR_STR, MATCAL_TEMPLATE_DIRECTORY, 
                                   FINAL_RESULTS_FILENAME)
from matcal.core.data import (DataCollection, convert_dictionary_to_data, 
                              RangeDataConditioner, MaxAbsDataConditioner)
from matcal.core.evaluation_set import StudyEvaluationSet
from matcal.core.models import   PythonModel
from matcal.core.objective import (CurveBasedInterpolatedObjective, Objective, ObjectiveCollection,
                                   SimulationResultsSynchronizer)
from matcal.core.parameter_studies import (ParameterStudy)
from matcal.core.parameters import ParameterCollection, UserDefinedParameterPreprocessor, Parameter
from matcal.core.plotting import _NullPlotter, StandardAutoPlotter

from matcal.core.pruner import (DirectoryPrunerKeepBestXPercent, 
                DirectoryPrunerKeepLastTwentyPercent, DirectoryPrunerKeepLast)
from matcal.core.qoi_extractor import MaxExtractor
from matcal.core.state import State, StateCollection, SolitaryState
from matcal.core.study_base import (StudyBase, StudyResults, _get_obj_name_if_not_string) 
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest
from matcal.core.tests.unit.test_pruner import _make_matcal_workdirs_with_objectives
from matcal.core.tests.utilities_for_tests import (_generate_singe_model_single_state_mock_eval_hist,
                                    _generate_singe_model_single_state_mock_eval_hist_given_params)


def model_func(coeff, power, offset):
    import numpy as np

    x = np.linspace(0, 10, 100)
    y = coeff*x**power+offset
    
    return {"x":x, "y": y}


def simple_func(a):
    return {"x":[0, 1], "y": [0,1]}


def do_nothing_load_disp_function(a, **params):
    disp = np.linspace(0, 5, 6)
    load = disp*-1
    return {"displacement":disp, "load":load}


def linear_load_disp_function(a, **params):
    disp = np.linspace(0.1, 5, 6)
    load = disp*-1*a+np.random.uniform(0,0.1,6)
    return {"displacement":disp, "load":load}


def param_preprocessor_func(params):
    for key in params:
        params[key] += 1
    params["new"] = 0
    return params


class StudyBaseUnitTests(object):
    class CommonSetup(MatcalUnitTest):
        @property
        @abstractmethod
        def _study_class(self):
            """"""

        def setUp(self, filename):
            super().setUp(filename)

            self.param = Parameter("a", 0, 10, 5+np.random.uniform()*1e-1)
            self.uncert_param = Parameter("a", 0, 10, 5+np.random.uniform()*1e-1, 'uniform_uncertain')

            self.parameter_collection = ParameterCollection("Test", self.param)
            self.uncertain_parameter_collection = ParameterCollection("TestUNCERT", self.uncert_param)

            self.state1 = State('state1')
            self.state2 = State('state2')
            self.state3 = State('state3')

            self.gold_results = convert_dictionary_to_data(do_nothing_load_disp_function(self.param.get_current_value()))
            self.gold_results.set_state(self.state1)

            self.gold_results2 = convert_dictionary_to_data(do_nothing_load_disp_function(self.param.get_current_value()))
            self.gold_results2.set_state(self.state2)

            self.gold_results3 = convert_dictionary_to_data(do_nothing_load_disp_function(self.param.get_current_value()))
            self.gold_results3.set_state(self.state3)

            self.data_collection = DataCollection("Test", self.gold_results, self.gold_results2)
            self.data_collection2 = DataCollection("Test", self.gold_results3)

            self.mock_model = PythonModel(do_nothing_load_disp_function)
            self.mock_model.set_name("TestPython")
            self.mock_model2 = PythonModel(do_nothing_load_disp_function)

            self.independent_variable = "displacement"
            self.dependent_variable = "load"
            self.objective = CurveBasedInterpolatedObjective(self.independent_variable, self.dependent_variable)
            self.objective.set_name("TestObj")
            self.max_disp_objective = Objective(self.dependent_variable)
            self.max_disp_objective.set_qoi_extractors(MaxExtractor(self.independent_variable))
            self.max_load_objective = Objective(self.dependent_variable)
            self.max_load_objective.set_qoi_extractors(MaxExtractor(self.dependent_variable))
            
            self.objective_collection = ObjectiveCollection('one objective', self.objective)
            self.objective_collection2 = ObjectiveCollection('two objective',self.max_disp_objective)
            self.objective_collection3 = ObjectiveCollection('three objective',self.max_load_objective)

        def _set_study_specific_options(self, study):
            pass


    class CommonTests(CommonSetup):

        def test_launch_with_no_evaluation_set(self):
            study = self._study_class(self.parameter_collection)
            with self.assertRaises(StudyBase.StudyError):
                results = study.launch()

        def test_launching_a_study_twice_raises_error(self):
            study = self._study_class(self.parameter_collection)
            study.add_evaluation_set(self.mock_model, self.objective, self.data_collection)
            self._set_study_specific_options(study)
            study.launch()

            with self.assertRaises(StudyBase.RepeatLaunchError):
                study.launch()

        def test_plot_progress(self):
            study = self._study_class(self.parameter_collection)
            self.assertIsInstance(study._plotter, _NullPlotter)
            study.plot_progress()
            self.assertIsInstance(study._plotter, StandardAutoPlotter)

        def test_results_filename(self):
            study = self._study_class(self.parameter_collection)
            study._set_final_results_name()
            res_filename = study.final_results_filename
            self.assertEqual(f"{FINAL_RESULTS_FILENAME}.joblib", res_filename)   
            study.set_working_directory("test")
            study._set_final_results_name()
            res_filename = study.final_results_filename
            self.assertEqual(os.path.abspath(os.path.join("test",f"{FINAL_RESULTS_FILENAME}.joblib")),
                              res_filename)   
            
        def test_get_input_deck_from_args(self):
            study = self._study_class(self.parameter_collection)
            input_deck_string = 'my_input_deck.i'
            args = ['a', 1, 'd', '-1', '-i', input_deck_string, '--debug']
            input = study._get_input_file_path(args)
            self.assertEqual(input, os.path.abspath(input_deck_string))
            args = ['a', 1, '--input', input_deck_string, '--debug', 'd', '-1']
            input = study._get_input_file_path(args)
            self.assertEqual(input, os.path.abspath(input_deck_string))
            input = study._get_input_file_path([1])
            self.assertIsNone(input)
            input = study._get_input_file_path(["html"])
            self.assertIsNone(input)
            input = study._get_input_file_path(["unittest"])
            self.assertIsNone(input)

        def test_set_use_threads(self):
            study = self._study_class(self.parameter_collection)
            study.add_evaluation_set(self.mock_model, self.objective, self.gold_results)
            self._set_study_specific_options(study)

            study.set_use_threads()
            param_batch_evaluator = study._initialize_study_and_batch_evaluator()    
            study_use_threads = study._use_threads
            study_always_use_threads = study._always_use_threads

            self.assertEqual(study_use_threads, True)
            self.assertEqual(param_batch_evaluator._use_threads, True)

            self.assertEqual(study_always_use_threads, False)
            self.assertEqual(param_batch_evaluator._always_use_threads, False)

        def test_set_always_use_threads(self):
            study = self._study_class(self.parameter_collection)
            study.add_evaluation_set(self.mock_model, self.objective, self.gold_results)
            self._set_study_specific_options(study)
            with self.assertRaises(StudyBase.StudyTypeError):
                study.set_use_threads("not valid")

            study.set_use_threads(always_use_threads=True)
            param_batch_evaluator = study._initialize_study_and_batch_evaluator()    
            study_use_threads = study._use_threads
            study_always_use_threads = study._always_use_threads

            self.assertEqual(study_use_threads, True)
            self.assertEqual(study_always_use_threads, True)

            self.assertEqual(param_batch_evaluator._use_threads, True)
            self.assertEqual(param_batch_evaluator._always_use_threads, True)

        def test_pass_list_of_parameters(self):
            ps = []
            p_names = ['a', 'b', 'c']
            for ni, name in enumerate(p_names):
                ps.append(Parameter(name, 0, ni+1))
            study = self._study_class(*ps)
            study_names = study._parameter_collection.keys()
            for name in p_names:
                self.assertIn(name, study_names)

        def test_core_limit_incorrect_type(self):
            study = self._study_class(self.parameter_collection)
            with self.assertRaises(StudyBase.StudyTypeError):
                study.set_core_limit(1.0)
            with self.assertRaises(StudyBase.StudyTypeError):
                study.set_core_limit("a")

        def test_set_core_limit(self):
            study = self._study_class(self.parameter_collection)
            study.set_core_limit(10)
            self.assertEqual(study._total_cores_available, 10)
            with self.assertRaises(StudyBase.StudyInputError):
                study.set_core_limit(501)   
            study.set_core_limit(501, override_max_limit=True)
            self.assertEqual(study._total_cores_available, 501)

        def test_bad_init(self):
            with self.assertRaises(TypeError):
                study = self._study_class(1)
            with self.assertRaises(TypeError):
                study = self._study_class("a")
            with self.assertRaises(TypeError):
                study = self._study_class(self.data_collection)

        def test_single_param_init(self):
            param = Parameter("a", 0,1)
            study = self._study_class(param)
            self.assertEqual(len(study._parameter_collection.items()), 1)
            self.assertEqual(list(study._parameter_collection.keys()), ["a"])
            
        def test_add_evaluation_set_invalid_states(self):
            study = self._study_class(self.parameter_collection)

            states = StateCollection('states', State('state3'))

            with self.assertRaises(StudyBase.StudyError):
                study.add_evaluation_set(self.mock_model, 
                                         self.objective, self.gold_results, states=states)

        def test_add_evaluation_set_bad_types(self):
            study = self._study_class(self.parameter_collection)
            with self.assertRaises(study.StudyTypeError):
                study.add_evaluation_set(self.mock_model, 
                                         self.objective, "not data")
            with self.assertRaises(study.StudyTypeError):
                study.add_evaluation_set("not model", 
                                         self.objective, self.gold_results)
            with self.assertRaises(study.StudyTypeError):
                study.add_evaluation_set(self.mock_model, 
                                         "not objective", self.gold_results)
            with self.assertRaises(study.StudyTypeError):
                study.add_evaluation_set(self.mock_model, 
                                         self.objective, self.gold_results,
                                         "not states")

        def test_add_evaluation_set_no_data(self):
            study = self._study_class(self.parameter_collection)
            with self.assertRaises(ValueError):
                study.add_evaluation_set(self.mock_model, 
                                        self.objective)
            res_sync = SimulationResultsSynchronizer("a", [0,1], "b")

            study.add_evaluation_set(self.mock_model, 
                                     res_sync)
            
            with self.assertRaises(ValueError):
                study.add_evaluation_set(self.mock_model, 
                                     ObjectiveCollection("objs", 
                                                         res_sync, 
                                                         self.objective))
            with self.assertRaises(ValueError):
                study.add_evaluation_set(self.mock_model, 
                                     ObjectiveCollection("objs", 
                                                        self.objective,
                                                         res_sync))
            res_sync2 = SimulationResultsSynchronizer("a", [0,1], "c")
            with self.assertRaises(ValueError):
                study.add_evaluation_set(self.mock_model, 
                                     ObjectiveCollection("objs", 
                                                        res_sync,
                                                         res_sync2))

        def test_add_evaluation_set_sim_res_synch_data_gen(self):
            res_sync = SimulationResultsSynchronizer("a", [0,1], "b")
            study = self._study_class(self.parameter_collection)

            study.add_evaluation_set(self.mock_model, 
                                     res_sync)

            eval_set = study._evaluation_sets[self.mock_model]
            obj_set = eval_set._objective_sets[0]
            data = obj_set._experiment_data_collection
            self.assertTrue(len(data.states) == 1)
            self.assertTrue(isinstance(list(data.states.values())[0], SolitaryState))
            self.assertTrue(len(data["matcal_default_state"])==1)
            self.assertTrue("a" in data["matcal_default_state"][0].field_names)
            self.assertTrue("b" in data["matcal_default_state"][0].field_names)
            self.assertTrue(data.name == "Sim results synchronizer generated 2")

        def test_add_evaluation_set_sim_res_synch_data_gen_with_states(self):
            res_sync = SimulationResultsSynchronizer("a", [0,1], "b")
            study = self._study_class(self.parameter_collection)

            state_1 = State("one")
            state_2 = State("two")
            sc = StateCollection("states", state_1, state_2)
            study.add_evaluation_set(self.mock_model, 
                                     res_sync, states=sc)

            eval_set = study._evaluation_sets[self.mock_model]
            obj_set = eval_set._objective_sets[0]
            data = obj_set._experiment_data_collection
            self.assertTrue(len(data.states) == 2)
            self.assertTrue("one" in data.states)
            self.assertTrue("two" in data.states)
            self.assertTrue(state_1 in data.states.values())
            self.assertTrue(state_2 in data.states.values())

        def test_add_repeated_eval_set(self):
            study = self._study_class(self.parameter_collection)
            study.add_evaluation_set(self.mock_model, 
                                     self.objective_collection, 
                                     self.data_collection)
            with self.assertRaises(StudyBase.StudyError):
                study.add_evaluation_set(self.mock_model, 
                                         self.objective_collection, 
                                         self.data_collection)

        def test_add_repeated_singleton_objective(self):
            study = self._study_class(self.parameter_collection)
            study.add_evaluation_set(self.mock_model, 
                                     self.objective, self.data_collection)
            study.add_evaluation_set(self.mock_model, 
                                     self.max_disp_objective, self.data_collection)
        
        def test_add_repeated_objective_same(self):
            study = self._study_class(self.parameter_collection)
            study.add_evaluation_set(self.mock_model, 
                                     self.objective, 
                                     self.data_collection, )
            with self.assertRaises(StudyEvaluationSet.InputError):
                study.add_evaluation_set(self.mock_model, 
                                        self.objective, self.data_collection)    

        def test_add_repeated_objective_same_names(self):
            study = self._study_class(self.parameter_collection)
            self.objective.set_name("same")
            study.add_evaluation_set(self.mock_model, 
                                     self.objective, self.data_collection)
            new_obj = Objective(self.dependent_variable)
            new_obj.set_name("same")
            with self.assertRaises(StudyEvaluationSet.InputError):
                study.add_evaluation_set(self.mock_model, 
                                         new_obj, self.data_collection)    

        def test_add_empty_collections(self):
            study = self._study_class(self.parameter_collection)
            empty_dc = DataCollection("empty")
            empty_obj_c = ObjectiveCollection("empty")
            empty_sc = StateCollection("empty")

            with self.assertRaises(StudyBase.StudyError):
                study.add_evaluation_set(self.mock_model, self.objective_collection, 
                                         empty_dc)
            with self.assertRaises(StudyBase.StudyError):
                study.add_evaluation_set(self.mock_model, empty_obj_c, 
                                         self.data_collection)
            with self.assertRaises(StudyBase.StudyError):
                study.add_evaluation_set(self.mock_model,
                                         self.objective_collection, self.data_collection,
                                         empty_sc)

        def test_add_parameter_preprocessor(self):
            param_preprocessor = UserDefinedParameterPreprocessor(param_preprocessor_func)
            study = self._study_class(self.parameter_collection)
            study.add_parameter_preprocessor(param_preprocessor)

            with self.assertRaises(StudyBase.StudyTypeError):
                study.add_parameter_preprocessor("not valid preprocessor")

            pc_dict = self.parameter_collection.get_current_value_dict()
            processed_params = study._preprocess_parameters(pc_dict)
            self.assertTrue("new" in processed_params)
            self.assertEqual(processed_params["new"], 0)
            self.assertEqual(processed_params["a"]-1, pc_dict["a"])

        def test_default_assessor(self):
            study = self._study_class(self.parameter_collection)
            dirs = _make_matcal_workdirs_with_objectives(self.build_dir, 25)

        def test_custom_assessor(self):
            """
            makes some directories with objective files and 
            tests functionality of a couple different assessors
            """
            study = self._study_class(self.parameter_collection)
            dirs = _make_matcal_workdirs_with_objectives(self.build_dir, 25)
            #dirs.sort()

            self.assertEqual(study._assessor.assess(), []) #starts as KeepAll

            study.set_cleanup_mode(DirectoryPrunerKeepLastTwentyPercent())
            test_list = study._assessor.assess()
            
            self.assertEqual(test_list.sort(), dirs[1:-5].sort())

            study.set_cleanup_mode(DirectoryPrunerKeepBestXPercent(40))
            self.assertEqual(study._assessor.assess().sort(), dirs[10:].sort())

            study.set_cleanup_mode(DirectoryPrunerKeepLast())
            self.assertEqual(study._assessor.assess().sort(), dirs[1:-1].sort())

            study._purge_unneeded_matcal_information()
            workdirs = glob.glob(MATCAL_WORKDIR_STR+'.*')
            self.assertEqual([os.path.join(os.getcwd(), x) for x in workdirs].sort(), 
                              [dirs[0], dirs[-1]].sort())

        def test_add_different_model_with_same_name(self):
            study = self._study_class(self.parameter_collection)
            study.set_core_limit(10)
            self.mock_model.set_name("mock1")   
            study.add_evaluation_set(self.mock_model, self.objective, self.gold_results,)
            self.mock_model2.set_name("mock1")
            with self.assertRaises(StudyBase.StudyInputError):
                study.add_evaluation_set(self.mock_model2, self.objective, self.gold_results2,)
            
        def test_get_model_names(self):
            study = self._study_class(self.parameter_collection)
            study.set_core_limit(10)
            study.add_evaluation_set(self.mock_model, self.objective, self.gold_results)
            self.mock_model.set_name("mock1")
            self.mock_model2.set_name("mock2")
            study.add_evaluation_set(self.mock_model2, self.objective, self.gold_results2)

            goal_names = ["mock1", "mock2"]
            for goal_name, name in zip(goal_names, study._get_model_names()):
                self.assertEqual(goal_name, name)
            
        def test_get_model_by_name(self):
            study = self._study_class(self.parameter_collection)
            study.set_core_limit(10)
            study.add_evaluation_set(self.mock_model, self.objective, self.gold_results)
            self.mock_model.set_name("mock1")
            self.mock_model2.set_name("mock2")
            study.add_evaluation_set(self.mock_model2, self.objective, self.gold_results2)

            self.assertEqual(self.mock_model, study._get_model_by_name("mock1"))
            self.assertEqual(self.mock_model2, study._get_model_by_name("mock2"))
            self.assertEqual(None, study._get_model_by_name("invalid"))

        def test_add_two_eval_sets_one_model(self):
            study = self._study_class(self.parameter_collection)
            study.set_core_limit(10)
            study.add_evaluation_set(self.mock_model, self.objective, self.gold_results)
            study.add_evaluation_set(self.mock_model, self.max_load_objective, self.gold_results)
            study.add_evaluation_set(self.mock_model, self.max_disp_objective, self.gold_results2)
            self.assertEqual(len(study._evaluation_sets.keys()),1)

        def test_add_eval_sets_with_conditioner(self):
            study = self._study_class(self.parameter_collection)
            study.set_core_limit(10)
            study.add_evaluation_set(self.mock_model, self.objective, self.gold_results)
            eval_set = study._evaluation_sets[self.mock_model]
            obj_set = eval_set._objective_sets[-1]
            self.assertEqual(obj_set._conditioner_class, MaxAbsDataConditioner)   
            study = self._study_class(self.parameter_collection)
            study.set_core_limit(10)        
            study.add_evaluation_set(self.mock_model, self.objective, self.gold_results, 
                                     data_conditioner_class=RangeDataConditioner)
            eval_set = study._evaluation_sets[self.mock_model]
            obj_set = eval_set._objective_sets[-1]
            self.assertEqual(obj_set._conditioner_class, RangeDataConditioner)   

        def test_add_eval_sets_with_bad_conditioner(self):
            study = self._study_class(self.parameter_collection)
            study.set_core_limit(10)
            study.add_evaluation_set(self.mock_model, self.objective, self.gold_results)
            with self.assertRaises(TypeError):
                study.add_evaluation_set(self.mock_model, self.objective, self.gold_results,
                                     data_conditioner_class=1)
            with self.assertRaises(TypeError):
                study.add_evaluation_set(self.mock_model, self.objective, self.gold_results,
                                     data_conditioner_class=self._study_class)
            
        def test_study_set_working_directory_bad(self):
            study = self._study_class(self.parameter_collection)
            with self.assertRaises(study.StudyTypeError):
                study.set_working_directory(1)
            with self.assertRaises(study.StudyInputError):
                study.set_working_directory("./sub_dir/goal_dir")
            
        def test_study_set_working_directory(self):
            study = self._study_class(self.parameter_collection)
            study.set_working_directory("my_study_dir")
            self.assertEqual(os.path.abspath("my_study_dir"), study._working_directory)
            self.assertEqual(os.getcwd(), study._initial_directory)
            study._go_to_working_directory()
            self.assertEqual(os.getcwd(),study._working_directory)
            study._return_to_initial_directory()
            self.assertEqual(os.getcwd(), study._initial_directory)
            os.mkdir("sub_dir")
            study.set_working_directory("./sub_dir/goal_dir")
            study._go_to_working_directory()
            self.assertEqual(os.getcwd(),study._working_directory)

        def test_study_set_working_directory_initialize(self):
            model = PythonModel(model_func)
            data = convert_dictionary_to_data({"x":1})
            objective = Objective("x")
            study = self._study_class(self.parameter_collection)
            self._set_study_specific_options(study)
            study.set_working_directory("my_study_dir")
            study.add_evaluation_set(model, objective, data)
            self.assertEqual(os.path.abspath("my_study_dir"), study._working_directory)
            self.assertEqual(os.getcwd(), study._initial_directory)
            study._initialize_study_and_batch_evaluator()
            self.assertEqual(os.getcwd(), study._working_directory)
            study._return_to_initial_directory()
            self.assertEqual(os.getcwd(), study._initial_directory)

        def test_study_purge_working_directory_initialize(self):
            model = PythonModel(model_func)
            data = convert_dictionary_to_data({"x":1})
            objective = Objective("x")
            study = self._study_class(self.parameter_collection)
            self._set_study_specific_options(study)
            study.set_working_directory("my_study_dir")
            study.add_evaluation_set(model, objective, data)
            study._initialize_study_and_batch_evaluator()
            os.chdir(study._initial_directory)
            self.assertTrue(os.path.exists("my_study_dir"))
            os.chdir("my_study_dir")
            os.mkdir(MATCAL_WORKDIR_STR+".1")

            study._delete_old_study_files()
            self.assertTrue(not os.path.exists(MATCAL_TEMPLATE_DIRECTORY))
            self.assertTrue(not os.path.exists(MATCAL_WORKDIR_STR+".1"))

        def test_study_purge_working_directory_initialize_workdir_exists(self):
            model = PythonModel(model_func)
            data = convert_dictionary_to_data({"x":1})
            objective = Objective("x")
            study = self._study_class(self.parameter_collection)
            self._set_study_specific_options(study)
            study.set_working_directory("my_study_dir")
            study.add_evaluation_set(model, objective, data)
            os.mkdir("my_study_dir")
            with self.assertRaises(FileExistsError):
                study._initialize_study_and_batch_evaluator()


class StudyResultsBaseUnitTests(object):
    class CommonSetup(MatcalUnitTest):
        @property
        @abstractmethod
        def _random_init(self)->StudyResults:
            """"""

        def setUp(self, filename):
            super().setUp(filename)

    class CommonTests(CommonSetup):
        def test_return_outcome(self):
            rc = self._random_init()
            goal_outcome = {"Best:a":0, "Best:b":1}
            n_parameters = 2
            self.assertEqual(len(rc.outcome), n_parameters)
            self.assert_close_dicts_or_data(rc.outcome, goal_outcome)
            
        @unittest.skip("need to decide if we still want this in how we are packaging results") 
        def test_raise_error_if_outcome_does_not_have_parameters_in_name(self):
            rc = self._random_init()
            bad_outcome = {"best1":0, "best2":123}
            with self.assertRaises(KeyError):
                rc._set_outcome(bad_outcome)
                
        def test_setting_outcome_adds_nested_attributes(self):
            rc = self._random_init()
            twoD = np.array([[1, .5],[.2, 1]])
            outcome_dict = {'best:a': 1.43, 'best:b':5454,
                            'twoD:a:b':twoD[0,1], "twoD:a:a":twoD[0,0],
                            'twoD:b:a':twoD[1,0], "twoD:b:b":twoD[1,1]}
            rc._set_outcome(outcome_dict)
            self.assertEqual(rc.best.a,  outcome_dict['best:a'])
            self.assertEqual(rc.best.b, outcome_dict['best:b'])
            self.assertEqual(rc.twoD.a.a, twoD[0,0])
            self.assertEqual(rc.twoD.a.b, twoD[0,1])
            self.assertEqual(rc.twoD.b.a, twoD[1,0])
            self.assertEqual(rc.twoD.b.b, twoD[1,1])
            
        def test_calling_set_outcome_overwrites_old_values(self):
            rc = self._random_init()
            twoD = np.array([[1, .5],[.2, 1]])
            outcome_dict = {'best:a': 1.43, 'best:b':5454,
                            'twoD:a:b':twoD[0,1], "twoD:a:a":twoD[0,0],
                            'twoD:b:a':twoD[1,0], "twoD:b:b":twoD[1,1]}
            rc._set_outcome(outcome_dict)
            twoD = np.array([[10, 1.5],[4.2, 21]])
            outcome_dict = {'best:a': 22.4, 'best:b':-5454,
                            'twoD:a:b':twoD[0,1], "twoD:a:a":twoD[0,0],
                            'twoD:b:a':twoD[1,0], "twoD:b:b":twoD[1,1]}
            rc._set_outcome(outcome_dict)
            self.assertEqual(rc.best.a,  outcome_dict['best:a'])
            self.assertEqual(rc.best.b, outcome_dict['best:b'])
            self.assertEqual(rc.twoD.a.a, twoD[0,0])
            self.assertEqual(rc.twoD.a.b, twoD[0,1])
            self.assertEqual(rc.twoD.b.a, twoD[1,0])
            self.assertEqual(rc.twoD.b.b, twoD[1,1])
        
        def test_raise_error_if_outcome_is_not_dict(self):
            rc = self._random_init()
            with self.assertRaises(TypeError):
                rc._set_outcome([1, 2])
            with self.assertRaises(TypeError):
                rc._set_outcome("ab1x")
            with self.assertRaises(TypeError):
                rc._set_outcome(str)
                
        def test_raise_error_if_outcome_not_set_when_accessed(self):
            rc = StudyResults()
            with self.assertRaises(RuntimeError):
                rc.outcome
            
        def test_can_return_termination_status(self):
            rc = self._random_init()
            rc.exit_status
            
        def test_can_return_eval_sets(self):
            rc = self._random_init()
            self.assertTrue(len(rc.evaluation_sets) > 0)
        
        def test_return_some_objective_details(self):
            rc = self._random_init()
            eval_set_0 = rc.evaluation_sets[0]
            self.assertTrue(len(rc.objective_history)>0)

        def test_return_some_objective_details(self):
            rc = self._random_init()
            eval_set_0 = rc.evaluation_sets[0]
            self.assertTrue(len(rc.objective_history)>0)

        def test_return_best_evaluation_set_objective(self):
            rc = self._specified_init()
            eval_obj = rc.best_evaluation_set_objective("MockModel", "MockObj")
            self.assertEqual(eval_obj[0], 0)
            self.assertEqual(eval_obj[1], 5)

        def test_return_best_simulation_data(self):
            rc = self._specified_init()
            best_sim_data = rc.best_simulation_data("MockModel", "MockState")
            self.assertEqual(best_sim_data["time"][-1], 0)
            self.assertEqual(best_sim_data["c"][-1], 0)

        def test_return_get_experiment_qois(self):
            rc = self._specified_init()
            exp_qoi_list = rc.get_experiment_qois("MockModel", "MockObj", "MockState")
            self.assertEqual(len(exp_qoi_list), 1)
            self.assertEqual(exp_qoi_list[0]["time"], 0)
            self.assertEqual(exp_qoi_list[0]["c"], 0)
            
            exp_qoi = rc.get_experiment_qois("MockModel", "MockObj", "MockState", 0)
            self.assertEqual(exp_qoi_list[0], exp_qoi)
            eval_name = rc.get_eval_set_name("MockModel", "MockObj")
            self.assertEqual(rc.qoi_history[eval_name].experiment_qois["MockState"], exp_qoi_list)
            with self.assertRaises(IndexError):
                exp_qoi = rc.get_experiment_qois("MockModel", "MockObj", "MockState", 1)

        def test_return_get_experiment_data(self):
            rc = self._specified_init()
            exp_qoi_list = rc.get_experiment_data("MockModel", "MockObj", "MockState")
            self.assertEqual(len(exp_qoi_list), 1)
            self.assertEqual(exp_qoi_list[0]["time"], 0)
            self.assertEqual(exp_qoi_list[0]["c"], 0)
            eval_name = rc.get_eval_set_name("MockModel", "MockObj")
            self.assertEqual(rc.qoi_history[eval_name].experiment_data["MockState"], exp_qoi_list)

            exp_qoi = rc.get_experiment_data("MockModel", "MockObj", "MockState", 0)
            self.assertEqual(exp_qoi_list[0], exp_qoi)

        def test_return_best_simulation_qois(self):
            rc = self._specified_init()
            sim_qoi_list = rc.best_simulation_qois("MockModel", "MockObj", "MockState")
            self.assertEqual(len(sim_qoi_list), 1)
            self.assertEqual(sim_qoi_list[0]["time"], 0)
            self.assertEqual(sim_qoi_list[0]["c"], 0)
            eval_name = rc.get_eval_set_name("MockModel", "MockObj")
            self.assertEqual(rc.qoi_history[eval_name].simulation_qois[5]["MockState"], sim_qoi_list)
            sim_qoi = rc.best_simulation_qois("MockModel", "MockObj", "MockState", 0)
            self.assertEqual(sim_qoi_list[0], sim_qoi)

        def test_return_best_simulation_data(self):
            rc = self._specified_init()
            sim_data = rc.best_simulation_data("MockModel", "MockState")
            self.assertEqual(sim_data["time"], 0)
            self.assertEqual(sim_data["c"], 0)

        def test_return_best_residuals(self):
            rc = self._specified_init()
            resids_list = rc.best_residuals("MockModel", "MockObj",  "MockState") 
            self.assertEqual(len(resids_list), 1)
            self.assertEqual(resids_list[0]["time"], 0)
            self.assertEqual(resids_list[0]["c"], 0)
            eval_name = rc.get_eval_set_name("MockModel", "MockObj")
            self.assertEqual(rc.objective_history[eval_name].residuals[5]["MockState"], resids_list)
            resid = rc.best_residuals("MockModel", "MockObj", "MockState", 0)
            self.assertEqual(resids_list[0], resid)

        def test_return_best_weighted_conditioned_residuals(self):
            rc = self._specified_init(record_weighted_conditioned=True)
            resids_list = rc.best_weighted_conditioned_residuals("MockModel", "MockObj",  "MockState") 
            self.assertEqual(len(resids_list), 1)
            self.assertEqual(resids_list[0]["time"], 0)
            self.assertEqual(resids_list[0]["c"], 0)
            eval_name = rc.get_eval_set_name("MockModel", "MockObj")
            self.assertEqual(rc.objective_history[eval_name].weighted_conditioned_residuals[5]["MockState"], resids_list)
            resid = rc.best_weighted_conditioned_residuals("MockModel", "MockObj", "MockState", 0)
            self.assertEqual(resids_list[0], resid)

        def test_get_obj_name_if_not_string(self):
            class TestObj:
                def __init__(self, name):
                    self.name = name
                
            name = "test"
            res = _get_obj_name_if_not_string(name)
            self.assertEqual(name, res)

            test_obj = TestObj(name)
            res = _get_obj_name_if_not_string(test_obj)
            self.assertEqual(res, name)

        def test_dump_and_load_joblib(self):
            rc = self._random_init()
            save_file = "test_results.joblib"
            rc.save(save_file)
            same_rc = matcal_load(save_file)
            self.assert_close_dicts_or_data(rc.parameter_history, same_rc.parameter_history)
            self.assert_close_dicts_or_data(rc.outcome, same_rc.outcome)
        
        def test_turn_off_parameter_record_when_nothing_recorded(self):
            sr = StudyResults()
            self.assertTrue(sr.should_record_parameters)
            sr = StudyResults(True, False, False, False, False)
            self.assertTrue(sr.should_record_parameters)
            sr = StudyResults(False, True, False, False, False, )
            self.assertTrue(sr.should_record_parameters)
            sr = StudyResults(False, False, True, False, False)
            self.assertTrue(sr.should_record_parameters)
            sr = StudyResults(False, False, False, True, False)
            self.assertTrue(sr.should_record_parameters)
            sr = StudyResults(False, False, False, False, True)
            self.assertTrue(sr.should_record_parameters)

            sr = StudyResults(False, False, False, False, False)
            self.assertFalse(sr.should_record_parameters)

        def test_access_simulation_information(self):
            n_eval = 5
            sr = self._random_init(n_eval, record_weighted_conditioned=True)
            first_eval = sr.evaluation_sets[0]
            qoi_name = 'c'
            self.assertEqual(len(sr.qoi_history[first_eval].simulation_qois),
                              n_eval)
            self.assertEqual(len(sr.qoi_history[first_eval].experiment_qois["MockState"]), 1)
            self.assertEqual(len(sr.qoi_history[first_eval].simulation_weighted_conditioned_qois), n_eval)
            self.assertEqual(len(sr.qoi_history[first_eval].experiment_weighted_conditioned_qois["MockState"]), 1)

            data_name = 'c'
            self.assertEqual(len(sr.simulation_history["MockModel"]["MockState"]), n_eval)
            self.assertEqual(len(sr.qoi_history[first_eval].experiment_data["MockState"]), 1)
            
        def test_access_objective_information(self):
            n_eval = 5
            sr = self._random_init(n_eval)
            first_eval = sr.evaluation_sets[0]
            
            self.assertEqual(len(sr.total_objective_history), n_eval)

        def test_results_save_frequency(self):
            neval = 26
            sr = self._random_init(neval, results_save_frequency=5)
            self.assertEqual(len(sr.evaluation_ids), 6)
            self.assertEqual(sr.evaluation_ids, [0+1,5+1,10+1,15+1,20+1,25+1])
            self.assertEqual(sr.number_of_evaluations, 26)

        def test_results_save_frequency_best_index(self):
            sr = self._specified_init(results_save_frequency=5)
            self.assertEqual(len(sr.evaluation_ids), 3)
            self.assertEqual(sr.evaluation_ids, [0+1,5+1,10+1])
            self.assertEqual(sr.number_of_evaluations, 11)
            self.assertEqual(sr.best_evaluation_id, 6)
            self.assertEqual(sr.best_evaluation_index, 1)

        def test_results_attributes(self):
            res_attr = StudyResults.ResultsAttribute()
            res_attr.a = 1
            res_attr.b = 2
            res_attr.c = "a"

            self.assertEqual({'a':1, 'b':2, 'c':"a"}, res_attr.to_dict())
            self.assertEqual([1,2,'a'], res_attr.to_list())
            string_val = str(res_attr)
            self.assertTrue("a" in string_val)
            self.assertTrue("1" in string_val)
            self.assertTrue("b" in string_val)
            self.assertTrue("2" in string_val)
            self.assertTrue("c" in string_val)
            

class TestStudyResults(StudyResultsBaseUnitTests.CommonTests):

    def setUp(self):
        self._required_fields = ["a", "b"]
        self._fields_of_interest = ["a"]
        return super().setUp(__file__)
    
    def _random_init(self, n_samples=2, 
                     record_weighted_conditioned=False, 
                     results_save_frequency=1)->StudyResults:
        param_names = ['a', 'b']
        param_means = [0, 1]
        param_stds = [1, 2]
        n_samples = n_samples
        def simple_fun(a,b):
            out = [a + b - 1]
            return {'time':np.array([0]), 
                    'c':np.array(out)}
        sr = _generate_singe_model_single_state_mock_eval_hist(param_names, param_means,
                                                               param_stds, n_samples, 
                                                               simple_fun, 
                                                               record_weighted_conditioned, 
                                                               results_save_frequency=results_save_frequency)
        sr._initialize_exit_status(True, '')
        outcome = {'Best:a':0, "Best:b":1}
        sr._set_outcome(outcome)
        return sr

    def _specified_init(self, record_weighted_conditioned=False, 
                        results_save_frequency=1)->StudyResults:
        params = {}
        params['a'] = [-5,-4,-3,-2,-1.1, 0,1,2,3,4,5]
        params['b'] = [ 4, 3, 2,1,    0, 1,2,3,4,5,6]
        
        def simple_fun(a,b):
            out = [a + b - 1]
            return {'time':np.array([0]), 
                    'c':np.array(out)}
        best_dict = simple_fun(0,1)
        best = convert_dictionary_to_data(best_dict)
        sr = _generate_singe_model_single_state_mock_eval_hist_given_params(params, 
                                                               simple_fun, 
                                                               record_weighted_conditioned, 
                                                               best, 
                                                               results_save_frequency)
        sr._initialize_exit_status(True, '')
        outcome = {'Best:a':0, "Best:b":1}
        sr._set_outcome(outcome)
        return sr
        
    def test_populated_results_one_model(self):

        state_1 = State("one", offset=1)
        state_2 = State("minus_one", offset=-1)
        coeff = Parameter("coeff", 0, 10)
        pow = Parameter("power", 0, 10)
        
        params = ParameterCollection("test params", coeff, pow)
        study = ParameterStudy(params)    

        gold_params = {"coeff":2.125, "power":3.2}
        state_1_dict = {"offset":1}
        state_2_dict = {"offset":-1}

        data_state_1 =  convert_dictionary_to_data(model_func(**gold_params, **state_1_dict))
        data_state_1.set_state(state_1)
        data_state_2 =  convert_dictionary_to_data(model_func(**gold_params, **state_2_dict))
        data_state_2.set_state(state_2)

        data_collection = DataCollection("test", data_state_1, data_state_2)

        py_model = PythonModel(model_func)

        py_model.set_name("test_py_model_1")
        obj = CurveBasedInterpolatedObjective("x", "y")
        obj.set_name("x y obj")
        study.add_evaluation_set(py_model, obj, data_collection)
        study.add_parameter_evaluation(coeff=1.125, power=3.2)
        study.add_parameter_evaluation(coeff=2.125, power=3.2)
        study.add_parameter_evaluation(coeff=2.125, power=2.2)
        
        study.set_core_limit(6)
        results = study.launch()
    
        self.assertTrue(isinstance(results, StudyResults))
        self.assertEqual(len(results.parameter_history), 2)
        
    def test_export_eval_result_to_file(self):
        n_pts = 11
        time = np.linspace(0, 1, n_pts)
        a = np.linspace(50, 60, n_pts)
        b = np.linspace(-10, 5, n_pts)
        param1 = 4
        param2 = 6
        data = OrderedDict({'time':time, 'a':a, 'b':b})
        filename = 'test_record.csv'
        StudyResults._export_parameter_results(filename, data, {'param1':param1, 'param2':param2})
        goal = f"{{\"param1\":{param1}, \"param2\":{param2}}}\n"
        goal += "time, a, b\n"
        for row_i in range(n_pts):
            goal += f"{time[row_i]}, {a[row_i]}, {b[row_i]}\n"
        self.assert_file_equals_string(goal, filename)
