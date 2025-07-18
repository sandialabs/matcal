from collections import OrderedDict
import numpy as np
import os
import h5py

from matcal.core.constants import (BATCH_RESTART_FILENAME, 
                                   DESIGN_PARAMETER_FILE)
from matcal.core.data import DataCollection, convert_dictionary_to_data
from matcal.core.evaluation_set import StudyEvaluationSet
from matcal.core.models import PythonModel
from matcal.core.objective import (CurveBasedInterpolatedObjective, 
                                   ObjectiveCollection, ObjectiveSet)
from matcal.core.objective_results import ObjectiveResults
from matcal.core.parameter_batch_evaluator import (BatchRestartHDF5, MissingKeyError, 
                                                   ParameterBatchEvaluator, BatchRestartCSV, SelectedBatchRestartClass,
                                                    _calculate_total_objective, 
                                                    _combine_objective_results, 
                                                    _combine_residual_results, _setup_workdir, 
                                                    flatten_evaluation_batch_results, 
                                                    write_parameter_include_file,
                                                    EvaluationFailureDefaults)
from matcal.core.reporter import MatCalParameterReporterIdentifier
from matcal.core.state import SolitaryState, State
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest
import gc

def line(**parameters):
    x = 1
    y = parameters['a'] * x + parameters['b']
    out = {'x':[0,1], 'y':[parameters['b'],y]}
    return out 

def line_with_failure(**parameters):
    if parameters['a'] < 0:
        raise RuntimeError("Intentional Failure")
    x = 1
    y = parameters['a'] * x + parameters['b']
    out = {'x':[0,1], 'y':[parameters['b'],y]}
    return out 

def quad(**parameters):
    import numpy as np
    time = np.array([0,1,2,3])
    temp = 273 + parameters['a'] * time
    disp = np.power(time, parameters['b'])/10
    return {"time":time, "temp":temp, "disp": disp}


def param_preprocessor(params):
    return {"a":params["a"]+1.0, "b":params["b"]*2.0}


def _make_linear_data(line_func=line, count_on_failure=False):
    template_dir = "matcal_template"
    if not os.path.exists(template_dir):
        os.mkdir(template_dir)
    data_dict = {"x":[0,1], "y":[1,2]}
    exp_data = convert_dictionary_to_data(data_dict)
    dc = DataCollection('test', exp_data)
    objective = CurveBasedInterpolatedObjective("x", "y")
    n_cores = 2
    model = PythonModel(line_func)
    if count_on_failure:
        model.continue_when_simulation_fails()
    eval_set = StudyEvaluationSet(model, ObjectiveSet(ObjectiveCollection("one_obj", objective), 
                                                      dc, dc.states))
    eval_set.prepare_model_and_simulators(template_dir)
    eval_sets = {model:eval_set}
    return exp_data,n_cores,model,eval_sets


def _make_quadratic_data():
    template_dir = "matcal_template"
    if not os.path.exists(template_dir):
        os.mkdir(template_dir)

    data_dict = {"time":[0, 1, 2, 3], 
                 "temp":[273, 283, 293, 303], 
                 "disp":[0.0, 0.1, 0.4, 0.9]
                 }
    exp_data = convert_dictionary_to_data(data_dict)
    dc = DataCollection('test', exp_data)
    objective = CurveBasedInterpolatedObjective("time", "temp", 'disp')
    n_cores = 2
    model = PythonModel(quad)
    eval_set = StudyEvaluationSet(model, ObjectiveSet(ObjectiveCollection("one_obj", objective),
                                                       dc, dc.states))
    eval_set.prepare_model_and_simulators(template_dir)
    return n_cores,model,{model:eval_set}


def _make_more_linear_data(npts=2):
    template_dir = "matcal_template"
    if not os.path.exists(template_dir):
        os.mkdir(template_dir)

    data_dict = {"x":np.linspace(0,1, npts), "y":np.linspace(1,2, npts)}
    exp_data1 = convert_dictionary_to_data(data_dict)

    data_dict2 = {"x":np.linspace(0,1, npts), "y": np.linspace(1.01, 2.02, npts)}
    exp_data2 = convert_dictionary_to_data(data_dict2)
    dc = DataCollection('test', exp_data1, exp_data2)
    objective = CurveBasedInterpolatedObjective("x", "y")
    n_cores = 2
    model = PythonModel(line)
    eval_set = StudyEvaluationSet(model, ObjectiveSet(ObjectiveCollection("one_obj", objective), 
                                                      dc, dc.states))
    eval_set.prepare_model_and_simulators(template_dir)
    return n_cores,model,{model:eval_set}


class BatchRestartTests(MatcalUnitTest):

    class CommonSetUp(MatcalUnitTest):

        @property
        def _batch_restart_class(self):
            """"""

        def setUp(self):
            super().setUp(__file__)

    class CommonTests(CommonSetUp):

        def test_init(self):
            save_only = True
            br = self._batch_restart_class(save_only)
            br.close()

        def test_record_and_retrieve_if_exists_else_None_and_not_save_only(self):
            empty_br = self._batch_restart_class(True)
            empty_br.close()
            gc.collect()

            save_only = False
            br = self._batch_restart_class(save_only)
            eval_name = 'eval.1'
            model_name = 'model'
            state_name = 'matcal_default_state'
            results_filename = 'results.csv'
            job_key = [eval_name, model_name, state_name]
            none_job_key = ['eval.2', model_name, state_name]
            goal_file= results_filename
            br.record(job_key, results_filename)
            self.assertEqual(br.retrieve_results_file(job_key), goal_file)
            self.assertIsNone(br.retrieve_results_file(none_job_key))
            br.close()

        def test_if_save_only_retrieve_returns_None_always(self):
            save_only = True
            br = self._batch_restart_class(save_only)
            eval_name = 'eval.1'
            model_name = 'model'
            state_name = 'matcal_default_state'
            results_filename = 'results.csv'
            job_key = [eval_name, model_name, state_name]
            none_job_key = ['eval.2', model_name, state_name]
            goal_file= results_filename
            br.record(job_key, results_filename)
            self.assertIsNone(br.retrieve_results_file(job_key), goal_file)
            self.assertIsNone(br.retrieve_results_file(none_job_key))
            br.close()

        def test_None_filename_does_not_get_written(self):
            save_only = True
            br = self._batch_restart_class(save_only)

            restart_file = f"{BATCH_RESTART_FILENAME}"+br.file_extension()
            old_file_size = os.path.getsize(restart_file)

            br.record(['a', 'b', 'c'], None)
            self.assertIsNone(br.retrieve_results_file(['a', 'b', 'c']))
            os.path.getsize(restart_file)
            new_file_szie = os.path.getsize(restart_file)
            self.assertEqual(new_file_szie, old_file_size)

        def test_write_to_file_durring_a_record(self):
            save_only = True
            br = self._batch_restart_class(save_only)
            restart_file = f"{BATCH_RESTART_FILENAME}"+br.file_extension()

            old_file_size = os.path.getsize(restart_file)
            br.record(['a', 'b', 'c'], 'a.txt')
            new_file_size = os.path.getsize(restart_file)
            self.assertGreater(new_file_size, old_file_size)
            old_file_size = new_file_size
            br.record(['a', 'b', 'd'], 'b.txt')
            new_file_size = os.path.getsize(restart_file)
            self.assertGreater(new_file_size, old_file_size)
            old_file_size = new_file_size
            br.record(['a', 'b', '3'], 'c.txt')
            new_file_size = os.path.getsize(restart_file)
            self.assertGreater(new_file_size, old_file_size)
            old_file_size = new_file_size


class TestBatchRestartHDF5(BatchRestartTests.CommonTests):

    _batch_restart_class = BatchRestartHDF5


class TestBatchRestartCSV(BatchRestartTests.CommonTests):

    _batch_restart_class = BatchRestartCSV


class TestFailureDefaults(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def test_init(self):
        exp_data, n_cores, model, eval_sets = _make_linear_data()
        fail_defaults = EvaluationFailureDefaults(eval_sets)

    def test_assemble_fall_back_qois_from_eval_sets_one_set(self):
        exp_data, n_cores, model, eval_sets = _make_linear_data()
        fail_defaults = EvaluationFailureDefaults(eval_sets)
        goal_fields = ['x', 'y']
        self.assertEqual(len(fail_defaults.all_fallback_fieldnames), len(goal_fields))
        for field in fail_defaults.all_fallback_fieldnames:
            self.assertIn(field, goal_fields)

    def test_assemble_fall_back_qois_from_multiple_eval_sets(self):
        template_dir = "matcal_template"
        if not os.path.exists(template_dir):
            os.mkdir(template_dir)
        data_dict_lin = {"x":[0,1], "y":[1,2]}
        exp_data_lin = convert_dictionary_to_data(data_dict_lin)
        dc_lin = DataCollection('lin', exp_data_lin)
        objective_lin = CurveBasedInterpolatedObjective("x", "y")
        model_lin = PythonModel(line)
        eval_set_lin = StudyEvaluationSet(model_lin, ObjectiveSet(ObjectiveCollection("one_obj", objective_lin), dc_lin, dc_lin.states))
        eval_set_lin.prepare_model_and_simulators(template_dir)

        exp_data_quad = convert_dictionary_to_data({'t':[0, 1, 2], 'z':[0, 1, 4], 'b':[0, -1, -4]})
        dc_quad = DataCollection('quad', exp_data_quad)
        obj_quad = CurveBasedInterpolatedObjective('t', 'z', 'b')
        model_quad = PythonModel(quad)
        eval_set_quad = StudyEvaluationSet(model_quad, ObjectiveSet(ObjectiveCollection('quad', obj_quad), dc_quad, dc_quad.states))
        eval_set_quad.prepare_model_and_simulators(template_dir)

        eval_sets = {model_lin:eval_set_lin, model_quad:eval_set_quad}
        fail_defaults = EvaluationFailureDefaults(eval_sets)
        goal_fields = ['x', 'y', 't', 'z', 'b']
        self.assertEqual(len(fail_defaults.all_fallback_fieldnames), len(goal_fields))
        for field in fail_defaults.all_fallback_fieldnames:
            self.assertIn(field, goal_fields)

    def test_extract_fall_back_data_from_model(self):
        template_dir = "matcal_template"
        if not os.path.exists(template_dir):
            os.mkdir(template_dir)
        data_dict_lin = {"x":[0,1], "y":[1,2]}
        exp_data_lin = convert_dictionary_to_data(data_dict_lin)
        dc_lin = DataCollection('lin', exp_data_lin)
        objective_lin = CurveBasedInterpolatedObjective("x", "y")
        model_lin = PythonModel(line)
        num_pts = 10
        goal_model_fields = {'x':np.linspace(-1, 1, num_pts), 'y':np.zeros(num_pts), 'z':np.ones(num_pts)}
        model_lin.continue_when_simulation_fails(**goal_model_fields)
        eval_set_lin = StudyEvaluationSet(model_lin, ObjectiveSet(ObjectiveCollection("one_obj", objective_lin), dc_lin, dc_lin.states))
        eval_set_lin.prepare_model_and_simulators(template_dir)
        eval_sets = {model_lin:eval_set_lin}

        fail_defaults = EvaluationFailureDefaults(eval_sets)
        all_goal_fields = ['x', 'y']
        self.assertEqual(len(fail_defaults.all_fallback_fieldnames), len(all_goal_fields))
        for field in fail_defaults.all_fallback_fieldnames:
            self.assertIn(field, all_goal_fields)
        self.assertEqual(len(fail_defaults.model_specific_fallbacks[model_lin.name]), len(goal_model_fields))
        for field_name, value in fail_defaults.model_specific_fallbacks[model_lin.name].items():
            self.assert_close_arrays(value, goal_model_fields[field_name])

    def test_make_generic_default_data(self):
        exp_data, n_cores, model, eval_sets = _make_linear_data()
        fail_defaults = EvaluationFailureDefaults(eval_sets)
        state = State("my_state", a=2)
        data = fail_defaults.make_failure_data(model.name, state)
        generic_default_values = np.linspace(-1, 1, 20)
        goal_values = {'x':generic_default_values, 'y':generic_default_values}
        self.assertEqual(state, data.state)
        for name, value in goal_values.items():
            self.assert_close_arrays(data[name], value)

    def test_extract_general_and_model_specific_data_and_match_model_length(self):
        template_dir = "matcal_template"
        if not os.path.exists(template_dir):
            os.mkdir(template_dir)
        data_dict_lin = {"x":[0,1], "y":[1,2]}
        exp_data_lin = convert_dictionary_to_data(data_dict_lin)
        dc_lin = DataCollection('lin', exp_data_lin)
        objective_lin = CurveBasedInterpolatedObjective("x", "y")
        model_lin = PythonModel(line)
        num_pts = 10
        goal_z_value = np.linspace(0, 10, num_pts)
        model_lin.continue_when_simulation_fails(z=goal_z_value)
        eval_set_lin = StudyEvaluationSet(model_lin, ObjectiveSet(ObjectiveCollection("one_obj", objective_lin), dc_lin, dc_lin.states))
        eval_set_lin.prepare_model_and_simulators(template_dir)
        eval_sets = {model_lin:eval_set_lin}

        fail_defaults = EvaluationFailureDefaults(eval_sets)
        state = SolitaryState()
        data = fail_defaults.make_failure_data(model_lin.name, state)

        generic_resized_values = np.linspace(-1, 1, num_pts)
        goal_values = {'x':generic_resized_values, 'y':generic_resized_values, 'z':goal_z_value}
        for name, value in goal_values.items():
            self.assert_close_arrays(data[name], value)
        

class TestParameterBatchEvaluator(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)
        MatCalParameterReporterIdentifier._registry = {}

    def test_write_parameter_include_file(self):
        parameters = {"Y":1.0, "CAT":102.0, "cheese":-7.2}
        write_parameter_include_file(parameters, ".")

        goal = ""
        goal += "Y=1.0\n"
        goal += "CAT=102.0\n"
        goal += "cheese=-7.2\n"

        self.assert_file_equals_string(goal, DESIGN_PARAMETER_FILE)

    def test_add_parameters(self):
        exp_data, n_cores, model, eval_sets = _make_linear_data()
        eval_set = eval_sets[model]
        obj_name = list(eval_set._objective_sets[-1].objectives.values())[-1].name
        pbe = ParameterBatchEvaluator(n_cores, eval_sets, False)
        params = {"eval.1":{"a":0.0, "b":0.0}}
        results = pbe.run(params, False)
        residual_dict = results['objectives'][0][model.name][obj_name].residuals[exp_data.state]
        residual = residual_dict[0]['y']
        goal_residual  = np.array([-1, -2])
        self.assert_close_arrays(goal_residual, residual)
                
    def test_add_parameters_in_serial(self):
        exp_data, n_cores, model, eval_sets = _make_linear_data()
        eval_set = eval_sets[model]
        obj_name = list(eval_set._objective_sets[-1].objectives.values())[-1].name
        pbe = ParameterBatchEvaluator(n_cores, eval_sets, False, run_async=False)
        params = {"eval.1":{"a":0.0, "b":0.0}}
        results = pbe.run(params, False)
        residual_dict = results['objectives'][0][model.name][obj_name].residuals[exp_data.state]
        residual = residual_dict[0]['y']
        goal_residual  = np.array([-1, -2])
        self.assert_close_arrays(goal_residual, residual)

    class ParameterBatchEvaluator_SPY(ParameterBatchEvaluator):

        @property
        def fallback_fields(self):
            return self._failure_defaults.all_fallback_fieldnames

    def test_assemble_fall_back_qois_from_eval_sets_one_set(self):
        exp_data, n_cores, model, eval_sets = _make_linear_data()
        eval_set = eval_sets[model]
        obj_name = list(eval_set._objective_sets[-1].objectives.values())[-1].name
        pbe = self.ParameterBatchEvaluator_SPY(n_cores, eval_sets, False, run_async=False)
        goal_fields = ['x', 'y']
        self.assertEqual(len(pbe.fallback_fields), len(goal_fields))
        for field in pbe.fallback_fields:
            self.assertIn(field, goal_fields)

    def test_assemble_fall_back_qois_from_multiple_eval_sets(self):
        template_dir = "matcal_template"
        if not os.path.exists(template_dir):
            os.mkdir(template_dir)
        data_dict_lin = {"x":[0,1], "y":[1,2]}
        exp_data_lin = convert_dictionary_to_data(data_dict_lin)
        dc_lin = DataCollection('lin', exp_data_lin)
        objective_lin = CurveBasedInterpolatedObjective("x", "y")
        model_lin = PythonModel(line)
        eval_set_lin = StudyEvaluationSet(model_lin, ObjectiveSet(ObjectiveCollection("one_obj", objective_lin), dc_lin, dc_lin.states))
        eval_set_lin.prepare_model_and_simulators(template_dir)

        exp_data_quad = convert_dictionary_to_data({'t':[0, 1, 2], 'z':[0, 1, 4], 'b':[0, -1, -4]})
        dc_quad = DataCollection('quad', exp_data_quad)
        obj_quad = CurveBasedInterpolatedObjective('t', 'z', 'b')
        model_quad = PythonModel(quad)
        eval_set_quad = StudyEvaluationSet(model_quad, ObjectiveSet(ObjectiveCollection('quad', obj_quad), dc_quad, dc_quad.states))
        eval_set_quad.prepare_model_and_simulators(template_dir)


        eval_sets = {model_lin:eval_set_lin, model_quad:eval_set_quad}
        pbe = self.ParameterBatchEvaluator_SPY(1, eval_sets, False, run_async=False)
        goal_fields = ['x', 'y', 't', 'z', 'b']
        self.assertEqual(len(pbe.fallback_fields), len(goal_fields))
        for field in pbe.fallback_fields:
            self.assertIn(field, goal_fields)

    def test_plot_get_common_dofs(self):
        from matcal.core.plotting import _get_common_fields

        sim_qoi_list = [convert_dictionary_to_data({'a':[1],'b':[2], 'c':[3]})]

        exp_qoi_list = [convert_dictionary_to_data({'a':[1]}), convert_dictionary_to_data({'c':[2]})]

        result = _get_common_fields(sim_qoi_list, exp_qoi_list)

        self.assertEqual(result, ['a', 'c'])

    def test_setup_workdir_removes_existing_file(self):
        wrk_dir = 'to_be_removed'
        os.mkdir(wrk_dir)
        is_restart = False
        _setup_workdir(wrk_dir, is_restart)
        self.assertFalse(os.path.exists(wrk_dir))

    def test_setup_workdir_keeps_existing_file_on_restart(self):
        wrk_dir = 'to_be_removed'
        os.mkdir(wrk_dir)
        is_restart = True
        _setup_workdir(wrk_dir, is_restart)
        self.assertTrue(os.path.exists(wrk_dir))

    def test_create_restart_file_on_serial_run(self):
        exp_data, n_cores, model, eval_sets = _make_linear_data()
        eval_set = eval_sets[model]
        obj_name = list(eval_set._objective_sets[-1].objectives.values())[-1].name
        pbe = ParameterBatchEvaluator(n_cores, eval_sets, False, run_async=False)
        params = {"eval.1":{"a":0.0, "b":0.0}, 'eval.2':{'a':1.1, "b":-1.}}
        results = pbe.run(params, False)
        self.assert_file_exists(BATCH_RESTART_FILENAME+SelectedBatchRestartClass.file_extension())
        job_keys = [['eval.1', model.name, 'matcal_default_state'],
                ['eval.2', model.name, 'matcal_default_state']]
        goals = [ os.path.join("matcal_python_results_archive",model.name+'_a=0.0_b=0.0.joblib'),
                 os.path.join("matcal_python_results_archive",model.name+'_a=1.1_b=-1.0.joblib')]
        br = SelectedBatchRestartClass(False)
        for key, goal in zip(job_keys, goals):
            self.assertEqual(br.retrieve_results_file(key), goal)
        br.close()

    def test_restarts_are_not_written_on_failures_serial(self):
        exp_data, n_cores, model, eval_sets = _make_linear_data(line_with_failure, True)
        eval_set = eval_sets[model]
        obj_name = list(eval_set._objective_sets[-1].objectives.values())[-1].name
        pbe = ParameterBatchEvaluator(n_cores, eval_sets, False, run_async=False)
        params = {"eval.1":{"a":2.0, "b":0.0}, 'eval.2':{'a':-1.1, "b":-1.},'eval.3':{'a':1.1, "b":-1.}}
        results = pbe.run(params, False)
        self.assert_file_exists(BATCH_RESTART_FILENAME+SelectedBatchRestartClass.file_extension())
        job_keys = [['eval.1', model.name, 'matcal_default_state'],
                ['eval.2', model.name, 'matcal_default_state'],
                ['eval.3', model.name, 'matcal_default_state']]
        goals = [ os.path.join("matcal_python_results_archive", model.name+'_a=2.0_b=0.0.joblib'), None,
                 os.path.join("matcal_python_results_archive",model.name+'_a=1.1_b=-1.0.joblib')]
        br = SelectedBatchRestartClass(False)
        for key, goal in zip(job_keys, goals):
            print(key, br.retrieve_results_file(key))
        for key, goal in zip(job_keys, goals):
            self.assertEqual(br.retrieve_results_file(key), goal)
        br.close()

    def test_restarts_are_not_written_on_failures_async(self):
        exp_data, n_cores, model, eval_sets = _make_linear_data(line_with_failure, True)
        eval_set = eval_sets[model]
        obj_name = list(eval_set._objective_sets[-1].objectives.values())[-1].name
        pbe = ParameterBatchEvaluator(n_cores, eval_sets, False, run_async=True)
        params = {"eval.1":{"a":2.0, "b":0.0}, 'eval.2':{'a':-1.1, "b":-1.},'eval.3':{'a':1.1, "b":-1.}}
        results = pbe.run(params, False)
        self.assert_file_exists(BATCH_RESTART_FILENAME+SelectedBatchRestartClass.file_extension())
        job_keys = [['eval.1', model.name, 'matcal_default_state'],
                ['eval.2', model.name, 'matcal_default_state'],
                ['eval.3', model.name, 'matcal_default_state']]
        goals = [ os.path.join("matcal_python_results_archive",model.name+'_a=2.0_b=0.0.joblib'), None,
                 os.path.join("matcal_python_results_archive",model.name+'_a=1.1_b=-1.0.joblib')]
        br = SelectedBatchRestartClass(False)
        for key, goal in zip(job_keys, goals):
            self.assertEqual(br.retrieve_results_file(key), goal)
        br.close()

    def test_create_restart_run_only_reads(self):
        exp_data, n_cores, model, eval_sets = _make_linear_data()
        eval_set = eval_sets[model]
        obj_name = list(eval_set._objective_sets[-1].objectives.values())[-1].name
        pbe = ParameterBatchEvaluator(n_cores, eval_sets, False, run_async=False)
        params = {"eval.1":{"a":0.0, "b":0.0}, 'eval.2':{'a':1.1, "b":-1.}}
        results = pbe.run(params, False)
        restart_results = pbe.restart_run(params, False)
        for eval_name, eval_params in results['parameters'].items():
            restart_params = restart_results['parameters'][eval_name]
            self.assert_close_dicts_or_data(eval_params, restart_params)
        for eval_qoi, restart_qoi in zip(results['qois'], restart_results['qois']):
            eval_sim = eval_qoi[model.name][obj_name]._simulation_data
            restart_sim = restart_qoi[model.name][obj_name]._simulation_data
            self.assert_close_dicts_or_data(eval_sim, restart_sim)

    def _get_fake_results_dict(self):
        res = OrderedDict()
        res["model1"] = {}
        res["model1"]["obj1"] = ObjectiveResults("x", ["x", "y"])
        res["model1"]["obj1"].set_objective(1.0)
        data = convert_dictionary_to_data({"x":np.ones(3), "y":np.ones(3)})
        res_dc = DataCollection("test", data)
        res["model1"]["obj1"].set_weighted_conditioned_normalized_residuals(res_dc)
        res["model2"] = {}
        res["model2"]["obj1"] = ObjectiveResults("x", ["x", "y"])
        res["model2"]["obj1"].set_objective(2.0)
        data_2 = convert_dictionary_to_data({"x":2*np.ones(3), "y":2*np.ones(3)})
        res_dc_2 = DataCollection("test", data_2)
        res["model2"]["obj1"].set_weighted_conditioned_normalized_residuals(res_dc_2)

        param_list_res = {}
        param_list_res["eval.1"] = res

        return param_list_res

    def test_combine_objective_and_residual_results(self):
        res = self._get_fake_results_dict()["eval.1"]
        combined_obj = _combine_objective_results(res)
        self.assert_close_arrays(combined_obj, np.array([1.0, 2.0]))
        combined_resids = _combine_residual_results(res)
        self.assert_close_arrays(combined_resids, np.hstack((np.ones(6), 2*np.ones(6))))

    def test_flatten_evaluation_batch_results(self):
        res = self._get_fake_results_dict()

        combined_obj, combined_resids, eval_names = flatten_evaluation_batch_results(res)
        self.assert_close_arrays(combined_obj, np.array([1.0, 2.0]))
        self.assert_close_arrays(combined_resids, np.hstack((np.ones(6), 2*np.ones(6))))
        self.assertEqual(eval_names, ["eval.1"])

    def test_calculate_total_objective(self):
        res = self._get_fake_results_dict()
        total_objectives = _calculate_total_objective(res, False)
        self.assertEqual(total_objectives["eval.1"], 3.0)
        total_objectives = _calculate_total_objective(res, True)
        self.assertEqual(total_objectives["eval.1"], 
                         np.linalg.norm(np.hstack((np.ones(6), 2*np.ones(6))))**2)
        

class ResultsSerializerTestBase(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def _make_obj_results(self, obj_val):
        obj = ObjectiveResults(["req_fields"], ["fields_of_interest"])
        obj.set_objective(obj_val)
        obj.set_weighted_conditioned_normalized_residuals(np.ones(2)*obj_val/np.sqrt(2))
        state_1_obj_dict= {"fields_of_interest":obj_val/3}
        state1 = State("state1")
        data1 = convert_dictionary_to_data(state_1_obj_dict)
        data1.set_state(state1)

        state_2_obj_dict= {"fields_of_interest":2*obj_val/3}
        state2 = State("state2")
        data2 = convert_dictionary_to_data(state_2_obj_dict)
        data2.set_state(state2)

        obj.add_weighted_conditioned_objective(data1)
        obj.add_weighted_conditioned_objective(data2)
        
        return obj
    

