from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

from matcal.core.data import DataCollection, convert_dictionary_to_data
from matcal.core.evaluation_set import StudyEvaluationSet
from matcal.core.models import PythonModel
from matcal.core.objective import (CurveBasedInterpolatedObjective, 
                                   ObjectiveCollection, ObjectiveSet)
from matcal.core.parameter_batch_evaluator import (ParameterBatchEvaluator)
from matcal.core.plotting import (
    _NullPlotter,
    _ObjectiveProgressPlotJob,
    _ParameterModelObjectivePlotJob,
    _PlotEvaluationIdJob,
    _TotalObjectiveProgressPlotJob,
    _UserAutoPlotter,
    StandardAutoPlotter,
    _get_common_fields,
    _get_study_results_evaluation_ids,
    make_standard_plots,
)
from matcal.core.restart_file import BatchRestartNone
from matcal.core.study_base import StudyResults, _record_results, _unpack_evaluation
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest


def line(**parameters):
    x = 1
    y = parameters['a'] * x + parameters['b']
    out = {'x':[0,1], 'y':[parameters['b'],y]}
    return out 


def quad(**parameters):
    time = np.array([0,1,2,3])
    temp = 273 + parameters['a'] * time
    disp = np.power(time, parameters['b'])/10
    return {"time":time, "temp":temp, "disp": disp}


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


class TestNullPlotter(MatcalUnitTest):
    
    def setUp(self):
        super().setUp(__file__)
        
    def test_plot_does_nothing(self):
        null_ap = _NullPlotter()
        null_ap.plot()
        glob_search = "user_plots/*.pdf"
        plot_files = glob(glob_search)
        self.assertEqual(len(plot_files), 0)


class TestMakeStandardPlots(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)
        self._batch_restart = BatchRestartNone(None, None)

    def test_standard_plot_jobs(self):
        sap = StandardAutoPlotter()
        plot_jobs = sap._get_plot_jobs()
        job_names = []
        goal_names = ["objective_", 'parameter_model_objective_', "evaluation_best", 
                      "total_objective", "parameter_total_objective"]
        for job in plot_jobs:
            job_names.append(job.filename_root)
        self.assertEqual(len(goal_names), len(job_names))
        for g_name in goal_names:
            self.assertIn(g_name, job_names)

    def test_plots_created_show_indep_field_no_show(self):

        plt.close("all")
        n_eval = 5
        vals = np.linspace(-5, 2, n_eval)
        vals = np.concatenate((vals,[1.05]))
        param_evals = {}
        for index, v in enumerate(vals): 
            pt = {'a':v, 'b':v}         
            param_evals[f"eval.{index}"] = pt
        n_cores, model, eval_sets = _make_more_linear_data()
        pbe = ParameterBatchEvaluator(n_cores, eval_sets, False)
        batch_results = pbe.evaluate_parameter_batch(param_evals, False, self._batch_restart)
        raw_obj, total_obj, qoi = _unpack_evaluation(batch_results)
        sr = StudyResults()
        _record_results(sr, param_evals, raw_obj, total_obj, qoi, False)

        make_standard_plots("x", show=False)
        glob_search = "user_plots/*.pdf"
        plot_files = glob(glob_search)
        self.assertEqual(len(plot_files), 3)
        plt.close("all")

    def test_plots_bad_independent_field(self):

        plt.close("all")
        n_eval = 5
        vals = np.linspace(-5, 2, n_eval)
        vals = np.concatenate((vals,[1.05]))
        param_evals = {}
        for index, v in enumerate(vals): 
            pt = {'a':v, 'b':v}         
            param_evals[f"eval.{index}"] = pt
        n_cores, model, eval_sets = _make_more_linear_data()
        pbe = ParameterBatchEvaluator(n_cores, eval_sets, False)
        batch_results = pbe.evaluate_parameter_batch(param_evals, False, self._batch_restart)
        raw_obj, total_obj, qoi = _unpack_evaluation(batch_results)
        sr = StudyResults()
        _record_results(sr, param_evals, raw_obj, total_obj, qoi, False)
        with self.assertRaises(ValueError):
            make_standard_plots("not a valid field", show=False)

    def test_plots_created_show(self):

        plt.close("all")
        n_eval = 5
        vals = np.linspace(-5, 2, n_eval)
        vals = np.concatenate((vals,[1.05]))
        param_evals = {}
        for index, v in enumerate(vals): 
            pt = {'a':v, 'b':v}         
            param_evals[f"eval.{index}"] = pt

        n_cores, model, eval_sets = _make_more_linear_data()
        pbe = ParameterBatchEvaluator(n_cores, eval_sets, False)
        batch_results = pbe.evaluate_parameter_batch(param_evals, False, self._batch_restart)
        raw_obj, total_obj, qoi = _unpack_evaluation(batch_results)
        sr = StudyResults()
        _record_results(sr, param_evals, raw_obj, total_obj, qoi, False)

        make_standard_plots("x", block=False)
        self.assertEqual(len(plt.get_fignums()), 3)
        plt.close("all")

    def test_plots_created_show_exp_data(self):

        plt.close("all")
        n_eval = 5
        vals = np.linspace(-5, 2, n_eval)
        vals = np.concatenate((vals,[1.05]))
        param_evals = {}
        for index, v in enumerate(vals): 
            pt = {'a':v, 'b':v}         
            param_evals[f"eval.{index}"] = pt

        n_cores, model, eval_sets = _make_more_linear_data()
        pbe = ParameterBatchEvaluator(n_cores, eval_sets, False)
        batch_results = pbe.evaluate_parameter_batch(param_evals, False, self._batch_restart)
        raw_obj, total_obj, qoi = _unpack_evaluation(batch_results)
        sr = StudyResults()
        _record_results(sr, param_evals, raw_obj, total_obj, qoi, False)

        make_standard_plots("x", block=False, plot_exp_data=True)
        self.assertEqual(len(plt.get_fignums()), 3)
        plt.close("all")

    def test_plots_created_show_data_no_qois_or_resids(self):

        plt.close("all")
        n_eval = 5
        vals = np.linspace(-5, 2, n_eval)
        vals = np.concatenate((vals,[1.05]))
        param_evals = {}
        for index, v in enumerate(vals): 
            pt = {'a':v, 'b':v}         
            param_evals[f"eval.{index}"] = pt

        n_cores, model, eval_sets = _make_more_linear_data()
        pbe = ParameterBatchEvaluator(n_cores, eval_sets, False)
        batch_results = pbe.evaluate_parameter_batch(param_evals, False, self._batch_restart)
        raw_obj, total_obj, qoi = _unpack_evaluation(batch_results)
        sr = StudyResults(record_qois=False, record_residuals=False)
        _record_results(sr, param_evals, raw_obj, total_obj, qoi, False)

        make_standard_plots("x", block=False, plot_exp_data=True, plot_sim_data=True)
        self.assertEqual(len(plt.get_fignums()), 3)
        plt.close("all")

    def test_plots_created_show_sim_data(self):

        plt.close("all")
        n_eval = 5
        vals = np.linspace(-5, 2, n_eval)
        vals = np.concatenate((vals,[1.05]))
        param_evals = {}
        for index, v in enumerate(vals): 
            pt = {'a':v, 'b':v}         
            param_evals[f"eval.{index}"] = pt

        n_cores, model, eval_sets = _make_more_linear_data()
        pbe = ParameterBatchEvaluator(n_cores, eval_sets, False)
        batch_results = pbe.evaluate_parameter_batch(param_evals, False, self._batch_restart)
        raw_obj, total_obj, qoi = _unpack_evaluation(batch_results)
        sr = StudyResults()
        _record_results(sr, param_evals, raw_obj, total_obj, qoi, False)

        make_standard_plots("x", block=False, plot_sim_data=True)
        self.assertEqual(len(plt.get_fignums()), 3)
        plt.close("all")

    def test_plots_created_show_sim_and_exp_data(self):

        plt.close("all")
        n_eval = 5
        vals = np.linspace(-5, 2, n_eval)
        vals = np.concatenate((vals,[1.05]))
        param_evals = {}
        for index, v in enumerate(vals): 
            pt = {'a':v, 'b':v}         
            param_evals[f"eval.{index}"] = pt

        n_cores, model, eval_sets = _make_more_linear_data()
        pbe = ParameterBatchEvaluator(n_cores, eval_sets, False)
        batch_results = pbe.evaluate_parameter_batch(param_evals, False, self._batch_restart)
        raw_obj, total_obj, qoi = _unpack_evaluation(batch_results)
        sr = StudyResults()
        _record_results(sr, param_evals, raw_obj, total_obj, qoi, False)

        make_standard_plots("x", block=False, plot_sim_data=True, plot_exp_data=True)
        self.assertEqual(len(plt.get_fignums()), 3)
        plt.close("all")

    def test_plots_created_show_sim_and_exp_data_no_qois_no_resids(self):

        plt.close("all")
        n_eval = 5
        vals = np.linspace(-5, 2, n_eval)
        vals = np.concatenate((vals,[1.05]))
        param_evals = {}
        for index, v in enumerate(vals): 
            pt = {'a':v, 'b':v}         
            param_evals[f"eval.{index}"] = pt

        n_cores, model, eval_sets = _make_more_linear_data()
        pbe = ParameterBatchEvaluator(n_cores, eval_sets, False)
        batch_results = pbe.evaluate_parameter_batch(param_evals, False, self._batch_restart)
        raw_obj, total_obj, qoi = _unpack_evaluation(batch_results)
        sr = StudyResults(record_qois=False, record_residuals=False)
        _record_results(sr, param_evals, raw_obj, total_obj, qoi, False)

        make_standard_plots("x", block=False, plot_sim_data=True, plot_exp_data=True)
        self.assertEqual(len(plt.get_fignums()), 3)
        plt.close("all")

    def test_plots_created_show_no_data_no_qois_no_resids(self):

        plt.close("all")
        n_eval = 5
        vals = np.linspace(-5, 2, n_eval)
        vals = np.concatenate((vals,[1.05]))
        param_evals = {}
        for index, v in enumerate(vals): 
            pt = {'a':v, 'b':v}         
            param_evals[f"eval.{index}"] = pt

        n_cores, model, eval_sets = _make_more_linear_data()
        pbe = ParameterBatchEvaluator(n_cores, eval_sets, False)
        batch_results = pbe.evaluate_parameter_batch(param_evals, False, self._batch_restart)
        raw_obj, total_obj, qoi = _unpack_evaluation(batch_results)
        sr = StudyResults(record_qois=False, record_residuals=False, record_data=False)
        _record_results(sr, param_evals, raw_obj, total_obj, qoi, False)

        make_standard_plots("x", block=False, plot_sim_data=True, plot_exp_data=True)
        self.assertEqual(len(plt.get_fignums()), 2)
        plt.close("all")

    def test_plots_no_crash_when_qois_disabled_default_plot(self):
        """Regression test: plotting with default flags (no -psd/-ped) should
        not crash when qois were not recorded (qois=False).  It should
        gracefully skip the evaluation plots and still produce the
        objective/parameter plots."""

        plt.close("all")
        n_eval = 5
        vals = np.linspace(-5, 2, n_eval)
        vals = np.concatenate((vals, [1.05]))
        param_evals = {}
        for index, v in enumerate(vals):
            pt = {'a': v, 'b': v}
            param_evals[f"eval.{index}"] = pt

        n_cores, model, eval_sets = _make_more_linear_data()
        pbe = ParameterBatchEvaluator(n_cores, eval_sets, False)
        batch_results = pbe.evaluate_parameter_batch(param_evals, False,
                                                     self._batch_restart)
        raw_obj, total_obj, qoi = _unpack_evaluation(batch_results)
        sr = StudyResults(record_qois=False)
        _record_results(sr, param_evals, raw_obj, total_obj, qoi, False)

        # Should NOT raise -- previously this raised a KeyError
        make_standard_plots("x", block=False)
        # Objective + parameter plots should still be created (2),
        # but evaluation QoI plots should be skipped
        self.assertGreaterEqual(len(plt.get_fignums()), 2)
        plt.close("all")

    def test_plots_no_crash_exp_data_flag_with_data_disabled(self):
        """Requesting -ped when data was not recorded should warn,
        not crash."""
        plt.close("all")
        n_eval = 5
        vals = np.linspace(-5, 2, n_eval)
        vals = np.concatenate((vals, [1.05]))
        param_evals = {}
        for index, v in enumerate(vals):
            pt = {'a': v, 'b': v}
            param_evals[f"eval.{index}"] = pt

        n_cores, model, eval_sets = _make_more_linear_data()
        pbe = ParameterBatchEvaluator(
            n_cores, eval_sets, False,
        )
        batch_results = pbe.evaluate_parameter_batch(
            param_evals, False, self._batch_restart,
        )
        raw_obj, total_obj, qoi = _unpack_evaluation(
            batch_results,
        )
        sr = StudyResults(
            record_qois=False, record_data=False,
        )
        _record_results(
            sr, param_evals, raw_obj, total_obj,
            qoi, False,
        )

        # -ped and -psd with nothing stored should not crash
        make_standard_plots(
            "x", block=False,
            plot_exp_data=True, plot_sim_data=True,
        )
        # Only objective plots, no eval plots
        self.assertEqual(len(plt.get_fignums()), 2)
        plt.close("all")

    def test_plots_no_crash_sim_data_flag_with_qois_disabled(self):
        """Requesting -psd when qois disabled but data recorded
        should still produce evaluation plots using raw data."""
        plt.close("all")
        n_eval = 5
        vals = np.linspace(-5, 2, n_eval)
        vals = np.concatenate((vals, [1.05]))
        param_evals = {}
        for index, v in enumerate(vals):
            pt = {'a': v, 'b': v}
            param_evals[f"eval.{index}"] = pt

        n_cores, model, eval_sets = _make_more_linear_data()
        pbe = ParameterBatchEvaluator(
            n_cores, eval_sets, False,
        )
        batch_results = pbe.evaluate_parameter_batch(
            param_evals, False, self._batch_restart,
        )
        raw_obj, total_obj, qoi = _unpack_evaluation(
            batch_results,
        )
        sr = StudyResults(record_qois=False)
        _record_results(
            sr, param_evals, raw_obj, total_obj,
            qoi, False,
        )

        # -psd -ped with data=True, qois=False
        make_standard_plots(
            "x", block=False,
            plot_exp_data=True, plot_sim_data=True,
        )
        # Should produce eval plots from raw data + obj plots
        self.assertEqual(len(plt.get_fignums()), 3)
        plt.close("all")

    def test_plots_created_show_selected_index(self):

        plt.close("all")
        n_eval = 5
        vals = np.linspace(-5, 2, n_eval)
        vals = np.concatenate((vals,[1.05]))
        param_evals = {}
        for index, v in enumerate(vals): 
            pt = {'a':v, 'b':v}         
            param_evals[f"eval.{index}"] = pt

        n_cores, model, eval_sets = _make_more_linear_data()
        pbe = ParameterBatchEvaluator(n_cores, eval_sets, False)
        batch_results = pbe.evaluate_parameter_batch(param_evals, False, self._batch_restart)
        raw_obj, total_obj, qoi = _unpack_evaluation(batch_results)
        sr = StudyResults()
        _record_results(sr, param_evals, raw_obj, total_obj, qoi, False)

        make_standard_plots("x", plot_id=2, block=False)
        self.assertEqual(len(plt.get_fignums()), 3)
        plt.close("all")

    def test_plots_selected_index_too_high(self):

        plt.close("all")
        n_eval = 5
        vals = np.linspace(-5, 2, n_eval)
        vals = np.concatenate((vals,[1.05]))
        param_evals = {}
        for index, v in enumerate(vals): 
            pt = {'a':v, 'b':v}         
            param_evals[f"eval.{index}"] = pt

        n_cores, model, eval_sets = _make_more_linear_data()
        pbe = ParameterBatchEvaluator(n_cores, eval_sets, False)
        batch_results = pbe.evaluate_parameter_batch(param_evals, False, self._batch_restart)
        raw_obj, total_obj, qoi = _unpack_evaluation(batch_results)
        sr = StudyResults()
        _record_results(sr, param_evals, raw_obj, total_obj, qoi, False)
        with self.assertRaises(ValueError):
            make_standard_plots("x", plot_id=20, block=False)
        
    def test_plots_selected_index_not_in_saved_results(self):

        plt.close("all")
        n_eval = 5
        vals = np.linspace(-5, 2, n_eval)
        vals = np.concatenate((vals,[1.05]))
        param_evals = {}
        for index, v in enumerate(vals): 
            pt = {'a':v, 'b':v}         
            param_evals[f"eval.{index}"] = pt

        n_cores, model, eval_sets = _make_more_linear_data()
        pbe = ParameterBatchEvaluator(n_cores, eval_sets, False)
        batch_results = pbe.evaluate_parameter_batch(param_evals, False, self._batch_restart)
        raw_obj, total_obj, qoi = _unpack_evaluation(batch_results)
        sr = StudyResults(results_save_frequency=5)
        _record_results(sr, param_evals, raw_obj, total_obj, qoi, False)
        with self.assertRaises(ValueError):
            make_standard_plots("x", plot_id=2, block=False)
        make_standard_plots("x", plot_id=5, block=False)
        self.assertEqual(len(plt.get_fignums()), 3)
        plt.close("all")

    def test_plots_created_show_no_idependent_fields(self):

        plt.close("all")
        n_eval = 5
        vals = np.linspace(-5, 2, n_eval)
        vals = np.concatenate((vals,[1.05]))
        param_evals = {}
        for index, v in enumerate(vals): 
            pt = {'a':v, 'b':v}         
            param_evals[f"eval.{index}"] = pt

        n_cores, model, eval_sets = _make_more_linear_data()
        pbe = ParameterBatchEvaluator(n_cores, eval_sets, False)
        batch_results = pbe.evaluate_parameter_batch(param_evals, False, self._batch_restart)
        raw_obj, total_obj, qoi = _unpack_evaluation(batch_results)
        sr = StudyResults()
        _record_results(sr, param_evals, raw_obj, total_obj, qoi, False)
        make_standard_plots(block=False)
        self.assertEqual(len(plt.get_fignums()), 3)
        plt.close("all")

    def test_plots_cleared(self):

        plt.close("all")
        n_eval = 25
        vals = np.linspace(-5, 2, n_eval)
        vals = np.concatenate((vals,[1.05]))
        param_evals = {}
        for index, v in enumerate(vals): 
            pt = {'a':v, 'b':v}         
            param_evals[f"eval.{index}"] = pt

        n_cores, model, eval_sets = _make_more_linear_data(npts=30)

        pbe = ParameterBatchEvaluator(n_cores, eval_sets, False)
        batch_results = pbe.evaluate_parameter_batch(param_evals, False, self._batch_restart)
        raw_obj, total_obj, qoi = _unpack_evaluation(batch_results)
        sr = StudyResults()
        _record_results(sr, param_evals, raw_obj, total_obj, qoi, False)

        plotter = _UserAutoPlotter("x")
        plotter.plot()
        eval_set = eval_sets[model]
        obj_name = list(eval_set._objective_sets[-1].objectives.values())[-1].name
        state_name = "matcal_default_state"
        eval_name = f"{model.name}_{obj_name}"
        obj_plot_filename = f"user_plots/objective_{eval_name}.pdf"
        best_eval_filename = f'user_plots/evaluation_best_{eval_name}_{state_name}.pdf'
        param_obj_filename = f"user_plots/parameter_objective_{eval_name}.pdf"
        total_obj_plot_filename = f"user_plots/total_objective.pdf"
        tot_param_obj_plot_filename = f"user_plots/parameter_total_objective.pdf"   
        self.assert_file_exists(best_eval_filename)
        self.assert_file_exists(total_obj_plot_filename)
        self.assert_file_exists(tot_param_obj_plot_filename)
        self.assertFalse(os.path.exists(obj_plot_filename))
        self.assertFalse(os.path.exists(param_obj_filename))
        

        plotter = _UserAutoPlotter("fields")
        plotter._clean_plot_dir()
        self.assertTrue(os.path.exists(f"user_plots"))
        self.assertFalse(os.path.exists(total_obj_plot_filename))
        self.assertFalse(os.path.exists(best_eval_filename))
        self.assertFalse(os.path.exists(tot_param_obj_plot_filename))

    def test_plots_multiple_eval_sets(self):

        plt.close("all")
        n_eval = 25
        vals = np.linspace(-5, 2, n_eval)
        vals = np.concatenate((vals,[1.05]))
        param_evals = {}
        for index, v in enumerate(vals): 
            pt = {'a':v, 'b':v}         
            param_evals[f"eval.{index}"] = pt

        n_cores, model, eval_sets_linear = _make_more_linear_data(npts=30)
        n_cores, model2, eval_sets_quadratic = _make_quadratic_data()

        eval_sets = eval_sets_linear
        eval_sets.update(eval_sets_quadratic)

        pbe = ParameterBatchEvaluator(n_cores, eval_sets_linear, False)
        batch_results = pbe.evaluate_parameter_batch(param_evals, False, self._batch_restart)
        raw_obj, total_obj, qoi = _unpack_evaluation(batch_results)
        sr = StudyResults()
        _record_results(sr, param_evals, raw_obj, total_obj, qoi, False)
        fields = ["x", "time"]
        plotter = _UserAutoPlotter(fields, plot_model_objectives=True)
        plotter.plot()
        eval_set = eval_sets_linear[model]
        obj_name = list(eval_set._objective_sets[-1].objectives.values())[-1].name
        state_name = "matcal_default_state"
        eval_name = f"{model.name}_{obj_name}"
        obj_plot_filename_mod1 = f"user_plots/objective_{eval_name}.pdf"
        param_obj_filename_mod1 = f"user_plots/parameter_model_objective_{eval_name}.pdf"
        eval_set2 = eval_sets_quadratic[model2]
        obj_name2 = list(eval_set2._objective_sets[-1].objectives.values())[-1].name
        eval_name2 = f"{model2.name}_{obj_name2}"
        obj_plot_filename_mod2 = f"user_plots/objective_{eval_name2}.pdf"
        param_obj_filename_mod2 = f"user_plots/parameter_model_objective_{eval_name2}.pdf"
        best_eval_filename = f'user_plots/evaluation_best_{eval_name2}_{state_name}.pdf'
        total_obj_plot_filename = f"user_plots/total_objective.pdf"
        tot_param_obj_plot_filename = f"user_plots/parameter_total_objective.pdf"   
        self.assert_file_exists(best_eval_filename)
        self.assert_file_exists(total_obj_plot_filename)
        self.assert_file_exists(tot_param_obj_plot_filename)
        self.assert_file_exists(obj_plot_filename_mod1)
        self.assert_file_exists(obj_plot_filename_mod2)
        self.assert_file_exists(param_obj_filename_mod1)
        self.assert_file_exists(param_obj_filename_mod2)

        plotter._clean_plot_dir()
        self.assertTrue(os.path.exists(f"user_plots"))
        self.assertTrue(len(glob(os.path.join("user_plots", "*.pdf"))) == 0)
        

class TestObjectivePlotJobsWithObjectivesDisabled(MatcalUnitTest):
    """Cover the RuntimeError guard paths in
    _ObjectiveProgressPlotJob and _ParameterModelObjectivePlotJob
    when objectives recording is disabled."""

    def setUp(self) -> None:
        super().setUp(__file__)
        self._batch_restart = BatchRestartNone(None, None)

    def _build_study_results_no_objectives(
        self,
    ) -> StudyResults:
        """Build a StudyResults with record_objectives=False."""
        n_eval = 5
        vals = np.linspace(-5, 2, n_eval)
        vals = np.concatenate((vals, [1.05]))
        param_evals = {}
        for index, v in enumerate(vals):
            pt = {'a': v, 'b': v}
            param_evals[f"eval.{index}"] = pt
        n_cores, _, eval_sets = _make_more_linear_data()
        pbe = ParameterBatchEvaluator(
            n_cores, eval_sets, False,
        )
        batch_results = pbe.evaluate_parameter_batch(
            param_evals, False, self._batch_restart,
        )
        raw_obj, total_obj, qoi = _unpack_evaluation(
            batch_results,
        )
        sr = StudyResults(record_objectives=False)
        _record_results(
            sr, param_evals, raw_obj, total_obj,
            qoi, False, in_progress_save=False,
        )
        return sr

    def test_objective_progress_skips_when_disabled(
        self,
    ) -> None:
        """Lines 230-234: early return when objectives
        not recorded."""
        plt.close("all")
        sr = self._build_study_results_no_objectives()
        job = _ObjectiveProgressPlotJob(
            plot_directory="user_plots",
        )
        job.plot(sr)
        plt.close("all")

    def test_param_model_objective_skips_when_disabled(
        self,
    ) -> None:
        """Lines 297-301: early return when objectives
        not recorded."""
        plt.close("all")
        sr = self._build_study_results_no_objectives()
        job = _ParameterModelObjectivePlotJob(
            plot_directory="user_plots",
        )
        job.plot(sr)
        plt.close("all")


class TestSubplotLengthInchesProperties(MatcalUnitTest):
    """Cover the subplot_length_inches abstract-property
    implementations that are never called in normal flow."""

    def setUp(self) -> None:
        super().setUp(__file__)

    def test_objective_progress_subplot_length(
        self,
    ) -> None:
        """Line 225."""
        job = _ObjectiveProgressPlotJob(
            plot_directory="user_plots",
        )
        self.assertEqual(job.subplot_length_inches, 4)

    def test_total_objective_subplot_length(
        self,
    ) -> None:
        """Line 272."""
        job = _TotalObjectiveProgressPlotJob(
            plot_directory="user_plots",
        )
        self.assertEqual(job.subplot_length_inches, 4)


class TestGetStudyResultsEvaluationIdsFallback(MatcalUnitTest):
    """Cover the AttributeError fallback in
    _get_study_results_evaluation_ids (lines 254-255)."""

    def setUp(self) -> None:
        super().setUp(__file__)

    def test_fallback_when_no_evaluation_ids(self) -> None:
        """An object without evaluation_ids should fall back
        to range(len(total_objective_history))."""
        fake_sr = SimpleNamespace(
            total_objective_history=[1.0, 0.5, 0.3],
        )
        result = _get_study_results_evaluation_ids(fake_sr)
        self.assertEqual(list(result), [0, 1, 2])


class TestGetCommonFieldsEdgeCases(MatcalUnitTest):
    """Cover the retry (line 368) and ValueError (line 370)
    paths in _get_common_fields."""

    def setUp(self) -> None:
        super().setUp(__file__)

    def _make_mock_qoi(
        self,
        field_names: list[str],
    ) -> MagicMock:
        """Build a mock QoI with given field names and dict
        keys."""
        mock = MagicMock()
        mock.field_names = field_names
        mock.keys.return_value = field_names
        return mock

    def test_retry_when_all_fields_excluded(self) -> None:
        """Line 368: first pass excludes everything, retry
        finds the fields without exclusion."""
        sim_qoi = self._make_mock_qoi(["x", "y"])
        exp_qoi = self._make_mock_qoi(["x", "y"])
        result = _get_common_fields(
            [sim_qoi], [exp_qoi], excluded_qois=["x", "y"],
        )
        self.assertIn("x", result)
        self.assertIn("y", result)

    def test_raises_when_no_common_fields(self) -> None:
        """Line 370: sim and exp share zero field names."""
        sim_qoi = self._make_mock_qoi(["alpha", "beta"])
        exp_qoi = self._make_mock_qoi(["gamma", "delta"])
        with self.assertRaises(ValueError):
            _get_common_fields([sim_qoi], [exp_qoi])


class TestGetIndexFallback(MatcalUnitTest):
    """Cover the AttributeError fallback for
    best_evaluation_id in get_index (lines 455-456)."""

    def setUp(self) -> None:
        super().setUp(__file__)
        self._batch_restart = BatchRestartNone(None, None)

    def test_get_index_without_evaluation_id(self) -> None:
        """When best_evaluation_id is missing, get_index
        should still return the best index."""
        n_eval = 3
        vals = np.linspace(-5, 2, n_eval)
        param_evals = {}
        for index, v in enumerate(vals):
            pt = {'a': v, 'b': v}
            param_evals[f"eval.{index}"] = pt
        n_cores, _, eval_sets = _make_more_linear_data()
        pbe = ParameterBatchEvaluator(
            n_cores, eval_sets, False,
        )
        batch_results = pbe.evaluate_parameter_batch(
            param_evals, False, self._batch_restart,
        )
        raw_obj, total_obj, qoi = _unpack_evaluation(
            batch_results,
        )
        sr = StudyResults()
        _record_results(
            sr, param_evals, raw_obj, total_obj,
            qoi, False, in_progress_save=False,
        )
        del sr._evaluation_ids
        job = _PlotEvaluationIdJob(
            plot_dir="user_plots", plot_id="best",
        )
        result = job.get_index(sr)
        expected = int(np.argmin(sr._total_objective_history))
        self.assertEqual(result, expected)


class TestPlotQoiListMarkerPop(MatcalUnitTest):
    """Cover line 879: the 'marker' kwarg pop branch in
    _plot_qoi_list."""

    def setUp(self) -> None:
        super().setUp(__file__)

    def test_marker_popped_when_less_than_10(self) -> None:
        """When both less_than_10_marker and marker kwarg are
        passed with <10 data points, marker should be popped
        and less_than_10_marker used instead."""
        plt.close("all")
        fig, ax = plt.subplots()
        mock_qoi = MagicMock()
        mock_qoi.__getitem__ = lambda self, k: [1, 2, 3]
        mock_qoi.__len__ = lambda self: 3
        job = _PlotEvaluationIdJob(
            plot_dir="user_plots", plot_id="best",
        )
        # Should not raise; marker='o' in kwargs is popped
        # and replaced by less_than_10_marker='x'
        job._plot_qoi_list(
            ax, [mock_qoi], "x", "y", "label",
            less_than_10_marker='x',
            marker='o', color='red',
        )
        plt.close("all")