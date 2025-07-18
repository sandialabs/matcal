from matcal.core.state import SolitaryState
from matcal.core.study_base import StudyResults, _record_results, _unpack_evaluation
import numpy as np
from glob import glob
import os

from matcal.core.parameter_batch_evaluator import (ParameterBatchEvaluator)
from matcal.core.plotting import (_NullPlotter, _UserAutoPlotter, 
                                  StandardAutoPlotter, 
                                  make_standard_plots)
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest

from matcal.core.tests.unit.test_parameter_batch_evaluator import (_make_more_linear_data, 
                                                                   _make_quadratic_data)


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
        import matplotlib.pyplot as plt
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
        batch_results = pbe.evaluate_parameter_batch(param_evals, False)
        raw_obj, total_obj, qoi = _unpack_evaluation(batch_results)
        sr = StudyResults()
        _record_results(sr, param_evals, raw_obj, total_obj, qoi, False)

        make_standard_plots("x", show=False)
        glob_search = "user_plots/*.pdf"
        plot_files = glob(glob_search)
        self.assertEqual(len(plot_files), 3)
        plt.close("all")

    def test_plots_bad_independent_field(self):
        import matplotlib.pyplot as plt
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
        batch_results = pbe.evaluate_parameter_batch(param_evals, False)
        raw_obj, total_obj, qoi = _unpack_evaluation(batch_results)
        sr = StudyResults()
        _record_results(sr, param_evals, raw_obj, total_obj, qoi, False)
        with self.assertRaises(ValueError):
            make_standard_plots("not a valid field", show=False)

    def test_plots_created_show(self):
        import matplotlib.pyplot as plt
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
        batch_results = pbe.evaluate_parameter_batch(param_evals, False)
        raw_obj, total_obj, qoi = _unpack_evaluation(batch_results)
        sr = StudyResults()
        _record_results(sr, param_evals, raw_obj, total_obj, qoi, False)

        make_standard_plots("x", block=False)
        self.assertEqual(len(plt.get_fignums()), 3)
        plt.close("all")

    def test_plots_created_show_exp_data(self):
        import matplotlib.pyplot as plt
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
        batch_results = pbe.evaluate_parameter_batch(param_evals, False)
        raw_obj, total_obj, qoi = _unpack_evaluation(batch_results)
        sr = StudyResults()
        _record_results(sr, param_evals, raw_obj, total_obj, qoi, False)

        make_standard_plots("x", block=False, plot_exp_data=True)
        self.assertEqual(len(plt.get_fignums()), 3)
        plt.close("all")

    def test_plots_created_show_data_no_qois_or_resids(self):
        import matplotlib.pyplot as plt
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
        batch_results = pbe.evaluate_parameter_batch(param_evals, False)
        raw_obj, total_obj, qoi = _unpack_evaluation(batch_results)
        sr = StudyResults(record_qois=False, record_residuals=False)
        _record_results(sr, param_evals, raw_obj, total_obj, qoi, False)

        make_standard_plots("x", block=False, plot_exp_data=True, plot_sim_data=True)
        self.assertEqual(len(plt.get_fignums()), 3)
        plt.close("all")

    def test_plots_created_show_sim_data(self):
        import matplotlib.pyplot as plt
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
        batch_results = pbe.evaluate_parameter_batch(param_evals, False)
        raw_obj, total_obj, qoi = _unpack_evaluation(batch_results)
        sr = StudyResults()
        _record_results(sr, param_evals, raw_obj, total_obj, qoi, False)

        make_standard_plots("x", block=False, plot_sim_data=True)
        self.assertEqual(len(plt.get_fignums()), 3)
        plt.close("all")

    def test_plots_created_show_sim_and_exp_data(self):
        import matplotlib.pyplot as plt
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
        batch_results = pbe.evaluate_parameter_batch(param_evals, False)
        raw_obj, total_obj, qoi = _unpack_evaluation(batch_results)
        sr = StudyResults()
        _record_results(sr, param_evals, raw_obj, total_obj, qoi, False)

        make_standard_plots("x", block=False, plot_sim_data=True, plot_exp_data=True)
        self.assertEqual(len(plt.get_fignums()), 3)
        plt.close("all")

    def test_plots_created_show_sim_and_exp_data_no_qois_no_resids(self):
        import matplotlib.pyplot as plt
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
        batch_results = pbe.evaluate_parameter_batch(param_evals, False)
        raw_obj, total_obj, qoi = _unpack_evaluation(batch_results)
        sr = StudyResults(record_qois=False, record_residuals=False)
        _record_results(sr, param_evals, raw_obj, total_obj, qoi, False)

        make_standard_plots("x", block=False, plot_sim_data=True, plot_exp_data=True)
        self.assertEqual(len(plt.get_fignums()), 3)
        plt.close("all")

    def test_plots_created_show_no_data_no_qois_no_resids(self):
        import matplotlib.pyplot as plt
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
        batch_results = pbe.evaluate_parameter_batch(param_evals, False)
        raw_obj, total_obj, qoi = _unpack_evaluation(batch_results)
        sr = StudyResults(record_qois=False, record_residuals=False, record_data=False)
        _record_results(sr, param_evals, raw_obj, total_obj, qoi, False)

        make_standard_plots("x", block=False, plot_sim_data=True, plot_exp_data=True)
        self.assertEqual(len(plt.get_fignums()), 2)
        plt.close("all")

    def test_plots_created_show_selected_index(self):
        import matplotlib.pyplot as plt
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
        batch_results = pbe.evaluate_parameter_batch(param_evals, False)
        raw_obj, total_obj, qoi = _unpack_evaluation(batch_results)
        sr = StudyResults()
        _record_results(sr, param_evals, raw_obj, total_obj, qoi, False)

        make_standard_plots("x", plot_id=2, block=False)
        self.assertEqual(len(plt.get_fignums()), 3)
        plt.close("all")

    def test_plots_selected_index_too_high(self):
        import matplotlib.pyplot as plt
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
        batch_results = pbe.evaluate_parameter_batch(param_evals, False)
        raw_obj, total_obj, qoi = _unpack_evaluation(batch_results)
        sr = StudyResults()
        _record_results(sr, param_evals, raw_obj, total_obj, qoi, False)
        with self.assertRaises(ValueError):
            make_standard_plots("x", plot_id=20, block=False)
        
    def test_plots_selected_index_not_in_saved_results(self):
        import matplotlib.pyplot as plt
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
        batch_results = pbe.evaluate_parameter_batch(param_evals, False)
        raw_obj, total_obj, qoi = _unpack_evaluation(batch_results)
        sr = StudyResults(results_save_frequency=5)
        _record_results(sr, param_evals, raw_obj, total_obj, qoi, False)
        with self.assertRaises(ValueError):
            make_standard_plots("x", plot_id=2, block=False)
        make_standard_plots("x", plot_id=5, block=False)
        self.assertEqual(len(plt.get_fignums()), 3)
        plt.close("all")

    def test_plots_created_show_no_idependent_fields(self):
        import matplotlib.pyplot as plt
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
        batch_results = pbe.evaluate_parameter_batch(param_evals, False)
        raw_obj, total_obj, qoi = _unpack_evaluation(batch_results)
        sr = StudyResults()
        _record_results(sr, param_evals, raw_obj, total_obj, qoi, False)
        make_standard_plots(block=False)
        self.assertEqual(len(plt.get_fignums()), 3)
        plt.close("all")

    def test_plots_cleared(self):
        import matplotlib.pyplot as plt
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
        batch_results = pbe.evaluate_parameter_batch(param_evals, False)
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
        import matplotlib.pyplot as plt
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
        batch_results = pbe.evaluate_parameter_batch(param_evals, False)
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
        
    