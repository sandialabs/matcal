"""
This module contains base classes used for plotting. It also 
includes user facing functions for plotting and retrieving results from 
serialized archive files.
"""

from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

from matcal.core.constants import (EVALUATION_EXTENSION, IN_PROGRESS_RESULTS_FILENAME, 
                                   MATCAL_WORKDIR_STR)
from matcal.core.data import DataCollection
from matcal.core.logger import initialize_matcal_logger
from matcal.core.serializer_wrapper import matcal_load


logger = initialize_matcal_logger(__name__)
MATCAL_PLOT_DIR = "plots"
MATCAL_USER_PLOT_DIR = "user_plots"
LOG_TOL = 1e-14


def clean_plot_dir(plot_dir):
    if os.path.exists(plot_dir):
        shutil.rmtree(plot_dir)
    os.mkdir(plot_dir)


class _AutoPlotterBase(ABC):
    """
    Base class for automatic plotting routines. 
    Users should not need to interact with AutoPlotterBase in order to run a parameter 
    study or plot study results.
    """

    @abstractmethod
    def _get_plot_jobs(self)->list:
        """"""

    def __init__(self, plot_dir = MATCAL_PLOT_DIR, plot_id='best', 
                plot_exp_data=False, plot_sim_data=False):
        self._plot_dir = plot_dir
        self._plot_id=plot_id
        self._plot_exp_data = plot_exp_data
        self._plot_sim_data = plot_sim_data

    def plot(self):
        clean_plot_dir(self._plot_dir)
        study_results = matcal_load(IN_PROGRESS_RESULTS_FILENAME+'.'+EVALUATION_EXTENSION)
        plot_jobs = self._get_plot_jobs()
        for plot_job in plot_jobs:
            plot_job.plot(study_results)


class _NullPlotter:
    """
    Plotter which does not plot.
    """
    def plot(self):
        pass
    

def make_standard_plots(*independent_fields, show=True, block=True,
                        plot_dir=MATCAL_USER_PLOT_DIR, plot_id='best', 
                        plot_model_objectives=False, plot_exp_data=False, 
                        plot_sim_data=False)->None:
    """
    Makes a series of standardized plots based on the best parameter evaluation, and the 
    evaluation history of the study. 

    :param independent_fields: Optional parameters to pass to 
        specify the name of field to be used
        as the independent field for the purposes of plotting. 
        Multiple field names can be passed if
        different data sets use different independent variables. 
        Priority is given to fields specified 
        earlier in the passed fields. If not passed an array of plots will
        be generated to cover all possible plotting combinations. 
    :type independent_field: list(str)

    :param show: Specify where to show the plots
    :type show: bool

    :param block: stops Python from executing code after the plot 
        figure is created. Follow-on code
        will not execute until the figure is closed.
        Default is to block (e.g. block=True).
    :type block: bool

    :param plot_dir: specify a folder to output the plot files to
    :type plot_dir: str

    :param plot_id: evaluation id number to plot. Default is 'best' only other 
        valid options are evaluation ids that have completed and have been saved.
    :type param_index: int

    :param plot_model_objectives: Plot the objectives versus parameter and 
        evaluation indices for each model. 
    :type plot_model_objectives: bool

    :param plot_exp_data: Plot the experimental data instead of the QoIs extracted
        from the data. 
    :type plot_exp_data: bool

    :param plot_sim_data: Plot the simulation data instead of the QoIs extracted
        from the data.  
    :type plot_sim_data: bool
    """
    plotter = _UserAutoPlotter(independent_fields, 
                               plot_dir=plot_dir, 
                               plot_id=plot_id, 
                               plot_model_objectives=plot_model_objectives, 
                               plot_exp_data=plot_exp_data, plot_sim_data=plot_sim_data)
    plotter.plot()
    if show:
        plt.show(block=block)


class StandardAutoPlotter(_AutoPlotterBase):
    """
    Class used to create automatic plots at the end of an evaluation set.
    """
    def _get_plot_jobs(self)->list:
        plot_jobs = [_ObjectiveProgressPlotJob(plot_directory=self._plot_dir)]
        plot_jobs += [_TotalObjectiveProgressPlotJob(plot_directory=self._plot_dir)]
        plot_jobs += [_ParameterModelObjectivePlotJob(plot_directory=self._plot_dir)]
        plot_jobs += [_PlotEvaluationIdJob(plot_dir=self._plot_dir, plot_id="best", )]
        plot_jobs += [_ParameterTotalObjectivePlotJob(plot_directory=self._plot_dir)]
        return plot_jobs


class _UserAutoPlotter(_AutoPlotterBase):
    """
    Class wrapped by :func:`matcal.core.plotting.make_standard_plots` to generate plots.
    Users should not need to interact with _UserAutoPlotter.
    """

    def __init__(self, independent_fields=(), plot_dir=MATCAL_USER_PLOT_DIR, 
                 plot_id='best', plot_model_objectives=True, plot_exp_data=False, 
                 plot_sim_data=False):
        super().__init__(plot_dir, plot_id, plot_exp_data, plot_sim_data)
        self._independent_fields = independent_fields
        self._plot_model_objectives = plot_model_objectives

    def _clean_plot_dir(self):
        clean_plot_dir(self._plot_dir)

    def _get_plot_jobs(self)->list:
        jobs = [_TotalObjectiveProgressPlotJob(plot_directory=self._plot_dir)]
        jobs += [_PlotEvaluationIdJob(plot_dir=self._plot_dir, 
                                         plot_id=self._plot_id, 
                                         indep_fields=self._independent_fields, 
                                         plot_exp_data=self._plot_exp_data, 
                                         plot_sim_data=self._plot_sim_data)]
        jobs += [_ParameterTotalObjectivePlotJob(plot_directory=self._plot_dir)]
        if self._plot_model_objectives:
            jobs += [_ObjectiveProgressPlotJob(plot_directory=self._plot_dir)]
            jobs += [_ParameterModelObjectivePlotJob(plot_directory=self._plot_dir)]
        return jobs
    

class _PlotJobBase(ABC):
    """
    Interface for creating a new automatic plotting plugin.
    Users should not need to interact with PlotJobBase.
    """

    @property
    @abstractmethod
    def filename_root(self):
        """"""

    @property
    @abstractmethod
    def subplot_length_inches(self):
        """"""

    @abstractmethod
    def plot(self, study_results):
        """"""

    def __init__(self, plot_directory):
        self._export_file_root = f"{plot_directory}/{self.filename_root}"

    def _set_up_figure_and_axis(self, n_row, n_col, figname, sharey=False):
        fig, ax_set = plt.subplots(n_row, n_col, num=figname, sharey=sharey, 
                                   constrained_layout=True)
        fig.set_size_inches(self.subplot_length_inches*n_col, self.subplot_length_inches*n_row)
        if not isinstance(ax_set, np.ndarray):
            ax_set = np.array([ax_set])
        return fig, ax_set

    def _export(self, filename):
        plt.savefig(filename, dpi=300)


class _ObjectiveProgressPlotJob(_PlotJobBase):
    """
    Plotting plugin that plots objective values over the history of the parameter study. 
    Users should not need to interact with this plugin, it is wrapped by make_standard_plots.

    This class is intended to provide a convergence history for calibration studies. 
    """
    @property
    def filename_root(self):
        return "objective_"

    @property
    def subplot_length_inches(self):
        return 4

    def plot(self,  study_results):
        if len(study_results.evaluation_sets) > 1:
            for eval_set_name in study_results.evaluation_sets:
                model_name, obj_name = study_results.decompose_evaluation_name(eval_set_name)
                objective_history = study_results.get_evaluation_set_objectives(model_name, 
                                                                                obj_name)
                plt.figure(f"{model_name}:\n {obj_name}", constrained_layout=True)
                objs_to_plot = np.maximum(objective_history, LOG_TOL)
                eval_ids = _get_study_results_evaluation_ids(study_results)
                plt.semilogy(eval_ids, objs_to_plot)
                plt.xlabel("evaluation id")
                plt.ylabel(f"{obj_name}")
                model_name, obj_name = study_results.decompose_evaluation_name(eval_set_name)
                plt.title(f"{model_name}:\n{obj_name}")
                self._export(self._export_file_root+f"{model_name}_{obj_name}.pdf")


def _get_study_results_evaluation_ids(study_results):
    try:
        eval_ids = study_results.evaluation_ids
    except AttributeError:
        eval_ids = range(len(study_results.total_objective_history))
    return eval_ids


class _TotalObjectiveProgressPlotJob(_PlotJobBase):
    """
    Plotting plugin that plots objective values over the history of the parameter study. 
    Users should not need to interact with this plugin, it is wrapped by make_standard_plots.

    This class is intended to provide a convergence history for calibration studies. 
    """
    @property
    def filename_root(self):
        return "total_objective"

    @property
    def subplot_length_inches(self):
        return 4

    def plot(self,  study_results):
        plt.figure("total objective", constrained_layout=True)
        objs_to_plot = np.maximum(study_results.total_objective_history, LOG_TOL)
        eval_ids = _get_study_results_evaluation_ids(study_results)
        plt.semilogy(eval_ids, objs_to_plot)
        plt.xlabel("evaluation id")
        plt.ylabel("total objective")
        self._export(self._export_file_root+".pdf")


class _ParameterModelObjectivePlotJob(_PlotJobBase):
    """"""
    @property
    def filename_root(self):
        return "parameter_model_objective_"

    @property
    def subplot_length_inches(self):
        return 5
    
    def plot(self, study_results):
        if len(study_results.evaluation_sets) > 1:
            parameter_names = list(study_results.parameter_history.keys())
            n_param = len(parameter_names)
            for model_name in study_results.models_in_results:
                objs = study_results.get_objectives_for_model(model_name)
                fig, ax_set = self._set_up_figure_and_axis(len(objs), n_param, 
                                f"objective vs parameters: {model_name}", sharey=True)
                for obj_index, obj_name in enumerate(objs):
                    objs = study_results.get_evaluation_set_objectives(model_name, 
                                                                    obj_name)
                    for param_index, param_name in enumerate(parameter_names):
                        ax = _lookup_ax(ax_set, obj_index, param_index)

                        x_vals = study_results.parameter_history[param_name]
                        y_vals = np.maximum(objs, LOG_TOL)
                        eval_ids = _get_study_results_evaluation_ids(study_results)
                        sc = ax.scatter(x_vals, y_vals, c=eval_ids)
                        ax.set_xlabel(f"{param_name}")
                        ax.set_yscale('log')
                        if param_index == 0:
                            ax.set_ylabel(f"{obj_name}")
                cbar = fig.colorbar(sc,ax=ax_set)
                cbar.set_label("evaluation id")
                self._export(self._export_file_root+f"{model_name}_{obj_name}"+".pdf") 
                        

class _ParameterTotalObjectivePlotJob(_PlotJobBase):
    """"""
    @property
    def filename_root(self):
        return "parameter_total_objective"

    @property
    def subplot_length_inches(self):
        return 5
    
    def plot(self, study_results):
        parameter_names = list(study_results.parameter_history.keys())
        n_param = len(parameter_names)
        objs = study_results.total_objective_history
        fig, ax_set = self._set_up_figure_and_axis(1, n_param, 
                            f"total objective vs parameters", sharey=True)
        for param_index, param_name in enumerate(parameter_names):
            ax = ax_set[param_index]
            x_vals = study_results.parameter_history[param_name]
            y_vals = np.maximum(objs, LOG_TOL)
            eval_ids = _get_study_results_evaluation_ids(study_results)
            sc = ax.scatter(x_vals, y_vals, c=eval_ids)
            ax.set_xlabel(f"{param_name}")
            ax.set_yscale('log')
            if param_index == 0:
                ax.set_ylabel(f"total objective")
        cbar = fig.colorbar(sc, ax=ax_set)
        cbar.set_label("evaluation id")
        self._export(self._export_file_root+".pdf") 


def _get_common_fields(sim_qoi_list, exp_qois_list, excluded_qois=[], attempts=None):
    fields = list(sim_qoi_list[0].keys())
    common_fields = []
    for field in fields:
        for exp_qois in exp_qois_list:
            if (field in exp_qois.field_names and _not_already_added(common_fields, field)
                and _not_excluded(excluded_qois, field)):
                common_fields.append(field)
    if len(common_fields) == 0 and attempts is None:
        common_fields = _get_common_fields(sim_qoi_list, exp_qois_list, attempts=1)
    elif len(common_fields) == 0 and attempts is not None:
        raise ValueError("Outputs being plotted have no common fields.")
    return common_fields


def _not_already_added(common_dofs, dof):
    return dof not in common_dofs


def _not_excluded(excluded_qois, dof):
    return dof not in excluded_qois


class _PlotEvaluationIdJob(_PlotJobBase):
    
    @property           
    def filename_root(self):
        return f"evaluation_{self.plot_id}"

    @property
    def subplot_length_inches(self):
        return 10./3.
    
    def __init__(self, plot_dir, plot_id, indep_fields=(), plot_exp_data=False,
                 plot_sim_data=False):
        self.plot_id = plot_id
        super().__init__(plot_dir)
        self.x_fields = indep_fields
        self._plot_exp_data=plot_exp_data
        self._plot_sim_data=plot_sim_data
        self._exp_label = None
        self._sim_label = None
        self.set_labels()

    def set_labels(self):
        if self._plot_exp_data:
            self._exp_label = "exp data"
        else:
            self._exp_label = "exp qois"
        if self._plot_sim_data:
            self._sim_label = "sim data"
        else:
            self._sim_label = "sim qois"
        
    def get_index(self, study_results):
        if self.plot_id == "best":
            best_index = study_results.best_evaluation_index
            best_obj = study_results.best_total_objective
            try:
                best_id = study_results.best_evaluation_id
                logger.info(f"Best evaluation: {MATCAL_WORKDIR_STR}.{best_id}\n"
                         f"Best objective: {best_obj}")
            except AttributeError:
                logger.info(f"Best evaluation index: {best_index}\n"
                         f"Best objective: {best_obj}")
            return best_index
        else: 
            num_evals = study_results.number_of_evaluations
            if self.plot_id > num_evals:
                raise ValueError("Invalid evaluation index requested. Only "
                                 f"{num_evals} have been performed")
            eval_ids = _get_study_results_evaluation_ids(study_results)
            if self.plot_id in eval_ids:
                index  = int(np.where(np.array(eval_ids) == self.plot_id)[0][0])
            else:
                raise ValueError(f"Evaluation id {self.plot_id} is not stored in the results. "
                                 f"Ids {eval_ids} have been saved.")
            obj = study_results.total_objective_history[index]
            logger.info(f"Plotting evaluation: {MATCAL_WORKDIR_STR}.{self.plot_id}\n"
                         f"Objective: {obj}")
            return index

    def _get_states(self, study_results, eval_set_name, model_name, index):
        try:
            qois = study_results.qoi_history[eval_set_name]
            sim_hist = study_results.simulation_history[model_name]
            if qois.simulation_qois:
                return qois.simulation_qois[index].states
            elif sim_hist:
                return sim_hist.states
        except KeyError:
            logger.warning("No simulation data or QoIs to plot. Skipping")
            return {}

    def plot(self, study_results):
        index = self.get_index(study_results)
        for eval_set_name in study_results.evaluation_sets:
            model_name, obj_name = study_results.decompose_evaluation_name(eval_set_name)
            states = self._get_states(study_results, eval_set_name, model_name, index)
            for state in states.values():
                state_exp_results, state_sim_results = self.get_results_to_plot(study_results, 
                                                                               eval_set_name, 
                                                                               model_name, state, 
                                                                               index)
                fig_name = f"{model_name} {obj_name} {state.name}"
                y_qois = _get_common_fields(state_sim_results, state_exp_results, self.x_fields)
                x_qois = self._determine_x_qois(y_qois, state_sim_results, state_exp_results)

                fig, ax_set = self._set_up_figure_and_axis(len(x_qois), len(y_qois), 
                                                           figname=fig_name)
                for x_idx, x_qoi in enumerate(x_qois):
                    for y_idx, y_qoi in enumerate(y_qois):
                        ax = _lookup_ax(ax_set, x_idx, y_idx)
                        self._plot_qois(ax, x_qoi, y_qoi, state_sim_results, 
                                        state_exp_results)
                self._export(self._export_file_root+"_"+fig_name.replace(" ", "_")+".pdf")

    def get_results_to_plot(self, study_results, eval_set_name, model_name, state, index):
        qoi_hist = study_results.qoi_history[eval_set_name]
        if self._plot_exp_data:
            exp_results = qoi_hist.experiment_data[state]
        else:
            exp_results = qoi_hist.experiment_qois[state]
        if self._plot_sim_data:
            sim_hist = study_results.simulation_history[model_name]
            sim_results = [sim_hist[state][index]]
        else:
            sim_results = qoi_hist.simulation_qois[index][state]
        return exp_results, sim_results

    def _determine_x_qois(self, common_qois, state_sim_qoi_list, state_exp_qoi_list):
        if len(self.x_fields) < 1 or self.x_fields is None:
            return common_qois
        else:
            selected_field = self._select_indep_field(state_sim_qoi_list, 
                                                      state_exp_qoi_list)
            return selected_field

    def _plot_qois(self, ax, x_label, y_label, sim_qois, exp_qois):
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        self._plot_qoi_list(ax, exp_qois, x_label, y_label,  self._exp_label,
                            linestyle='-', 
                            marker='o', color="#bdbdbd")
        self._plot_qoi_list(ax, sim_qois, x_label, y_label, self._sim_label,
                            linestyle='-',
                            less_than_10_marker='x', color="tab:blue")
        ax.legend()

    def _plot_qoi_list(self, ax, qoi_list, x_label, y_label, 
                       legend_label, less_than_10_marker=None,
                        *args, **kwargs):
        for idx, qois in enumerate(qoi_list):
            markevery_percentage = self._set_markevery(qois[x_label])

            if idx == 0:
                label=legend_label
            else:
                label=None   
            if len(qois) < 10 and less_than_10_marker is not None: 
                if "marker" in kwargs:
                    kwargs.pop("marker")
                ax.plot(qois[x_label], qois[y_label], label=label, 
                        markevery=markevery_percentage, marker=less_than_10_marker, *args, **kwargs)
            else:
                ax.plot(qois[x_label], qois[y_label], label=label, *args, **kwargs)

    def _set_markevery(self, qoi_values):
        if len(qoi_values) < 20:
            return 1
        else:
            return int(len(qoi_values)/20)

    def _select_indep_field(self, sim_qoi_list, exp_qoi_list):
        x_field = None
        sim_keys = list(sim_qoi_list[0].field_names)
        for potential_field in self.x_fields:
            if potential_field in sim_keys:
                for exp_qoi in exp_qoi_list:
                    if potential_field in exp_qoi.keys():
                        x_field = potential_field
                        break
        if x_field is None:
            error_msg = get_independent_field_not_found_err_msg(self.x_fields, 
                                                                sim_keys)
            raise ValueError(error_msg)
        return [x_field]


def get_independent_field_not_found_err_msg(indep_fields, exp_fields):
    message = "Independent field not found for plotting. Potential fields supplied:\n"
    message += f"{indep_fields}\n"
    message += "Fields in experimental data:\n"
    message += f"{exp_fields}"
    return message
        

def _lookup_ax(ax_set, x_idx, y_idx):
    if ax_set.ndim < 2:
        return ax_set[y_idx]
    else:
        return ax_set[x_idx, y_idx]
