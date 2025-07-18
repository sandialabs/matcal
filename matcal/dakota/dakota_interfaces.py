from collections import OrderedDict
from matcal.core.data import convert_data_to_dictionary
from matcal.core.data_importer import FileData
from matcal.core.serializer_wrapper import matcal_save, matcal_load
from matcal.core.utilities import _find_smallest_rect
from matcal.dakota.dakota_constants import DAKOTA_MCMC_CHAIN_FILE, \
    DAKOTA_COVARIANCE_FILE, MATCAL_MCMC_CHAIN_FILE
import numpy as np
import os
import re
import copy

from matcal.core.logger import initialize_matcal_logger

logger = initialize_matcal_logger(__name__)


def read_dakota_mcmc_chain(chain_file, n_params):
    begin_parameter_column = 2
    chain_data = FileData(chain_file, file_type='csv', delimiter=None, 
          usecols=np.arange(begin_parameter_column,begin_parameter_column+n_params))
    return chain_data


def make_pairwise_plots(chain_file, plot_dir, downsample_rate):
    chain_data = matcal_load(chain_file)
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    _make_triangle_plot(plot_dir, chain_data, downsample_rate, pd, plt, sns)
    _make_density_plot(plot_dir, chain_data, downsample_rate, plt, sns)
    _make_chain_plot(plot_dir, chain_data, downsample_rate, plt)
    
    plt.show()

def _make_chain_plot(plot_dir, chain_data, downsample_rate, plt):
    n_parameters = len(chain_data.keys())
    short_axis_length, long_axis_length = _find_smallest_rect(n_parameters)
    fig, ax_set = plt.subplots(short_axis_length, long_axis_length, 
                               constrained_layout=True)
    
    for p_name, ax in zip(list(chain_data.keys()), ax_set.flatten()):
        ax.set_xlabel("Evaluation")
        ax.set_ylabel(p_name)
        full_chain_field_values = chain_data[p_name]
        n_eval  = len(full_chain_field_values)
        ax.plot(np.arange(0, n_eval, downsample_rate), full_chain_field_values[::downsample_rate])
    filename = f"{plot_dir}/chain_plot.png"
    plt.savefig(filename, dpi=400)

def _make_density_plot(plot_dir, chain_data, downsample_rate, plt, sns):
    n_parameters = len(chain_data.keys())
    short_axis_length, long_axis_length = _find_smallest_rect(n_parameters)
    fig, ax_set = plt.subplots(short_axis_length, long_axis_length, 
                               constrained_layout=True)
    for p_name, ax in zip(list(chain_data.keys()), ax_set.flatten()):
        ax.set_xlabel(p_name)
        ax.set_ylabel("Density[-]")
        sns.kdeplot(chain_data[p_name][::downsample_rate], ax=ax)
    filename = f"{plot_dir}/density_plot.png"
    plt.savefig(filename, dpi=400)

def _make_triangle_plot(plot_dir, chain_data, downsample_rate, pd, plt, sns):
    chain_dataframe = pd.DataFrame.from_dict(chain_data[::downsample_rate])
    pair_plot = sns.PairGrid(chain_dataframe, diag_sharey=False)
    pair_plot.map_diag(sns.kdeplot)
    pair_plot.map_lower(sns.kdeplot, fill=True, cmap="rocket_r")

    filename = f"{plot_dir}/triangle_plot.png"
    plt.savefig(filename, dpi=400)




class DakotaOutputReader(object):
  """
  Reads dakota log file to report results back to MatCal and the user. Will soon be deprecated for a new reader that
  reads in the HDF5 file. The text based reader is too error prone.
  """

  def __init__(self, dakota_filename):

    if not os.path.isfile(dakota_filename):
      raise FileNotFoundError("Unable to open {}".format(dakota_filename))

    self.dakota_file = dakota_filename

    self.info = None
    self.settings = None
    self.iterations = None
    self.results = None
    self._param_pattern = None
    self._eval_pattern = None

    self._current_line_number = 0
    self._total_lines = None
    self._lines = None


  def parse_calibration(self):
    with open(self.dakota_file, "r") as fdta:
      self._lines = fdta.read().split("\n")
      self._param_pattern = "Best parameters"
      self._best_obj_pattern = "Best objective"
      self._best_residual_pattern = "Best residual"

      self._eval_pattern = "Best evaluation ID"
      
      self._total_lines = len(self._lines)

      while self._current_line_number < self._total_lines:
        found_params = self._param_pattern in self._lines[self._current_line_number]
        if found_params:
          all_parameters = self._extract_parameters()
        found_params = False
        self._current_line_number+= 1

    return all_parameters

  def _find_key_phrase_in_lines(self, key_phrase):
    current_line = self._lines[self._current_line_number]
    while key_phrase not in current_line and self._current_line_number < self._total_lines:
      self._current_line_number += 1
      current_line = self._lines[self._current_line_number]

  def _find_residuals_or_objectives_in_lines(self):
    current_line = self._lines[self._current_line_number]
    while (self._best_obj_pattern not in current_line and self._best_residual_pattern not in current_line) and self._current_line_number < self._total_lines:
      self._current_line_number += 1
      current_line = self._lines[self._current_line_number]

  def _extract_parameters(self):
    self._current_line_number += 1
    parameters = OrderedDict()
    while self._lines[self._current_line_number].startswith(" ") and self._current_line_number < self._total_lines:
      value, label = self._lines[self._current_line_number].strip().split(" ") 
      parameters["best:"+label] = float(value)
      self._current_line_number += 1
    return parameters
  
  def _extract_objectives(self):
    self._current_line_number += 1
    objectives = []
    while self._lines[self._current_line_number].startswith(" ") and self._current_line_number < self._total_lines:
      objectives.append(float(self._lines[self._current_line_number].strip()))
      self._current_line_number += 1
    return objectives

  def _extract_eval_id(self):
    eval_id = None
    if self._eval_pattern in self._lines[self._current_line_number] and self._current_line_number < self._total_lines:
      eval_value = self._lines[self._current_line_number].split("Best evaluation ID")[-1].strip()
      try:
        eval_id = int(eval_value)
      except Exception:
        eval_id = eval_value
    return eval_id

  def parse_sobol(self):
    with open(self.dakota_file, "r") as fdta:
      lines = fdta.read().split("\n")
      line_nb = 0
      match = None
      pattern = re.compile(r"Global sensitivity indices")

      while not match:
        line_nb += 1
        match = pattern.match(lines[line_nb])

      line_nb += 3
      parameters = OrderedDict()
      while not lines[line_nb].startswith("<"):
        line = lines[line_nb]
        line_nb += 1
        if line.startswith("obj") or line.startswith("least_sq"):
          continue
        split_line = line.strip().split()
        if len(split_line)==0 or split_line[0] == "Main":
          continue
        main, total, label = line.split()
        if label not in parameters.keys():
          parameters[label] = [[float(main), float(total)]]
        else:
          parameters[label].append([float(main), float(total)])

      final_parameters = OrderedDict()
      for label, value in parameters.items():
        final_parameters[f"sobol:{label}"] = np.array(value)

    self.results = final_parameters
    return final_parameters


  def parse_pearson(self):
    with open(self.dakota_file, "r") as fdta:
      lines = fdta.read().split("\n")
      line_nb = 0
      match = None
      pattern = re.compile(r"Partial Correlation Matrix between input and output:")

      while not match:
        line_nb += 1
        match = pattern.match(lines[line_nb])

      line_nb += 2
      parameters = OrderedDict()
      while lines[line_nb].startswith(" "):
        split_line = lines[line_nb].split()
        label = split_line[0]
        coeffs = split_line[1:]
        parameters[f'pearson:{label}'] = np.array(coeffs, dtype=float)
        line_nb += 1

    self.results = parameters
    return self.results


  def process_chain(self, chain_file=DAKOTA_MCMC_CHAIN_FILE):
    chain_data = read_dakota_mcmc_chain(chain_file, self.nparameters)
    matcal_save(MATCAL_MCMC_CHAIN_FILE, chain_data)
    indiv_stats_to_run = OrderedDict()
    indiv_stats_to_run['mean'] = np.mean
    indiv_stats_to_run['stddev'] = np.std
    chain_stats = OrderedDict()
    for stat_name in indiv_stats_to_run:
      chain_stats[stat_name] = OrderedDict()
    all_parmaeters = []
    chain_data = convert_data_to_dictionary(chain_data)
    p_order = []
    p_order_name = ""
    for p_name, p_values in chain_data.items():
      all_parmaeters.append(p_values)
      p_order.append(p_name)
      if len(p_order_name) < 1:
        p_order_name += p_name
      else:
        p_order_name += f":{p_name}"
      for stat_name, stat_func in indiv_stats_to_run.items():
        chain_stats[stat_name][p_name] = stat_func(p_values)
    
    all_parmaeters = np.array(all_parmaeters).T
    cor = correlate(all_parmaeters)
    co_var = self._calculate_covariance(all_parmaeters)


    chain_stats[f'pearson'] = OrderedDict({p_order_name:cor})
    chain_stats[f'covariance'] = OrderedDict({p_order_name:co_var})

    self._calculate_covariance(all_parmaeters)
    self.results.update(chain_stats)



  def _calculate_covariance(self, all_parmaeters):
      if self.nparameters > 1:
        co_var  = np.cov(all_parmaeters, rowvar=False)
        np.savetxt(DAKOTA_COVARIANCE_FILE, co_var)
      else:
        co_var= np.array([[1]])
      return co_var


  def parse_bayes(self):
    with open(self.dakota_file, "r") as fh:
      lines = fh.readlines()
      pattern = re.compile(r"<<<<< Best parameters ")
      match = None
      line_nb = 0
      while not match:
        line_nb += 1
        match = pattern.match(lines[line_nb])
      if not match:
        raise EOFError
      line_nb += 1
      parameters = OrderedDict()
      while lines[line_nb].startswith(" "):
        items = lines[line_nb].split()
        value = items[0]
        label = items[1]
        parameters[label] = float(value)
        line_nb += 1
    self.nparameters = len(parameters)
    self.results = OrderedDict()
    self.results["MAP"] = parameters
    self.process_chain()
    self.report_bayes()
    
    study_results_format = OrderedDict()
    for stat_name, param_results in self.results.items():
      for param_name, param_value in param_results.items():
        out_name = f"{stat_name}:{param_name}"
        study_results_format[out_name] = param_value
    self.results = study_results_format
    return self.results

  def report_bayes(self, fname="bayes_results.txt"):
    MAPs = self.results["MAP"]
    AVs = self.results["mean"]
    SDs = self.results["stddev"]
    with open(fname, "w") as fh:
      msg = "# MAP mean stddev"
      fh.write(msg+"\n")
      for p in MAPs:
        msg = "{0:12s} {1:12.6e} {2:12.6e} {3:12.6e}".format(
            p, MAPs[p], AVs[p], SDs[p])
        fh.write(msg+"\n")
      fh.close()

def correlate(data):
  pearson_coef = np.corrcoef(data.T)
  return pearson_coef

