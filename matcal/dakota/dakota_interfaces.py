import os
import re
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from matcal.core.data import convert_data_to_dictionary
from matcal.core.data_importer import FileData
from matcal.core.logger import initialize_matcal_logger
from matcal.core.serializer_wrapper import matcal_load, matcal_save
from matcal.core.utilities import _find_smallest_rect
from matcal.dakota.dakota_constants import (
    DAKOTA_COVARIANCE_FILE,
    DAKOTA_MCMC_CHAIN_FILE,
    MATCAL_MCMC_CHAIN_FILE,
)

logger = initialize_matcal_logger(__name__)


def read_dakota_mcmc_chain(chain_file, n_params):
    begin_parameter_column = 2
    chain_data = FileData(chain_file, file_type='csv', delimiter=None, 
          usecols=np.arange(begin_parameter_column,begin_parameter_column+n_params))
    return chain_data


def make_pairwise_plots(chain_file, plot_dir, downsample_rate):
    chain_data = matcal_load(chain_file)
    import matplotlib.pyplot as plt
    import pandas as pd
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

@dataclass(frozen=True)
class DakotaTermination:
    """
    Termination information reported by a Dakota solver.

    :param success: True if the solver reports convergence, False if it
        stopped for another reason, and None if no termination text was found.
    :param status: the solver's exit code or status string, if it reports one.
    :param message: a human readable termination message.
    """

    success: bool | None
    status: int | str | None
    message: str


_TERMINATION_NOT_FOUND = DakotaTermination(
    None, None, "Termination reason not found in the Dakota or solver output."
)

_NPSOL_INFORM_MEANINGS = {
    0: "optimal solution found",
    1: "optimal solution found, but not to the requested accuracy",
    2: "no feasible point for the linear constraints",
    3: "no feasible point for the nonlinear constraints",
    4: "iteration limit reached",
    6: "current point cannot be improved",
    7: "derivative check failed",
    9: "invalid input parameter",
}

_NL2SOL_FAILURES = ("FALSE", "SINGULAR", "LIMIT")

_ROL_SUCCESS_STATUSES = ("Converged", "Step Tolerance Met")


def _last_match(pattern: str, text: str) -> re.Match | None:
    matches = list(re.finditer(pattern, text, flags=re.MULTILINE))
    return matches[-1] if matches else None


def _parse_nl2sol(text: str) -> DakotaTermination | None:
    match = _last_match(r"\*{5} ([A-Z][A-Z\- ]*?) \*{5}", text)
    if match is None:
        return None
    reason = match.group(1)
    success = "CONVERGENCE" in reason and not any(
        failure in reason for failure in _NL2SOL_FAILURES
    )
    return DakotaTermination(success, None, f"NL2SOL: {reason}")


def _parse_npsol(text: str) -> DakotaTermination | None:
    match = _last_match(r"(NPSOL|NLSSOL) exits with INFORM code = (-?\d+)", text)
    if match is None:
        return None
    solver, code = match.group(1), int(match.group(2))
    meaning = _NPSOL_INFORM_MEANINGS.get(code, "see the NPSOL manual")
    return DakotaTermination(code == 0, code, f"{solver} INFORM {code}: {meaning}")


def _parse_optpp(text: str) -> DakotaTermination | None:
    default_out = r"Return code\s+=\s+(-?\d+) \((.*)\)"
    dakota_out = r"Return Code\s+(-?\d+)\s*\n.*\n(Algorithm[^\n]*)"
    match = _last_match(default_out, text) or _last_match(dakota_out, text)
    if match is None:
        return None
    code = int(match.group(1))
    return DakotaTermination(code > 0, code, f"OPT++: {match.group(2).strip()}")


def _parse_rol(text: str) -> DakotaTermination | None:
    match = _last_match(r"Optimization Terminated with Status: ([^\n]+)", text)
    if match is None:
        return None
    status = match.group(1).strip()
    success = status in _ROL_SUCCESS_STATUSES
    return DakotaTermination(success, status, f"ROL: {status}")


def _parse_dot(text: str) -> DakotaTermination | None:
    match = _last_match(r"^\s*TERMINATION CRITERIA\s*\n\s*([^\n]*\S)", text)
    if match is None:
        return None
    reason = match.group(1).strip()
    success = "CONVERGENCE CRITERION WAS MET" in reason
    return DakotaTermination(success, None, f"DOT: {reason}")


def _parse_conmin(text: str) -> DakotaTermination | None:
    match = _last_match(r"^\s*TERMINATION CRITERION\s*\n\s*([^\n]*\S)", text)
    if match is None:
        return None
    reason = match.group(1).strip()
    success = "ITMAX" not in reason
    return DakotaTermination(success, None, f"CONMIN: {reason}")


_TERMINATION_PARSERS: dict[str, Callable[[str], DakotaTermination | None]] = {
    "nl2sol": _parse_nl2sol,
    "npsol_": _parse_npsol,
    "nlssol_": _parse_npsol,
    "optpp_": _parse_optpp,
    "rol": _parse_rol,
    "dot_": _parse_dot,
    "conmin_": _parse_conmin,
}


def _select_termination_parser(
    method: str,
) -> Callable[[str], DakotaTermination | None] | None:
    for prefix, parser in _TERMINATION_PARSERS.items():
        if method.startswith(prefix):
            return parser
    return None


def parse_termination_text(text: str, method: str) -> DakotaTermination:
    """
    Extract the termination reason of a Dakota gradient solver from its output.

    :param text: the combined Dakota and solver output to search.
    :param method: the Dakota method keyword, used to select the solver's
        termination message format.
    :return: the last termination message found for the solver, or a
        termination with ``success`` None if none is found.
    """
    parser = _select_termination_parser(method)
    termination = parser(text) if parser is not None else None
    return termination if termination is not None else _TERMINATION_NOT_FOUND


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

  def parse_termination(self, method: str,
                        additional_texts: Iterable[str] = ()) -> DakotaTermination:
    """
    Return the solver termination reason found in the Dakota output file
    and any additional solver output, such as captured standard output.

    :param method: the Dakota method keyword.
    :param additional_texts: other solver output to search after the Dakota
        output file.
    :return: the termination information.
    """
    with open(self.dakota_file, "r") as file_handle:
      texts = [file_handle.read(), *additional_texts]
    return parse_termination_text("\n".join(texts), method)

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

