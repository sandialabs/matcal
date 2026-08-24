from abc import ABC, abstractmethod
from collections import OrderedDict
from numbers import Integral, Real
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score
from typing import Any, Callable, List, Optional


from matcal.core.data import convert_dictionary_to_data
from matcal.core.logger import initialize_matcal_logger
from matcal.core.object_factory import BasicIdentifier
from matcal.core.serializer_wrapper import matcal_save
from matcal.core.state import State
from matcal.core.utilities import (check_value_is_nonempty_str, 
                                   check_item_is_correct_type, 
                                   _find_smallest_rect, 
                                   check_value_is_bool)


logger = initialize_matcal_logger(__name__)


class _DoNothingDataTransformer:
    def inverse_transform(self, source_data):
        return source_data

    def transform(self, source_data):
        return source_data

class _VarianceDecomposition:
    
    def __init__(self, goal_variance):
        self._goal_variance = goal_variance
        
    def generate(self, source_data, make_log_scale, logger_on=True):
        return _convert_data_and_make_basis(source_data, self._goal_variance, make_log_scale, 
                                            logger_on)
    

class _ReconstructionDecomposition:
    
    def __init__(self, reconstruction_tol:float):
        self._reconstruction_tol = reconstruction_tol
        
    def generate(self, source_data, make_log_scale, logger_on=True):
        return _tune_data_decomposition(source_data, make_log_scale,  self._reconstruction_tol, 
                                        logger_on=logger_on)


class SurrogateGenerator:
    """
    This class is responsible for taking source data and a parameter set 
    and generating an efficient surrogate 
    for predicting probe based quantities of interest. The generator uses
    Principal Component Analysis(PCA) to generate an efficient representation 
    of the data and then trains 
    a predictor in the latent space identified by the PCA. 
    To perform these calculations sklearn is 
    leveraged to perform the correct scaling, PCA, and predictor training required. 
    """

    def __init__(self, evaluation_information, interpolation_field=None, 
                interpolation_locations=200, 
                training_fraction=.8, surrogate_type = "PCA Multiple Regressors", 
                regressor_type="Gaussian Process", test_eval_info=None, **regressor_kwargs):
        """
        :param evaluation_information: A container of the relevant information used 
            to form a surrogate from MatCal study data.    
        :type evaluation_information: :class:`~matcal.core.study_base.StudyResults`,
            :class:`~matcal.core.study_base.StudyBase`, or dict

        :param training_fraction: What fraction of the source data to use as training data. 
            Value should be 0 < training_fraction <= 1. If training_fraction == 1, 
            test_eval_info must be provided.
        :type training_fraction: float

        :param interpolation_field: the field that will be the independent field for surrogate results.            
        :type interpolation_field: str

        :param interpolation_locations: the number of interpolation locations for the 
            surrogate to output at or an array-like of values for the interpolation locations.
            If a number of locations is given, the surrogate will linearly space the points
            over the min and max value for the interpolation field for all evaluations.
        :type interpolation_locations: int or Array-like
        
        :param surrogate_type: What type of surrogate to run. Details of each are detailed in the 
            surrogate's documentation. Currently the only available 
            options are "PCA Multiple Regressors" 
            and "PCA Monolithic Regressor". The Default is set to 
            "PCA Multiple Regressors" as it has
            better performance but uses more memory than the monolithic surrogate. 
        :type surrogate_type: str

        :param regressor_type: The identifier key for what core regressor 
            form to use as the predictor. 
            Only "Random Forest", "Gaussian Process" and "RBF" are accepted. Currently, MatCal
            uses the implementations of the random forest and Gaussian Process tools
            from the sklearn library. For "RBF", MatCal uses scipy.interpolate.RBFInterpolator with a default
            local-neighbor count of 20. This can be changed by passing neighbors=<int>
            through regressor_kwargs.
        :type regressor_type: str

        :param test_eval_info: A container of the relevant
            information to test a surrogate generated
            from a MatCal sampling study. This data is only used and must
            be provided if training_fraction == 1.0.
        :type test_eval_info: :class:`~matcal.core.study_base.StudyResults`
        
        :param regressor_kwargs: A keyword selection of parameters to pass to the predictor used. 
            Please refer to the scikit-learn documentation for ``"Random Forest"`` and
            ``"Gaussian Process"`` options, and the SciPy ``RBFInterpolator`` documentation
            for ``"RBF"`` options.
        :type regressor_kwargs: dict
        """
        self._interpolation_field = interpolation_field
        self._input_parameter_history = None
        self._interpolation_locations = interpolation_locations
        self._eval_info = evaluation_information
        self._test_eval_info = test_eval_info
        self._model_name = None
        self._state = None
        self._training_fraction  = training_fraction
        self._surrogate_type = surrogate_type
        self._regressor_type = regressor_type
        self._regressor_kwargs = regressor_kwargs
        self._decomp_tool = _assign_decomp(.99, None)
        self._logger_on=True

        self._fields_to_log_scale = []
        self._fields_of_interest = None
        self._train_score = OrderedDict()
        self._test_score = OrderedDict()
        self._check_test_evaluation_information_provided()

    def set_model_and_state(self, model_name=None, state=None):
        """
        Set the evaluation set and state to select from the study results.

        :param model_name: This is the model name for which the surrogate will
           generate results. 
           If no argument is passed, the surrogate generator will 
           expect the study to have a single model. 
        :type model_name: str or None 

        :param state: This specifies the state for the model for which the surrogate 
            will generate results. It can be either a :class:`~matcal.core.state.State` 
            object or a state name. If no argument is provided, 
            this method will assume that only a single state is 
            associated with the model for which the surrogate is being generated.
        :type state: str or :class:`~matcal.core.state.State`
        """

        if model_name is not None:
            check_value_is_nonempty_str(model_name, "model_name")
            self._model_name = model_name
        if state is not None:
            check_item_is_correct_type(state, (str, State), "state")
            self._state = state
   
    def set_PCA_details(self, decomp_var=.99, reconstruction_error=None):
        """
        Set options that control how many PCA modes are retained.

        :param decomp_var: What level of the total variance should be accounted for in the PCA
            decomposition. Values closer to 1 will keep more modes than lower values. The more modes
            kept the more difficult it can become to train the predictors. A default value of .99 is 
            chosen because it is a common conventional choice, and explains the vast majority of the 
            seen behavior, and for an appropriate data set can lead
            to very few modes being retained. 
        :type decomp_var: float

        :param reconstruction_error: Optional reconstruction-error tolerance used to
            tune the number of retained modes. If provided, this overrides variance-based
            mode selection.
        :type reconstruction_error: float or None
        """
        self._decomp_tool = _assign_decomp(decomp_var, reconstruction_error)
        

    def set_surrogate_details(self, surrogate_type="PCA Multiple Regressors", 
                              regressor_type="Gaussian Process", 
                              training_fraction=.8, interpolation_locations=None, 
                              test_eval_info=None, **regressor_kwargs):
        """
        This method provides another avenue to alter the surrogate 
        generation parameters after initialization. 

        :param surrogate_type: What type of surrogate to run. Details of each are detailed in the 
            surrogate's documentation. Currently the only available options 
            are "PCA Multiple Regressors" 
            and "PCA Monolithic Regressor". The Default is set to 
            "PCA Multiple Regressors" as it has
            better performance but uses more memory than the monolithic surrogate. 
        :type surrogate_type: str

        :param regressor_type: The identifier key for what core regressor 
            form to use as the predictor. 
            Only "Random Forest", "Gaussian Process" and "RBF" are accepted. 
            Currently, MatCal uses the implementations of the "Random Forest" and
            "Gaussian Process" regressors from the sklearn library. The "RBF" option
            uses the RBFInterpolator from SciPy.
        :type regressor_type: str

        :param training_fraction: Fraction of source data used for training. Must satisfy
            ``0 < training_fraction <= 1``. If equal to ``1.0``, ``test_eval_info`` must
            be provided.
        :type training_fraction: float

        :param interpolation_locations: Optional replacement interpolation locations.
            If provided, updates the interpolation locations used during surrogate
            generation.
        :type interpolation_locations: int, array-like, or None
        
        :param test_eval_info: A container of the relevant
            information to test a surrogate generated
            from a MatCal sampling study. This data is only used and must
            be provided if training_fraction == 1.0.
        :type test_eval_info: :class:`~matcal.core.study_base.StudyResults`
        
        :param regressor_kwargs: Keyword arguments passed to the selected regressor.
            Refer to the scikit-learn documentation for ``"Random Forest"`` and
            ``"Gaussian Process"`` options, and the SciPy ``RBFInterpolator``
            documentation for ``"RBF"`` options. 
        :type regressor_kwargs: dict
        """
        self._training_fraction  = training_fraction
        self._surrogate_type = surrogate_type
        self._regressor_type = regressor_type
        self._regressor_kwargs = regressor_kwargs
        if test_eval_info is not None:
            self._test_eval_info = test_eval_info
        self._check_test_evaluation_information_provided()
        if (interpolation_locations is not None):
            self._interpolation_locations = interpolation_locations

    def set_fields_to_log_scale(self, *field_names):
        """
        For fields of interest that span over orders of magnitude it can be easier
        to train a base-10 logarithmic scale rather than the raw data. 
        Passing fields here will inform the surrogate and the generator that 
        these fields should be evaluated on a base-10 logarithmic scale. Any predictions
        given by the surrogate will be at the original scale. This just adds an 
        additional scaling/descaling step within it.
         
        The current implementation applies an internal feature-wise offset before the
        base-10 logarithm so that the minimum fitted value maps to ``log10(1)``.
        Predictions are transformed back to the original scale before being returned.

        :param field_names: a series of field names to train on the log scale
        :type field_names: str
        """
        self._fields_to_log_scale = field_names

    def set_fields_of_interest(self, *fields_of_interest):
        """
        Specify which data fields the surrogate should model.

        By default the surrogate generator attempts to build a model for every
        field present in the source data (aside from the independent
        interpolation field).  Use this method to limit the surrogate to a
        user‑selected subset of fields.

        :param fields_of_interest: One or more field names that should be
            included in the surrogate model. 
        :type fields_of_interest: ``*str``

        .. note::
            * The independent interpolation field (if any) is never treated as a
            field of interest and is automatically excluded; you should not 
            pass it here.
            * Fields that are **not** listed will be ignored during surrogate
            generation and will not appear in the surrogate’s output.
        """
        if fields_of_interest:
            for field in fields_of_interest:
                check_value_is_nonempty_str(field, "field_of_interest")
            self._fields_of_interest = fields_of_interest

    def generate(self, save_filename:Optional[str]=None, preprocessing_function:Optional[Callable]=None, 
                 plot_n_worst:int=0)->Callable:
        """
        Generates a surrogate based on the information passed to it upon initialization

        :param save_filename: The base of a filename without any extensions
            used to save the surrogate. If ``None``, the surrogate is returned
            but not serialized to disk. A filename is required when
            ``plot_n_worst > 0`` because the worst-recreation plot uses this
            filename as its output prefix.
        :type save_filename: str or None

        :param preprocessing_function: an optional function that modifies
            the model data before it is passed to the tools that generate the 
            surrogate model.
        :type preprocessing_function: Callable

        :param plot_n_worst: Generate a number of plots that show the worst 
            recreations made by the surrogate. The number of plots made is equal to the 
            value passed to this argument. Any values less than 1 will result in no
            plots being generated or worst analysis being performed.
        :type plot_n_worst: int
            
        :return: a callable surrogate
        :rtype: :class:`~matcal.core.surrogates.MatCalPCASurrogateBase` 
        """
        if save_filename is not None:
            check_value_is_nonempty_str(save_filename, "save_filename")
        elif plot_n_worst > 0:
            raise ValueError(
                "save_filename must be provided when plot_n_worst > 0 because "
                "the worst-recreation plots require an output filename prefix."
            )
        self._normalize_test_evaluation_information_names()

        results = _package_surrogate_generator_input_data(self._eval_info, self._model_name, 
                                                          self._state)
        source_data, params = results
        self._fields_of_interest = _identify_fields_of_interest(source_data, 
                                                                self._interpolation_field, 
                                                                self._fields_of_interest)
        self._interpolation_locations = _process_interpolation_locations(source_data, 
                                                                         self._interpolation_locations, 
                                                                         self._interpolation_field)
        source_dict = _process_data_for_surrogate(source_data, self._fields_of_interest,
                                                  self._interpolation_locations, 
                                                  self._interpolation_field, preprocessing_function)
        test_train_split_results = self._select_training_and_test_data(source_dict, params, 
                                                                preprocessing_function)
        train_data, test_data, train_params, test_params = test_train_split_results
        combined_params = _combine_parameters(test_params, train_params)
        param_ranges = _package_parameter_ranges(combined_params)
        if self._logger_on:
            logger.info(f'Generating and scoring {self._regressor_type} surrogates. '+
                    'The ideal score is 1.0.')
        surrogate_class = _surrogate_selection.identify(self._surrogate_type)
        new_surrogate = surrogate_class.fit(train_data, test_data, train_params, test_params,
                                            self._fields_to_log_scale,
                                            self._decomp_tool, self, param_ranges, 
                                            self._logger_on)
        if save_filename is not None:
            matcal_save(save_filename + ".joblib", new_surrogate)
        self._plot_worst_recreations(new_surrogate, params, source_dict, 
                                     plot_n_worst, save_filename)
        return new_surrogate

    def _check_test_evaluation_information_provided(self):
        if self._training_fraction == 1.0 and self._test_eval_info is None:
            raise ValueError("Test evaluations must be provided when training_fraction = 1.0.")

    def _normalize_test_evaluation_information_names(self):
        """
        Normalize supplied test-evaluation information to match training data names.

        This is primarily needed when training and test StudyResults were generated
        by equivalent model objects that received different runtime-generated names.
        """
        if self._test_eval_info is None:
            return

        required_model_name = _get_model_name_from_evaluation_information(
            self._eval_info,
            self._model_name,
        )

        if required_model_name is None:
            return

        self._test_eval_info = _normalize_evaluation_information_names(
            self._test_eval_info,
            required_model_name=required_model_name,
            required_objective_name=None,
            data_set_name="surrogate test_eval_info",
            logger_on=self._logger_on,
        )

    def _plot_worst_recreations(self, surrogate, parameters, source_data, n_worst, save_filename):
        if n_worst < 1:
            return
        import matplotlib.pyplot as plt
        plt.close('all')
        n_eval = len(parameters[list(parameters.keys())[0]])
        sur_predict = surrogate(parameters)
        worst_sets = self._get_worst_recreations(source_data, n_worst, n_eval, sur_predict)
        short, long = _find_smallest_rect(n_worst)
        size_per_plt = 2
        fig, ax_set = plt.subplots(short, long, figsize=(size_per_plt*long,size_per_plt*short), 
                                   constrained_layout=True)
        ax_set = self._format_ax_set(n_worst, ax_set)
        for ax, (field, eval_idx) in zip(ax_set, worst_sets):
            self._plot_set(surrogate, source_data, sur_predict, ax, field, eval_idx)
        filename = f"{save_filename}_worst.png"
        plt.savefig(filename, dpi=400)

    def _format_ax_set(self, n_worst, ax_set):
        if n_worst > 1:
            ax_set = ax_set.flatten()
        else:
            ax_set = [ax_set]
        return ax_set

    def _plot_set(self, surrogate, source_data, sur_predict, ax, field, eval_idx):
        prediction_locations=surrogate.prediction_locations
        if prediction_locations is not None:
            ax.plot(prediction_locations, sur_predict[field][eval_idx,:], '--', 
                    lw=3, label='surrogate')
            ax.plot(prediction_locations, source_data[field][eval_idx,:], '-', 
                    lw=3, label='source')
            ax.set_xlabel(surrogate.independent_field)
        else:
            ax.plot(sur_predict[field][eval_idx,:], '--', lw=3, label='surrogate')
            ax.plot(source_data[field][eval_idx,:], '-', lw=3, label='source')
            ax.set_xlabel(surrogate.independent_field)
        ax.set_title(f"{field} eval index{eval_idx}")
        ax.set_ylabel(field)
        ax.legend()

    def _get_worst_recreations(self, source_data, n_worst, n_eval, sur_predict):
        worst = _WorstEvaluations(n_worst)
        for field in source_data:
            field_prediction = sur_predict[field]
            for eval_idx in range(n_eval):
                sur_values = field_prediction[eval_idx,:]
                source_values = source_data[field][eval_idx, :]
                misfit = _score_recreation(sur_values, source_values)
                worst.update(field, eval_idx, misfit)
        worst_sets = worst.get_set()
        return worst_sets

    def _select_training_and_test_data(self, source_dict, params, 
                                       preprocessing_function):
        if self._training_fraction == 1.0:
            results = _package_surrogate_generator_input_data(self._test_eval_info, 
                                                              self._model_name, self._state)
            test_data, test_params = results
            test_data = _process_data_for_surrogate(test_data, self._fields_of_interest,
                                                    self._interpolation_locations,
                                                    self._interpolation_field, 
                                                    preprocessing_function) 
            train_data = source_dict
            train_params = params
            _check_fields_in_keys_list(self._fields_of_interest, test_data.keys(), "test data set")
        else:
            from sklearn.model_selection import train_test_split
            first_param_key = list(params.keys())[0]
            indices = np.arange(len(params[first_param_key]))
            data_split_results = train_test_split(indices, train_size=self._training_fraction)
            train_indices, test_indices = data_split_results
            train_params, test_params = _split_dict_data_into_test_train_data_dicts(train_indices, 
                                                                                    test_indices, 
                                                                                    params)
            train_data, test_data = _split_dict_data_into_test_train_data_dicts(train_indices, 
                                                                                test_indices, 
                                                                                source_dict)
        return train_data, test_data, train_params, test_params

def _check_fields_in_keys_list(fields, data_fields, data_set_name):
    for field in fields:
        if field not in data_fields:
            raise KeyError(f"The field of interest {field} for the surrogate was "+
                            f"not in the provided the {data_set_name}.")

def _split_dict_data_into_test_train_data_dicts(train_indices, test_indices, data_dict):
    test_data = OrderedDict()
    train_data = OrderedDict()
    for key in data_dict:
        data_key_array =  np.array(data_dict[key])
        test_data[key] = data_key_array[test_indices]
        train_data[key] =  data_key_array[train_indices]
    return train_data, test_data

def _package_surrogate_generator_input_data(eval_info, model_name, state):
        data_history, input_parameter_history = _select_relevant_study_data(eval_info, model_name, state)
        param_history = _import_parameter_hist(input_parameter_history)
        return data_history, param_history
    

def _select_relevant_study_data(evaluation_information, model_name, state):
    parsed_eval_info = _parse_evaluation_info(evaluation_information, model_name)  
    input_parameter_history, _sim_hist_data_collection = parsed_eval_info                                                          
    data_history = _select_state_data(state, _sim_hist_data_collection)
    return data_history, input_parameter_history


def _select_state_data(state, sim_history_dc):
    if state is None:
        states = list(sim_history_dc.state_names)
        if len(states) > 1:
            raise ValueError(f"There are {len(states)} in the results data for the "
                             "surrogate generator. Specify a state for the surrogate.")
        else:
            state = states[0]
    return sim_history_dc[state]


def _select_model(simulation_history, model_name):
    if model_name is None:
        model_name = list(simulation_history.keys())[0]
    return model_name

def _parse_study_results(study_results, model_name, ):
    input_hist = study_results.parameter_history
    sim_history = study_results.simulation_history
    model_name = _select_model(sim_history, model_name)
    output_hist = sim_history[model_name]
    return input_hist, output_hist


def _parse_evaluation_info(eval_info, model_name):
    from matcal.core.study_base import StudyResults, StudyBase
    if isinstance(eval_info, StudyResults):
        input_hist, output_hist = _parse_study_results(eval_info, model_name,
                                                       )
        
    elif isinstance(eval_info, StudyBase):
        input_hist, output_hist = _parse_study_results(eval_info.results,
                                                       model_name)

    elif isinstance(eval_info, dict):
        input_hist = eval_info['input']
        output_hist = eval_info['output']
    else:
        raise TypeError(f"Surrogate Generator can not process data of type {type(eval_info)}")

    return input_hist, output_hist


def _get_results_like_from_evaluation_information(eval_info):
    """
    Return a StudyResults-like object from supported evaluation information.

    This intentionally accepts StudyResults-like objects by attribute rather than
    by strict type so the name-normalization helpers can be unit tested with
    lightweight objects.
    """
    if eval_info is None:
        return None

    if hasattr(eval_info, "simulation_history") or hasattr(eval_info, "qoi_history"):
        return eval_info

    if hasattr(eval_info, "results"):
        return eval_info.results

    return None


def _get_model_name_from_evaluation_information(eval_info, model_name=None):
    """
    Determine the model name that should be used for surrogate data extraction.

    If ``model_name`` is supplied, it is returned directly. If not supplied and
    the evaluation information contains exactly one model, that model name is
    returned. If the model name cannot be determined unambiguously, ``None`` is
    returned and no automatic test-data model-name normalization is attempted.
    """
    if model_name is not None:
        return model_name

    results = _get_results_like_from_evaluation_information(eval_info)

    if results is None or not hasattr(results, "simulation_history"):
        return None

    simulation_history = results.simulation_history

    if simulation_history is None:
        return None

    model_names = list(simulation_history.keys())

    if len(model_names) == 1:
        return model_names[0]

    return None


def _split_qoi_history_key(qoi_key):
    """
    Split a QoI-history key into model and objective names.

    Expected keys have the form ``'<model_name>:<objective_name>'``.
    """
    if not isinstance(qoi_key, str) or ":" not in qoi_key:
        raise RuntimeError(
            "Surrogate test-data QoI-history keys must have the form "
            f"'<model_name>:<objective_name>'. Received key '{qoi_key}'."
        )

    model_name, objective_name = qoi_key.split(":", 1)

    if model_name == "" or objective_name == "":
        raise RuntimeError(
            "Surrogate test-data QoI-history keys must have nonempty model "
            f"and objective names. Received key '{qoi_key}'."
        )

    return model_name, objective_name


def _get_qoi_history_model_and_objective_names(qoi_history):
    """
    Extract model and objective names from QoI-history keys.
    """
    model_names = set()
    objective_names = set()

    for qoi_key in qoi_history.keys():
        model_name, objective_name = _split_qoi_history_key(qoi_key)
        model_names.add(model_name)
        objective_names.add(objective_name)

    return model_names, objective_names


def _rename_mapping_key(mapping, old_key, new_key):
    """
    Rename one key in a mutable mapping while preserving its value.
    """
    if new_key in mapping:
        raise RuntimeError(
            f"Renaming surrogate test-data key '{old_key}' to '{new_key}' "
            "would create a duplicate key."
        )

    mapping[new_key] = mapping.pop(old_key)


def _normalize_single_key_mapping_name(
    mapping,
    required_name,
    entry_kind,
    data_set_name,
    logger_on=True,
):
    """
    Rename a one-entry mapping key to ``required_name`` when unambiguous.

    If the required key already exists, no change is made. If the required key is
    absent and exactly one key is present, that key is renamed. If the required
    key is absent and multiple keys are present, an error is raised.
    """
    if mapping is None:
        return

    names = list(mapping.keys())

    if required_name in names:
        return

    if len(names) == 0:
        raise RuntimeError(
            f"The supplied {data_set_name} does not contain any {entry_kind} "
            "entries and cannot be used for surrogate testing."
        )

    if len(names) > 1:
        raise RuntimeError(
            f"The supplied {data_set_name} contains more than one {entry_kind} "
            f"name. MatCal cannot determine which {entry_kind} should be used "
            f"for surrogate testing. {entry_kind.capitalize()} names found: "
            f"{names}. The required {entry_kind} name is '{required_name}'."
        )

    current_name = names[0]

    if logger_on:
        logger.warning(
            f"The supplied {data_set_name} uses {entry_kind} name "
            f"'{current_name}', but the surrogate requires {entry_kind} name "
            f"'{required_name}'. Because the supplied data contains exactly one "
            f"{entry_kind} name, MatCal is renaming the {data_set_name} "
            f"{entry_kind} entry at runtime."
        )

    _rename_mapping_key(mapping, current_name, required_name)


def _qoi_history_is_optional(required_objective_name):
    """
    Return True when missing QoI history can be ignored.

    SurrogateGenerator only needs simulation-history model-name normalization.
    It does not require QoI history, so missing/empty QoI history is acceptable
    when no required objective name is supplied.
    """
    return required_objective_name is None


def _raise_missing_qoi_history_error(data_set_name):
    raise RuntimeError(
        f"The supplied {data_set_name} does not contain a QoI history and "
        "cannot be used for adaptive surrogate testing."
    )


def _raise_empty_qoi_history_error(data_set_name):
    raise RuntimeError(
        f"The supplied {data_set_name} does not contain any QoI-history "
        "entries and cannot be used for adaptive surrogate testing."
    )


def _qoi_history_has_required_content(
    qoi_history,
    required_objective_name,
    data_set_name,
):
    """
    Validate whether a QoI-history mapping must be processed.

    Returns False when there is no required objective name and the QoI history is
    absent or empty, because this is acceptable for non-adaptive surrogate
    normalization.
    """
    if qoi_history is None:
        if _qoi_history_is_optional(required_objective_name):
            return False
        _raise_missing_qoi_history_error(data_set_name)

    if len(qoi_history) == 0:
        if _qoi_history_is_optional(required_objective_name):
            return False
        _raise_empty_qoi_history_error(data_set_name)

    return True


def _raise_multiple_qoi_model_names_error(
    model_names,
    required_model_name,
    data_set_name,
):
    raise RuntimeError(
        f"The supplied {data_set_name} contains more than one model "
        "name in its QoI history. MatCal cannot determine which model "
        "should be used for surrogate testing. Model names found: "
        f"{sorted(model_names)}. The required model name is "
        f"'{required_model_name}'."
    )


def _raise_multiple_qoi_objective_names_error(
    objective_names,
    required_objective_name,
    data_set_name,
):
    raise RuntimeError(
        f"The supplied {data_set_name} contains more than one "
        "objective name in its QoI history. MatCal cannot determine "
        "which objective should be used for surrogate testing. "
        f"Objective names found: {sorted(objective_names)}. The "
        f"required objective name is '{required_objective_name}'."
    )


def _require_single_qoi_model_name(model_names, required_model_name, data_set_name):
    """
    Return the only QoI-history model name when model-name normalization is requested.
    """
    if required_model_name is None:
        return None

    if len(model_names) > 1:
        _raise_multiple_qoi_model_names_error(
            model_names,
            required_model_name,
            data_set_name,
        )

    return next(iter(model_names))


def _require_single_qoi_objective_name(
    objective_names,
    required_objective_name,
    data_set_name,
):
    """
    Return the only QoI-history objective name when objective-name normalization is requested.
    """
    if required_objective_name is None:
        return None

    if len(objective_names) > 1:
        _raise_multiple_qoi_objective_names_error(
            objective_names,
            required_objective_name,
            data_set_name,
        )

    return next(iter(objective_names))


def _required_name_was_supplied(required_name):
    return required_name is not None


def _name_change_required(current_name, required_name):
    """
    Return True if a supplied required name differs from the current name.
    """
    if not _required_name_was_supplied(required_name):
        return False

    return current_name != required_name


def _get_qoi_model_name_change(
    model_names,
    required_model_name,
    data_set_name,
):
    """
    Return ``(old_model_name, new_model_name, model_changed)``.
    """
    current_model_name = _require_single_qoi_model_name(
        model_names,
        required_model_name,
        data_set_name,
    )

    if not _required_name_was_supplied(required_model_name):
        return None, None, False

    return (
        current_model_name,
        required_model_name,
        _name_change_required(current_model_name, required_model_name),
    )


def _get_qoi_objective_name_change(
    objective_names,
    required_objective_name,
    data_set_name,
):
    """
    Return ``(old_objective_name, new_objective_name, objective_changed)``.
    """
    current_objective_name = _require_single_qoi_objective_name(
        objective_names,
        required_objective_name,
        data_set_name,
    )

    if not _required_name_was_supplied(required_objective_name):
        return None, None, False

    return (
        current_objective_name,
        required_objective_name,
        _name_change_required(current_objective_name, required_objective_name),
    )


def _get_qoi_history_name_changes(
    qoi_history,
    required_model_name,
    required_objective_name,
    data_set_name,
):
    """
    Determine required model/objective renames for a QoI-history mapping.

    Returns:
        old_model_name, old_objective_name, new_model_name, new_objective_name,
        model_changed, objective_changed
    """
    model_names, objective_names = _get_qoi_history_model_and_objective_names(
        qoi_history
    )

    old_model_name, new_model_name, model_changed = _get_qoi_model_name_change(
        model_names,
        required_model_name,
        data_set_name,
    )

    old_objective_name, new_objective_name, objective_changed = (
        _get_qoi_objective_name_change(
            objective_names,
            required_objective_name,
            data_set_name,
        )
    )

    return (
        old_model_name,
        old_objective_name,
        new_model_name,
        new_objective_name,
        model_changed,
        objective_changed,
    )


def _qoi_history_name_changes_required(model_changed, objective_changed):
    """
    Return True if any QoI-history key component must be renamed.
    """
    return model_changed or objective_changed


def _log_qoi_model_name_change(
    old_model_name,
    new_model_name,
    data_set_name,
):
    logger.warning(
        f"The supplied {data_set_name} uses QoI-history model name "
        f"'{old_model_name}', but the surrogate requires model name "
        f"'{new_model_name}'. Because the supplied data contains exactly "
        "one QoI-history model name, MatCal is renaming the QoI-history "
        "key at runtime."
    )


def _log_qoi_objective_name_change(
    old_objective_name,
    new_objective_name,
    data_set_name,
):
    logger.warning(
        f"The supplied {data_set_name} uses objective name "
        f"'{old_objective_name}', but the surrogate requires objective "
        f"name '{new_objective_name}'. Because the supplied data contains "
        "exactly one objective name, MatCal is renaming the QoI-history "
        "key at runtime."
    )


def _log_qoi_history_name_changes(
    old_model_name,
    old_objective_name,
    new_model_name,
    new_objective_name,
    model_changed,
    objective_changed,
    data_set_name,
    logger_on=True,
):
    """
    Log warnings for requested QoI-history key renames.
    """
    if not logger_on:
        return

    if model_changed:
        _log_qoi_model_name_change(
            old_model_name,
            new_model_name,
            data_set_name,
        )

    if objective_changed:
        _log_qoi_objective_name_change(
            old_objective_name,
            new_objective_name,
            data_set_name,
        )


def _validate_qoi_key_component(
    actual_name,
    expected_name,
    component_name,
    old_key,
):
    """
    Validate one old QoI-history key component against an expected name.

    ``expected_name=None`` means any actual name is accepted.
    """
    if expected_name is None:
        return

    if actual_name == expected_name:
        return

    raise RuntimeError(
        f"Unexpected {component_name} name encountered while renaming "
        "surrogate test-data QoI history. Expected {component_name} name "
        f"'{expected_name}', but found '{actual_name}' in key '{old_key}'."
    )


def _select_qoi_key_component_name(actual_name, new_name):
    """
    Return the renamed QoI-history key component, or preserve the original.
    """
    if new_name is None:
        return actual_name

    return new_name


def _make_updated_qoi_history_key(
    old_key,
    old_model_name=None,
    old_objective_name=None,
    new_model_name=None,
    new_objective_name=None,
):
    """
    Build the updated QoI-history key for one existing key.
    """
    model_name, objective_name = _split_qoi_history_key(old_key)

    _validate_qoi_key_component(
        model_name,
        old_model_name,
        "model",
        old_key,
    )
    _validate_qoi_key_component(
        objective_name,
        old_objective_name,
        "objective",
        old_key,
    )

    final_model_name = _select_qoi_key_component_name(
        model_name,
        new_model_name,
    )
    final_objective_name = _select_qoi_key_component_name(
        objective_name,
        new_objective_name,
    )

    return f"{final_model_name}:{final_objective_name}"


def _insert_renamed_qoi_history_item(qoi_history, new_key, value):
    """
    Insert a renamed QoI-history item, rejecting duplicate keys.
    """
    if new_key in qoi_history:
        raise RuntimeError(
            "Renaming surrogate test-data QoI history would create a "
            f"duplicate QoI-history key '{new_key}'."
        )

    qoi_history[new_key] = value


def _rename_qoi_history_keys(
    qoi_history,
    old_model_name=None,
    old_objective_name=None,
    new_model_name=None,
    new_objective_name=None,
):
    """
    Rename model/objective components of QoI-history keys in place.

    Any component with ``None`` is treated as a wildcard/preserve operation:

    * ``old_model_name=None``: accept any existing model name.
    * ``old_objective_name=None``: accept any existing objective name.
    * ``new_model_name=None``: preserve each key's existing model name.
    * ``new_objective_name=None``: preserve each key's existing objective name.

    This supports the SurrogateGenerator use case where only the model-name
    component needs to be renamed while multiple objective names may be present.
    """
    original_items = list(qoi_history.items())
    qoi_history.clear()

    for old_key, value in original_items:
        new_key = _make_updated_qoi_history_key(
            old_key,
            old_model_name=old_model_name,
            old_objective_name=old_objective_name,
            new_model_name=new_model_name,
            new_objective_name=new_objective_name,
        )
        _insert_renamed_qoi_history_item(qoi_history, new_key, value)


def _normalize_qoi_history_names(
    qoi_history,
    required_model_name=None,
    required_objective_name=None,
    data_set_name="test data",
    logger_on=True,
):
    """
    Normalize model/objective names in a QoI-history mapping.

    If ``required_model_name`` is supplied, the QoI history must contain exactly
    one model name. If the single model name differs from the required model
    name, it is renamed.

    If ``required_objective_name`` is supplied, the QoI history must contain
    exactly one objective name. If the single objective name differs from the
    required objective name, it is renamed.

    If ``required_objective_name`` is ``None``, objective names are preserved.
    This permits SurrogateGenerator to normalize test-data model names while
    preserving multiple objective names.
    """
    if not _qoi_history_has_required_content(
        qoi_history,
        required_objective_name,
        data_set_name,
    ):
        return

    (
        old_model_name,
        old_objective_name,
        new_model_name,
        new_objective_name,
        model_changed,
        objective_changed,
    ) = _get_qoi_history_name_changes(
        qoi_history,
        required_model_name,
        required_objective_name,
        data_set_name,
    )

    if not _qoi_history_name_changes_required(
        model_changed,
        objective_changed,
    ):
        return

    _log_qoi_history_name_changes(
        old_model_name,
        old_objective_name,
        new_model_name,
        new_objective_name,
        model_changed,
        objective_changed,
        data_set_name,
        logger_on=logger_on,
    )

    _rename_qoi_history_keys(
        qoi_history,
        old_model_name=old_model_name,
        old_objective_name=old_objective_name,
        new_model_name=new_model_name,
        new_objective_name=new_objective_name,
    )


def _normalize_evaluation_information_names(
    eval_info,
    required_model_name=None,
    required_objective_name=None,
    data_set_name="test data",
    logger_on=True,
):
    """
    Normalize model/objective names in StudyResults-like evaluation information.

    This is used to make externally supplied test data compatible with surrogate
    generation and adaptive surrogate studies when the test data were generated
    with equivalent model/objective objects that have different runtime names.

    The normalization is intentionally conservative:

    * simulation-history model names are renamed only when the supplied data has
      exactly one model name and the required model name is missing;
    * QoI-history model names are renamed only when exactly one QoI-history model
      name is present;
    * if ``required_objective_name`` is supplied, the QoI history must contain
      exactly one objective name; otherwise an error is raised;
    * every runtime rename logs a warning.
    """
    results = _get_results_like_from_evaluation_information(eval_info)

    if results is None:
        return eval_info

    if required_model_name is not None and hasattr(results, "simulation_history"):
        _normalize_single_key_mapping_name(
            results.simulation_history,
            required_model_name,
            "model",
            data_set_name,
            logger_on=logger_on,
        )

    if hasattr(results, "qoi_history"):
        _normalize_qoi_history_names(
            results.qoi_history,
            required_model_name=required_model_name,
            required_objective_name=required_objective_name,
            data_set_name=data_set_name,
            logger_on=logger_on,
        )
    elif required_objective_name is not None:
        raise RuntimeError(
            f"The supplied {data_set_name} does not have a qoi_history "
            "attribute and cannot be used for adaptive surrogate testing."
        )

    return eval_info


def _apply_preprocessing_function(preprocessing_function, training_data_history):
    if preprocessing_function is not None:
        check_item_is_correct_type(preprocessing_function, Callable, "preprocessing_function")
        for idx, data in enumerate(training_data_history):
            processed_data = preprocessing_function(data)
            if isinstance(processed_data, (dict, OrderedDict)):
                processed_data = convert_dictionary_to_data(processed_data)
                processed_data.set_state(data.state)
            training_data_history[idx] = processed_data
    return training_data_history    


def _process_data_for_surrogate(source_data_list, fields_of_interest, 
                                interpolation_locations, interpolation_field, 
                                preprocessing_function=None):
    source_data_list = _apply_preprocessing_function(preprocessing_function, source_data_list)
    processed_data = _initialize_processed_data(source_data_list, fields_of_interest,
                                                 interpolation_locations)
    for idx, data in enumerate(source_data_list):
        for field in fields_of_interest:
            data_field = data[field]
            if interpolation_locations is not None and interpolation_field is not None:
                data_field =  np.interp(interpolation_locations, 
                                        data[interpolation_field], data_field)
            processed_data[field][idx, :] = data_field
    return processed_data


def _initialize_processed_data(training_data_list, fields_of_interest,
                               interpolation_locations):
    processed_data = OrderedDict()
    n_evals = len(training_data_list)
    for field in fields_of_interest:
        n_points = _get_n_points(interpolation_locations, 
                            training_data_list, field)
        processed_data[field] = np.zeros([n_evals, n_points])
    return processed_data


def _get_n_points(interpolation_locations, training_data_list, field):
    if interpolation_locations is None:
        return len(training_data_list[0][field])
    else:
        return len(interpolation_locations)


class _WorstEvaluations:
    
    def __init__(self, track_n):
        self._n = track_n
        self._scores = []
        self._field_eval_sets =[]

    def update(self, field, eval_idx, score):        
        self._scores.append(score)
        self._field_eval_sets.append((field, eval_idx))
        if len(self._scores) > self._n:
            n_worst_args = np.argsort(self._scores).flatten()[-self._n:]
            _new_scores = []
            _new_sets = []
            for idx in n_worst_args:
                _new_scores.append(self._scores[idx])
                _new_sets.append(self._field_eval_sets[idx])
            self._scores = _new_scores
            self._field_eval_sets = _new_sets
    
    def get_set(self):
        return self._field_eval_sets


class _RBFInterpolatorRegressor(BaseEstimator):
    """
    sklearn-like wrapper around scipy.interpolate.RBFInterpolator.

    By default, this uses a local RBF interpolant with a limited number of
    nearest neighbors so that prediction cost does not grow too aggressively
    for large training sets.
    """

    def __init__(self, neighbors=20, **rbf_kwargs):
        self.neighbors = neighbors
        self.rbf_kwargs = rbf_kwargs
        self._rbf = None
        self._effective_neighbors = None
        self._single_output = False

    def fit(self, input_values, output_values):
        from scipy.interpolate import RBFInterpolator

        input_values = np.asarray(input_values, dtype=float)
        output_values = np.asarray(output_values, dtype=float)

        if input_values.ndim != 2:
            raise ValueError("RBFInterpolator input values must be a 2D array.")

        if output_values.ndim == 1:
            self._single_output = True
        elif output_values.ndim == 2 and output_values.shape[1] == 1:
            self._single_output = True

        n_samples = input_values.shape[0]

        if self.neighbors is None:
            effective_neighbors = None
        else:
            effective_neighbors = min(int(self.neighbors), n_samples)
            if effective_neighbors < 1:
                raise ValueError("RBFInterpolator neighbors must be at least 1.")

        self._effective_neighbors = effective_neighbors
        self._rbf = RBFInterpolator(
            input_values,
            output_values,
            neighbors=effective_neighbors,
            **self.rbf_kwargs,
        )

        return self

    def predict(self, input_values):
        if self._rbf is None:
            raise RuntimeError("RBFInterpolatorRegressor must be fit before predict is called.")

        input_values = np.asarray(input_values, dtype=float)
        prediction = self._rbf(input_values)

        if self._single_output:
            prediction = np.asarray(prediction).ravel()

        return prediction

    def score(self, input_values, output_values):
        prediction = np.asarray(self.predict(input_values))
        output_values = np.asarray(output_values)

        if prediction.shape != output_values.shape and prediction.size == output_values.size:
            prediction = prediction.reshape(output_values.shape)

        if output_values.ndim == 2 and output_values.shape[1] == 1:
            output_values = output_values.ravel()
            prediction = prediction.ravel()

        return r2_score(output_values, prediction)


def _init_rbf_surrogate(n_inputs, **kwargs):
    return _RBFInterpolatorRegressor(**kwargs)


def _init_random_forest_surrogate(n_inputs, **kwargs):
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(**kwargs)


def _init_gp_surrogate(n_inputs, **kwargs):
    from sklearn.gaussian_process import GaussianProcessRegressor

    if "kernel" not in kwargs:
        from sklearn.gaussian_process.kernels import (
            RBF, 
            ConstantKernel, 
        )

        kernel = ConstantKernel(1.0, (1e-5, 1e5)) * RBF(
            length_scale=0.1*np.ones(n_inputs),
            length_scale_bounds=(1e-5, 1e5),
        )
        kwargs["kernel"] = kernel
    #if "alpha" not in kwargs:
    #    kwargs["alpha"] = 1e-8
    gpr = GaussianProcessRegressor(**kwargs)
    return gpr


_regressor_lookup = {
    "Random Forest":_init_random_forest_surrogate,
    "Gaussian Process":_init_gp_surrogate, 
    "RBF": _init_rbf_surrogate
}


def _initialize_regressor(regressor_type, n_inputs, regressor_kwargs):
    return _regressor_lookup[regressor_type](n_inputs, **regressor_kwargs)


def _scale_data_for_surrogate(data_array, make_log=False):
    """
    Expects the data as n_samples x n_features
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    if make_log:
        scaler = Pipeline([('log', _MatCalLogScaler()), ('standard', StandardScaler())])
    else:
        scaler = StandardScaler()
        
    scaler.fit(data_array) 
    scaled_data = scaler.transform(data_array)
    return scaled_data, scaler


def _decompose_with_pca(data, var_tol, logger_on=True):
    """
    Expects data as n_samples x n_features
    """
    from sklearn.decomposition import PCA
    pca = PCA(n_components=var_tol, svd_solver='full')
    transformed_data = pca.fit_transform(data)
    if isinstance(var_tol, Integral) and logger_on:
        logger.info(f"\tGenerated PCA decomposition with {pca.n_components_} components.")
    elif isinstance(var_tol, Real) and logger_on:
        logger.info(f"\tGenerated PCA decomposition with {pca.n_components_}"
                    f" components using {var_tol} variance explanation.")
    elif logger_on:
        logger.info(f"\tGenerated PCA decomposition with {pca.n_components_}"
                    f" components using option \'{var_tol}\'.")
    return transformed_data, pca


def _use_pca_to_decompose_if_many_features(data, var_tol=.99, logger_on=True):
    """
    Expects data as n_samples x n_features
    """
    if data.shape[1] > 15:
        return _decompose_with_pca(data, var_tol, logger_on)
    else:
        return data, _DoNothingDataTransformer()


def _import_parameter_hist(parameter_history):
    return OrderedDict(parameter_history)


def _package_parameter_ranges(param_history):
    out_dict = {}
    for name, p in param_history.items():
        out_dict[name] = (np.min(p), np.max(p))
    return out_dict


def _convert_data_and_make_basis(source_data, decomp_variance, make_log_scale, logger_on=True):
    scaled_data, data_scaler = _scale_data_for_surrogate(source_data, make_log_scale)
    latent_data, decomposer = _use_pca_to_decompose_if_many_features(scaled_data, decomp_variance, 
                                                                     logger_on)
    latent_data = _ensure_2d_array(latent_data)
    scaled_latent_data, latent_scaler = _scale_data_for_surrogate(latent_data)
    return data_scaler,decomposer,scaled_latent_data,latent_scaler


def _tune_data_decomposition(source_data, make_log_scale, reconstruction_error_tol:float=1e-3, 
                             max_modes:int=10, logger_on=True):
    scaled_data, data_scaler = _scale_data_for_surrogate(source_data, make_log_scale)
    logger.info("  Tuning decomposition to meet recreation error tolerance of "+
                f"{reconstruction_error_tol}, up to a limit of {max_modes} modes")
    for mode_count in range(max_modes):
        kept_modes = mode_count + 1
        logger.info(f"    Analyzing {kept_modes} mode decomposition")
        latent_data, decomposer = _use_pca_to_decompose_if_many_features(scaled_data, kept_modes, 
                                                                         logger_on)
        recreated_data = decomposer.inverse_transform(latent_data)
        error = scaled_data - recreated_data
        max_error_rel = np.amax(error) / np.amax(scaled_data)
        logger.info(f"      Recreation has max relative error of {max_error_rel}")
        if max_error_rel < reconstruction_error_tol:
            logger.info(f"      Error below tolerance using {kept_modes} modes")
            break
        elif kept_modes == max_modes:
            message = ("      Recreation error tolerance not met, but max modes reached, "+
                       f"using {max_modes} mode decomposition")
            logger.info(message)
        else:
            logger.info("      Recreation error tolerance not met.\n")
    latent_data = _ensure_2d_array(latent_data)
    scaled_latent_data, latent_scaler = _scale_data_for_surrogate(latent_data)
    return data_scaler,decomposer,scaled_latent_data,latent_scaler


def _record_variance_behaviors(decomposer, filename_base, field_name):
    individual_variance = decomposer.explained_variance_ratio_
    missing_variance = np.ones_like(individual_variance)
    for i in range(len(missing_variance)):
        missing_variance[i:] -= individual_variance[i]
    logger.info(f"    Decomposition Modes Explained Variance Ratios: {missing_variance}")
    variance_filename = f"{filename_base}_{field_name}_pca_variance.png"
    marker_levels = [.05, .01]
    import matplotlib.pyplot as plt
    plt.figure()
    plt.title(field_name)
    plt.xlabel('PCA Mode')
    plt.ylabel('Missing Variance ratio [log10]')
    plt.semilogy(missing_variance, label='PCA')
    for marker in marker_levels:
        plt.semilogy(np.ones_like(missing_variance) *  marker, label=f"{int(marker*100)}%")
    plt.legend()
    plt.savefig(variance_filename, dpi=400)


class MatCalSurrogateBase(ABC):
    @abstractmethod
    def fit(parameters, predictions):
        """"""
    
    @property
    def scores(self):
        """
        The test and train R2 scores for the surrogate.
        """
        return self._r2_scores

    @property
    def max_errors(self):
        """
        The test and train max errors for the surrogate in 
        the given field's units.
        """
        return self._max_scores

    @property
    def rmse_errors(self):
        """
        The test and train root mean squared errors for the surrogate in the
        given field's original units.

        The RMSE is calculated as

        .. math::

            \\mathrm{RMSE}
            =
            \\sqrt{
            \\frac{1}{N}
            \\sum_{i=1}^{N}
            \\left(R_i - \\hat{R}_i\\right)^2
            }

        where :math:`N` is the total number of scalar response values.
        """
        return self._rmse_scores

    @abstractmethod
    def __call__(self, parameters)-> OrderedDict:
        """"""
        
    def __init__(self, latent_scores,  
                 fields_to_log_scale, interp_field, interp_locs, 
                 parameter_scaler, regressors, decomposers, data_scalers, 
                 latent_scalers, param_ranges):
        """Surrogate abstract base class from which all surrogates should be derived 
        in MatCal.
        """
        self._latent_scores = OrderedDict()
        self._rmse_scores = OrderedDict()
        self._max_scores = OrderedDict()
        self._r2_scores = OrderedDict()

        self._latent_scores['train'] = latent_scores[0]
        self._latent_scores['test'] = latent_scores[1]

        self._fields_to_log_scale = fields_to_log_scale
        self._interpolation_field = interp_field
        self._interpolation_locations = interp_locs
        self._parameter_scaler = parameter_scaler
        self._regressors = regressors
        self._decomposers = decomposers
        self._data_scalers = data_scalers
        self._latent_scalers = latent_scalers
        self._param_ranges = param_ranges
        self._enforce_training_data_parameter_range = True

    def enforce_training_data_parameter_range(self, enforce_training_data_parameter_range=True):
        """
        By default, the surrogate raises an error if called with parameter values
        outside the stored admissible parameter ranges. For surrogates generated by
        SurrogateGenerator, these ranges are initially built from the combined training
        and test parameter data used during surrogate generation. They can be updated
        with set_parameter_ranges().
        
        :param enforce_training_data_parameter_range: If ``True``, reject surrogate
            calls outside the stored parameter ranges. If ``False``, allow calls outside
            the stored parameter ranges.
        :type enforce_training_data_parameter_range: bool
        """
        check_value_is_bool(enforce_training_data_parameter_range, 
                            "enforce_training_data_parameter_range")
        self._enforce_training_data_parameter_range = enforce_training_data_parameter_range

    def _set_original_data_space_scores(self, rmse_scores, max_scores, r2_scores):
        self._rmse_scores['train'] = rmse_scores[0]
        self._rmse_scores['test'] = rmse_scores[1]

        self._max_scores['train'] = max_scores[0]
        self._max_scores['test'] = max_scores[1]

        self._r2_scores['train'] = r2_scores[0]
        self._r2_scores['test'] = r2_scores[1]

    def set_parameter_ranges(self, *args, **param_ranges):
        """
        Update the admissible parameter ranges that the user can call the surrogate to evaluate.

        The surrogate stores, for each input parameter, a lower and upper bound that
        define the region of parameter space where the surrogate is considered valid.  When
        the surrogate is called, values that fall outside of these ranges trigger a
        ``RuntimeError`` unless :meth:`enforce_training_data_parameter_range` has been
        disabled.

        Only keyword arguments are accepted; each keyword corresponds to a parameter
        name and must map to a two‑element sequence ``(lower, upper)`` describing the
        allowed range for that parameter.

        :param param_ranges: Mapping of parameter names to (lower, upper) bounds.
        :type param_ranges: ``dict`` or ``OrderedDict`` where each value is an
            iterable of two ``float``/``int`` values.

        :raises RuntimeError: If any positional arguments are supplied, or if a
            required parameter is missing from ``param_ranges``.
        :raises RuntimeError: If a supplied parameter name is not part of the
            surrogate’s ``parameter_order`` (i.e., it was not present in the training
            data).
        :raises ValueError: If the lower bound is greater than the upper bound for any
            parameter.
        :raises TypeError: If either bound is not a real number (i.e., not an instance
            of :class:`numbers.Real`).

        **Example**

        >>> surrogate.set_parameter_ranges(
        ...     temperature=(300.0, 800.0),
        ...     pressure=(1e5, 5e5)
        ... )
        """
        valid_params = self._parameter_scaler.parameter_order
        if args:
            raise RuntimeError(f"{self.__class__.__name__}.set_parameter_ranges "+
                               "does not accept positional arguments. "+
                               "All inputs must be keyword arguments.")
        for param in param_ranges:
            if param not in valid_params:
                raise RuntimeError(f"The parameter '{param}' is not a valid "+
                                   "parameter for the surrogate. Valid parameters include "+
                                    f"{valid_params}.")
            range_values = np.asarray(param_ranges[param])
            if range_values.shape != (2,):
                raise RuntimeError("Each parameter range must only have two values. "+
                                   f"Received values with shape {range_values.shape} "+
                                   f"for parameter '{param}'.")
            for idx, value in enumerate(range_values):
                hi_low = ["lower", "upper"]
                if not isinstance(value, Real):
                    raise TypeError(f"The {hi_low[idx]} bound for parameter '{param}' "+
                                     f"must be a real number. Received '{value}' of type {type(value)}.")
            if range_values[1] < range_values[0]:
                raise ValueError(f"The range for parameter '{param}' has a lower bound greater "+
                                 "than the upper bound. The lower bound is specified first! "+
                                 f"Received {range_values[0]} and then {range_values[1]} as "
                                  "the lower bound and upper bound, respectively.")
        for param in self._parameter_scaler.parameter_order:
            if param not in param_ranges:
                raise RuntimeError(f"The parameter '{param}' is required for the surrogate "+
                                   "and was not provided for the desired updated parameter ranges. "
                                   f"Received ranges for parameters {list(param_ranges.keys())}.")
        self._param_ranges = param_ranges


def _get_decomp_results(train_data, test_data, make_log_scale, decomposition_tool, logger_on=True):
    combined_data = np.vstack([train_data, test_data])
    decomp_results = decomposition_tool.generate(combined_data, make_log_scale, logger_on)
    data_scaler, decomposer, scaled_latent_data, latent_scaler = decomp_results
    scaled_latent_test_data = _apply_decomposing_and_scaling_to_data(test_data, data_scaler, 
                                                                     decomposer, latent_scaler)
    scaled_latent_train_data = _apply_decomposing_and_scaling_to_data(train_data, data_scaler, 
                                                                     decomposer, latent_scaler)
    return scaled_latent_test_data, scaled_latent_train_data, data_scaler, decomposer, latent_scaler


def _apply_decomposing_and_scaling_to_data(data, data_scaler, decomposer, 
                             latent_scaler):
    """Transform data using previously fitted response scaling, 
    decomposition, and latent-space scaling tools."""
    scaled_data = data_scaler.transform(data)
    latent_data = decomposer.transform(scaled_data)
    latent_data = _ensure_2d_array(latent_data)
    scaled_latent_test_data = latent_scaler.transform(latent_data)
    return scaled_latent_test_data


def _scale_parameters(test_params, train_params, fields_to_log_scale):
    combined_params = _combine_parameters(test_params, train_params)
    parameter_scaler_set = _make_parameter_scaler_set(combined_params, fields_to_log_scale)
    scaled_test_parameters = parameter_scaler_set.transform_as_array(test_params)
    scaled_train_parameters = parameter_scaler_set.transform_as_array(train_params)
    return parameter_scaler_set, scaled_test_parameters, scaled_train_parameters


def _combine_parameters(test_params, train_params):
    combined_params = OrderedDict()
    combined_params.update(train_params)
    for field in combined_params:
        combined_params[field] = np.hstack((combined_params[field], test_params[field]))
    return combined_params


def _train_parameter_to_pca_weight_regressor(scaled_train_params, scaled_latent_train_data, 
                                            regressor_type, regressor_kwargs,
                                            regressor_init_func):
    n_parameters = scaled_train_params.shape[1]
    regressor = regressor_init_func(regressor_type, n_parameters, regressor_kwargs)
    scaled_latent_train_data = _ensure_2d_array(scaled_latent_train_data)
    regressor.fit(scaled_train_params, scaled_latent_train_data)
    return regressor


def _score_regressor_in_latent_space(regressor, scaled_train_params, 
                                     scaled_latent_train_data, scaled_test_params, 
                                     scaled_latent_test_data, logger_on):
    train_score = _calculate_performance_metrics(regressor, scaled_train_params, 
                                                 scaled_latent_train_data)
    scaled_latent_test_data = _ensure_2d_array(scaled_latent_test_data)
    test_score = _calculate_performance_metrics(regressor, scaled_test_params, 
                                                scaled_latent_test_data)
    training_fraction = scaled_train_params.shape[0]/(scaled_train_params.shape[0]+
                                                      scaled_test_params.shape[0])
    if logger_on:
        logger.info(f"\tTraining Complete: {training_fraction*100} % of data used for training")
    return train_score, test_score


def _field_uses_pca(decomposers, field):
    """
    Return True if the field was actually decomposed with PCA.

    For fields with <= 15 features, MatCal uses _DoNothingDataTransformer,
    so the so-called latent-space scores are really scaled response-space
    regressor scores and should not be reported as PCA latent-space scores.
    """
    return not isinstance(decomposers[field], _DoNothingDataTransformer)


def _print_scores(latent_train_score, latent_test_score, 
                  data_space_train_score, data_space_test_score, decomposers=None):
    for field in latent_train_score:
        logger.info(f"\nSurrogate scores for {field}: ")

        score_message = "\tTrain:\n"

        if _field_uses_pca(decomposers, field):
            score_message += (
                f"\t\tPCA latent space score: "
                f"{latent_train_score[field]['score']}\n"
            )

        score_message += f"\t\toriginal data space score: {data_space_train_score[field]}\n"

        score_message += "\tTest:\n"

        if _field_uses_pca(decomposers, field):
            score_message += (
                f"\t\tPCA latent space score: "
                f"{latent_test_score[field]['score']}\n"
            )

        score_message += f"\t\toriginal data space score: {data_space_test_score[field]}\n"

        logger.info(score_message)


def _calculate_additional_score_metrics(train_score, test_score):
    train_score = _convert_instances_to_stats(train_score)
    test_score = _convert_instances_to_stats(test_score)
    return train_score, test_score


def _calculate_performance_metrics(regressor, param, data):
    metrics = []
    if param.shape[0] > 1:
        # R2 score only valid for more than one sample
        metrics.append(regressor.score(param, data))
    else:
        metrics.append(None)
    metrics.append(_regressor_nlpd(regressor, param, data))
    metrics.append(_regressor_rmse(regressor, param, data))
    return metrics

def _safe_regressor_metric_return(reg, y_true, handle_errors, 
    metric_func, input_values, error_value
):
    if handle_errors:
        try:
            return metric_func(reg, input_values, y_true)
        except Exception:
            return error_value

    return metric_func(reg, input_values, y_true)


def _apply_regressor_metric(
    regressor,
    input_values,
    evals,
    metric_func,
    *,
    handle_errors=False,
    error_value=np.nan,
):
    """
    Apply a metric to either a single regressor or a modal regressor.
    """
    if isinstance(regressor, _modal_regressor):
        results = np.full(evals.shape[1], error_value, dtype=float)

        for idx, reg in enumerate(regressor._mode_regressors):
            results[idx] = _safe_regressor_metric_return(reg, evals[:, idx], handle_errors, 
                metric_func, input_values, error_value
            )
        return results

    return _safe_regressor_metric_return(regressor, evals, handle_errors, 
        metric_func, input_values, error_value
    )


def _regressor_nlpd(regressor, input_values, evals):
    """
    Negative Log Predictive Density.

    Only applicable for GPR-like regressors that support
    predict(..., return_std=True).
    """
    return _apply_regressor_metric(
        regressor,
        input_values,
        evals,
        _calculate_nlpd,
        handle_errors=True,
        error_value=np.nan,
    )


def _regressor_rmse(regressor, input_values, evals):
    return _apply_regressor_metric(
        regressor,
        input_values,
        evals,
        _calculate_rmse,
    )
 
    
def _convert_instances_to_stats(scores):
    score_stats = OrderedDict()
    score_stats['score'] = np.array(scores[0])
    score_stats['nlpd'] = np.array(scores[1])
    score_stats['rmse'] = np.array(scores[2])
    return score_stats


class _modal_regressor:
    
    def __init__(self, regressor_type:str, n_inputs, regressor_kwargs):
        self._mode_regressors: List[Any] = []
        self._regressor_type = regressor_type
        self._regressor_kwargs = regressor_kwargs
        self._n_inputs = n_inputs

    def _initialize_regressors(self, n_inputs, n_modes):
        for mode_idx in range(n_modes):
            self._mode_regressors.append(_initialize_regressor(self._regressor_type, n_inputs,
                                                               self._regressor_kwargs))
    
    def fit(self, input_values, mode_values):
        n_modes = mode_values.shape[1]
        n_inputs = input_values.shape[1]
        if self._n_inputs != n_inputs:
            err_msg = f"Inconsistent input size for regressor {self._n_inputs} vs {n_inputs}."
            raise ValueError(err_msg)
        self._initialize_regressors(n_inputs, n_modes)
        for mode_idx, regressor in enumerate(self._mode_regressors):
            regressor.fit(input_values, np.atleast_2d(mode_values[:, mode_idx]).T)
    
    @property
    def num_modes(self):
        return len(self._mode_regressors)
    
    def score(self, input_values, mode_values):
        mode_scores = np.zeros(self.num_modes)
        for mode_idx, regressor in enumerate(self._mode_regressors):
            mode_scores[mode_idx] = regressor.score(input_values, mode_values[:, mode_idx])
        return mode_scores
    
    def predict(self, input_values):
        n_predictions = input_values.shape[0]
        prediction = np.zeros([n_predictions, self.num_modes])
        for mode_idx, regressor in enumerate(self._mode_regressors):
            prediction[:, mode_idx] = regressor.predict(input_values)
        return prediction        
        

class MatCalPCASurrogateBase(MatCalSurrogateBase):
               
    @property
    def parameter_order(self):
        """
        A list of strings that describe the correct order to input parameters 
        into the surrogate prediction.
        """
        return self._parameter_scaler.parameter_order

    @property
    def independent_field(self):
        """
        The name of the independent field used in the surrogate prediction
        """
        return self._interpolation_field

    @property
    def prediction_locations(self):
        """
        The array of locations that the surrogate predicts at
        """
        return self._interpolation_locations
    
    def __call__(self, *args, batch_evaluate=False, transpose=False, **kwargs)-> OrderedDict:
        """
        By executing a call on the surrogate object. [Example my_surrogate(my_parameters)]
        return a dictionary of the different field predictions

        If passing a batch array with shape ``(n_samples, n_parameters)``, call with
        ``batch_evaluate=True``. For a single sample, pass one positional value per
        parameter, keyword arguments for all parameters, or a parameter dictionary.

        :param batch_evaluate: If ``True``, treat the first positional argument as a
            batch parameter array.
        :type batch_evaluate: bool

        :param transpose: If ``True``, transpose the batch array before evaluation.
        :type transpose: bool

        :param parameters: parameter values to evaluate the surrogate at.
            If not a dict, the parameters are expected to be in an order as detailed by 
            :meth:`~matcal.core.surrogates.MatCalPCASurrogateBase.parameter_order`. 
            As an array, the input should have shape (n_samples, n_parameters).
        :type parameters: np.ndarray or list or dict

        :return: Ordered dictionary containing predicted fields and, when applicable,
            the interpolation field.
        :rtype: OrderedDict
        """
        param_names = self._parameter_scaler.parameter_order
        params_array = _process_surrogate_args_call(param_names, *args, 
                                     batch_evaluate=batch_evaluate, transpose=transpose, **kwargs)
        params_dict = _convert_param_array_to_dict(params_array, param_names)
        _check_params_in_range(params_dict, self._param_ranges, 
                               self._enforce_training_data_parameter_range)
        scaled_params = self._parameter_scaler.transform_as_array(params_dict)
        multiple_samples = False
        if scaled_params.shape[0] > 1:
            multiple_samples=True
        results = OrderedDict()
        if self._interpolation_field is not None:
            results[self._interpolation_field] = self._interpolation_locations
        for field in self._regressors:
            scaled_latent_prediction = self._regressors[field].predict(scaled_params)
            scaled_latent_prediction = scaled_latent_prediction.reshape(scaled_params.shape[0], -1)
            results[field] = self._transform_data_to_original_data_space(field, scaled_latent_prediction)
            if not multiple_samples:
                results[field] = results[field].flatten()
        return results
    
    def _transform_data_to_original_data_space(self, field, scaled_latent_data):
        latent_scaler = self._latent_scalers[field]
        latent_prediction = latent_scaler.inverse_transform(scaled_latent_data)
        scaled_prediction  = self._decomposers[field].inverse_transform(latent_prediction)
        prediction = self._data_scalers[field].inverse_transform(scaled_prediction)
        return prediction

    def _fit(train_data, test_data, train_params, test_params, fields_to_log_scale,
             decomposition_tool, surrogate_generator, param_ranges, 
             regressor_initializer, surrogate_class, logger_on=True):
        
        regressors = OrderedDict()
        decomposers = OrderedDict()
        data_scalers = OrderedDict()
        latent_scalers = OrderedDict()
        latent_train_scores = OrderedDict()
        latent_test_scores = OrderedDict()
        param_scaler, scaled_test_params, scaled_train_params = _scale_parameters(test_params, 
                                                                                  train_params, 
                                                                                  fields_to_log_scale)
        for field in train_data:
            if logger_on:
                logger.info(f"\nGenerating Surrogate for {field}")
            make_log_scale = field in fields_to_log_scale
            decomp_results = _get_decomp_results(train_data[field], test_data[field], 
                                                 make_log_scale, decomposition_tool, 
                                                 logger_on=logger_on)
            scaled_latent_test_data, scaled_latent_train_data = decomp_results[0:2]
            data_scaler, decomposer, latent_scaler = decomp_results[2:5]
            decomposers[field] = decomposer
            data_scalers[field] = data_scaler
            latent_scalers[field] = latent_scaler

            regressor_type = surrogate_generator._regressor_type
            regressor_kwargs = surrogate_generator._regressor_kwargs
            regressor = _train_parameter_to_pca_weight_regressor(scaled_train_params,
                                                                scaled_latent_train_data, 
                                                                regressor_type, regressor_kwargs, 
                                                                regressor_initializer)
            regressors[field] = regressor
            decomposers[field] = decomposer
            data_scalers[field] = data_scaler
            latent_scalers[field] = latent_scaler
            latent_scores = _score_regressor_in_latent_space(regressor, scaled_train_params, 
                                             scaled_latent_train_data, scaled_test_params, 
                                             scaled_latent_test_data, logger_on)
            latent_scores = _calculate_additional_score_metrics(latent_scores[0], latent_scores[1])
            latent_train_scores[field], latent_test_scores[field] = latent_scores
        latent_scores = [latent_train_scores, latent_test_scores]
        surrogate = surrogate_class(latent_scores, fields_to_log_scale, 
                                    surrogate_generator._interpolation_field, 
                                    surrogate_generator._interpolation_locations, 
                                    param_scaler, regressors, 
                                    decomposers, data_scalers, latent_scalers, param_ranges) 

        original_data_space_scores = _get_scores_in_original_data_space(surrogate, test_params, test_data, 
                                                               train_params, train_data)
        rmse_scores, max_scores, r2_scores = original_data_space_scores
        surrogate._set_original_data_space_scores(rmse_scores, max_scores, r2_scores)
        if logger_on:
            _print_scores(*latent_scores, *r2_scores, decomposers=decomposers)
        return surrogate


def _process_surrogate_args_call(param_names, *args,  
                                 batch_evaluate=False, transpose=False, **kwargs,):
    if batch_evaluate:
        processed_args = np.asarray(args[0], dtype=float)
        if transpose:
            processed_args = processed_args.T
    elif len(args)==1 and isinstance(args[0], (dict, OrderedDict)):
        if _all_params_exist_dict(param_names, args[0]):
            params = _convert_param_dict_to_array(args[0], param_names)
        batch_evaluate=True
        return _process_surrogate_args_call( param_names, params, batch_evaluate=batch_evaluate, 
                                            transpose=transpose)
    elif len(args) == len(param_names) and len(kwargs) == 0:
        processed_args =  np.asarray(args, dtype=float)
        if transpose:
            processed_args = processed_args.T
    elif len(args) == 0 and len(kwargs) == len(param_names):
        param_ordered_list = []
        if _all_params_exist_dict(param_names, kwargs):
            for param_name in param_names:
                param_ordered_list.append(kwargs[param_name])        
        return _process_surrogate_args_call(param_names, *param_ordered_list,  transpose=transpose)
    else:
        raise RuntimeError("Surrogate model was not called correctly. The input parameters "+
                            "are likely of the incorrect format. Check input")
    return processed_args


def _all_params_exist_dict(param_names, data_dict):
    for param_name in param_names:
        if param_name not in data_dict:
            error_message = ("All required parameters were not passed to the surrogate. "+
                f"Required parameters include:\n{param_names}\n"+
                f"Received parameters include:\n{data_dict.keys()}")
            raise RuntimeError(error_message)
    return True

def _check_params_in_range( params_dict, param_ranges, enforce_range=True):
    if not isinstance(param_ranges, (dict, OrderedDict)):
        param_ranges = _convert_param_array_to_dict(param_ranges, params_dict.keys())
    for param in params_dict:
        param_values = params_dict[param]
        bad_values = param_values > np.max(param_ranges[param])
        bad_values = (param_values < np.min(param_ranges[param])) | bad_values
        if bad_values.any() and enforce_range:
            raise RuntimeError(f"The passed parameter values for parameter '{param}' contains "+ 
                                "values outside of the trained parameter range of "+
                                f"{param_ranges[param][0]} to "+
                                f"{param_ranges[param][1]}.\n{param_values}")
   

def _get_scores_in_original_data_space(surrogate, test_params, test_data, train_params, train_data):
    rmse_train_score = _get_field_scores(surrogate, train_params, train_data, 
                                        _root_mean_squared_error)
    max_train_score = _get_field_scores(surrogate, train_params, train_data, 
                                        _max_error_inf_norm)
    r2_train_score = _get_field_scores(surrogate, train_params, train_data, _global_r2_score)

    rmse_test_score = _get_field_scores(surrogate, test_params, test_data, 
                                        _root_mean_squared_error)
    max_test_score = _get_field_scores(surrogate, test_params, test_data, 
                                        _max_error_inf_norm)
    r2_test_score = _get_field_scores(surrogate, test_params, test_data, _global_r2_score)

    rmse_scores = (rmse_train_score, rmse_test_score)
    max_scores = (max_train_score, max_test_score)
    r2_scores = (r2_train_score, r2_test_score)
    return rmse_scores, max_scores, r2_scores


def _get_field_scores(surrogate, params, data, score_function):
    """
    Compute one original data space score per predicted field.

    The score function is responsible for returning ``nan`` when a metric is not
    defined, such as global R2 with fewer than two scalar comparisons.
    """
    surrogate_data = surrogate(params)
    scores = OrderedDict()

    for field in surrogate_data:
        if field == surrogate._interpolation_field:
            continue

        surrogate_data_field = np.atleast_2d(surrogate_data[field])
        scores[field] = score_function(data[field], surrogate_data_field)

    return scores


#TODO - make function for surrogate class
#            if not isinstance(decomposer, _DoNothingDataTransformer):
#                _record_variance_behaviors(decomposer, support_information['save_filename'], field)


class MatCalMonolithicPCASurrogate(MatCalPCASurrogateBase):
    """
    This class takes the results of the :meth:`~matcal.core.surrogates.SurrogateGenerator.generate` 
    and create a callable object that can generate predictions.

    :param surrogate_information: The file path to or the lists of information generated by 
        :meth:`~matcal.core.surrogates.SurrogateGenerator.generate`.        
    """    
    name = "PCA Monolithic Regressor"
    
    def fit(train_data, test_data, train_params, test_params, 
            fields_to_log_scale, decomposition_tool,
            surrogate_generator, param_ranges, print_score=True):
        return MatCalPCASurrogateBase._fit(train_data, test_data, train_params, 
                                           test_params, fields_to_log_scale, 
                                           decomposition_tool, surrogate_generator, 
                                           param_ranges, 
                                           _initialize_regressor, __class__, 
                                           print_score)


class MatCalMultiModalPCASurrogate(MatCalPCASurrogateBase):
    """
    This class takes the results of the :meth:`~matcal.core.surrogates.SurrogateGenerator.generate`
    and create a callable object that can generate predictions.

    :param surrogate_information: The file path to or the lists of information generated by 
        :meth:`~matcal.core.surrogates.SurrogateGenerator.generate`.        
    """
    name = "PCA Multiple Regressors"
    
    def fit(train_data, test_data, train_params, test_params, 
            fields_to_log_scale, decomposition_tool,
            surrogate_generator, param_ranges, print_score=True):
        return MatCalPCASurrogateBase._fit(train_data, test_data, train_params, test_params, 
                                            fields_to_log_scale, decomposition_tool,
                                            surrogate_generator, param_ranges, _modal_regressor,
                                           __class__, print_score)


_surrogate_selection = BasicIdentifier()
_surrogate_selection.register(MatCalMultiModalPCASurrogate.name, MatCalMultiModalPCASurrogate)
_surrogate_selection.register(MatCalMonolithicPCASurrogate.name, MatCalMonolithicPCASurrogate)


def _ensure_2d_array(active_array):
    if not isinstance(active_array, np.ndarray):
        active_array = np.array(active_array)    
    if active_array.ndim == 1:
        #Reshape 1D vector to be column vector (nsamples, 1) - single feature
        active_array = active_array.reshape(-1, 1)
    return np.atleast_2d(active_array)
   

class _MatCalSurrogateWrapper:
    
    def __init__(self, surrogate):
        self._surrogate = surrogate
    
    def __call__(self, **parameters):
        """
        Wrapper to make MatCal surrogates compatible with PythonModel interface.
        
        PythonModel expects functions that accept parameters as keyword arguments.
        MatCal surrogates (both MatCalSurrogateBase and AdaptiveSurrogate) accept
        parameters as keyword arguments directly, so we pass them through.
        
        For AdaptiveSurrogate, the default surrogate_index="best" is used.
        """
        # Filter parameters to only those the surrogate knows about
        if hasattr(self._surrogate, '_parameter_scaler'):
            # MatCalSurrogateBase instances
            known_params = self._surrogate._parameter_scaler.parameter_order
            filtered_params = {k: v for k, v in parameters.items() if k in known_params}
        elif hasattr(self._surrogate, 'param_names'):
            # AdaptiveSurrogate instances (public attribute set by Study)
            known_params = self._surrogate.param_names
            filtered_params = {k: v for k, v in parameters.items() if k in known_params}
        elif hasattr(self._surrogate, '_param_names'):
            # AdaptiveSurrogate instances (private attribute, e.g. when retrieved
            # directly from study.surrogate rather than created by the study wrapper)
            known_params = self._surrogate._param_names
            filtered_params = {k: v for k, v in parameters.items() if k in known_params}
        else:
            # Unknown surrogate type or regular function - pass all parameters
            filtered_params = parameters
        
        results = self._surrogate(**filtered_params)
        return results
    
    def __getstate__(self):
        """Support pickling by returning the surrogate state."""
        return {'surrogate': self._surrogate}
    
    def __setstate__(self, state):
        """Support unpickling by restoring the surrogate."""
        self._surrogate = state['surrogate']


    
def _score_recreation(sur_values, source_values):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(source_values.reshape(1, -1))
    scaled_source = scaler.transform(source_values.reshape(1, -1))
    scaled_sur = scaler.transform(sur_values.reshape(1, -1))
    delta = scaled_source - scaled_sur
    return np.linalg.norm(delta.flatten())


def _assign_decomp(decomp_var, reconstruction_error):
    if reconstruction_error == None:
        if not (isinstance(decomp_var, str) and decomp_var == "mle"):
            if decomp_var <=0 or (decomp_var > 1 and not isinstance(decomp_var, int)):
                err_str = ("Total Explained Variance Decomposition ratio must be between 0 and 1,"+
                    f"if a float, or 1 or greater if an integer.\nPassed {decomp_var}.")
                raise RuntimeError(err_str)
        return _VarianceDecomposition(decomp_var)
    else:
        if reconstruction_error >= 1 or reconstruction_error <=0 :
            err_str = ("Reconstruction tolerance must be between 0 and 1, "+
                       f"passed {reconstruction_error}.")
            raise RuntimeError(err_str)
        return _ReconstructionDecomposition(reconstruction_error)
    

def _process_interpolation_locations(output_history, interpolation_locations, interpolation_field):
    if interpolation_field is None:
        return None

    if isinstance(interpolation_locations, Integral):
        return _get_interpolation_field(
            output_history,
            interpolation_field,
            interpolation_locations,
        )

    try:
        return np.asarray(interpolation_locations, dtype=float)
    except Exception:
        raise ValueError(
            "The surrogate generator expects an integer or array-like "
            f"set of values. Received variable of type {type(interpolation_locations)}."
        )
    

def _get_interpolation_field(output_history, interpolation_field, n_interp):
    start, end = _identify_common_region(output_history, interpolation_field)
    return np.linspace(start, end, n_interp) 


def _identify_common_region(output_history, interpolation_field):
    start = None
    end = None

    for current_array in output_history:
        cur_max = np.max(current_array[interpolation_field])
        cur_min = np.min(current_array[interpolation_field])
        if start is None:
            start = cur_min
        if end is None:
            end = cur_max
        start = np.max([start, cur_min])
        end = np.min([end, cur_max])
    return start,end   
    

def _identify_fields_of_interest(sim_list, indep_field, user_fields_of_interest):
    sim_data_fields = list(sim_list[0].field_names)

    if user_fields_of_interest is not None:
        fields_of_interest = list(user_fields_of_interest)
        _check_fields_in_keys_list(
            fields_of_interest,
            sim_data_fields,
            "training data set",
        )
    else:
        fields_of_interest = list(sim_data_fields)

    if indep_field is not None and indep_field in fields_of_interest:
        fields_of_interest.remove(indep_field)

    return fields_of_interest


class _MatCalLogScaler(BaseEstimator):
    
    def __init__(self):
        self._offset = None
        self._lower_limit = 1
        
    def fit(self, data, y=None, **fit_params):
        # interface designed to align with that of sklearn's preprocessors
        self._check_data(data)
        self._offset = np.min(data, axis=0)
    
    def transform(self, data):
        self._check_data(data)
        return np.log10(data - self._offset + self._lower_limit)

    def fit_transform(self, data, y=None, **fit_params):
        self.fit(data, y, **fit_params)
        return self.transform(data)
    
    def inverse_transform(self, trans_data):
        self._check_data(trans_data)
        return np.power(10, trans_data) + self._offset - self._lower_limit
    
    def _check_data(self, data):
        if not isinstance(data, np.ndarray):
            raise TypeError("Passed data must be of type np.ndarray")
        if data.ndim != 2:
            msg = f"Data must be of dimension 2, passed array of dimension {data.ndim}."
            if data.ndim == 1:
                msg += ("\nOne dimensional data can be mapped by using reshape(-1, 1) and "+
                       "reshape(1, -1), making it an array of multiple samples or multiple "+
                        "features, respectively. ")
            raise IndexError(msg)
        
        
class _ParameterScalerSet:
    
    def __init__(self):
        self._scalers = OrderedDict()
        
    def add_scaler(self, field_name, scaler):
        self._scalers[field_name] = scaler
    
    @property
    def parameter_order(self):
        return list(self._scalers.keys())

    def _arbitrary_transform_to_array(self, parameters, transform_method_name):
        if isinstance(parameters, (dict, OrderedDict)):
            parameters = _convert_param_dict_to_array(parameters, self.parameter_order)
        for param_index, (param_name, scaler) in enumerate(self._scalers.items()):
            param_data = parameters[:, param_index]
            param_data = _ensure_2d_array(param_data)
            method_to_call = getattr(scaler, transform_method_name)
            parameters[:, param_index] = method_to_call(param_data).flatten()
        return parameters

    def transform_as_array(self, parameter_dict):
        return self._arbitrary_transform_to_array(parameter_dict, "transform")

    def inverse_transform_as_array(self, parameter_dict):
        return self._arbitrary_transform_to_array(parameter_dict, "inverse_transform")
           

def _make_parameter_scaler_set(parameter_fields, fields_to_log_scale):
    parameter_scaler_set = _ParameterScalerSet()
    for parameter_name, parameter_values in parameter_fields.items():
        use_log_scale = parameter_name in fields_to_log_scale
        prepared_params = _ensure_2d_array(parameter_values)
        s_parameters, field_scaler = _scale_data_for_surrogate(prepared_params, use_log_scale)
        parameter_scaler_set.add_scaler(parameter_name, field_scaler)
    return parameter_scaler_set  


def _init_param_array(parameter_dict):
    n_params = len(parameter_dict)
    n_evals = _get_eval_count(parameter_dict)
    return np.zeros((n_evals, n_params))


def _get_eval_count(parameter_dict):
    first_key = list(parameter_dict.keys())[0]
    first_param_vals = parameter_dict[first_key]
    if isinstance(first_param_vals, (float, int)):
        n_evals = 1
    else:
        n_evals = len(first_param_vals)
    return n_evals


def _convert_param_array_to_dict(passed_params, parameter_order):
    if isinstance(passed_params, (dict, OrderedDict)):
        return passed_params
    else:
        passed_params = np.array(passed_params)
        out = OrderedDict()
        for param_i, param_name  in enumerate(parameter_order):
            out[param_name] = passed_params.reshape(-1, len(parameter_order))[:,param_i]
        return out


def _convert_param_dict_to_array(passed_params_dict, parameter_order):
    array = _init_param_array(passed_params_dict)
    for param_i, param_name  in enumerate(parameter_order):
        array[:,param_i] = passed_params_dict[param_name]
    return array


def _root_mean_squared_error(test_values, surrogate_values):
    """
    Compute the root mean squared error (RMSE) between reference values and
    surrogate predictions.

    The arrays are expected to represent responses with shape
    ``(n_samples, n_qois)``.

    The RMSE is calculated as

    .. math::

        \\mathrm{RMSE}
        =
        \\sqrt{
        \\frac{1}{N}
        \\sum_{i=1}^{N}
        \\left(R_i - \\hat{R}_i\\right)^2
        }

    where :math:`N` is the total number of scalar response values,
    :math:`R_i` are the reference values, and :math:`\\hat{R}_i` are the
    surrogate predictions.
    """
    test_values, surrogate_values = _prepare_metric_arrays(
        test_values,
        surrogate_values,
    )

    return float(np.sqrt(np.mean((test_values - surrogate_values) ** 2)))


def _max_error_inf_norm(test_values, surrogate_values):
    """
    Compute the maximum absolute scalar error.
    """
    test_values, surrogate_values = _prepare_metric_arrays(
        test_values,
        surrogate_values,
    )

    return float(np.linalg.norm((test_values - surrogate_values).flatten(), ord=np.inf))


def _global_r2_score(test_responses, surrogate_values):
    """
    Compute a global R2 score over all scalar response values.

    This treats all test samples and QoI locations as one pooled set of scalar
    observations. The score is defined when at least two scalar values are
    available.
    """
    test_responses, surrogate_values = _prepare_metric_arrays(
        test_responses,
        surrogate_values,
    )

    test_responses = test_responses.ravel()
    surrogate_values = surrogate_values.ravel()

    if test_responses.size < 2:
        return np.nan

    return r2_score(test_responses, surrogate_values)


def _prepare_metric_arrays(test_values, surrogate_values):
    """
    Convert metric inputs to comparable floating-point arrays.

    Metric arrays must have identical shape, except that a one-dimensional array
    with shape ``(n_samples,)`` is treated as equivalent to a singleton-column
    array with shape ``(n_samples, 1)``. This accommodates common single-output
    regressor prediction conventions without allowing arbitrary reshaping based
    only on total array size.

    Shape normalization beyond this singleton-column case should be performed by
    the caller, where sample and QoI orientation are known.
    """
    test_values = np.asarray(test_values, dtype=float)
    surrogate_values = np.asarray(surrogate_values, dtype=float)

    test_values, surrogate_values = _match_single_column_and_1d_metric_arrays(
        test_values,
        surrogate_values,
    )

    if test_values.shape != surrogate_values.shape:
        raise RuntimeError(
            "Metric arrays have incompatible shapes. "
            f"Reference shape: {test_values.shape}. "
            f"Prediction shape: {surrogate_values.shape}."
        )

    return test_values, surrogate_values


def _match_single_column_and_1d_metric_arrays(test_values, surrogate_values):
    """
    Match the safe singleton-column/1D metric-array convention.

    Some regressors return a single-output prediction with shape ``(n_samples,)``
    even when the reference data are stored as ``(n_samples, 1)``. These two
    shapes are semantically equivalent for scalar-output, per-sample metrics.

    This helper intentionally does not perform arbitrary reshaping based only on
    total array size.
    """
    if test_values.shape == surrogate_values.shape:
        return test_values, surrogate_values

    if (
        test_values.ndim == 2
        and test_values.shape[1] == 1
        and surrogate_values.ndim == 1
        and surrogate_values.shape[0] == test_values.shape[0]
    ):
        surrogate_values = surrogate_values.reshape(test_values.shape)
        return test_values, surrogate_values

    if (
        surrogate_values.ndim == 2
        and surrogate_values.shape[1] == 1
        and test_values.ndim == 1
        and test_values.shape[0] == surrogate_values.shape[0]
    ):
        test_values = test_values.reshape(surrogate_values.shape)
        return test_values, surrogate_values

    return test_values, surrogate_values


def _mean_absolute_error(test_values, surrogate_values):
    """
    Compute mean absolute error between reference values and predictions.
    """
    test_values, surrogate_values = _prepare_metric_arrays(
        test_values,
        surrogate_values,
    )
    return float(np.mean(np.abs(test_values - surrogate_values)))


def _sum_absolute_error(test_values, surrogate_values):
    """
    Compute sum of absolute errors between reference values and predictions.
    """
    test_values, surrogate_values = _prepare_metric_arrays(
        test_values,
        surrogate_values,
    )
    return float(np.sum(np.abs(test_values - surrogate_values)))


def _normalized_root_mean_squared_error(test_values, surrogate_values):
    """
    Compute normalized root-mean-squared error.

    The normalization is

    .. math::

        \\sqrt{
        \\frac{
            \\sum_i (y_i - \\hat{y}_i)^2
        }{
            \\sum_i y_i^2
        }}

    If the reference norm is zero, this falls back to RMSE.
    """
    test_values, surrogate_values = _prepare_metric_arrays(
        test_values,
        surrogate_values,
    )

    residual = test_values - surrogate_values
    denom = np.sum(test_values ** 2)

    if denom <= 0:
        return _root_mean_squared_error(test_values, surrogate_values)

    return float(np.sqrt(np.sum(residual ** 2) / denom))


def _calculate_response_error_metric(test_values, surrogate_values, metric):
    """
    Compute a deterministic response-space error/score metric.

    Supported metrics are:

    * ``"rmse"``
    * ``"mae"`` or ``"abs"``
    * ``"sum_abs"``
    * ``"nrmse"``
    * ``"max_error"``, ``"max_abs_error"``, or ``"linf"``
    * ``"r2"`` or ``"score"``

    ``"nlpd"`` is intentionally not handled here because NLPD requires
    predictive variances, not only deterministic predictions.
    """
    check_value_is_nonempty_str(metric, "metric")
    metric = metric.lower().strip()

    if metric == "rmse":
        return float(_root_mean_squared_error(test_values, surrogate_values))

    if metric in ("mae", "abs", "mean_abs_error"):
        return _mean_absolute_error(test_values, surrogate_values)

    if metric == "sum_abs":
        return _sum_absolute_error(test_values, surrogate_values)

    if metric == "nrmse":
        return _normalized_root_mean_squared_error(test_values, surrogate_values)

    if metric in ("max_error", "max_abs_error", "linf"):
        return float(_max_error_inf_norm(test_values, surrogate_values))

    if metric in ("r2", "score"):
        return float(_global_r2_score(test_values, surrogate_values))

    raise ValueError(
        "Unsupported response error metric. Supported metrics are "
        "'rmse', 'mae', 'abs', 'mean_abs_error', 'sum_abs', 'nrmse', "
        "'max_error', 'max_abs_error', 'linf', 'r2', and 'score'. "
        f"Received '{metric}'."
    )


def _calculate_nlpd(gpr, input_values, y_true):
    """
    Calculate Gaussian negative log predictive density.

    The value returned is the mean scalar NLPD over all supplied samples and
    outputs:

    .. math::

        \\mathrm{NLPD}
        =
        \\frac{1}{2}
        \\operatorname{mean}
        \\left[
            \\log(2\\pi\\sigma^2)
            +
            \\frac{(y - \\mu)^2}{\\sigma^2}
        \\right]

    ``gpr`` must support ``predict(..., return_std=True)``.
    """
    variance_floor = 1e-12

    mu, std = gpr.predict(input_values, return_std=True)

    y_true = np.asarray(y_true, dtype=float)
    mu = np.asarray(mu, dtype=float)
    std = np.asarray(std, dtype=float)

    if mu.shape != y_true.shape:
        if mu.size == y_true.size:
            mu = mu.reshape(y_true.shape)
        else:
            raise RuntimeError(
                "Gaussian Process mean prediction shape is incompatible with "
                "the supplied true values for NLPD calculation. "
                f"mu shape: {mu.shape}. y_true shape: {y_true.shape}."
            )

    if std.shape != y_true.shape:
        if std.size == y_true.size:
            std = std.reshape(y_true.shape)
        elif y_true.ndim == 2 and std.size == y_true.shape[0]:
            std = np.repeat(std.reshape(-1, 1), y_true.shape[1], axis=1)
        else:
            raise RuntimeError(
                "Gaussian Process standard-deviation prediction shape is "
                "incompatible with the supplied true values for NLPD "
                "calculation. "
                f"std shape: {std.shape}. y_true shape: {y_true.shape}."
            )

    var = std ** 2
    var = np.maximum(var, variance_floor)

    residuals = y_true - mu

    nlpd_terms = np.log(2.0 * np.pi * var) + (residuals ** 2) / var
    return float(0.5 * np.mean(nlpd_terms))


def _calculate_rmse(regressor, input_values, y_true):
    y_pred = regressor.predict(input_values)
    return _root_mean_squared_error(y_true, y_pred)
