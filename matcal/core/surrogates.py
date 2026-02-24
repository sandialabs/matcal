from abc import ABC, abstractmethod
from collections import OrderedDict
from numbers import Integral, Real
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score
from typing import Callable


from matcal.core.data import convert_dictionary_to_data
from matcal.core.logger import initialize_matcal_logger
from matcal.core.object_factory import BasicIdentifier
from matcal.core.serializer_wrapper import matcal_save
from matcal.core.state import State
from matcal.core.utilities import (check_value_is_nonempty_str, 
                                   check_item_is_correct_type, 
                                   _time_interpolate, 
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
                                        logger_on)


class SurrogateGenerator:
    """
    This class is responsible for taking source data and a parameter set 
    and generating an efficient surrogate 
    for predicting probe based quantities of interest. The generator uses
    Principal Component Analysis(PCA) to generate an efficient representation 
    of the data and then trains 
    a predictor in the latent space identified by the PCA. 
    To preform these calculations sklearn is 
    leveraged to perform the correct scaling, PCA, and predictor training required. 
    """

    def __init__(self, evaluation_information, interpolation_field=None, 
                interpolation_locations=200, 
                training_fraction=.8, surrogate_type = "PCA Multiple Regressors", 
                regressor_type="Gaussian Process", test_eval_info=None, **regressor_kwargs):
        """
        :param evaluation_information: A container of the relevant 
            information to form a surrogate off of 
            a body of data. This is intended to be based off of the results of a MatCal conducted 
            sampling study.
            In addition, previously run surrogates joblib files can be passed to rerun the surrogate
            generation process with new settings.            
        :type evaluation_information: :class:`~matcal.core.study_base.StudyResults`

        :param training_fraction: What fraction of the source data to use as training data. 
            Value should be 0 < training_fraction <= 1. If training_fraction == 1, 
            test_evaluation_information
            must be provided.
        :type training_fraction: float

        :param interpolation_field: the field that will be the independent field for surrogate results.            
        :type interpolation_field: str

        :interpolation_locations: the number of interpolation locations for the 
            surrogate to output at or an array-like of values for the interpolation locations.
            If a number of locations is given, the surrogate will linearly space the points
            over the min and max value for the interpolation field for all evaluations.
        :interpolation_locations: int or Array-like
        
        :param surrogate_type: What type of surrogate to run. Details of each are detailed in the 
            surrogate's documentation. Currently the only available 
            options are "PCA Multiple Regressors" 
            and "PCA Monolithic Regressor". The Default is set to 
            "PCA Multiple Regressors" as it has
            better performance but uses more memory than the monolithic surrogate. 
        :type surrogate_type: str

        :param regressor_type: The identifier key for what core regressor 
            form to use as the predictor. 
            Only "Random Forest" and "Gaussian Process" are accepted. Currently, MatCal
            uses the implementations of these tools from the sklearn library. 
        :type regressor_type: str

        :param test_eval_info: A container of the relevant
            information to test a surrogate generated
            from a MatCal sampling study. This data is only used and must
            be provided if training_fraction == 1.0.
        :type test_evaluation_information: :class:`~matcal.core.study_base.StudyResults`
        
        :param regressor_kwargs: A keyword selection of parameters to pass to the predictor used. 
            Please refer to the sklearn documentation for more information for what can be passed to 
            the predictors. 
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
        :type eval_set_key: str 

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
        :param decomp_var: What level of the total variance should be accounted for in the PCA
            decomposition. Values closer to 1 will keep more modes than lower values. The more modes
            kept the more difficult it can become to train the predictors. A default value of .99 is 
            chosen because it is a common conventional choice, and explains the vast majority of the 
            seen behavior, and for an appropriate data set can lead
            to very few modes being retained. 
        :type decomp_var: float
        """
        self._decomp_tool = _assign_decomp(decomp_var, reconstruction_error)

    def set_surrogate_details(self, surrogate_type="PCA Multiple Regressors", 
                              regressor_type="Gaussian Process", 
                              training_fraction=.8, interpolation_locations=None, 
                              test_eval_info=None, **regressor_kwargs):
        """
        This method provides an other avenue to alter the surrogate 
        generation parameters after initialization. 

        :param surrogate_type: What type of surrogate to run. Details of each are detailed in the 
            surrogate's documentation. Currently the only available options 
            are "PCA Multiple Regressors" 
            and "PCA Monolithic Regressor". The Default is set to 
            "PCA Multiple Regressors" as it has
            better performance but uses more memory than the monolithic surrogate. 
        :type surrogate_type: str

        :param training_fraction: What fraction of the source data to use as 
            training data. Value should be 0 < training_fraction < 1. 
        :type training_fraction: float

        :param regressor_type: The identifier key for what core regressor 
            form to use as the predictor. 
            Only "Random Forest" and "Gaussian Process" are accepted. Currently, MatCal
            uses the implementations of these tools from the sklearn library. 
        :type regressor_type: str

        :param test_eval_info: A container of the relevant
            information to test a surrogate generated
            from a MatCal sampling study. This data is only used and must
            be provided if training_fraction == 1.0.
        :type test_evaluation_information: :class:`~matcal.core.study_base.StudyResults`
        
        :param regressor_kwargs: A keyword selection of parameters to pass to the predictor used. 
            Please refer to the sklearn documentation for more information for what can be passed to 
            the predictors. 
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
        to train to the natural log of the data rather than the raw data. 
        Passing fields here will inform the surrogate and the generator that 
        these fields should be evaluated on the natural log scale. Any predictions
        given by the surrogate will be at the original scale. This just adds an 
        additional scaling/descaling step within it. Note that data that has values
        less than or equal to zero will need to be scaled or modified by the user 
        prior to selecting them as an option for log scaling.

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

    def generate(self, save_filename:str, preprocessing_function:Callable=None, 
                 plot_n_worst:int=0)->Callable:
        """
        Generates a surrogate based on the information passed to it upon initialization

        :parameter save_filename: the base of a filename without any extensions 
            to be used to record the surrogate. 
        :type save_filename: str

        :parameter preprocessing_function: an optional function that modifies
            the model data before it is passed to the tools that generate the 
            surrogate model.
        :type preprocessing_function: Callable

        :parameter source_data_dict: a dictionary of training data from which to generate
            the surrogate. Its keys are the field names for the data, rows contain
            data samples and  and columns are the data pts at each independent variable
            data point. Not intended to be an argument for users. Passing data this way 
            will take the place of any other data source. 
        :type source_data_dict: dict(str, Array-Like)
            
        :parameter plot_n_worst: Generate a number of plots that show the worst 
            recreations made by the surrogate. The number of plots made is equal to the 
            value passed to this argument. Any values less than 1 will result in no
            plots being generated or worst analysis being performed.
        :type plot_n_worst: int
            
        :return: a callable surrogate
        :rtype: :class:`~matcal.core.surrogates.MatCalPCASurrogateBase` 
        """
        check_value_is_nonempty_str(save_filename, "save_filename")
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
        matcal_save(save_filename+".joblib", new_surrogate)
        self._plot_worst_recreations(new_surrogate, params, source_dict, 
                                     plot_n_worst, save_filename)
        return new_surrogate

    def _check_test_evaluation_information_provided(self):
        if self._training_fraction == 1.0 and self._test_eval_info is None:
            raise ValueError("Test evaluations must be provided when training_fraction = 1.0.")
        
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


def _init_random_forest_surrogate(n_inputs, **kwargs):
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(**kwargs)


def _init_gp_surrogate(n_inputs, **kwargs):
    from sklearn.gaussian_process import GaussianProcessRegressor
    # reference for later for anisotropic kernel generation
    # aniso_kernel = RBF(1e-1 * np.ones(n_inputs), length_scale_bounds=(1e-5, 1e5))
    gpr = GaussianProcessRegressor(**kwargs)
    return gpr


_regressor_lookup = {"Random Forest":_init_random_forest_surrogate,
                        "Gaussian Process":_init_gp_surrogate}


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
            message = (f"      Recreation error tolerance not met, but max modes reached, "+
                       "using {max_modes} mode decomposition")
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
        self._average_scores = OrderedDict()
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
        By default the surrogate will error if called with a parameter set outside of the 
        parameter ranges used in the training data set. To call the surrogate for parameters 
        outside of the training data range, call this method with the argument set to False. 
        Adherence to the training data range can be reactivated by calling this method 
        with the argument set to True.
        
        :param ignore_training_range: bool flag to ignore training data range.
        :type ignore_training_range:
        """
        check_value_is_bool(enforce_training_data_parameter_range, 
                            "enforce_training_data_parameter_range")
        self._enforce_training_data_parameter_range = enforce_training_data_parameter_range

    def _set_native_space_scores(self, average_scores, max_scores, r2_scores):
        self._average_scores['train'] = average_scores[0]
        self._average_scores['test'] = average_scores[1]

        self._max_scores['train'] = max_scores[0]
        self._max_scores['test'] = max_scores[1]

        self._r2_scores['train'] = r2_scores[0]
        self._r2_scores['test'] = r2_scores[1]

    def set_parameter_ranges(self, *args, **param_ranges):
        """
        Update the admissible parameter ranges that the user can call the surrogate to evaluate.

        The surrogate stores, for each input parameter, a lower and upper bound that
        define the region of parameter space where the surrogate is considerd valid.  When
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
                                   f"parameter for the surrogate. Valid parameters include "+
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
    """Transform test data after scalers and decomposition tool have already 
        been trained on training data."""
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


def _print_scores(latent_train_score, latent_test_score, native_train_score, native_test_score):
    for field in latent_train_score:
        logger.info(f"\nSurrogate scores for {field}: ")
        score_message = f"\tTrain:\n"
        score_message += f"\t\tlatent space score: {latent_train_score[field]['score']}\n"    
        score_message += f"\t\tnative space score: {native_train_score[field]}\n"    
        score_message += f"\tTest:\n"
        score_message += f"\t\tlatent space score: {latent_test_score[field]['score']}\n"    
        score_message += f"\t\tnative space score: {native_test_score[field]}\n"    
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
    metrics.append(nlpd(regressor, param, data))
    metrics.append(rmse(regressor, param, data))
    return metrics


def nlpd(regressor, input_values, evals):
    """ Negative Log Predictive Density
        Only applicable for GPR
    """
    if isinstance(regressor, _modal_regressor):
        nlpd = np.zeros(evals.shape[1])
        for idx, reg in enumerate(regressor._mode_regressors):
            try:
                nlpd[idx] = _calculate_nlpd(reg, input_values, evals[:, idx])
            except:
                nlpd[idx] = None
        return nlpd 
    elif not isinstance(regressor, _modal_regressor):
        try:
            nlpd = _calculate_nlpd(regressor, input_values, evals)
        except:
            return None
        return nlpd


def _calculate_nlpd(gpr, input_values, y_true):
    mu, std = gpr.predict(input_values, return_std=True)

    y_true = y_true.ravel()
    mu = mu.ravel()

    var = std ** 2
    residuals = y_true - mu
    nlpd = 0.5 * np.mean( np.log(2 * np.pi * var) + (residuals ** 2) / var)
    
    return nlpd


def _mse(regressor, input_values, evals):
    if isinstance(regressor, _modal_regressor):
        mse = np.zeros(evals.shape[1])
        for idx, reg in enumerate(regressor._mode_regressors):
            mse[idx] = _calculate_mse(reg, input_values, evals[:, idx])
    else:
        mse = _calculate_mse(regressor, input_values, evals)
    return mse


def _calculate_mse(regressor, input_values, y_true):
    y_pred = regressor.predict(input_values)
    residuals = y_true - y_pred
    mse = np.mean( residuals ** 2)
    return mse
    
    
def rmse(regressors, input_values, evals):
    rmse = _mse(regressors, input_values, evals) ** 0.5
    return rmse
    
    
def _convert_instances_to_stats(scores):
    score_stats = OrderedDict()
    score_stats['score'] = np.array(scores[0])
    score_stats['nlpd'] = np.array(scores[1])
    score_stats['rmse'] = np.array(scores[2])
    return score_stats


class _modal_regressor:
    
    def __init__(self, regressor_type:str, n_inputs, regressor_kwargs):
        self._mode_regressors = []
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

        :param parameters: parameter values to evaluate the surrogate at.
            If not a dict, the parameters are expected to be in an order as detailed by 
            :meth:`~matcal.core.surrogates.MatCalPCASurrogateBase.parameter_order`. 
            As an array, the input should have shape (n_samples, n_parameters).
        :type parameters: np.ndarray or list or dict

        :return: A dictionary of the various field predictions.
        :rtype: dict
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
            results[field] = self._transform_data_to_native_space(field, scaled_latent_prediction)
            if not multiple_samples:
                results[field] = results[field].flatten()
        return results
    
    def _transform_data_to_native_space(self, field, scaled_latent_data):
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

        native_space_scores = _get_scores_in_native_data_space(surrogate, test_params, test_data, 
                                                               train_params, train_data)
        average_scores, max_scores, r2_scores = native_space_scores
        surrogate._set_native_space_scores(average_scores, max_scores, r2_scores)
        if logger_on:
            _print_scores(*latent_scores, *r2_scores)
        return surrogate


def _process_surrogate_args_call(param_names, *args,  
                                 batch_evaluate=False, transpose=False, **kwargs,):
    if batch_evaluate:
        processed_args = np.asarray(args[0], dtype=float)
        if transpose:
            processed_args = processed_args.T
    elif len(args)==1 and isinstance(args[0], dict or OrderedDict):
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
            error_message = (f"All required parameters were not passed to the surrogate. "+
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
   

def _get_scores_in_native_data_space(surrogate, test_params, test_data, train_params, train_data):
    average_train_score = _get_field_scores(surrogate, train_params, train_data, 
                                        _average_l2_error_norm)
    max_train_score = _get_field_scores(surrogate, train_params, train_data, 
                                        _max_error_inf_norm)
    r2_train_score = _get_field_scores(surrogate, train_params, train_data, r2_score)

    average_test_score = _get_field_scores(surrogate, test_params, test_data, 
                                        _average_l2_error_norm)
    max_test_score = _get_field_scores(surrogate, test_params, test_data, 
                                        _max_error_inf_norm)
    r2_test_score = _get_field_scores(surrogate, test_params, test_data, r2_score, 
                                          needs_length=True)


    average_scores = (average_train_score, average_test_score)
    max_scores = (max_train_score, max_test_score)
    r2_scores = (r2_train_score, r2_test_score)
    return average_scores, max_scores, r2_scores


def _get_field_scores(surrogate, params, data, score_function, needs_length=False):
    surrogate_data = surrogate(params)
    scores = OrderedDict()
    eval_score=True
    for field in surrogate_data:
        surogate_data_field = np.atleast_2d(surrogate_data[field])
        if needs_length:
            if len(surogate_data_field) > 1:
                eval_score = surogate_data_field.shape[1] > 1
            else:
                eval_score = len(surogate_data_field) > 1
        if field != surrogate._interpolation_field:
            if eval_score:
                scores[field] = score_function(data[field], surogate_data_field)
            else:
                scores[field] = np.nan
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
    name = "PCA Monolythic Regressor"
    
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
        results = self._surrogate(parameters)
        for key, value in results.items():
            results[key] = value.flatten()
        return results 
   
    
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
    elif isinstance(interpolation_locations, (np.ndarray)):
        return interpolation_locations
    elif isinstance(interpolation_locations, Integral):
        return _get_interpolation_field(output_history, interpolation_field, 
                                 interpolation_locations)
    else:
        raise ValueError("The surrogate generator expects an integer or array-like "
            f"set of values. Received variable of type {type(interpolation_locations)}.")
    

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
    

def _identify_fields_of_interest(sim_list,  indep_field, user_fields_of_interest):
    sim_data_fields = sim_list[0].field_names
    if user_fields_of_interest is not None:
        fields_of_interest = user_fields_of_interest
        _check_fields_in_keys_list(fields_of_interest, sim_data_fields, "training data set")
    else:
        fields_of_interest = sim_data_fields
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


def _average_l2_error_norm(test_values, surrogate_values):
    #expects arrays to be sized (n_samples, n_qois)
    return (np.linalg.norm(test_values - surrogate_values)
            / test_values.shape[1])


def _max_error_inf_norm(test_values, surrogate_values):
    #expects arrays to be sized (n_samples, n_qois)
    return np.linalg.norm((test_values - surrogate_values).flatten(), ord=np.inf)
