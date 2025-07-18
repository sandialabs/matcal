from abc import ABC, abstractmethod
from collections import OrderedDict
from glob import glob
from numbers import Integral, Real
import numpy as np
from typing import Callable

from matcal.core.data import convert_dictionary_to_data
from matcal.core.data_importer import FileData
from matcal.core.logger import initialize_matcal_logger
from matcal.core.object_factory import BasicIdentifier
from matcal.core.serializer_wrapper import matcal_save, matcal_load
from matcal.core.state import State
from matcal.core.utilities import (check_value_is_nonempty_str, 
                                   check_item_is_correct_type, 
                                   _sort_numerically, _time_interpolate, 
                                   _find_smallest_rect)

logger = initialize_matcal_logger(__name__)

surrogate_restart_suffix = 'source_information'

class _DoNothingDataTransformer:
    def inverse_transform(self, source_data):
        return source_data

class _VarianceDecomposition:
    
    def __init__(self, goal_variance):
        self._goal_variance = goal_variance
        
    def generate(self, source_data, make_log_scale):
        return _convert_data_and_make_basis(source_data, self._goal_variance, make_log_scale)
    

class _ReconstructionDecomposition:
    
    def __init__(self, reconstruction_tol:float):
        self._reconstruction_tol = reconstruction_tol
        
    def generate(self, source_data, make_log_scale):
        return _tune_data_decomposition(source_data, make_log_scale,  self._reconstruction_tol)


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
                regressor_type="Gaussian Process", **regressor_kwargs):
        """
        :param evaluation_information: A container of the relevant 
            information to form a surrogate off of 
            a body of data. This is intended to be based off of the results of a MatCal conducted 
            sampling study.
            In addition, previously run surrogates joblib files can be passed to rerun the surrogate
            generation process with new settings.            
        :type evaluation_information: Union[:class:`~matcal.core.surrogates.StudyToSurrogateInfo`, 
            :class:`~matcal.core.study_base.StudyBase`]

        :param training_fraction: What fraction of the source data to use as training data. 
            Value should be 0 < training_fraction < 1. 
        :type training_fraction: float

        :param interpolation_field: the field that will be t
            he independent field for surrogate results.            
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

        :param regressor_kwargs: A keyword selection of parameters to pass to the predictor used. 
            Please refer to the sklearn documentation for more information for what can be passed to 
            the predictors. 
        """
        self._interpolation_field = interpolation_field
        self._input_parameter_history = None
        self._training_data_history = None
        self._interpolation_locations = interpolation_locations
        self._eval_info = evaluation_information
        self._model_name = None
        self._state = None
        self._training_fraction  = training_fraction
        self._surrogate_type = surrogate_type
        self._regressor_type = regressor_type
        self._regressor_kwargs = regressor_kwargs
        self._decomp_tool = _assign_decomp(.99, None)

        self._fields_to_log_scale = []
        self._train_score = OrderedDict()
        self._test_score = OrderedDict()
        self._need_to_update_data = True

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
            check_value_is_nonempty_str(model_name, "model_name", 
                                        "SurrogateGenerator.set_model_and_state")
            self._model_name = model_name
        if state is not None:
            check_item_is_correct_type(state, (str, State), 
                                       "SurrogateGenerator.set_model_and_state", 
                                       "state")
            self._state = state
        self._need_to_update_data = True
        
    def _update_source_data(self, evaluation_information, interpolation_field, 
                            interpolation_locations):
        parsed_eval_info = _parse_evaluation_info(evaluation_information, self._model_name)  
        self._input_parameter_history, _sim_hist_data_collection = parsed_eval_info                                                          
        training_data_history = _select_state_data(self._state, _sim_hist_data_collection)
        
        self._interpolation_locations = _process_interpolation_locations(training_data_history, 
                                                                         interpolation_locations, 
                                                                         interpolation_field)
        self._training_data_history = training_data_history
        self._need_to_update_data = False
    def set_PCA_details(self, decomp_var=.99, reconstruction_error = None):
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
                              **regressor_kwargs):
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

        :param regressor_kwargs: A keyword selection of parameters to pass to the predictor used. 
            Please refer to the sklearn documentation for more information for what can be passed to 
            the predictors. 
        """
        self._training_fraction  = training_fraction
        self._surrogate_type = surrogate_type
        self._regressor_type = regressor_type
        self._regressor_kwargs = regressor_kwargs
        if (interpolation_locations is not None):
            self._need_to_update_data = True
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
        self._fields_to_log_scale += field_names

    def generate(self, save_filename:str, preprocessing_function:Callable=None, 
                 plot_n_worst:int=12)->Callable:
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
        check_value_is_nonempty_str(save_filename, "save_filename", "SurrogateGenerator.generate")
        if self._need_to_update_data:
            self._update_source_data(self._eval_info, self._interpolation_field, 
                                     self._interpolation_locations)
        self._training_data_history = _apply_preprocessing_function(preprocessing_function, 
                                                                    self._training_data_history)

        fields_of_interest = _identify_fields_of_interest(self._training_data_history, 
                                                          self._interpolation_field)
        source_dict = _process_training_data(self._training_data_history, fields_of_interest,
                                           self._interpolation_locations, self._interpolation_field)
        param_fields = _import_parameter_hist(self._input_parameter_history)
        param_ranges = _package_parameter_ranges(self._input_parameter_history)
        support_information = {'parameter_ranges':param_ranges, 
                "interpolation_field":self._interpolation_field,
                'interpolation_locations':self._interpolation_locations, 
                'training_fraction': self._training_fraction,
                'regressor_type': self._regressor_type, 
                'regressor_kwargs': self._regressor_kwargs, 'save_filename': save_filename}
        logger.info(f'Generating and scoring {self._regressor_type} surrogates. '+
                    'The ideal score is 1.0.')
        surrogate_class = _surrogate_selection.identify(self._surrogate_type)
        new_surrogate = surrogate_class.fit(param_fields, source_dict, self._fields_to_log_scale,
                                            self._decomp_tool, support_information)
        self._plot_worst_recreations(new_surrogate, param_fields, source_dict, 
                                     plot_n_worst, save_filename)
        return new_surrogate
    
    def _plot_worst_recreations(self, surrogate, parameters, source_data, n_worst, save_filename):
        if n_worst < 1:
            return
        import matplotlib.pyplot as plt
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
        check_item_is_correct_type(preprocessing_function, Callable,
                                    "SurrogateGenerator.generate",
                                    "preprocessing_function")
        for idx, data in enumerate(training_data_history):
            processed_data = preprocessing_function(training_data_history[idx])
            if isinstance(processed_data, (dict, OrderedDict)):
                processed_data = convert_dictionary_to_data(processed_data)
            training_data_history[idx] = processed_data
    return training_data_history    


def _process_training_data(training_data_list, 
                         fields_of_interest,
                         interpolation_locations, 
                         interpolation_field):
    
    processed_data = _initialize_processed_data(training_data_list, fields_of_interest,
                                                 interpolation_locations)
    
    for idx, data in enumerate(training_data_list):
        for field in fields_of_interest:
            data_field = data[field]
            if interpolation_locations is not None and interpolation_field is not None:
                data_field =  _time_interpolate(interpolation_locations, 
                                                data[interpolation_field], 
                                                data_field)
            processed_data[field][idx, :] = data_field
    return processed_data


def _initialize_processed_data(training_data_list, processed_fields,
                               interpolation_locations):
    processed_data = OrderedDict()
    n_evals = len(training_data_list)
    for field in processed_fields:
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
    from sklearn.gaussian_process.kernels import RBF
    iso_kernel = RBF(1e-1, length_scale_bounds=(1e-5, 1e5))
    # reference for later for anisotropic kernel generation
    # aniso_kernel = RBF(1e-1 * np.ones(n_inputs), length_scale_bounds=(1e-5, 1e5))
    #gpr = GaussianProcessRegressor(kernel=iso_kernel, **kwargs)
    gpr = GaussianProcessRegressor(**kwargs)
    return gpr


_regressor_lookup = {"Random Forest":_init_random_forest_surrogate,
                        "Gaussian Process":_init_gp_surrogate}


def _initialize_regressor(regressor_type, n_inputs, regressor_kwargs):
    return _regressor_lookup[regressor_type](n_inputs, **regressor_kwargs)


def _import_and_interpolate(file_search_string, 
                            fields_of_interest, 
                            interp_field, 
                            interp_loc):
    files_to_import =_sort_numerically(_get_file_list(file_search_string))
    interp_data = _initialize_storage(fields_of_interest, interp_loc, files_to_import)
    for file_idx, filename in enumerate(files_to_import):
        current_data = FileData(filename)
        _add_current_data_to_storage(fields_of_interest, interp_data, file_idx, current_data, 
                                     interp_loc, interp_field)
    return interp_data


def _add_current_data_to_storage(fields_of_interest, interp_data, file_idx, current_data, 
                                 interp_loc, interp_field):
    for field in fields_of_interest:
        interp_data[field][file_idx,:] = _time_interpolate(interp_loc, current_data[interp_field], 
                                                           current_data[field])


def _get_file_list(file_search_string):
    files_to_import  = glob(file_search_string)
    return files_to_import


def _initialize_storage(fields_of_interest, interp_loc, files_to_import):
    interp_data = {}
    n_features = len(interp_loc)
    n_samples = len(files_to_import)
    for field in fields_of_interest:
        interp_data[field] = np.zeros([n_samples, n_features])
    return interp_data


def _scale_data_for_surrogate(data_array, make_log=False):
    """
    Expects the data as n_samples x n_features
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    if make_log:
        scaler = Pipeline([('log', MatCalLogScaler()), ('standard', StandardScaler())])
    else:
        scaler = StandardScaler()
        
    scaler.fit(data_array) 
    scaled_data = scaler.transform(data_array)
    return scaled_data, scaler


def _decompose_with_pca(data, var_tol):
    """
    Expects data as n_samples x n_features
    """
    from sklearn.decomposition import PCA
    pca = PCA(n_components=var_tol, svd_solver='full')
    transformed_data = pca.fit_transform(data)
    if isinstance(var_tol, Integral):
        logger.info(f"Generated PCA decomposition with {pca.n_components_} components.")
    elif isinstance(var_tol, Real):
        logger.info(f"Generated PCA decomposition with {pca.n_components_}"
                    f" components using {var_tol} variance explanation.")
    else:
        logger.info(f"Generated PCA decomposition with {pca.n_components_}"
                    f" components using option \'{var_tol}\'.")
    return transformed_data, pca


def _use_pca_to_decompose_if_many_features(data, var_tol=.99):
    """
    Expects data as n_samples x n_features
    """
    if data.shape[1] > 15:
        return _decompose_with_pca(data, var_tol)
    else:
        return data, _DoNothingDataTransformer()


def _import_parameter_hist(parameter_history):
    return OrderedDict(parameter_history)


def _package_parameter_ranges(param_history):
    out_dict = {}
    for name, p in param_history.items():
        out_dict[name] = (np.min(p), np.max(p))
    return out_dict


def _convert_data_and_make_basis(source_data, decomp_variance, make_log_scale):
    scaled_data, data_scaler = _scale_data_for_surrogate(source_data, make_log_scale)
    latent_data, decomposer = _use_pca_to_decompose_if_many_features(scaled_data, decomp_variance)
    latent_data = _ensure_2d_array(latent_data, 1)
    scaled_latent_data, latent_scaler = _scale_data_for_surrogate(latent_data)
    return data_scaler,decomposer,scaled_latent_data,latent_scaler


def _tune_data_decomposition(source_data, make_log_scale, reconstruction_error_tol:float=1e-3, 
                             max_modes:int=10):
    scaled_data, data_scaler = _scale_data_for_surrogate(source_data, make_log_scale)
    logger.info("  Tuning decomposition to meet recreation error tolerance of "+
                f"{reconstruction_error_tol}, up to a limit of {max_modes} modes")
    for mode_count in range(max_modes):
        kept_modes = mode_count + 1
        logger.info(f"    Analyzing {kept_modes} mode decomposition")
        latent_data, decomposer = _use_pca_to_decompose_if_many_features(scaled_data, kept_modes)
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
    latent_data = _ensure_2d_array(latent_data, 1)
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
    
    @abstractmethod
    def _save():
        """"""
        
    @abstractmethod
    def _load(self, tool_package):
        """"""
    @property
    def scores(self):
        return self._scores
        
    @abstractmethod
    def __call__(self, parameters)-> OrderedDict:
        """"""
        
    def __init__(self, surrogate_information):
        self._scores = None
        self._load(surrogate_information) 


def _get_data(source_dict, field):
    source_data = source_dict[field]
    return source_data

def _train_parameter_to_pca_weight_regressor(scaled_parameters, field, scaled_latent_data,
                        training_fraction, regressor_type, regressor_kwargs, regressor_init_func):
    from sklearn.model_selection import train_test_split
    n_fold_validation = 1
    _train_score = []
    _test_score = []
    best_regressor = None
    best_test_score = -1e4
    for training_repeat in range(n_fold_validation):
        data_split_results = train_test_split(scaled_parameters, 
                                                scaled_latent_data, 
                                                train_size=training_fraction)
        param_train, param_test, data_train, data_test = data_split_results
        n_parameters = scaled_parameters.shape[1]
        regressor = regressor_init_func(regressor_type, n_parameters, regressor_kwargs)
        data_train = _ensure_2d_array(data_train, 1)
        data_test = _ensure_2d_array(data_test, 1)
        regressor.fit(param_train, data_train)
        _train_score.append(regressor.score(param_train, data_train))
        new_test_scores = regressor.score(param_test, data_test)
        worst_new_test_score = np.min(new_test_scores)
        if worst_new_test_score > best_test_score:
            best_regressor = regressor
        _test_score.append(new_test_scores)
    if best_regressor == None:
        raise RuntimeError("Failed to train a regressor that performs well enough on test data.")

    logger.info(f"    Training Complete: {training_fraction*100} % of data used for training")
    logger.info(f"    Surrogate scores for {field} over {n_fold_validation} repeats:")
    _train_score = _convert_instances_to_stats(_train_score, "Train")
    _test_score = _convert_instances_to_stats(_test_score, "Test")
    return regressor, _train_score, _test_score


def _convert_instances_to_stats(scores, score_set_name):
    score_stats = OrderedDict()
    a_scores = np.array(scores)
    score_stats['mean'] = np.mean(a_scores, axis = 0)
    score_stats['max'] = np.max(a_scores, axis = 0)
    score_stats['min'] = np.min(a_scores, axis = 0)
    
    score_message = f"\t{score_set_name}:\n"
    for name, value in score_stats.items():
        score_message += f"\t {name} : {value}\n"    
    logger.info(score_message)
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
            regressor.fit(input_values, mode_values[:, mode_idx])
    
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
    
    def _load(self, parsed_information):
        param_info, score_info, self._field_surrogate_tools = parsed_information[0:3]
        self._log_data_fields, self._interp_field, self._interp_loc = parsed_information[3:]
        self._scores = {'train': score_info[0], 'test':score_info[1]}
        _, self._parameter_order, self._parameter_scaler = param_info

    def _save(source_dict, field_surrogate_tools, parameter_scaler,
              fields_to_log_scale, train_scores, test_scores, support_information, load_key):
        all_surrogate_tools = MatCalPCASurrogateBase._package_surrogate_for_export(source_dict, 
                            field_surrogate_tools, parameter_scaler,fields_to_log_scale,
                            train_scores, test_scores, support_information)
        full_save_filename = support_information['save_filename']+".joblib"
        matcal_save(full_save_filename, [load_key, all_surrogate_tools])
        return all_surrogate_tools

    def _package_surrogate_for_export(source_dict, field_surrogate_tools, parameter_scaler,
                                      fields_to_log_scale, train_scores, test_scores,
                                      support_information):
        packed_parameter_details = [support_information['parameter_ranges'], 
                                    parameter_scaler.parameter_order, parameter_scaler]
        scores = [train_scores, test_scores]
        all_surrogate_tools = [packed_parameter_details, scores,  field_surrogate_tools,
                               fields_to_log_scale, support_information['interpolation_field'], 
                               support_information['interpolation_locations']]
        return all_surrogate_tools
        
    @property
    def parameter_order(self):
        """
        A list of strings that describe the correct order to input parameters 
        into the surrogate prediction.
        """
        return self._parameter_order

    @property
    def independent_field(self):
        """
        The name of the independent field used in the surrogate prediction
        """
        return self._interp_field

    @property
    def prediction_locations(self):
        """
        The array of locations that the surrogate predicts at
        """
        return self._interp_loc
    
    def __call__(self, parameters)-> OrderedDict:
        """
        By executing a call on the surrogate object. [Example my_surrogate(my_parameters)]
        return a dictionary of the different field predictions

        :param parameters: a list or array of parameter values to evaluate the surrogate at.
            The parameters are expected to be in an order as detailed by 
        :meth:`~matcal.core.surrogates.MatCalPCASurrogateBase.parameter_order`

        :return: A dictionary of the various field predictions.
        :rtype: dict
        """
        params_dict = convert_array_to_dict(parameters, self.parameter_order)
        scaled_params = self._parameter_scaler.transform_to_array(params_dict)
        results = OrderedDict()
        if self._interp_field is not None:
            results[self._interp_field] = self._interp_loc
        for field, field_tools in self._field_surrogate_tools.items():
            surrogate, decomposer, data_scaler, latent_scaler = field_tools
            scaled_latent_prediction = surrogate.predict(scaled_params)
            scaled_latent_prediction = scaled_latent_prediction.reshape(scaled_params.shape[0], -1)
            latent_prediction = latent_scaler.inverse_transform(scaled_latent_prediction)
            scaled_prediction  = decomposer.inverse_transform(latent_prediction)
            prediction = data_scaler.inverse_transform(scaled_prediction)
            results[field] = prediction
        return results

    def _fit(parameter_fields, source_history, fields_to_log_scale, decomposition_tool,
             support_information, regressor_initializer, surrogate_class, load_key):
        parameter_scaler_set = _make_parameter_scaler_set(parameter_fields, fields_to_log_scale)
        scaled_parameters = parameter_scaler_set.transform_to_array(parameter_fields)
        field_surrogate_tools = OrderedDict()
        train_scores = OrderedDict()
        test_scores = OrderedDict()
        for field in list(source_history.keys()):
            logger.info(f"\nGenerating Surrogate for {field}")
            source_data = _get_data(source_history, field)
            make_log_scale = field in fields_to_log_scale
            decomp_results = decomposition_tool.generate(source_data, make_log_scale)
            data_scaler, decomposer, scaled_latent_data, latent_scaler = decomp_results
            if not isinstance(decomposer, _DoNothingDataTransformer):
                _record_variance_behaviors(decomposer, support_information['save_filename'], field)
            training_results = _train_parameter_to_pca_weight_regressor(scaled_parameters, 
                    field, scaled_latent_data, 
                    support_information['training_fraction'], support_information['regressor_type'],
                    support_information['regressor_kwargs'], regressor_initializer)
            regressor, train_scores[field], test_scores[field] = training_results
            packed_field_tools = [regressor, decomposer, data_scaler, latent_scaler]
            field_surrogate_tools[field] = packed_field_tools
        all_surrogate_tools = surrogate_class._save(source_history, field_surrogate_tools,
                                                    parameter_scaler_set, 
                                                    fields_to_log_scale, train_scores, test_scores, 
                                                    support_information, load_key)
        return surrogate_class(all_surrogate_tools) 

    
def load_matcal_surrogate(surrogate_savefile:str) -> MatCalPCASurrogateBase:
    """
    Load a MatCal PCA surrogate model from a saved file.

    This function loads the surrogate model information from the specified file and
    returns an instance of the appropriate surrogate class based on the loaded data.
    
    The function uses the `matcal_load` function to read the surrogate 
    model information from the file.
    The appropriate surrogate class is selected based on the `load_key` 
    obtained from the loaded data.

    :parameter surrogate_savefile: The path to the file where the surrogate model is saved.
    :type surrogate_savefile: str

    :return: An instance of a class derived from `MatCalPCASurrogateBase` containing 
        the loaded surrogate model.
    :rtype: :class:`~matcal.core.surrogates.MatCalPCASurrogateBase`


    """
    load_key, surrogate_info = matcal_load(surrogate_savefile)
    return _surrogate_selection.identify(load_key)(surrogate_info)


class MatCalMonolithicPCASurrogate(MatCalPCASurrogateBase):
    """
    This class takes the results of the :meth:`~matcal.core.surrogates.SurrogateGenerator.generate` 
    and create a callable object that can generate predictions.

    :param surrogate_information: The file path to or the lists of information generated by 
        :meth:`~matcal.core.surrogates.SurrogateGenerator.generate`.        
    """    
    name = "PCA Monolythic Regressor"
    
    def fit(parameter_history, source_history, fields_to_log_scale, decomposition_variance,
            support_information):
        return MatCalPCASurrogateBase._fit(parameter_history, source_history, fields_to_log_scale, 
                                           decomposition_variance, support_information,
                                           _initialize_regressor, __class__, 
                                           MatCalMonolithicPCASurrogate.name)


class MatCalMultiModalPCASurrogate(MatCalPCASurrogateBase):
    """
    This class takes the results of the :meth:`~matcal.core.surrogates.SurrogateGenerator.generate`
    and create a callable object that can generate predictions.

    :param surrogate_information: The file path to or the lists of information generated by 
        :meth:`~matcal.core.surrogates.SurrogateGenerator.generate`.        
    """
    name = "PCA Multiple Regressors"
    
    def fit(parameter_history, source_history, fields_to_log_scale, decomposition_tool,
            support_information):
        return MatCalPCASurrogateBase._fit(parameter_history, source_history, 
                                           fields_to_log_scale, decomposition_tool,
                                           support_information, _modal_regressor,
                                           __class__, MatCalMultiModalPCASurrogate.name)


_surrogate_selection = BasicIdentifier()
_surrogate_selection.register(MatCalMultiModalPCASurrogate.name, MatCalMultiModalPCASurrogate)
_surrogate_selection.register(MatCalMonolithicPCASurrogate.name, MatCalMonolithicPCASurrogate)


def _ensure_2d_array(active_array, constrained_dim=0):
    if not isinstance(active_array, np.ndarray):
        active_array = np.array([active_array])    
    if active_array.ndim == 1:
        if constrained_dim==0:
            active_array = active_array.reshape(1, -1)
        else:
            active_array = active_array.reshape(-1, 1)
    elif active_array.ndim == 2:
        aa_shape = active_array.shape
        if aa_shape[constrained_dim] > aa_shape[1-constrained_dim]:
            active_array = active_array.T
    return active_array


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
    

def _identify_fields_of_interest(sim_list,  indep_field):
    field_of_interest = sim_list[0].field_names
    if indep_field is not None:
        field_of_interest.remove(indep_field)
    return field_of_interest


class MatCalLogScaler:
    
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
        
        
class ParameterScalerSet:
    
    def __init__(self):
        self._scalers = OrderedDict()
        
    def add_scaler(self, field_name, scaler):
        self._scalers[field_name] = scaler
    
    @property
    def parameter_order(self):
        return list(self._scalers.keys())
    
    def transform_to_array(self, parameter_dict):
        n_params = len(self._scalers)
        first_key = list(self._scalers.keys())[0]
        n_evals = self._get_eval_count(parameter_dict, first_key)
        a = np.zeros((n_evals, n_params))
        for field_i, (field_name, scaler) in enumerate(self._scalers.items()):
            param_data = parameter_dict[field_name]
            param_data = _ensure_2d_array(param_data, 1)
            a[:, field_i] = scaler.transform(param_data).flatten()

        return a

    def _get_eval_count(self, parameter_dict, first_key):
        first_param_vals = parameter_dict[first_key]
        if isinstance(first_param_vals, (float, int)):
            n_evals = 1
        else:
            n_evals = len(first_param_vals)
        return n_evals
            

def _make_parameter_scaler_set(parameter_fields, fields_to_log_scale):
    parameter_scaler_set = ParameterScalerSet()
    for parameter_name, parameter_values in parameter_fields.items():
        use_log_scale = parameter_name in fields_to_log_scale
        prepared_params = _ensure_2d_array(parameter_values, 1)
        s_parameters, field_scaler = _scale_data_for_surrogate(prepared_params, use_log_scale)
        parameter_scaler_set.add_scaler(parameter_name, field_scaler)
    return parameter_scaler_set  


def convert_array_to_dict(passed_params, parameter_order):
    if isinstance(passed_params, (dict, OrderedDict)):
        return passed_params
    else:
        passed_params = np.array(passed_params)
        out = OrderedDict()
        for param_i, param_name  in enumerate(parameter_order):
            out[param_name] = passed_params.reshape(-1, len(parameter_order))[:,param_i]
        return out