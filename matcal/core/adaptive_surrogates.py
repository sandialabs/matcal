"""
This module contains adaptive surrogates. 
"""
from collections import OrderedDict
import copy
import numpy as np
import os
from scipy import stats
from sklearn.metrics import r2_score

from matcal.core.logger import initialize_matcal_logger
from matcal.core.objective import SimulationResultsSynchronizer
from matcal.core.parameter_studies import HaltonStudy
from matcal.core.state import State
from matcal.core.utilities import (check_value_is_positive_integer, 
                                   check_value_is_array_like_of_reals, 
                                   check_value_is_nonempty_str, 
                                   check_value_is_array_like_of_reals, 
                                   check_value_is_positive_real, 
                                   check_value_is_bool, 
                                   check_value_is_nonnegative_integer)
from matcal.core.serializer_wrapper import matcal_save
from matcal.core.surrogates import _average_l2_error_norm, _max_error_inf_norm

logger = initialize_matcal_logger(__name__)


def _get_parameter_bounds(parameters):
    param_bounds = []
    for name, parameter in parameters.items():
        param_bounds.append(np.atleast_2d([parameter.get_lower_bound(),
                                          parameter.get_upper_bound()]))
    bounds = np.r_[*param_bounds]
    return bounds


def _get_variable_from_bounds(bounds):
    from pyapprox.variables import IndependentMarginalsVariable

    marginals = [
        stats.uniform(bound[0], bound[1] - bound[0]) for bound in bounds
    ]
    return IndependentMarginalsVariable(marginals)


def _get_pyapprox_variable_transformer(bounds):
    from pyapprox.variables.transforms import AffineTransform
    variable = _get_variable_from_bounds(bounds)
    return AffineTransform(variable)


def _setup_sparse_grid_surrogate(n_parameters, n_qois):
    from pyapprox.surrogates.univariate.base import ClenshawCurtisQuadratureRule
    from pyapprox.surrogates.affine.multiindex import DoublePlusOneIndexGrowthRule
    from pyapprox.surrogates.univariate.lagrange import UnivariateLagrangeBasis
    from pyapprox.surrogates.sparsegrids.combination import (
        AdaptiveCombinationSparseGrid,
        MaxNSamplesSparseGridSubspaceAdmissibilityCriteria,
        VarianceRefinementCriteria
        )
    
    quad_rule = ClenshawCurtisQuadratureRule(store=True, bounds=[-1., 1.])
    bases_1d = [UnivariateLagrangeBasis(quad_rule, 3) for dim_id in range(n_parameters)]
    growth_rule = DoublePlusOneIndexGrowthRule()
    sg = AdaptiveCombinationSparseGrid(n_qois, n_parameters)
    sg.setup(
        MaxNSamplesSparseGridSubspaceAdmissibilityCriteria(np.inf),
        VarianceRefinementCriteria(),
        bases_1d,
        growth_rule
    )
    return sg


class AdaptiveSurrogate:
    """
    Stores the surrogate and training and test information regarding the surrogate
    and the progress of training for the surrogate.

    Can also be used to call the surrogate objects for predictions 
    using the surrogate models. Since all iterations of the surrogate are 
    stored, any version of the surrogate can be called.
    """

    def __init__(self, target_field_name, indep_variable_name, 
                 indep_variable_values, variable_transformer, 
                 test_params, test_responses, param_names):
        """
        Create an :class:`AdaptiveSurrogate` instance.

        :param str target_field_name: Name of the model field that the surrogate
            will approximate (e.g., ``"temperature"``).
        :param str indep_variable_name: Name of the auxiliary independent variable
            (e.g., ``"time"`` or ``"x_position"``) that will be attached to the
            surrogate output.
        :param indep_variable_values: The values of the independent variable at
            which the surrogate should be evaluated.
        :type indep_variable_values: array‑like of real numbers
        :param variable_transformer: Object that maps model parameters to the
            canonical space required by the surrogate library.
        :type variable_transformer: object with ``map_to_canonical`` and
            ``map_from_canonical`` methods
        :param test_params: Parameter samples used for testing the surrogate.
        :type test_params: :class:`numpy.ndarray` of shape ``(n_parameters, n_test)``  
        :param test_responses: Corresponding model responses for the test
            parameter samples.
        :type test_responses: :class:`numpy.ndarray` of shape
            ``(n_test, n_qois)``  
        :param param_names: Ordered list of parameter names that define the
            mapping between positional arguments and model parameters.
        :type param_names: list[str]

        The constructor stores the supplied information and prepares internal
        containers that will hold the surrogate objects, error histories and
        sample counts as the adaptive training proceeds.
        """

        self._surrogates: list = []         
        self._average_errors: list[float] = [] 
        self._max_errors: list[float] = []
        self._r2_scores: list[float] = []    
        self._sample_counts: list[int] = []    
        self._target_field_name: str = target_field_name
        self._indep_variable_name = indep_variable_name
        self._indep_variable_values = np.asarray(indep_variable_values)
        self._variable_transformer = variable_transformer
        self._test_params = test_params
        self._test_responses = test_responses
        self._param_names = param_names
        self._transpose_inputs = False
        self._transpose_outputs = False

    def _add_iteration(
        self,
        surrogate, 
        nsamples 
        ) -> None:
        self._surrogates.append(copy.deepcopy(surrogate))
        if self._variable_transformer is not None:
            params = self._variable_transformer.map_to_canonical(self._test_params)
        else:
            params = self._test_params
        surrogate_values = self(params, batch_evaluate=True)
        average_l2_error = _average_l2_error_norm(self._test_responses, surrogate_values)
        max_abs_error = _max_error_inf_norm( self._test_responses, surrogate_values)

        self._average_errors.append(average_l2_error)
        self._max_errors.append(max_abs_error)
        self._r2_scores.append(r2_score(self._test_responses, surrogate_values))
        self._sample_counts.append(nsamples)

    @property
    def current_surrogate(self):
        """Return the most recent surrogate (or ``None`` if no iteration yet)."""
        return self._surrogates[-1] if self._surrogates else None

    @property
    def average_error_history(self):
        """Returns the list of errors for the average error history. The average
        error is calculated using
       
        .. math::
            E_{avg}
            = \\frac{\\lVert \\mathbf{R}_{\\text{test}} - \\hat{\\mathbf{R}} \\rVert_{2}}
               {N}

        where :math:`{N`} is the number of QoIs in the response, :math:`{R}_{\\text{test}}` is the 
        test responses and :math:`{\\hat{R}}` is the surrogate responses. 
        """
        return self._average_errors

    @property
    def max_error_history(self):
        """Returns the list of errors for the max error history. The max
        error is calculated using
       
        .. math::
            E_{max}
            = \\lVert \\mathbf{R}_{\\text{test}} - \\hat{\\mathbf{R}} \\rVert_{\\infty}
              

        where :math:`{R}_{\text{test}}` is the 
        test responses and :math:`{\\hat{R}}` is the surrogate responses. 
        """
        return self._max_errors

    @property
    def sample_count_history(self):
        """Returns a list containing the number of samples used by each surrogate
           training step."""
        return self._sample_counts

    def __call__(self, *args, surrogate_index=-1, batch_evaluate=False, **kwargs):
        """
        Evaluate a stored surrogate model. This is represented in mathematical notation by 

        .. math::
            \\hat{\\mathbf{R}} = S_i\\bigl(\\mathbf{p}\\bigr),

        where :math:`\\mathbf{R}` is the vector (or matrix) of output responses,
        :math:`\\mathbf{p}` is the vector (or matrix) of input
        parameters and :math:`S_i` denotes the selected surrogate model.

        The surrogate objects includes all the
        models generated during the adaptive training process.  This method
        provides an interface for retrieving predictions from any
        version of the surrogate during training using the ``surrogate_index``
        keyword argument.

        :param *args: Positional arguments representing the model parameters.
            The accepted calling patterns are:

            * **Single‑sample evaluation** (``batch_evaluate=False``) – a
            tuple whose length equals the number of model parameters.
            The values are interpreted in the order defined by the 
            :class:`matcal.core.parameters.ParameterCollection` or order of parameters
            passed to the adaptive surrogate training study.

            * **Batch evaluation** (``batch_evaluate=True``) – a single argument
            that must be a two‑dimensional ``np.ndarray`` of shape
            ``(n_parameters, n_samples)``.  The array is forwarded unchanged
            to the surrogate.

        :type *args: tuple or np.ndarray

        :param surrogate_index: Index of the surrogate to use. ``-1`` selects the
            most recent surrogate. Any valid list index is accepted.
        :type surrogate_index: int, optional

        :param batch_evaluate: When ``True`` the call is interpreted as a *batch*
            evaluation; otherwise it is a *single‑sample* evaluation.
        :type batch_evaluate: bool, optional

        :param **kwargs: Keyword arguments that map each parameter name to the desired
            evaluation value. This calling style is mutually exclusive 
            with the positional ``*args`` form.
        :type **kwargs: dict

        :return: The surrogate prediction ``\\hat{\\mathbf{R}}``.  For a single
            sample, this is a dictionary 
            containing two one‑dimensional arrays of length ``n_qois`` 
            with the independent variable and the corresponding target variable
            response. For a batch evaluation,  it is a two‑dimensional array of shape
            ``(n_samples, n_qois)``.
        :rtype: np.ndarray or dict(str, np.ndarray)

        :raises RuntimeError: If the supplied arguments do not match any of the
            supported calling conventions (wrong number of positional arguments,
            missing or extra keyword arguments, etc.).
        """
        surrogate = self._surrogates[surrogate_index]
        if batch_evaluate:
            params_array = np.asarray([args])
            if self._transpose_inputs:
                params_array = params_array.T
            response = surrogate(params_array)
            if self._transpose_response:
                response = response.T
            return response
        elif len(args) == len(self._param_names) and len(kwargs) == 0:
            params_array = np.asarray([args])
            if self._transpose_inputs:
                params_array = params_array.T
            response = surrogate(params_array)[0]
            if self._transpose_response:
                response = response.T
        elif len(args) == 0 and len(kwargs) == len(self._param_names):
            param_ordered_list = []
            for param_name in self._param_names:
                if param_name not in kwargs:
                    error_message = (f"All required parameters were not passed to the surrogate."+
                        f"Required parameters:\n{self._param_names}\n"+
                        f"Received parameters:\n{kwargs.keys()}")
                    raise RuntimeError(error_message)
                param_ordered_list.append(kwargs[param_name])
            
            return self(*param_ordered_list, surrogate_index=surrogate_index)
        else:
            raise RuntimeError("Surrogate model was not called correctly. The input parameters "+
                               "are likely of the incorrect format. Check input")

        return {self._target_field_name:response, 
                self._indep_variable_name:self._indep_variable_values}


class AdaptiveSurrogateStudyBase(HaltonStudy):
    def __init__(self, *parameters):
        super().__init__(*parameters)

        self._bounds = _get_parameter_bounds(self._parameter_collection)

        self._target_field_name = None
        self._independent_variable=None
        self._independent_variable_values=None
        self._evaluation_set_added = False
        self._results_synchronizer = None

        self._surrogate = None
        self._variable_transformer = None

        self._max_training_samples=None
        self._number_of_test_samples=None
        self._training_batch_number = 1
        self.set_max_training_samples()

        self._average_l2_error_goal = 1e-2
        self._max_abs_error_goal = 1e-1

        self._surrogate_save_filename = None

    def set_error_stopping_criteria(self,
                                    average_l2_error_goal: float | None = None,
                                    max_abs_error_goal: float | None = None):
        """
        Set the error thresholds that determine when the adaptive surrogate
        training stops.

        The stopping criteria are examined in :meth:`_stopping_criterion_met`.  When
        the *average L2* error falls below ``average_l2_error_goal`` **or** the
        *maximum absolute* error falls below ``max_abs_error_goal`` the training
        loop terminates (provided at least two batches have been evaluated).

        :param average_l2_error_goal: Desired upper bound for the average L2
            error. Must be a positive number. If ``None`` the current goal is
            left unchanged. Default is 1e-2.
        :type average_l2_error_goal: float, optional

        :param max_abs_error_goal: Desired upper bound for the maximum absolute
            error. Must be a positive number. If ``None`` the current goal is
            left unchanged. Default is 1e-1.
        :type max_abs_error_goal: float, optional
        """
        if average_l2_error_goal is not None:
            check_value_is_positive_real(average_l2_error_goal, "average_l2_error_goal")
            self._average_l2_error_goal = float(average_l2_error_goal)

        if max_abs_error_goal is not None:
            check_value_is_positive_real(max_abs_error_goal, "max_abs_error_goal")
            self._max_abs_error_goal = float(max_abs_error_goal)

    def set_independent_variable(self, independent_variable, 
                                 independent_variable_values):
        """
        Specify an independent (auxiliary) variable and the values at which the surrogate
        will be evaluated.

        This variable is **not** a model input; it is a field that will be used later
        (for example, a spatial coordinate, a time step, or any other scalar quantity
        that the surrogate should be conditioned on).  The surrogate will be trained
        on the parameter samples generated by the study and then provide a response at each
        value supplied in ``independent_variable_values``.

        :param independent_variable: Name of the independent variable (e.g. ``"time"``,
            ``"x_position"``, …) that will be attached to the surrogate output.
        :type independent_variable: str
        :param independent_variable_values: A 1‑D array‑like collection of real numbers
            indicating the points at which the surrogate should be queried.
        :type independent_variable_values: array‑like of real numbers
        """
        check_value_is_nonempty_str(independent_variable, "independent_variable")
        self._independent_variable = independent_variable
        check_value_is_array_like_of_reals(independent_variable_values,
                                           "independent_variable_values")
        self._independent_variable_values = independent_variable_values
        logger.debug(f"Independent variable field set to {self._independent_variable}")
        logger.debug(f"Independent variable values set to {self._independent_variable_values}")

    def set_number_of_test_samples(self, number_of_test_samples):
        """
        Set the number of samples that will be used for testing.
        By default we test against ``max_training_samples``/20 or 
        the number of parameters*10, whichever is greater.
        
        :param max_training_samples: desired number of test samples
        :type max_training_samples: int
        """
        check_value_is_positive_integer(number_of_test_samples, "number_of_test_samples")
        self._number_of_test_samples = number_of_test_samples
        super().set_number_of_samples(self._number_of_test_samples)
        logger.debug(f"number_of_test_samples set to {self._number_of_test_samples}")

    def set_max_training_samples(self, max_training_samples=1000):
        """
        Set the maximum number of training samples you want to be run for 
        Sparse Grid surrogate generation. If the convergence criteria is not reached, 
        the training for the surrogate will stop after max_training_samples has been 
        reached.
        
        :param max_training_samples: desired maximum number of samples
        :type max_training_samples: int
        """
        check_value_is_positive_integer(max_training_samples, "max_training_samples")
        self._max_training_samples = max_training_samples
        logger.debug(f"max_training_samples set to {self._max_training_samples}")
        if self._number_of_test_samples is None:
            self.set_number_of_test_samples(self._set_default_number_of_test_samples())

    def _set_default_number_of_test_samples(self) -> int:
        """
        Compute the default number of test samples.

        The rule follows:
        * ``max_training_samples // 20`` (integer division) **or**
        * ``n_parameters * 10``

        Whichever of the two values is larger becomes the default.

        :returns: Default number of test samples for the current study.
        :rtype: int
        """
        candidate_a = int(self._max_training_samples // 20)  # floor division → int
        candidate_b = self._number_parameters * 10
        return max(candidate_a, candidate_b)

    def set_target_field_name(self, target_field_name):
        """Specify the field name for the response that the surrogate model 
        will seek to replicate. This is generally a model response such as temperature, 
        load, etc.

        :param target_field_name: the name of the field that the surrogate will 
            replicate
        :type target_field_name: str
        """
        check_value_is_nonempty_str(target_field_name, "target_field_name")
        self._target_field_name = target_field_name

    def _run_test_sampling(self):
        """
        Execute the initial test‑sampling phase in a dedicated sub‑directory.

        This method:

        1. Sets up a temporary working directory via
           :meth:`_setup_test_sampling_working_directory`.
        2. Calls the parent ``HaltonStudy`` ``launch`` method to generate the
           test samples.
        3. Restores the original working directory.
        4. Stores the formatted test parameters and responses for later use.

        :raises RuntimeError: If the test‑sampling launch fails.
        """
        orig_working_directory = self._update_work_dir_for_test_sampling()
        super().launch()
        test_params = self._format_params(self._results)
        test_responses = self._format_output(self._results)
        self._reset_study_after_test_sampling_generation(orig_working_directory)
        return test_params, test_responses

    def _update_work_dir_for_test_sampling(self):
        """
        Prepare a temporary working directory for the initial test‑sampling phase.

        The method follows the same logic that was previously in
        :meth:`launch`:

        * If the user has already set a working directory (``self._working_directory``)
          it is suffixed with ``"_test_samples"``.
        * If no working directory was set, a new directory named ``"test_samples"``
          is used.

        The original directory (or ``None`` if none was set) is returned so the
        caller can restore it after the test run has completed.

        :return: The original working‑directory path before it was modified,
                 or ``None`` if the study had no working directory set.
        :rtype: str | None
        """
        original_dir = None
        if self._working_directory is not None:
            original_dir = self._working_directory
            self._working_directory = self._working_directory + "_test_samples"
        else:
            self._working_directory = os.path.abspath("test_samples")
        return original_dir

    def _reset_study_after_test_sampling_generation(self, orig_working_directory):
        self._working_directory = orig_working_directory
        self._results = None
        self._next_evaluation_id_number = 1

    def add_evaluation_set(self, model, state=None):
        """
        Add an evaluation set that uses a 
        :class:`~matcal.core.objective.SimulationResultsSynchronizer`
        generated from the study’s independent variable, independent‑variable values,
        and target field name.

        .. warning::
            For adaptive surrogates, this can only be called **once** as the
            training points are adaptively chosen based on the response of
            interest.

        This method is a thin wrapper around :meth:`StudyBase.add_evaluation_set`.  
        It accepts only the *model* (required) and an optional *state* argument.
        ``state`` must be a single :class:`~matcal.core.state.State` instance; a
        collection of states is **not** supported.  If ``state`` is ``None`` MatCal's
        default state will be used.
        
        The synchronizer is built automatically from the attributes that
        were defined via :meth:`set_independent_variable` and
        :meth:`set_target_field_name`.

        :param model: The model that will generate the simulation data.
        :type model: :class:`~matcal.core.models.ModelBase`
        :param state: The single state to which the evaluation set should be applied.
            If ``None`` the model’s default state is used.
        :type state: :class:`~matcal.core.state.State`, optional

        :raises RuntimeError: If the required attributes for the synchronizer
            (independent variable, its values, or target field name) have not been set.
        """
        if self._evaluation_set_added:
            raise RuntimeError(
                "add_evaluation_set can only be called once for a "
                f"{self.__class__.__name__} instance because adaptivity "
                "is only supported for a single model and single response of interest."
            )
        self._evaluation_set_added = True

        if state is not None and not isinstance(state, State):
            raise TypeError(
                f"{self.__class__.__name__}.add_evaluation_set expects ``state`` "
                "to be a single `State` instance (or None)."
            )

        self._results_synchronizer = self._make_simulation_results_synchronizer()
        super().add_evaluation_set(
            model,
            objectives=self._results_synchronizer,
            data=None,
            states=state,
        )

    def _make_simulation_results_synchronizer(self):
        """
        Build a :class:`~matcal.core.objective.SimulationResultsSynchronizer`
        that will be used by the surrogate study.

        The synchronizer evaluates the *simulation* at the user‑provided
        independent‑variable locations and extracts the target field as the
        dependent quantity.

        :return: Configured ``SimulationResultsSynchronizer`` instance.
        :rtype: SimulationResultsSynchronizer
        :raises RuntimeError: If any of the required attributes have not been set.
        """
        if self._independent_variable is None:
            raise RuntimeError(
                "Independent variable name has not been set. Call "
                "`set_independent_variable` before creating the synchronizer."
            )

        if self._target_field_name is None:
            raise RuntimeError(
                "Target field name has not been set. Call "
                "`set_target_field_name` before creating the synchronizer."
            )
        return SimulationResultsSynchronizer(
            self._independent_variable, self._independent_variable_values,
            self._target_field_name          
        )

    def launch(self):
        """
        Run the initial test‑sampling study in a dedicated sub‑directory,
        then continue with the adaptive Sparse‑Grid workflow.

        The test‑sampling phase is performed by a standard **HaltonStudy** 
        to generate the required test points. If the user called
        :meth:`StudyBase.set_working_directory` before launching the study,
        the test‑sampling directory is created by appending the suffix
        ``\"_test_samples\"`` to the user‑provided path. Otherwise, the test 
        samples are run in a local directory named ``\"test_samples\"``.
        After the test sample study finishes, the original working directory is restored
        and the surrogate‑building routine is started.
        """
        test_params, test_responses = self._run_test_sampling()
        param_names = self._parameter_collection.get_item_names()
        self._variable_transformer = self._variable_transformer_factory(self._bounds)
        self._surrogate = AdaptiveSurrogate(self._target_field_name, self._independent_variable, 
                                            self._independent_variable_values, 
                                            self._variable_transformer, 
                                            test_params, test_responses, param_names)
        
        self._run_study = self._perform_adaptive_surrogate_batch_sampling
        super().launch()

    def _stopping_criterion_met(self, training_batch_number):
        stop = False
        if training_batch_number > 1:
            if np.abs(self._surrogate.average_error_history[-1]) <= self._average_l2_error_goal:
                logger.info(f"Average L2 norm score converged! "+
                            f"Final score: {self._surrogate.average_error_history[-1]}")
                stop=True
            elif np.abs(self._surrogate.max_error_history[-1]) <=self._max_abs_error_goal:
                logger.info(f"Max absolute error score converged! "+
                            f"Final score: {self._surrogate.max_error_history[-1]}")
                stop=True
        if self._results.number_of_evaluations >self._max_training_samples:
            logger.info("Surrogate not converged yet, but maximum training "+
                        "samples reached. Exiting.")
            stop=True
        return stop
        
    def _matcal_evaluate_parameter_sets_batch_adaptive_training(self, parameter_sets_from_pyapprox):
        self._populate_parameter_evaluations_adaptive(parameter_sets_from_pyapprox)
        current_batch = len(self._surrogate.sample_count_history)
        logger.info(f"Active Learning Batch {current_batch+1}. ")
        if current_batch > 1:
            logger.info(f"Currently the surrogate is trained on "+
                        f"{self._surrogate.sample_count_history[-1]} samples.")
        logger.info("................................................................")
        eval_meth = super()._matcal_evaluate_parameter_sets_batch
        batch_results = eval_meth(self._parameter_sets_to_evaluate, 
                                  is_restart=self._restart, ignore_missing_restart_file=True)
        return self._format_batch_results(batch_results, parameter_sets_from_pyapprox)

    def _add_parameter_evaluation(self, **p):
      super()._add_parameter_evaluation(**p)

    @property
    def surrogate(self):
        """
        Return the :class:`~matcal.core.adaptive_surrogates.AdaptiveSurrogate` 
        instance that holds the
        surrogate models and their training history.

        :return: The surrogate object, or ``None`` if the study has not yet
            created one (i.e., before :meth:`launch` is called).
        :rtype: :class:`~matcal.core.adaptive_surrogates.AdaptiveSurrogate` | None
        """
        return self._surrogate
    
    def set_surrogate_save_filename(self, filename):
        """
        Set the path used to save the surrogate object after each training batch.

        The surrogate (an :class:`~matcal.core.adaptive_surrogates.AdaptiveSurrogate` 
        instance) is periodically saved to
        disk with :func:`matcal.core.serializer_wrapper.matcal_save`.  The filename
        must be a non‑empty string that ends with the ``.joblib`` extension.
        The directory
        component of the path is not created automatically; it must already exist
        or be created by the user prior to calling this method.

        :param str filename: Full path (absolute or relative) to the file that will
            store the surrogate.  The filename **must** end with ``.joblib``.
            Example: ``"my_model_sparse_grid_surrogate.joblib"`` or
            ``"/tmp/surrogate.joblib"``.

        :raises ValueError: If *filename* does not contain the required ``.joblib``
            suffix or is empty.
        :raises TypeError: If *filename* is not a string.
        """
        check_value_is_nonempty_str(filename, "filename")
        if ".joblib" not in filename:
            raise ValueError("The save filename for the Adaptive Surrogate Study " +
                f"must end with \".joblib\". Passed filename is \"{filename}\".")
        self._surrogate_save_filename = filename

    @property
    def surrogate_save_filename(self):
        """
        Retrieve the filename (including the ``.joblib`` extension) that will be
        used to save the surrogate object after each training batch.

        :return: The absolute or relative path supplied via
            :meth:`set_surrogate_save_filename`, or ``None`` if no filename has been set.
        :rtype: str | None
        """
        return self._surrogate_save_filename

    def _format_params(self, results):
        params_formatted = []
        for param in results.parameter_history:
            params_formatted.append(results.parameter_history[param])
        return np.array(params_formatted)  


class SparseGridAdaptiveSurrogateStudy(AdaptiveSurrogateStudyBase):
    """
    The SparseGridAdaptiveSurrogateStudy builds a Sparse Grid adaptive surrogate
    using the PyApprox library. They generally behave well for larger parameter spaces 
    and problems with discontinuities in the response of interest. 
    Some downsides for these surrogates
    is that one must be trained independently for each response of interest. 
    As a result, this surrogate requires only a single model and state be passed to it.
    It also requires that a target field name be specified for building the surrogate that 
    signifies the response of interest for the surrogate.
    """

    _variable_transformer_factory= _get_pyapprox_variable_transformer

    def _format_output(self, results):
        model_name = self._get_model_names()[0]
        objective = self._results_synchronizer
        state_name = results.simulation_history[model_name].state_names[0]
        qois = results.qoi_history[f"{model_name}:{objective.name}"]
        sim_qois = qois.simulation_qois
        nsamples = results.number_of_evaluations
        nqois = len(self._independent_variable_values)
        data = np.zeros((nsamples, nqois))
        for idx, sim_qoi in enumerate(sim_qois):
            data[idx,:] = sim_qoi[state_name][0][self._target_field_name]
        return data
    
    def _perform_adaptive_surrogate_batch_sampling(self): 
        from pyapprox.interface.model import ModelFromVectorizedCallable

        n_qois = len(self._independent_variable_values)
        canonical_model = ModelFromVectorizedCallable(n_qois, self._number_parameters, 
                                    self._matcal_evaluate_parameter_sets_batch_adaptive_training)

        if self._surrogate_save_filename is None:
            self.set_surrogate_save_filename(f"{self._get_model_names()[0]}_sparse_grid_surrogate.joblib")
        sg = _setup_sparse_grid_surrogate(self._number_parameters, n_qois)    
        while sg.step(canonical_model):
            self._surrogate._add_iteration(sg, self._results.number_of_evaluations)
            logger.info(f"Training samples: {self._surrogate.sample_count_history[-1]}")
            training_batch_number = len(self._surrogate.sample_count_history)
            matcal_save(self._surrogate_save_filename, self._surrogate)
            if self._stopping_criterion_met(training_batch_number):
                break
            logger.info("Surrogate not converged yet.")
            logger.info(f"Average L2 norm error score: {self._surrogate.average_error_history[-1]}")
            logger.info(f"Max absolute error score: {self._surrogate.max_error_history[-1]}")

        return self._results

    def _populate_parameter_evaluations_adaptive(self, samples):
        samples = self._variable_transformer.map_from_canonical(samples)
        super()._populate_parameter_evaluations(samples.T)

    def _format_batch_results(self, batch_results, parameter_sets_from_pyapprox):
        model_name = self._get_model_names()[0]
        objective_name = self._results_synchronizer.name
        state_name = self._results.simulation_history[model_name].state_names[0]
        formatted_results = np.zeros((parameter_sets_from_pyapprox.shape[1],
                                      len(self._independent_variable_values)))
        for idx, qoi in enumerate(batch_results["qois"]):
            qoi = qoi[model_name][objective_name]
            formatted_results[idx, :] = qoi.simulation_qois[state_name][0][self._target_field_name]
        return formatted_results
      

def _fit_surrogate_model(eval_info, interpolation_field, interpolation_locations, 
                         test_eval_info, save_filename='voronoi_surrogate', 
                         cv_test_set=False):
    from matcal.core.surrogates import SurrogateGenerator
    surrogate_generator = SurrogateGenerator(eval_info, training_fraction=1.0, 
                                             interpolation_field=interpolation_field, 
                                             interpolation_locations=interpolation_locations, 
                                             test_eval_info=test_eval_info, 
                                             cv_test_set=cv_test_set)
    return surrogate_generator.generate(save_filename)
        

class VoronoiAdaptiveSurrogateStudy(AdaptiveSurrogateStudyBase):
    
    def _return_none(*args, **kwargs):
        return None
    
    _variable_transformer_factory = _return_none

    def __init__(self, *parameters):
        """Initialize the VoronoiAdaptiveSurrogateStudy
        """
        super().__init__(*parameters)

        self._num_initial_samples = None
        self._test_eval_info = None
        self.set_number_of_initial_samples()

        self._voronoi_type = None
        self._finite_only = None
        self._iterative_updates = None
        self._thin = None
        self._random_selection = None
        self.set_voronoi_sampling_options()

        self._nsplits = None
        self._nmax_folds = None
        self._nmax_loo = None
        self._cv_scale = None
        self._cv_metric = None
        self._group_kfold = None
        self.set_cross_validation_options()
        
        self._nbatch_samples = []
        self.test_eval_info = None

        self._convergence_metric = None
        self._eps = None
        self.set_convergence_criteria()

        self._current_surrogate_score = {}
        self._current_surrogate_score['score'] = []
        self._current_surrogate_score['nlpd'] = []
        self._current_surrogate_score['rmse'] = []

        self._seed = None
            
    def _update_surrogate_score(self):
        self._current_surrogate_score['score'].append(_get_surrogate_metric(self._surrogate, 'score'))
        self._current_surrogate_score['nlpd'].append(_get_surrogate_metric(self._surrogate, 'nlpd'))
        self._current_surrogate_score['rmse'].append(_get_surrogate_metric(self._surrogate, 'rmse'))
         
    def _build_boundary_hull(self):
        from scipy.spatial import ConvexHull, Delaunay
        self._boundary_points = self._make_nd_grid(2)
        self._boundary_hull = ConvexHull(self._boundary_points)
        self._boundary_hull_eq = self._boundary_hull.equations # (nfacet, ndim + 1)
        self._boundary_hull_V, self._boundary_hull_b = self._boundary_hull_eq[:, :-1],\
            self._boundary_hull_eq[:, -1] # normal, offset
        self._bhullD = Delaunay(self._boundary_points)

    def _make_nd_grid(self, npts_along_dim):
        grid_pts = []
        for param_index in np.arange(self._number_parameters):
            grid_pts.append(np.linspace(self._bounds[param_index][0], self._bounds[param_index][1],
                                        npts_along_dim))
        coords = np.meshgrid(*grid_pts)
        coords_ravel = [np.asarray(coords[i]).ravel() for i in np.arange(self._number_parameters)]
        return np.vstack(tuple(coords_ravel)).T

    def set_number_of_initial_samples(self, num_initial_samples=None):
        """
        :param num_initial_samples: The number of samples to initiate the algorithm with.
        The initial samples are used to train the initial surrogate and built the 
        initial voronoi tessellation. Default 10*ndim.
        :type initial_training_length: None or int
        """
        if num_initial_samples is None:
            self._num_initial_samples = 10*self._number_parameters
        else:
            check_value_is_positive_integer(num_initial_samples, "num_initial_samples")
            self._num_initial_samples = num_initial_samples

    def set_voronoi_sampling_options(self, voronoi_type='full', 
                                     finite_only=False, 
                                     iterative_updates = True, 
                                     thin=None,
                                     random_selection=None):
        """Set options pertaining to the voronoi sampling algorithm. Properties
        that can be altered are listed below.
        
        
        :param vornoi_type: Defines which Vornoi-based sampling strategy to use.
        Supported options are:
            * 'full': Constructs the full Voronoi tessellation over all points (Default)
            * 'local': Constructs a local Voronoi tessellation using only nearby
            points determined by k-nearest neighbors. This can reduce computational
            cost in high dimensions.
        :type voronoi_type: str
        
        :param finite_only: If True, only Vornoi vertices that lie inside the 
        convex hull defined by the boundary points are consided as candidate sample
        locations. If False, all vertices are considered, and those lying outside
        the parameter bounds are clipped back to the convex hull. This is more flexible but
        can be more computationally expensive, especially in high dimensions. Default False.
        :type finite_onlye: bool
        
        :param iterative_updates: If True, the Voronoi tessellation is recomputed 
        after each new sample is added, promoting a more space-filling design.
        If False, the tessellation is updated once per batch after all samples 
        in the batch are selected. This can be faster but may result in sample 
        clustering. Default True.
        :type iterative_updates: bool

        :param thin: If specified, every nth candidate sample location is selected as a new
        sample location. This can significantly reduce computational cost in high-dimensional spaces.
        Default None.
        :type thin: int or None
        
        :param random_selection: If sepecified, this defines the number of candidate sample
        locations that are randomly selected as new samples. This provides an 
        alternative way to reduce computational cost in high-dimensional problems. Default None.
        :type random_selection: int or None
        """
        check_value_is_nonempty_str(voronoi_type, "voronoi_type")
        if voronoi_type.lower() not in ['full', 'local']:
            raise ValueError(f"Voronoi type must be either 'full' or 'local', recieved '{voronoi_type}'.")
        else:
            self._voronoi_type = voronoi_type.lower()
        check_value_is_bool(finite_only, "finite_only")
        self._finite_only = finite_only
        check_value_is_bool(iterative_updates, "iterative_updates")
        self._iterative_updates = iterative_updates
        if thin is not None:
            check_value_is_bool(thin, "thin")
            self._thin = thin
        if random_selection is not None:
            check_value_is_bool(random_selection, "random_selection")
        if self._random_selection is not None and self._thin is not None:
            raise ValueError("Only one of 'thin' and 'random_selection' can be activated. Not both.")
    
    def set_convergence_criteria(self, eps=1e-4, convergence_metric='nlpd'):
        """
        Convergence is determined by comparing RMSE or NLPD of
        surrogate between two successive batches.

        :param convergence_metric: Choose from root mean squared error ('rmse') 
        or negative log posterior density ('nlpd') to track surrogate performance
        at each batch iteration. This metric is used to determine if the surrogate
        has converged according to eps. Default 'nlpd'.
        :type convergence metric: str
        
        :param eps: Tolerance for surrogate convergence. Default 1e-4.
        :type eps: float 
        """
        self._eps = eps
        self._convergence_metric = convergence_metric

    def set_cross_validation_options(self, nsplits=5, nmax_folds=3, nmax_loo=10, cv_scale=1.0,
                                     cv_metric='rmse', group_kfold=False):
        """Set options for cross validation. Properties that can be altered are listed below.
        
        :param nsplits: The number of folds to use in k-fold cross validation. 
        If nsplits = 0, k-fold cross-validation is skipped entirely and new samples
        are instead selected from every region of the Voronoi tessellation defined 
        by the current set of training samples.
        :type nsplits: int
        
        :param nmax_folds: Points in the folds with the highest k-fold error 
        (the top nmax_folds) define the Voronoi regions from which new samples 
        will be drawn. 
        :type nmax_folds: int
        
        :param nmax_loo: Points with the largest leave-one-out cross-validation (LOOCV)
        errors (the top nmax_loo). These define the Voronoi regions from which new 
        samples will be drawn. If nmax_loo = 'all', then new samples are drawn from
        all Voronoi regions defined by nmax_folds, and leave-one-out cross-validation
        is not performed.
        :type nmax_loo: int or 'all'
        
        :param cv_scale: Optional scaling applied to output before calculating errors in
        cross-validation and leave-one-out cross-validation. This can be used to 
        balance error magnitude across dimensions or outputs.
        :type scale: float
        
        :param cv_metric: Determines which metric is used when computing errors during
        cross-validation. Supported options are:
            * rmse -- root mean squared error (Default)
            * nlpd -- negative log posterior density
        :type cv_metric: str
        
        :param group_kfold: If True, samples are grouped using k-means clustering
        prior to k-fold cross-validation so that nearby points are allways assigned
        to the same fold. This prevents spatially correllated points from being split
        across training and validation sets. If False, folds are assigned randomly
        by the standard KFold algorithm. 
        :type group_kfold: bool
        """
        check_value_is_nonnegative_integer(nsplits, "nsplits")
        self._nsplits = nsplits
        check_value_is_positive_integer(nmax_folds, "nmax_folds")
        self._nmax_folds = nmax_folds
        if isinstance(nmax_loo, str):
            if nmax_loo != 'all':
                raise ValueError(f"If the {__class__} 'nmax_loo' parameter is a string, "+
                    "it must be 'all'.")
        else:
            try:   
                check_value_is_positive_integer(nmax_loo, "nmax_loo")
            except TypeError:
                raise TypeError(f"The {__class__} 'nmax_loo' parameter must be a positive integer "+
                                f"or the string 'all'. Recieved value {nmax_loo}.")
            except  ValueError:
                raise ValueError(f"The {__class__} 'nmax_loo' parameter must be a positive integer "+
                                f"recieved value {nmax_loo}.")
        self._nmax_loo = nmax_loo
        check_value_is_positive_real(cv_scale, "cv_scale")
        self._cv_scale = cv_scale
        check_value_is_nonempty_str(cv_metric, "cv_metric")
        self._cv_metric = cv_metric
        valid_cv_metrics = ['rmse', 'nlpd']
        if self._cv_metric not in valid_cv_metrics:
            raise ValueError("cv_metric not implemented. 'cv_metric' must one of"
                             " 'rmse', 'nlpd'")
        check_value_is_bool(group_kfold, "group_kfold")
        self._group_kfold = group_kfold
    
    def _format_output(self, results):
        from matcal.core.data import convert_data_to_dictionary
        model_name = self._get_model_names()[0]
        state_name = results.simulation_history[model_name].state_names[0]
        sim_history = self._results.simulation_history[model_name][state_name]
        nsamples = results.number_of_evaluations
        data = []
        for nn in np.arange(nsamples):
            data.append(convert_data_to_dictionary(sim_history[nn]))
        return data
    
    def _reset_study_after_test_sampling_generation(self, orig_working_directory):
        self._test_eval_info = copy.deepcopy(self._results)
        super()._reset_study_after_test_sampling_generation(orig_working_directory)
        
    def _perform_adaptive_surrogate_batch_sampling(self):
        if self._surrogate_save_filename is None:
            self.set_surrogate_save_filename(f"{self._get_model_names()[0]}_voronoi_adaptive_surrogate.joblib") 
        self._build_boundary_hull()
        self.param_names = self._parameter_collection.get_item_names()
        training_params, training_data = self._run_initial_training_samples()
        batch_number = 0
        while not self._stopping_criterion_met(batch_number):
            logger.info(f"Active Learning Batch {batch_number+1}."
                        f" Currently {self._nbatch_samples[-1]} samples.")
            logger.info("................................................................")
            new_points = self._create_voronoi_tess_and_choose_new_samples(batch_number, 
                                                                          training_params, 
                                                                          training_data)
            self._populate_parameter_evaluations(new_points)
            self._matcal_evaluate_parameter_sets_batch(self._parameter_sets_to_evaluate, 
                                                       is_restart=self._restart)
            training_params, training_data = self._train_surrogate_with_current_results()
            batch_number += 1
        return self._results

    def _stopping_criterion_met(self, training_batch_number):
        stop = super()._stopping_criterion_met(training_batch_number)
        scores = self._current_surrogate_score
        if training_batch_number > 1:
            this_score = scores[self._convergence_metric][training_batch_number]
            last_score = scores[self._convergence_metric][training_batch_number-1]
            if np.abs(this_score - last_score) <= self._eps:
                logger.info(f"Convergence from surrogate score.")
                logger.info(f"{self._convergence_metric}: \
                    {self._current_surrogate_score[self._convergence_metric]}")
                stop = True
        return stop

    def _run_initial_training_samples(self):
        super().set_number_of_samples(self._num_initial_samples)
        self._matcal_evaluate_parameter_sets_batch(self._parameter_sets_to_evaluate, 
                                                       is_restart=self._restart)        
        return self._train_surrogate_with_current_results()
    
    def _train_surrogate_with_current_results(self):
        training_params = self._format_params(self._results).T
        training_data = self._format_output(self._results)
        current_surrogate = _fit_surrogate_model(self, interpolation_field=self._independent_variable, 
                                               interpolation_locations=self._independent_variable_values, 
                                               test_eval_info=self._test_eval_info, 
                                               save_filename=self._surrogate_save_filename)
        self._surrogate._add_iteration(current_surrogate, self._results.number_of_evaluations)
        self._update_surrogate_score()
        self._nbatch_samples.append(self.results.number_of_evaluations)
        return training_params, training_data

    def _create_voronoi_tess_and_choose_new_samples(self, iteration, training_params, 
                                                    training_data):
        """Perform Voronoi batch sampling based on the specified algorithm.
        
        :return: Returns a list of the new samples selected.
        """
        if self._nsplits > 0:
            # Step 1: Randomly sort existing samples into K-folds and perform KFold Cross Validation
            self._perform_kfold_cross_validation(training_params, training_data)

            # Step 2: Select the fold(s) with the n largest K-fold CV error(s)
            self._find_kfold_max_errors()
            
            if self._nmax_loo == 'all':
                worst_sample_locations = training_params[self._max_fold_error_indices]
            else:
                # Step 3: Use LOOCV to evaluate each sample within the selected fold(s)
                self._perform_loo_cross_validation(training_params, training_data)
                # Step 4: Identify the n sample(s) with the highest LOOCV error(s)
                worst_sample_locations = self._find_loo_max_errors()
                
        else:
            # Do not perform kfold or loo CV. New samples drawn for all Voroni regions.
            worst_sample_locations = training_params

        if self._thin is not None:
            # thin the new samples locations according to "thin"
            worst_sample_locations = worst_sample_locations[::self._thin, ...]
        elif self._random_selection is not None:
            # randomly select the new sample locations from the candidates in worst_sample_locations
            draw_n = np.min((int(0.5 * worst_sample_locations.shape[0]), self._random_selection))
            random_rows = np.random.choice(worst_sample_locations.shape[0], 
                                           size=draw_n, replace=False)
            worst_sample_locations = worst_sample_locations[random_rows, ...]

        self._worst_sample_locations = worst_sample_locations
        logger.info(f"Initializing voronoi/tree for batch {iteration}")
        self._create_voronoi_tess(training_params)
        return self._find_new_sample_locations()
        
    def _create_voronoi_tess(self, training_params):
        if self._voronoi_type == 'full':
            # Initialize Voronoi tessellation
            self._voronoi_tessellation = VoronoiTessellation(training_params, self._bounds)
            self._voronoi_tessellation.build(finite_only=self._finite_only)

        elif self._voronoi_type == 'local':
            # Make a local voronoi tesselation for each new sample by using knearest neighbors
            # to determine the closest points
            from scipy.spatial import KDTree
            self._all_tree_points = training_params.copy()
            self._tree = KDTree(self._all_tree_points)
                
    def _find_new_sample_locations(self):
        new_points = []
        logger.info("Finding new sample locations")
        for loc_idx, location in enumerate(self._worst_sample_locations):
            if np.mod(loc_idx, 100) == 0:
                logger.info(f"Drawing new sample from region index {loc_idx}"
                            f" of {len(self._worst_sample_locations)}.")

            if self._voronoi_type == 'full':
                # Identify corresponding voronoi cell
                region_index = self._voronoi_tessellation.get_voronoi_region(location)[0][0]

                # Step 5: Select the point within this sample’s Voronoi cell that
                # is furthest from existing samples
                region_vertices, furthest_vertex_index = \
                    self._voronoi_tessellation.find_furthest_vertex(region_index)
                if region_vertices is None:
                    continue
                furthest_vertex = region_vertices[furthest_vertex_index]

                # Step 6: Add the new point and update Voronoi tessellation
                if self._iterative_updates:
                    self._voronoi_tessellation.add_points(furthest_vertex)

                # Step 7: Update X
                new_points.append(furthest_vertex)

            elif self._voronoi_type == 'local':
                nneighbors = int(self._all_tree_points.shape[0] * 0.25)
                nearest_neighbors = self._tree.query(location, k=nneighbors)
                nn_points = self._all_tree_points[nearest_neighbors[1].squeeze()]
                nn_vor = VoronoiTessellation(nn_points, self._bounds)
                nn_vor.build(**self._voronoi_tessellation_options)
                nn_region = nn_vor.get_voronoi_region(location)[0][0]
                try:
                    nn_vert, nn_fvi = nn_vor.find_furthest_vertex(nn_region)
                except:
                    continue

                if nn_vert is None:
                    continue
                furthest_vertex = nn_vert[nn_fvi]
                new_points.append(furthest_vertex)
                if self._iterative_updates:
                    from scipy.spatial import KDTree
                    self._all_tree_points = np.vstack((self._all_tree_points, furthest_vertex))
                    self._tree = KDTree(self._all_tree_points)

        new_points = np.asarray(new_points)
        # Make sure all new points are unique
        unique_points = set(tuple(row) for row in new_points)
        new_points = np.asarray([list(row) for row in unique_points])
        new_points = self._check_points_within_bounds(new_points)
        return new_points

    def _check_points_within_bounds(self, points):
        # verify that all samples are within bounds
        lb = self._bounds[:,0]
        ub = self._bounds[:,1]
        mask = ((points >= lb) & (points <= ub)).all(axis=1)
        return points[mask]
        
    def _perform_kfold_cross_validation(self, training_params, training_data):
        self._kf = None
        logger.info("Performing kfold cross-validation")
        kfcv = KFoldCrossValidation()
        groups = None

        if self._group_kfold:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=self._nsplits, random_state=42)
            groups = kmeans.fit_predict(training_params)
        
        kfcv_options = {'nsplits':self._nsplits, 'group_kfold':self._group_kfold,
                        'scale':self._cv_scale, 'metric':self._cv_metric,
                        'groups':groups, 'interpolation_field':self.interpolation_field,
                        'par_names':self.param_names}
        self._kf = kfcv.perform_kfold_cv(training_params, training_data, **kfcv_options)
    
    def _find_kfold_max_errors(self):
        self._max_fold_error_indices = None
        logger.info("Finding max kfold error")
        max_folds = self._find_indices_of_n_largest_kf_errors()
        self._max_fold_error_indices = np.concatenate(list(max_folds.values())) 
    
    def _perform_loo_cross_validation(self, training_params, training_data):
        self._loo_errors = None
        logger.info("Finding worst sample locations")
        loocv = LeaveOneOutCrossValidation()
        loo_options = {'scale':self._cv_scale,
                       'metric':self._cv_metric,
                       'interpolation_field':self.interpolation_field,
                       'par_names':self.param_names}
        
        self._loo_errors = loocv.perform_loocv(training_params, training_data,
                                               self._max_fold_error_indices,
                                               **loo_options)
    
    def _find_loo_max_errors(self, training_params):
            self._worst_sample_locations = None
            max_loo_indices = self._find_indices_of_n_largest_errors()
            return training_params[max_loo_indices]
        
    def _find_indices_of_n_largest_kf_errors(self):

        # Create a list of (key, error, sample_index) tuples
        items = [(key, value[0], value[1]) for key, value in self._kf.items()]

        # Sort the items based on the error in descending order
        sorted_items = sorted(items, key=lambda x: x[1], reverse=True)

        # Get the top n items
        top_n_items = sorted_items[:self._nmax_folds]

        # Extract the arrays associated with the top n largest floats
        result_arrays = {key: array for key, _, array in top_n_items}

        return result_arrays

    def _find_indices_of_n_largest_errors(self):
        """Find the indices of the n largest values in an array of errors.

        :return: Returns an array of indices corresponding to the n largest errors.
        """

        nkeep = int(self._nmax_loo)

        # Create a list of (key, error, sample_index) tuples
        items = [(key, value[0], value[1]) for key, value in self._loo_errors.items()]

        # Sort the items based on the error in descending order
        sorted_items = sorted(items, key=lambda x: x[1], reverse=True)

        # Get the top n items
        top_n_items = sorted_items[:nkeep]

        # Extract the indices associated with the top n largest floats
        indices = [item[2] for item in top_n_items]

        return np.array(indices)

    def _add_parameter_evaluation(self, **p):
      super()._add_parameter_evaluation(**p)

    def add_parameter_evaluation(self, **parameters):
        """"""
        raise self.StudyInputError("Users cannot add parameter evaluations to"
                                   " a VoronoiAdaptiveSurrogateStudy.")


class VoronoiTessellation:
    def __init__(self, points, bounds):
        """Initialize the VoronoiBatchSamplingStudy
        
        :param points: Array of points that are the seeds of the Voronoi tessellation
        :type points: nd_array

        :param bounds: Bounds for the parameter space,
            e.g., [(xmin, xmax), (ymin, ymax)] for a 2D space.
        :type bounds: list of tuples
        """
        self.points = np.array(points)
        self.ndim = self.points.shape[1]
        self.bounds = bounds
        
    def build(self, **options):
        """Initialize the Voronoi tessellation with given points and bounds.
        """
        from scipy.spatial import Voronoi, Delaunay, ConvexHull

        self._initialize_attributes()
        self._set_voronoi_options(**options)
        self.boundary_points = self._make_nd_grid(npts_along_dim=2)
        if not self.finite_only:
            self.boundary_hull = ConvexHull(self.boundary_points)
            self.boundary_hull_eq = self.boundary_hull.equations # (nfacet, ndim + 1)
            self.boundary_hull_V, self.boundary_hull_b = \
                self.boundary_hull_eq[:, :-1], self.boundary_hull_eq[:, -1] # normal, offset
            self.bhullD = Delaunay(self.boundary_points)
        else:
            self.boundary_hull = None
            self.bhullD = None
        self.create_ghost_points()
        self._all_points = np.vstack([self.points, self._ghost_points])
        
        self.vor = Voronoi(self._all_points, incremental=self.incremental)
        self.ghost_busters()
        self.boundary_regions = self.get_voronoi_region(self.boundary_points) # may need to update

    def _initialize_attributes(self):
        self.finite_only = False
        self.incremental = False
         
    def _set_voronoi_options(self, **voronoi_kwargs):
        """Set voronoi properties. See documentation for properties that
        an be altered and their options.
        
        :param incremental: Allow adding points incrementally. 
        This takes up additional resources. Default False.
        :type incremental: bool
        
        :param finite_only: When finite_only = True, only vertices which reside 
        inside convex hull defined by boundary points are returned as vertices
        of a voronoi region. With finite = False, all vertices are returned.
        In this case, vertices which fall outisde the parameter bounds are snipped 
            to the convex hull defined by boundary points, which requires more computational
            resources, especially in high dimensions. Default False.
        :type finite_only: bool
        """
        for key, value in voronoi_kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{key}'")
        
    def _make_nd_grid(self, npts_along_dim):
        grid_pts = []
        for dim in np.arange(self.ndim):
            grid_pts.append(np.linspace(self.bounds[dim,0], self.bounds[dim,1], npts_along_dim))
        coords = np.meshgrid(*grid_pts)
        coords_ravel = [np.asarray(coords[i]).ravel() for i in np.arange(self.ndim)]
        return np.vstack(tuple(coords_ravel)).T
        
    def create_ghost_points(self, stretchCoef=1.75, centCoef=1.5):
        """Reflect points nearest to the boundary hull across the nearest
        face of the boundary hull """

        boundary_points_stretched = self.boundary_points * stretchCoef
        self._ghost_points = boundary_points_stretched

        boundary_centroid = np.mean(self.boundary_points, axis=0)
        max_dist = np.max(np.linalg.norm(self.boundary_points - boundary_centroid, axis=1))
        self._ghost_points = \
            np.vstack([self._ghost_points, \
                boundary_centroid + centCoef * max_dist * np.eye(self.points.shape[1])])
        self._ghost_points =\
            np.vstack([self._ghost_points, \
                boundary_centroid - centCoef * max_dist * np.eye(self.points.shape[1])])

    def ghost_busters(self):
        """ Identify which points in self._all_points are ghost points"""
        self._boo = []
        for point in self._all_points:
            if point in self._ghost_points:
                self._boo.append(True)
            else:
                self._boo.append(False)

    def get_region_vertices(self, region_index, identify_outside_vertices=True):
        """Return the vertices of the Voronoi region."""
        region = self.vor.regions[region_index].copy()
        if -1 in region:
            logger.warning(f"Infinite vertice in Region {region_index}")
        
        if identify_outside_vertices:
            updated_region = self.identify_vertices_outside_bounds(region)
            if not -2 in updated_region and len(updated_region) > 0:
                region_vertices = self.vor.vertices[region]
            elif -2 in updated_region:
                if self.finite_only:
                    if max(updated_region) < 0:
                        region_vertices = None
                    else: 
                        region_vertices = \
                            np.asarray([self.vor.vertices[i]\
                                for i in updated_region if i > 0])
                else:
                    region_tuple_list = list(zip(region, updated_region))
                    region_vertices = \
                        self.replace_unbounded_vertices(updated_region, region_index, region_tuple_list)
            if region_vertices is not None:
                if not self.finite_only:
                    boundary_in_region = \
                        [i for i in np.arange(len(self.boundary_regions))\
                            if self.boundary_regions[i][0] == region_index]
                    if boundary_in_region:
                        boundary_vertices = self.boundary_points[boundary_in_region] 
                        region_vertices = np.concatenate((region_vertices, boundary_vertices))
                        unique_vertices = set(tuple(row) for row in region_vertices)
                        return np.asarray([list(row) for row in unique_vertices])
            return region_vertices

        elif not identify_outside_vertices:
            return self.vor.vertices[region]

    def get_voronoi_vertices(self, identify_outside_vertices=True):
        """Return the vertices of the Voronoi tessellation."""
        vertices = []
        for i, region in enumerate(self.vor.regions):
            try:
                region_point_index, = np.where(self.vor.point_region == i)[0]
            except:
                # empty region: Voronoi region for a point at infinity that was added internally
                continue
            if self._boo[region_point_index]:
                # region belongs to a ghost point
                continue
            elif -1 in region:
                logger.warning(f"Infinite vertice in Region {i}")

            if identify_outside_vertices:
                updated_region = self.identify_vertices_outside_bounds(region)
                if not -2 in updated_region and len(updated_region) > 0:
                    vertices.append(self.vor.vertices[region])
                elif -2 in updated_region:
                    if self.finite_only:
                        verts = np.asarray([self.vor.vertices[i] for i in updated_region if i > 0])
                        vertices.append(verts)
                    else:
                        region_tuple_list = list(zip(region, updated_region))
                        vertices.append(self.replace_unbounded_vertices(updated_region, i, region_tuple_list))
                        boundary_in_region =\
                            [ii for ii in np.arange(len(self.boundary_regions))\
                                if self.boundary_regions[ii][0] == i]
                        if boundary_in_region:
                            boundary_vertices = self.boundary_points[boundary_in_region] 
                            vertices.append(boundary_vertices)

            elif not identify_outside_vertices:
                vertices.append(self.vor.vertices[region])
        
        if vertices is not None:
            vertices = np.concatenate((vertices))
            unique_vertices = set(tuple(row) for row in vertices)
            return np.asarray([list(row) for row in unique_vertices])
        else: 
            return vertices
        
    def identify_vertices_outside_bounds(self, region):
        """
        Identify vertices that sit outside the bounding region

        :param region: A list of the voronoi regions. Each list contains indices of the voronoi 
        vertices forming each Voronoi region.
        :type region: list

        :return: Returns a new list of voronoi regions with vertices outside
        the bounding region replaced with -2.
        """

        #outside = lambda lb, ub, x: (x < lb) + (x > ub)
        # Create a boolean mask for vertices outside the bounds
        region = np.array(region)
        region_vertices = self.vor.vertices[region]
        outside_mask = np.zeros(region_vertices.shape, dtype=bool)

        for col_index in range(self.ndim):
            lb, ub = self.bounds[col_index,0], self.bounds[col_index, 1]
            #vert_outside, = np.where(outside(lb, ub, region_vertices[:, col_index]))
            outside_mask[:, col_index] |= \
                (region_vertices[:, col_index] < lb) | (region_vertices[:, col_index] > ub)

        # Get the indices of vertices that are outside the bounds
        vert_outside = np.where(outside_mask.any(axis=1))[0]
        if len(vert_outside) > 0:
            outside_vert_index = [region[i] for i in vert_outside]
            region[vert_outside] = -2
        return region.tolist()

    def replace_unbounded_vertices(self, region, region_index, region_tuple):
        """
        Replace the infinite vertices in a Voronoi region with new vertices on 
        the edge of the bounding box.
        ** vertices that sit outside the bounding region are considered infinite here

        :param region: A list of the voronoi regions. Each list contains indices
        of the Voronoi vertices forming each Voronoi region.
        :type region: list
        
        :param region_index: Region index
        :type region_index: int
        
        :return: Returns a new list of voronoi regions with infinite vertices replaced.
        """
        try:
            region_point_index, = np.argwhere(self.vor.point_region == region_index)[0]
        except:
            raise ValueError("No region point index found in VoronoiTessesllation"
                             " for Adaptive Surrogate Generation. Try a different random seed.")
        
        region_vertices = []
        if -2 in region:
            finite_indices = [v for v in region if v >= 0]
            if len(finite_indices) > 0:
                finite_vertices = self.vor.vertices[finite_indices]
            new_vertices = self.snip_ridge_vertices(\
                region_index, region_point_index, region_tuple)

            # Replace the infinite vertex
            if len(finite_indices) > 0:
                region_vertices = np.concatenate((finite_vertices, new_vertices))
            elif len(new_vertices) > 0:
                region_vertices = new_vertices
            else:
                return None

        else:
            region_vertices = self.vor.vertices[region]

        return region_vertices

    def snip_ridge_vertices(self, region_index, region_point_index, region_tuple):

        # Find the ridge vertices for the specified region
        region_dict = {x[0]: x[1] for x in region_tuple}
        
        # the voronoi points that are equidistant from the ridge that lies between them
        ridge_point_indices = np.argwhere(self.vor.ridge_points == region_point_index)[:, 0]
        
        # the vertices at the end of each ridge
        region_ridge_vertices = [self.vor.ridge_vertices[i] for i in ridge_point_indices]

        new_vertices = []
        for rv in region_ridge_vertices:
            urv = [region_dict.get(num) for num in rv]
            if len(urv) == 2: #2D Voronoi region
                u, v = np.argsort(urv)
                # and urv[v] > 0: # only one vertice is out of bounds - snip one end to the boundary hull
                if urv[u] == -2:
                    ray_end = self.vor.vertices[rv[u]]
                    ray_origin = self.vor.vertices[rv[v]]
                    norm_ray_direction =\
                        self.get_normal_ray_direction(ray_origin, ray_end)
                    new_vertice = \
                        self.find_boundary_hull_ray_crossings(norm_ray_direction, ray_origin)
                    # confirm new vertice is in given region
                    if new_vertice is not None and region_index in self.get_voronoi_region(new_vertice)[0]:
                        # confirm point is within boundary hull
                        if self.bhullD.find_simplex(new_vertice) >= 0:
                            new_vertices.append(new_vertice)
                # both vertices are out of bounds - snip both ends to the boundary hull
                if urv[v] == -2:
                    ray_end = self.vor.vertices[rv[v]]
                    ray_origin = self.vor.vertices[rv[u]]
                    norm_ray_direction = \
                        self.get_normal_ray_direction(ray_origin, ray_end)
                    new_vertice =\
                        self.find_boundary_hull_ray_crossings(norm_ray_direction, ray_origin)
                    if new_vertice is not None and region_index in self.get_voronoi_region(new_vertice)[0]:
                        if self.bhullD.find_simplex(new_vertice) >= 0:
                            new_vertices.append(new_vertice)

            elif len(urv) > 2: #3D + Voronoi region
                nunbounded_vert = urv.count(-1) + urv.count(-2)
                if nunbounded_vert > 0 and nunbounded_vert < len(urv):

                    edges = [[rv[i], rv[(i+1) % len(rv)]] for i in range(len(rv))]
                    updated_edges = \
                        [[urv[i], urv[(i+1) % len(urv)]] for i in range(len(urv))]
                    unbounded_edges =\
                        [[i, edge] for i, edge in enumerate(updated_edges) if -2 in edge]
                    for i, ev in unbounded_edges:
                        u, v = np.argsort(ev)
                        if ev[u] == -2: # and ev[v] > 0:
                            ray_end = self.vor.vertices[edges[i][u]]
                            ray_origin = self.vor.vertices[edges[i][v]]
                            norm_ray_direction = \
                                self.get_normal_ray_direction(ray_origin, ray_end)
                            new_vertice = \
                                self.find_boundary_hull_ray_crossings(norm_ray_direction, ray_origin)
                            if new_vertice is not None and region_index in self.get_voronoi_region(new_vertice)[0]:
                                if self.bhullD.find_simplex(new_vertice) >= 0:
                                            new_vertices.append(new_vertice)
                        if ev[v] == -2: # and ev[v] > 0:
                            ray_end = self.vor.vertices[edges[i][v]]
                            ray_origin = self.vor.vertices[edges[i][u]]
                            norm_ray_direction = \
                                self.get_normal_ray_direction(ray_origin, ray_end)
                            new_vertice = \
                                self.find_boundary_hull_ray_crossings(norm_ray_direction, ray_origin)
                            if new_vertice is not None and region_index in self.get_voronoi_region(new_vertice)[0]:
                                if self.bhullD.find_simplex(new_vertice) >= 0:
                                            new_vertices.append(new_vertice)

        return np.asarray(new_vertices)

    def get_normal_ray_direction(self, ray_origin, ray_end):
        ray_direction = ray_end - ray_origin
        return ray_direction / np.linalg.norm(ray_direction)
        
    def find_boundary_hull_ray_crossings(self, U, z):
        """Find where a ray crosses the convex hull of the boundary.

        :param U: Ray direction.
        :type U: np.ndarray
        
        :param z: Ray origin.
        :type z: np.ndarray

        :return: Returns a list of intersection points with the convex hull.
        """
        V = self.boundary_hull_V
        b = self.boundary_hull_b
        denom = np.dot(V, U)
        num = -(b + np.dot(V, z))
        alpha = num[denom!=0] / denom[denom!=0]
        if not np.any(alpha > 0):
           return None 
            
        return np.min(alpha[alpha >0]) * U + z

    def find_furthest_vertex(self, region_index, identify_outside_vertices=True):
        """Find the vertex that has the greatest distance from the cell centroid."""

        self.raise_if_invalid_region_index(region_index)
        vertices = self.get_region_vertices(region_index,\
            identify_outside_vertices=identify_outside_vertices)
        if vertices is not None:
            centroid = self.get_region_seed(region_index)
            distances = np.linalg.norm(vertices - centroid, axis=1)
            furthest_vertex_index = np.argmax(distances)
        else:
            furthest_vertex_index = None
        return vertices, furthest_vertex_index

    def get_region_seed(self, region_index):
        """Given a region_index, return the seed of the Voronoi tesselation that
        belongs to the region.

        :param region_index: Region index.
        :type region_index: int
        
        :return: Returns the Voronoi seed that belongs to the indexed region.
        """

        point_index, = np.where(self.vor.point_region == region_index)
        return np.atleast_2d(self.vor.points[point_index[0]])

    def get_voronoi_region(self, point_array):
        """Given an array of points, return the region of the Voronoi tesselation that the
        points belongs to. If a point lies on a ridge or vertice, multiple regions are 
        returned for that point.

        :param point: An array of points to find the region of.
        :type point: nd_array

        :return: Returns a list of lists, where each sublist contains the Voronoi 
        region(s) that contains the point. A point on a ridge has a sublist with 
        two regions (for 2D). A point on a vertice has a sublist
             with 3 regions (for 2D)
        """
        point_array = np.atleast_2d(point_array)
        region_index = []
        for point in point_array:
            point_already_exists = np.any(np.all(self.vor.points == point, axis=1))
            if point_already_exists:
                seed_index, = np.where(np.all(self.vor.points == point, axis=1))
            else:
                seed_index = self.get_closest_seed(point)
            
            # Get the region index for the point
            region_index.append(self.vor.point_region[seed_index].tolist())
        region_index = [sublist if sublist else [np.inf] for sublist in region_index]
        return region_index
    
    def get_closest_seed(self, point):
        """Return the index of the seed of the Voronoi cell that contains the given point.
            If the point lies on a ridge or vertex, multiple seeds are returned."""
        closest_seed_index = self.get_closest_point(self.vor.points, point)
        return closest_seed_index[0]

    def get_closest_point(self, candidates, target_point):
        """Return the index of the candidate point that has the min distance
           from the target point"""
        distances = np.linalg.norm(candidates - target_point, axis=1)
        min_dist = min(distances)
        closest_candidate_index = np.where(np.isclose(distances, min_dist, rtol=0, atol=1e-10))
        return closest_candidate_index        

    def remove_invalid_rows(self, arr):
        """ Remove points from NumPy array that contain NaN or infinite values."""
        if not isinstance(arr, np.ndarray):
            raise TypeError("Input to remove_invalid_rows must be a NumPy array.")
        arr = np.atleast_2d(arr)
        
        # Create a boolean mask for valid rows (no NaN, no inf, no -inf)
        mask = np.isfinite(arr).all(axis=1)
        return arr[mask]
    
    def add_points(self, points):
        """Process a set of additional points.
        
        Voronoi has a built in function to add points 
            -- self.vor.add_points(points, restart=True).
        However, 'incremental` must be set to True to use the built-in add_points() 
        method and is very slow. Qhull throws an error for dim>2 when 
        incremental=True and restart=False. This class method, which rebuilds 
        'manually` is faster.
        """
        from scipy.spatial import Voronoi 
        if not isinstance(points, np.ndarray):
            raise TypeError("Input to add_points must be a NumPy array.")
        points = np.atleast_2d(points)
        
        if not points.shape[-1] == self._all_points.shape[-1]:
            raise ValueError(f"Points in add_points have a different dimension"
                             " ({points.shape[-1]}) than points in voronoi"
                             " tessellation ({self._all_points.shape[-1]})")
        
        points = self.remove_invalid_rows(points)
        if points.size == 0:
            logger.warning("All input points were NaN or Inf."
                           " No new points added to voronoi tessellation.")
            return

        # make sure all new points are unique
        all_points = np.vstack((self._all_points, points))
        unique_points = set(tuple(row) for row in all_points)
        self._all_points = np.asarray([list(row) for row in unique_points])
        self.vor = Voronoi(self._all_points)

    def raise_if_invalid_region_index(self, region_index):
        if region_index > len(self.vor.regions) or region_index < 0:
            raise ValueError('Invalid region index. Index must be in (0, nregions]')


class KFoldCrossValidation:
    def __init__(self):
        """Initialize the K-Fold Cross-Validation with a given surrogate model.
        """
        self._initialize_attributes()

    def _initialize_attributes(self):
        self.nsplits = 5
        self.group_kfold = False
        self.scale = None
        self.metric = 'rmse'
        self.groups = None
        self.interpolation_field = 'x'
        self.par_names = None
         
    def _set_kfcv_options(self, **kfcv_kwargs):
        """Set KFold CV properties."""
        self._kfcv_kwargs = kfcv_kwargs
        for key, value in kfcv_kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{key}'")
        if self.nsplits > self.X.shape[0]:
            self.nsplits = int(self.X.shape[0]/2.0)
            logger.warning("Input parameter \"nsplits\" can't be greater than " +
                           "the number of samples in KFoldCrosValidation. Reducing " +
                           "number of splits to approximately half the number of samples.")
        
    def perform_kfold_cv(self, training_params, training_data, **kfcv_options):
        """
        Perform K-Fold Cross-Validation.

        :param X: Feature matrix (training samples).
        :type X: np.ndarray
        
        :param y: Target values (ground truth).
        :type y: np.ndarray
        
        :return: Returns the index of the sample with the greatest 
        prediction error and the corresponding error value.
        tuple (index_of_max_error, max_error)
        """
        from sklearn.model_selection import GroupKFold, KFold
        from joblib import Parallel, delayed
        self._set_kfcv_options(**kfcv_options)
        if self.group_kfold:
            assert self.groups is not None
            cv = GroupKFold(n_splits=self.nsplits)
            kf_results = Parallel(n_jobs=1)(
                delayed(self.evaluate_fold)(train_index, test_index, training_params, training_data)
                for train_index, test_index in cv.split(training_params, training_data, self.groups)
            )
        else:
            cv = KFold(n_splits=self.nsplits, shuffle=True, random_state=1)
            kf_results = Parallel(n_jobs=1)(
                delayed(self.evaluate_fold)(train_index, test_index, training_params, training_data)
                for train_index, test_index in cv.split(self.X)
            )

        # Convert the results to a dictionary
        kf = {k_idx: result for k_idx, result in enumerate(kf_results)}
        return kf

    def evaluate_fold(self, train_index, test_index, X, y):
        """Perform a single fold of cross-validation."""

        info = self.extract_fold_info(train_index, test_index, X, y)
        train_eval_info, test_eval_info, X_test, y_test = info
        surrogate_options = get_cv_surrogate_options(test_eval_info,
                                                     self.interpolation_field) 
        
        # Fit the model on the training data
        fold_surrogate = _fit_surrogate_model(train_eval_info, **surrogate_options)
        
        # Make predictions for the test set and calculate error
        error = _get_surrogate_metric(fold_surrogate, self.metric)
        return error, test_index
   
    def extract_fold_info(self, train_index, test_index, X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train = [y[i] for i in train_index]
        y_test = [y[i] for i in test_index]
        train_res, test_res = _setup_studies_for_cv(self.par_names,
                                                    X_train, X_test, y_train, y_test)
        return train_res, test_res, X_test, y_test


class LeaveOneOutCrossValidation:
    def __init__(self):
        """
        Initialize the LOOCV.
        """
        self._initialize_attributes()
        
    def _initialize_attributes(self):
        self.scale = None
        self.metric = 'rmse'
        self.interpolation_field = 'x'
        self.par_names = None
        
    def _set_loocv_options(self, **loocv_kwargs):
        """Set LeaveOneOut CV properties."""
        self._loocv_kwargs = loocv_kwargs
        for key, value in loocv_kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{key}'")

    def perform_loocv(self, X, y, indices, **loocv_options):

        """Perform Leave-One-Out Cross-Validation.

        :param X: Feature matrix (training samples).
        :type X: np.ndarray
        
        :param y: Target values (ground truth).
        :type y: np.ndarray

        :return: Returns the index of the sample with the greatest 
        prediction error and the corresponding error value.
        tuple: (index_of_max_error, max_error)
        """

        from joblib import Parallel, delayed
        self._set_loocv_options(**loocv_options)
        
        loo_results = Parallel(n_jobs=1)(
            delayed(self.evaluate_loo_sample)(X, y, i)
            for i in indices
        )

        loo = {loo_idx: result for loo_idx, result in enumerate(loo_results)}
        return loo
    
    def evaluate_loo_sample(self, X, y, index):
        # Leave one out: create training and test sets
        info = self.extract_loo_info(index, X, y)
        train_eval_info, test_eval_info, X_test, y_test = info
        surrogate_options = get_cv_surrogate_options(test_eval_info,
                                                     self.interpolation_field)
        # Fit the model on the training data
        loo_surrogate = _fit_surrogate_model(train_eval_info, **surrogate_options)

        # Make predictions for the left-out sample and calculate error
        error = _get_surrogate_metric(loo_surrogate, self.metric)
        return error, index

    def extract_loo_info(self, index, X, y):
        X_train = np.delete(X, index, axis=0)
        y_train = y.copy()
        del y_train[index]
        X_test = X[index].reshape(1, -1)  # Reshape for a single sample
        y_test = [y[index]]
        train_res, test_res = _setup_studies_for_cv(self.par_names,
                                                    X_train, X_test,
                                                    y_train, y_test)
        return train_res, test_res, X_test, y_test
    
    
def get_cv_surrogate_options(test_eval_info, interpolation_field):
    surrogate_options = {'interpolation_field':interpolation_field,
                         'training_fraction':1.0,
                         'test_eval_info':test_eval_info,
                         'cv_test_set':True}
    return surrogate_options


def _setup_studies_for_cv(p_names, train_samples, test_samples,
                               train_evals, test_evals):
    res = _get_parameter_and_simulation_hist(p_names, train_samples, train_evals)
    test_res = _get_parameter_and_simulation_hist(p_names, test_samples, test_evals)
    return res, test_res


def _get_parameter_and_simulation_hist(p_names, p_samples, m_evals):
    from matcal.core.study_base import StudyResults
    p_hist = _format_parameter_hist(p_names, p_samples)
    res_hist = _format_parameter_evaluations(p_names, m_evals)
    res = StudyResults()
    res._update_parameter_history(p_hist, list(p_hist.keys()))
    res._update_simulation_history(res_hist, 'cv')
    return res


def _format_parameter_hist(names, p_samples):
    from matcal.core.parameters import ParameterCollection
    n_samples = p_samples.shape[0]
    params = OrderedDict()
    pc = ParameterCollection('cv')
    for idx in range(n_samples):
        params[f"eval_{idx}"] = OrderedDict()
        for n_idx, param_name in enumerate(names):
            params[f"eval_{idx}"][param_name] = p_samples[idx, n_idx]
    return params 


def _format_parameter_evaluations(param_order, model_evals):
    from matcal.core.data import convert_dictionary_to_data, DataCollection
    results_hist = DataCollection("CV data collection")
    for eval in model_evals:
        results_hist.add(convert_dictionary_to_data(eval))
    return results_hist


def _get_surrogate_metric(surrogate, metric):
    test_score = surrogate.current_surrogate._latent_scores['test']
    combined_score = []
    for field_idx, field_name in enumerate(test_score):
        if isinstance(test_score[field_name], (dict, OrderedDict)):
            combined_score += list(test_score[field_name][metric])
    if metric == 'nlpd':
        return np.sum(combined_score)
    else:
        return np.mean(combined_score)