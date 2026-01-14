"""
This module contains adaptive surrogates. 
"""
import copy
import numpy as np
import os
from scipy import stats

from matcal.core.logger import initialize_matcal_logger
from matcal.core.objective import SimulationResultsSynchronizer
from matcal.core.parameter_studies import HaltonStudy
from matcal.core.state import State
from matcal.core.utilities import (check_value_is_positive_integer, 
                                   check_value_is_array_like_of_reals, 
                                   check_value_is_nonempty_str, 
                                   check_value_is_array_like_of_reals, 
                                   check_value_is_bool, 
                                   check_value_is_positive_real)
from matcal.core.serializer_wrapper import matcal_save

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
        self._sample_counts: list[int] = []    
        self._target_field_name: str = target_field_name
        self._indep_variable_name = indep_variable_name
        self._indep_variable_values = np.asarray(indep_variable_values)
        self._variable_transformer = variable_transformer
        self._test_params = test_params
        self._test_responses = test_responses
        self._param_names = param_names

    def _add_iteration(
        self,
        surrogate, 
        nsamples 
        ) -> None:
        self._surrogates.append(copy.deepcopy(surrogate))
        params = self._variable_transformer.map_to_canonical(self._test_params)
        surrogate_values = self(params, batch_evaluate=True)
        average_l2_error = (
            np.linalg.norm(self._test_responses - surrogate_values)
            / self._test_responses.shape[1]
        )
        max_abs_error = np.max(np.abs(self._test_responses - surrogate_values))
        self._average_errors.append(average_l2_error)
        self._max_errors.append(max_abs_error)
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
        test responses and :math:`{\hat{R}}` is the surrogate responses. 
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
        test responses and :math:`{\hat{R}}` is the surrogate responses. 
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
            response = surrogate(np.asarray(*args))
            return response
        elif len(args) == len(self._param_names) and len(kwargs) == 0:
            params_array = np.asarray([args]).T
            response = surrogate(params_array)[0]
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


class SparseGridAdaptiveSurrogateStudy(HaltonStudy):
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

        self._save_filename = None

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
            check_value_is_positive_real(average_l2_error_goal, "average_l2_error_goal", 
                "SparseGridAdaptiveSurrogateStudy.set_error_stopping_criteria"
            )
            self._average_l2_error_goal = float(average_l2_error_goal)

        if max_abs_error_goal is not None:
            check_value_is_positive_real(max_abs_error_goal, "max_abs_error_goal", 
                "SparseGridAdaptiveSurrogateStudy.set_error_stopping_criteria"
            )
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
        check_value_is_nonempty_str(independent_variable, "independent_variable", 
                                    "SparseGridAdaptiveSurrogateStudy.set_independent_variable")
        self._independent_variable = independent_variable
        check_value_is_array_like_of_reals(independent_variable_values,
                                           "independent_variable_values", 
                                    "SparseGridAdaptiveSurrogateStudy.set_independent_variable")
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
        check_value_is_positive_integer(number_of_test_samples, "number_of_test_samples", 
                                     "SparseGridAdaptiveSurrogateStudy.set_number_of_test_samples")
        self._number_of_test_samples = number_of_test_samples
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
        check_value_is_positive_integer(max_training_samples, "max_training_samples", 
                                        "SparseGridAdaptiveSurrogateStudy.set_max_training_samples")
        self._max_training_samples = max_training_samples
        logger.debug(f"max_training_samples set to {self._max_training_samples}")
        if self._number_of_test_samples is None:
            self._number_of_test_samples = self._set_default_number_of_test_samples()

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
        n_params = len(self._parameter_collection)
        candidate_b = n_params * 10
        return max(candidate_a, candidate_b)

    def set_target_field_name(self, target_field_name):
        """Specify the field name for the response that the surrogate model 
        will seek to replicate. This is generally a model response such as temperature, 
        load, etc.

        :param target_field_name: the name of the field that the surrogate will 
            replicate
        :type target_field_name: str
        """
        check_value_is_nonempty_str(target_field_name, "target_field_name", 
                                    "SparseGridAdaptiveSurrogateStudy.set_target_field_name")
        self._target_field_name = target_field_name

    def set_test_group_random_seed(self, seed, scramble=True):
        """
        Set the random seed for the ``Halton`` sampler that the study uses
        to generate the *test* samples (the ``HaltonStudy`` invoked inside
        :meth:`launch`).

        The seed is applied by (re)‑creating the internal ``HaltonSampler``
        with the same ``scramble`` flag that was used during construction
        of the base ``HaltonStudy``.  The method should be called **before**
        :meth:`launch` (or any other method that triggers sampling) to
        guarantee reproducibility.

        :param seed: Integer seed for the pseudo‑random number generator.
        :type seed: int

        :param scramble: set the scramble keyword for the numpy Halton object.
        :type scramble: bool
        """
        check_value_is_positive_integer(
            seed, "seed", "SparseGridAdaptiveSurrogateStudy.set_test_group_random_seed"
        )
        check_value_is_bool(scramble, "scramble", 
                            "SparseGridAdaptiveSurrogateStudy.set_test_group_random_seed")
        self.HaltonSampler = stats.qmc.Halton(d=self.dim, scramble=scramble, seed=seed)
        logger.debug(
            f"Halton sampler re‑initialised with seed {seed} (scramble={scramble})"
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
        self._variable_transformer = _get_pyapprox_variable_transformer(self._bounds)
        self._surrogate = AdaptiveSurrogate(self._target_field_name, self._independent_variable, 
                                            self._independent_variable_values, 
                                            self._variable_transformer, 
                                            test_params, test_responses, param_names)
        
        self._run_study = self._perform_sparse_grid_batch_sampling
        super().launch()

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
        super().launch(self._number_of_test_samples)
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

    def _format_params(self, results):
        params_formatted = []
        for param in results.parameter_history:
            params_formatted.append(results.parameter_history[param])
        return np.array(params_formatted)  

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

    def add_evaluation_set(self, model, state=None, left=None, right=None, period=None):
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
                "SparseGridAdaptiveSurrogateStudy instance because adaptivity "
                "is only supported for a single model and single response of interest."
            )
        self._evaluation_set_added = True

        if state is not None and not isinstance(state, State):
            raise TypeError(
                "SparseGridAdaptiveSurrogateStudy.add_evaluation_set expects ``state`` "
                "to be a single `State` instance (or None)."
            )

        self._results_synchronizer = self._make_simulation_results_synchronizer(left, right, period)
        super().add_evaluation_set(
            model,
            objectives=self._results_synchronizer,
            data=None,
            states=state,
        )

    def _make_simulation_results_synchronizer(self, left, right, period):
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
            self._target_field_name, left=left, right=right, period=period          
        )

    def _perform_sparse_grid_batch_sampling(self): 
        from pyapprox.interface.model import ModelFromVectorizedCallable

        n_parameters = len(self._parameter_collection.get_item_names())
        n_qois = len(self._independent_variable_values)
        canonical_model = ModelFromVectorizedCallable(n_qois, n_parameters, 
                                    self._matcal_evaluate_parameter_sets_batch_adaptive_training)

        if self._save_filename is None:
            self.set_save_filename(f"{self._get_model_names()[0]}_sparse_grid_surrogate.joblib")
        sg = _setup_sparse_grid_surrogate(n_parameters, n_qois)    
        while sg.step(canonical_model):
            self._surrogate._add_iteration(sg, self._results.number_of_evaluations)
            logger.info(f"Training samples: {self._surrogate.sample_count_history[-1]}")
            training_batch_number = len(self._surrogate.sample_count_history)
            matcal_save(self._save_filename, self._surrogate)
            if self._stopping_criterion_met(training_batch_number):
                break
            logger.info("Surrogate not converged yet.")
            logger.info(f"Average L2 norm error score: {self._surrogate.average_error_history[-1]}")
            logger.info(f"Max absolute error score: {self._surrogate.max_error_history[-1]}")

        return self._results

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

    def _populate_parameter_evaluations_adaptive(self, samples):
        samples = self._variable_transformer.map_from_canonical(samples)
        self._parameter_sets_to_evaluate = []
        param_order = self._parameter_collection.get_item_names() 
        for sample in samples.T:
            ss = { key:sample[i] for i, key in enumerate(param_order) }
            self._add_parameter_evaluation(**ss)
        self._check_parameter_sets_populated()

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
    
    def set_save_filename(self, filename):
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
        check_value_is_nonempty_str(filename, "filename", 
                                    "SparseGridAdaptiveSurrogateStudy.set_save_filename")
        if ".joblib" not in filename:
            raise ValueError("The save filename for the SparseGridAdaptiveSurrogateStudy " +
                f"must end with \".joblib\". Passed filename is \"{filename}\".")
        self._save_filename = filename

    @property
    def save_filename(self):
        """
        Retrieve the filename (including the ``.joblib`` extension) that will be
        used to save the surrogate object after each training batch.

        :return: The absolute or relative path supplied via
            :meth:`set_save_filename`, or ``None`` if no filename has been set.
        :rtype: str | None
        """
        return self._save_filename