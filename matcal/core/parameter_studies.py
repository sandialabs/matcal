"""
This module contains pure MatCal implementations of parameter studies. 
These do not invoke external algorithm libraries. 
"""
from abc import abstractmethod
from collections import OrderedDict
import numpy as np
from scipy.stats import qmc

from matcal.core.data import MaxAbsDataConditioner, DataCollectionStatistics
from matcal.core.logger import initialize_matcal_logger
from matcal.core.parameter_batch_evaluator import ParameterBatchEvaluator
from matcal.core.study_base import StudyBase
from matcal.core.utilities import (check_value_is_real_between_values, 
                                   check_value_is_positive_integer, 
                                   check_value_is_positive_real, 
                                   check_value_is_array_like_of_reals)


logger = initialize_matcal_logger(__name__)


class ParameterStudy(StudyBase):
    """
    Use the MatCal :class:`~matcal.core.parameter_studies.ParameterStudy` to run models and evaluate objectives 
    for a user specified set of parameters values. This can be used for brute-force manual calibrations, sensitivity studies
    when the user prefers to post process the results/chose evaluation parameters and building surrogates for the models using
    python based surrogate algorithms not directly supported in MatCal. 
    """
    study_class = "ParameterStudy"

    class NoEvaluationsDefinedError(RuntimeError):
        """"""

    def __init__(self, *parameters):
        super().__init__(*parameters)
        self._parameter_sets_to_evaluate = []
        self._num_evaluations = 0
        self._return_residuals = True
        self._batch_results = None

    @property
    def _needs_residuals(self):
        return self._return_residuals

    def add_parameter_evaluation(self, **parameters):
        """
        Add parameter sets to be evaluated to the study. This function can be called as many times as needed to 
        evaluate several different parameter sets. They will be evaluated in the order they are added. All parameters
        that were passed into the study on initialization must also have a value specified when adding 
        a parameter set to be evaluated with this function.

        :param parameters: the parameters values to be added as an evaluated parameter set for the study.
        :type parameters: dict(str, float)

        :raises ValueError: If all study parameters do not have a value passed to this function when called.
        """
        self._check_all_parameters_provided(parameters)
        self._num_evaluations += 1
        pc = self._parameter_collection
        for param, value in parameters.items():
            check_value_is_real_between_values(value, 
                                               pc[param].get_lower_bound(), 
                                               pc[param].get_upper_bound(), 
                                               param, "add_parameter_evaluation", 
                                               closed=True)
            parameters[param] = float(value)
        self._parameter_sets_to_evaluate.append(OrderedDict(**parameters))

    def _check_all_parameters_provided(self, new_param_set):
        if new_param_set.keys() != self._parameter_collection.keys():
            raise ValueError("The following parameters are required:\n{}\n"
            " The following were provided for a parameter study evaluation:\n{}\n".format(
                list(self._parameter_collection.keys()),
                list(new_param_set.keys())))

    @property
    def parameter_sets_to_evaluate(self):
        return self._parameter_sets_to_evaluate

    def _run_study(self):
        self._check_parameter_sets_populated()
        param_sets = self._parameter_sets_to_evaluate
        success = True
        exit_status = 0
        e=""
        try:
            self._batch_results = self._matcal_evaluate_parameter_sets_batch(param_sets, is_restart=self._restart)
        except Exception as e:
            success = False
            exit_status = -1
        self._results._set_exit_information(success, exit_status, f"{e}")
        return self._results

    def _check_parameter_sets_populated(self):
        if not self._parameter_sets_to_evaluate:
            raise RuntimeError("The parameter study has no evaluations defined."
                               " Please use the \"add_parameter_evaluation\" "
                               "method to add parameter sets to evaluate")

    def _format_parameter_batch_eval_results(self, batch_raw_objectives, 
                                             flattened_batch_results, 
                                             total_objs, parameter_sets, batch_qois):

        return ParameterBatchEvaluator.default_results_formatter(batch_raw_objectives, 
                                                                 total_objs, 
                                                                 parameter_sets, batch_qois)

    def make_total_objective_study(self):
        """
        This changes the stored total objectives to be a summation of 
        all metric function results. 
        """
        self._return_residuals = False

    def make_residuals_study(self):
        """
        This changes the stored total objectives to be the L2 norm of one 
        long concatenated residual from all objectives added using 
        :meth:`~matcal.core.parameter_studies.ParameterStudy.add_evaluation_set`
        """
        self._return_residuals = True

    def _format_parameters(self, params):
        return params

    def restart(self):
        """
        Sets the study to launch in restart mode. The study will use existing
        results from previous launches to populate the results instead of 
        running the simulations again. Note that this feature requires that no 
        changes to the study to be made in order for the study to produce correct
        results. 

        Files from previous runs are read in to this study, they should not be 
        deleted. Missing files may cause errors in the restart. 

        If any random number generation is used in the calculation. It is important to 
        set the same seed value as used previously
        """
        self._restart = True
            
    def _study_specific_postprocessing(self):
        """"""

class HaltonStudy(ParameterStudy):
    def __init__(self, *parameters, scramble=True, rng=None):
        """Initialize the HaltonStudy

        :param scramble: If True, Owen scrambling is used. Defaults to False.
        :type scrambel: bool
        
        :param rng: Pseudorandom numer generator state. When rng is None, a new generator
            is created using entropy from the operating system.
        :type rng: int
        """
        # optional: check that all parameters are continuous design or
        # uniform uncertain

        super().__init__(*parameters)
        self.l_bounds = []
        self.u_bounds = []
        for idx, key in enumerate(self._parameter_collection):
            self.l_bounds.append(self._parameter_collection[key].get_lower_bound())
            self.u_bounds.append(self._parameter_collection[key].get_upper_bound())
        self.dim = len(self._parameter_collection)
        self._design = None
        self._check_variable_type(scramble, 'scramble', bool)
        #if rng is None:
        #    warnings.warn("If design will be continued, we recommend setting rng to an integer value.", UserWarning)
        self.HaltonSampler = qmc.Halton(d=self.dim, scramble=scramble, seed=rng)


    def _check_variable_type(self, var, var_name, *var_types):
        """Assert variables are of the given type.

        :param var_name: name of parameter being checked
        :type var_name: str
        
        :param var: variable to be checked
        :type var: any
        
        :param var_types: type(s) that 'var_name' should be. If not, a TypeError is raised
        :type var_types: (__type__)
        """

        if not isinstance(var, var_types):
            if len(var_types) == 1:
                message = f"'{var_name}' must be of type {var_types[0].__name__}"
            else:
                message = f"'{var_name}' must be one of the types: {', '.join(t.__name__ for t in var_types)}"
            raise TypeError(message)

    def launch(self, nsamples=20, skip=None):
        """ Launch study, generates samples from Halton Sequence and
        scales to bounds if bounds are defined.

        :param nsamples: Number of parameter samples to generate from Halton sequence.
        :type nsamples: int
        
        :param skip: When continuing an existing design, the user may optionally skip ahead in the
                     Halton sequence by an amount determined by 'skip'.
        :type skip: int
        """
        self._set_number_of_samples(nsamples, skip)
        
        return super().launch()

    def _set_number_of_samples(self, nsamples, skip):
        """ generates samples from Halton Sequence and
        scales to bounds if bounds are defined.

        :param nsamples: number of parameter samples to generate from Halton sequence
        :type nsamples: int
        """

        self._check_variable_type(nsamples, 'nsamples', int)
        if skip is not None:
            self._check_variable_type(skip, 'skip', int)
            self._skip_ahead(skip)
        self._generate_samples(nsamples)

    def _generate_samples(self, nsamples):
        """ Generate sample from a Halton seqence
        """
        unscaled_samples = self.HaltonSampler.random(n=nsamples)
        scaled_samples = self._scale_samples_to_bounds(unscaled_samples)
        self._populate_parameter_evaluations(scaled_samples)
        
    def _populate_parameter_evaluations(self, scaled_samples):
        
        param_order = self._parameter_collection.get_item_names() 

        self._new_sample_start_index = len(self._parameter_sets_to_evaluate)
        for sample in scaled_samples:
            ss = { key:sample[i] for i, key in enumerate(param_order) }
            self._add_parameter_evaluation(**ss)
        self._check_parameter_sets_populated()

    def _skip_ahead(self, skip):
        _ = self.HaltonSampler.fast_forward(skip)

    def _scale_samples_to_bounds(self, samples):
        """
        Scale samples to be within defined bounds.

        :param samples: samples to be scaled
        :type samples: np.ndarray of size (nsamples x nfeatures)

        :return: scaled samples
        :type return: np.ndarray of size (nsamples x nfeatures)
        """

        return qmc.scale(samples, self.l_bounds, self.u_bounds)

    def _add_parameter_evaluation(self, **p):
      super().add_parameter_evaluation(**p)

    def add_parameter_evaluation(self, **parameters):
        """"""
        raise self.StudyInputError("Users cannot add parameter evaluations to a HaltonStudy.")

class FiniteDifference:

    def __init__(self, center_point, relative_step_size=1.e-3, 
                 epsilon=np.sqrt(np.finfo(float).eps)):
        self._center_point = np.array(center_point, dtype=float)
        self._number_of_variables = len(self._center_point)
        self._relative_step_size = relative_step_size
        self._step_sizes = []
        for x in self._center_point:
          dx = np.abs(x)*relative_step_size
          if dx < epsilon: 
            dx = epsilon
          self._step_sizes.append(dx)
        self._finite_difference_evaluation_points = None
        self._gradient_coefficients = None
        self._gradient_indices      = None
        self._function_values = None
        self._hessian_coefficients = None
        self._hessian_indices      = None

    def set_function_values(self,ys): 
        self._function_values = ys
        ndim = np.squeeze(ys).ndim
        self._function_shape = None
        if ndim > 1: 
          self._function_shape = ys[0].shape

    def gradient(self): 
        shape = [self._number_of_variables]
        if self._function_shape is not None: shape.extend(self._function_shape)
        G = np.zeros(shape)
        for i,c in enumerate(self._gradient_coefficients):
            for j,ii in enumerate(self._gradient_indices[i]):
                G[i] += c[j]*self._function_values[ii]
        return G
    
    def hessian(self):
        shape = [self._number_of_variables,self._number_of_variables]
        if self._function_shape is not None: shape.extend(self._function_shape)
        H = np.zeros(shape)
        k = 0
        for i in range(self._number_of_variables):
            for l,m in enumerate(self._hessian_indices[k]):
                H[i,i] += self._hessian_coefficients[k][l]*self._function_values[m]
            k += 1
            for j in range(i+1,self._number_of_variables):
                for l,m in enumerate(self._hessian_indices[k]):
                    H[i,j] += self._hessian_coefficients[k][l]*self._function_values[m]
                k += 1
                H[j,i] = H[i,j]
        return H

    def compute_gradient_evaluation_points(self, three_point_finite_diff=True):
        self._gradient_coefficients = []
        self._gradient_indices = []
        self._finite_difference_evaluation_points = [self._center_point]
        for i in range(self._number_of_variables):
            dx = self._step_sizes[i]
            new_coeffs = []
            new_indexes = []
            coef_plus, idx_plus = self._get_gradient_step_point_coefficients_indices(dx, i, 
                                                                                three_point_finite_diff)
            new_coeffs.append(coef_plus)
            new_indexes.append(idx_plus)
            
            if three_point_finite_diff:
                coef_minus, idx_minus = self._get_gradient_step_point_coefficients_indices(-dx, i, 
                                                                                three_point_finite_diff)
                new_coeffs.append(coef_minus)
                new_indexes.append(idx_minus)
            else:
                new_coeffs.append(-1/dx)
                new_indexes.append(0)
            
            self._gradient_coefficients.append(new_coeffs)
            self._gradient_indices.append(new_indexes)

        return self._finite_difference_evaluation_points
      
    def _get_gradient_step_point_coefficients_indices(self, dx, i, three_point_finite_diff):
        x= self._center_point.copy()
        x[i] += dx
        coeff = 1/dx
        if three_point_finite_diff:
            coeff *= 0.5
        self._finite_difference_evaluation_points.append(x)
        return coeff, len(self._finite_difference_evaluation_points)-1

    def compute_hessian_evaluation_points(self):
        self.compute_gradient_evaluation_points(three_point_finite_diff=True)
        self._hessian_coefficients = []
        self._hessian_indices      = []
        for i in range(self._number_of_variables):
            dxi = self._step_sizes[i]
            self._get_hessian_diagonal_term_step_point_coefficients_indices(dxi, i)
            for j in range(i+1,self._number_of_variables):
                dxj = self._step_sizes[j]
                coefs, idxs = self._get_all_hessian_cross_terms(dxi, dxj, i, j)
                self._hessian_coefficients.append(coefs)
                self._hessian_indices.append(idxs)
        return self._finite_difference_evaluation_points

    def _get_all_hessian_cross_terms(self, dxi, dxj, i, j):
        coefs = []
        idxs = []
        coef, idx = self._get_hessian_cross_term_step_point_coefficients_indices(-dxi, -dxj, i, j)
        coefs.append(coef)
        idxs.append(idx)
        coef, idx = self._get_hessian_cross_term_step_point_coefficients_indices(dxi, -dxj, i, j)
        coefs.append(coef)
        idxs.append(idx)        
        coef, idx = self._get_hessian_cross_term_step_point_coefficients_indices(-dxi, dxj, i, j)
        coefs.append(coef)
        idxs.append(idx) 
        coef, idx = self._get_hessian_cross_term_step_point_coefficients_indices(dxi, dxj, i, j)
        coefs.append(coef)
        idxs.append(idx) 

        return coefs, idxs

    def _get_hessian_diagonal_term_step_point_coefficients_indices(self, dxi, i):
        inv_eps2 = 1.0/(dxi*dxi)
        self._hessian_coefficients.append([-2.0*inv_eps2, inv_eps2, inv_eps2])
        ii = self._gradient_indices[i]
        self._hessian_indices.append([0, ii[0], ii[1]])

    def _get_hessian_cross_term_step_point_coefficients_indices(self, dxi, dxj, i, j):
        inv_eps2 = 0.25/(dxi*dxj)
        x = self._center_point.copy()
        x[i] += dxi
        x[j] += dxj
        self._finite_difference_evaluation_points.append(x)
        return inv_eps2, len(self._finite_difference_evaluation_points)-1


_small = 1e-12


def _estimate_parameter_covariance(residuals, sensitivities, noise_variance):
    has_replicas = len(residuals.shape) > 1
    if has_replicas:
        Sigma_y = _get_residual_covariance(residuals)
        Sigma_guess = _solve_for_parameter_covariance(Sigma_y, sensitivities, 
                                                        noise_variance)
    elif not has_replicas:
        raise RuntimeError("The LaplaceStudy has no repeats. Repeat data "
                            "are needed for the study.")
    return Sigma_guess


def _get_residual_covariance(residuals):
    Sigma_y = np.cov(residuals.T) 
    return np.atleast_2d(Sigma_y)


def _solve_for_parameter_covariance( output_covariance, 
                                    residual_sensitivities, 
                                    noise_variance=0.0):
    mineval, maxeval = _check_covariance(output_covariance)
    if np.abs(mineval) < _small:
        logger.warning("Residual "
                    "covariance is not positive definite!")
    n_y = output_covariance.shape[0]
    n_p = residual_sensitivities.shape[1]
    output_covariance -= np.eye(n_y)*noise_variance
    [U,d,V] = np.linalg.svd(output_covariance)
    UTA = U[:,:n_p].T@residual_sensitivities
    invUTA = np.linalg.inv(UTA)
    s = np.diag(d[:n_p])
    if (d[:n_p-1] < _small).any(): 
        raise ValueError("LaplaceStudy under determined. "
                         "System may be under determined.")
    S = invUTA@s@(invUTA.T)
    return S


def _check_covariance(Sigma):
    try:
        evals = np.linalg.eigvalsh(Sigma)
        min_eval = np.min(evals)
        max_eval = np.max(evals)
    except Exception as e:
        logger.warning("Residual covariance eigenvalues could not be computed. "
                       "The LaplaceStudy results are likely to be unreliable.\n"
                       f"{repr(e)}")
        min_eval = 0
        max_eval = 0
    return min_eval, max_eval


class _MinimizeCallbackWithCounter:
    def __init__(self, num_parameters):
        self._iteration = 0
        self._num_parameters = num_parameters

    def __call__(self, intermediate_result):
        self._iteration += 1
        if self._iteration % 20 == 0:
            r = intermediate_result
            cur_sig = _process_fitted_covariance_result(r.x, 
                                                        self._num_parameters)
            cur_sig_str = "\n\t\t".join([str(repr(row)) for row in cur_sig])
            logger.info(f"\tCurrent covariance:\n\t\t{cur_sig_str}")
            logger.info(f"\tCurrent LaplaceStudy objective:\t{r.fun}")
            logger.info(f"\tCurrent iteration:\t{self._iteration}\n")


def _fit_posterior(residuals, residual_sensitivities, sigma_estimate, 
                  noise_variance, method='nelder-mead'):
    nparameters = residual_sensitivities.shape[1]
    init_variances, init_correlation_coefficients = _decompose_covariance(sigma_estimate)
    init_theta = _to_theta(init_variances, init_correlation_coefficients)
    init_theta = [x for x in init_theta]
    from scipy.optimize import minimize
    args = (residuals, residual_sensitivities, noise_variance)
    logger.info("Improving posterior covariance estimate:")

    callback = _MinimizeCallbackWithCounter(nparameters)
    try:
        result = minimize(_fitted_posterior_objective, init_theta, args=args, method=method, 
                          tol=1e-3, callback=callback)
    except np.linalg.LinAlgError as e:
        logger.warning("Improving covariance failed. Try a different noise estimate. " +
                       f"Improvement optimization failed due to the:\n  \'{repr(e)}\'")
        return None
    
    theta = result.x
    optimized_sigma = _process_fitted_covariance_result(theta, 
                                                        nparameters)
    return optimized_sigma


def _fitted_posterior_objective(theta, residuals, 
                                residual_sensitivities, noise_estimate):
    obj = -_log_posterior_predictive(theta, residual_sensitivities, 
                                              residuals, noise_estimate)
    return obj


def _decompose_covariance(Sigma):
    n = Sigma.shape[0]
    variances = np.diag(Sigma)
    s = np.diag(1.0/np.sqrt(variances))
    sigma = s@Sigma@s
    correlation_coefficients = []
    for i in range(n):
        for j in range(i+1,n):
            correlation_coefficients.append(sigma[i,j])
    correlation_coefficients = np.array(correlation_coefficients)
    return variances,correlation_coefficients


def _process_fitted_covariance_result(theta, nparameters):
    variances = theta[:nparameters]
    correlation_coefficients = theta[nparameters:]
    optimized_sigma = _assemble_covariance_matrix(np.exp(variances),
                                    np.tanh(correlation_coefficients))
    return optimized_sigma
   

def _to_theta(variances, correlation_coefficients, clip=True):
    v =  np.log(variances)
    if clip:
        tol = 1.e-8
        correlation_coefficients = np.clip(correlation_coefficients,-1.0+tol,1.0-tol)
    else:
        assert np.all(np.abs(correlation_coefficients) <= 1.0)
    c =  np.arctanh(correlation_coefficients)
    theta = np.concatenate([v,c])
    return theta


def _log_posterior_predictive(theta, residual_sensitivities, residuals, noise):
    noise2 = noise*noise
    if noise2 == 0.0:
        noise2 = _small
    variances, correlation_coefficients = _from_theta(theta, residual_sensitivities.shape[1])
    Sigma_y = _pushed_forward_variances(variances,correlation_coefficients,residual_sensitivities)
    Sigma_y = Sigma_y + noise2*np.eye(Sigma_y.shape[0])
    
    sign, logdet =  np.linalg.slogdet(Sigma_y)
    logdetSigma_y = sign*logdet
    invSigma = np.linalg.solve(Sigma_y, np.eye(Sigma_y.shape[0]))
    mse = np.einsum("ki,ij,kj",residuals,invSigma,residuals) 

    n_repeats = residuals.shape[1]
    logp = -0.5*( logdetSigma_y + mse/n_repeats)
    return logp
    

def _from_theta(theta, nparameters, clip=True):
    variances                = np.exp (theta[:nparameters])
    correlation_coefficients = np.tanh(theta[nparameters:])
    if clip:
        tol = 1.e-8
        correlation_coefficients = np.clip(correlation_coefficients,-1.0+tol,1.0-tol)
    return variances,correlation_coefficients


def _pushed_forward_variances(variances, correlation_coefficients, 
                            parameter_sensitivities):
    A = parameter_sensitivities
    Sigma = _assemble_covariance_matrix(variances,correlation_coefficients) 
    ASAT = A@Sigma@A.T
    return ASAT

    
def _assemble_covariance_matrix(variances, correlation_coefficients): 
    n = len(variances);
    Vars = np.diag(variances);
    Cors = np.eye(n);
    indx = 0
    for i in range(n):
        for j in range(i+1,n):
            Cors[i,j] = correlation_coefficients[indx]
            Cors[j,i] = correlation_coefficients[indx]
            indx += 1
    Sigma = np.sqrt(Vars)@Cors@np.sqrt(Vars) 
    return Sigma


class _LaplaceStudyBase(ParameterStudy):

    def __init__(self, *parameters):
        super().__init__(*parameters)
        self._center = None
        self._finite_difference = None
        self._step_size = None
        self.set_step_size()

    def _check_parameter_sets_populated(self):
        if not self._parameter_sets_to_evaluate:
            raise RuntimeError("The LaplaceStudy has no parameter center defined. "
                 "Please use the \"set_parameter_center\" method before launching the study.")
    
    def set_parameter_center(self, **parameters):
        """
        Pass an unpacked dictionary of parameters with valid 
        values to set the center about which to calculate the Hessian for the 
        study objectives. These parameters must be valid for the study parameters 
        and all study parameters must be included. The values must be determined 
        from a calibration and must be located at an objective minimum. 

        :param parameters: keyword/value pair of parameter names and values for the 
          location about which to calculate the Hessian
        """
        self._check_all_parameters_provided(parameters)
        param_order = self._parameter_collection.get_item_names() 
        ordered_center = OrderedDict()
        for param in param_order:
            ordered_center[param] = parameters[param]
        self._center = ordered_center
        center = [ self._center[key] for key in param_order ]
        self.mean = np.array(center)
        self._setup_finite_difference()

    def _setup_finite_difference(self):
        self._parameter_sets_to_evaluate = []
        self._finite_difference = FiniteDifference(self.mean, relative_step_size=self._step_size)
        finite_difference_points = self._get_finite_difference_evaluation_points()
        param_order = self._parameter_collection.get_item_names()
        for pt in finite_difference_points:
            p = { key:pt[i] for i,key in enumerate(param_order) }
            self._add_parameter_evaluation(**p)

    def set_step_size(self, step_size=1e-3):
        """
        Sets the finite difference step sizes for the LaplaceStudy hessian 
        and gradient approximations. This is a relative step size.
        Default step size is a relative step of 1e-3. The value must be between
        zero and one.
        
        :param step_size: the desired step_size
        :type step_size: float
        """
        check_value_is_real_between_values(step_size, 0, 1, 
                                           "step_size", 
                                           "LaplaceStudy.set_step_size")
        self._step_size=step_size
        if self._finite_difference is not None:
            self._setup_finite_difference()

    def _add_parameter_evaluation(self, **p):
      super().add_parameter_evaluation(**p)

    def add_parameter_evaluation(self, **parameters):
        """"""
        raise self.StudyInputError("Users cannot add parameter evaluations to a LaplaceStudy.")

    def _get_center_eval_index(self):
        return 0

    def _gradient(self): 
        G = self._finite_difference.gradient()
        return G

    def _get_raw_residuals(self, model_name, obj_name, eval_index):
        batch_objectives = self._batch_results['objectives']
        return batch_objectives[eval_index][model_name][obj_name].residuals

    def _get_normalized_weighted_conditioned_residuals(self, model_name, obj_name, eval_index, 
        flatten=False):
        batch_objectives = self._batch_results['objectives']
        eval_model_obj_res = batch_objectives[eval_index][model_name][obj_name]
        result = eval_model_obj_res.weighted_conditioned_normalized_residuals
        if flatten:
            result = eval_model_obj_res.flatten_data_collection(result)
        return result
        
    def _log_total_sensitivity_information(self):
        logger.info("\n")
        logger.info("Parameter center:")
        logger.info(str(repr(self.mean)))
        logger.info("\n")
        logger.info("Estimated parameter covariance:")
        init_sigma = (self._results.outcome["estimated_parameter_covariance"])
        logger.info(str(repr(init_sigma)))
        logger.info("\n")

    def _get_parameter_specific_results(self, gradient_key):
        results = OrderedDict()
        results[gradient_key]   = self._gradient()
        results["mean"] = self.mean
        results = _package_parameter_specific_results(self._parameter_collection, results)
        return results

    @abstractmethod
    def _get_finite_difference_evaluation_points(self):
        """"""

    @abstractmethod
    def _get_overall_results(self):
        """"""

class LaplaceStudy(_LaplaceStudyBase):
    """
    Use the MatCal :class:`~matcal.core.parameter_studies.LaplaceStudy` to evaluate the gradient of the 
    calibration residuals
    at an optimal point in parameter space. The residual gradient can then be used to form a modified Laplace 
    approximation to estimate the parameter covariance matrix for use in uncertainty quantification. We perform this
    assuming uncertainty is due to model form error. 
    """
    study_class = "LaplaceStudy"
    _laplace_results_key = "laplace results"

    def __init__(self, *parameters):
        super().__init__(*parameters)
        self._calibrate_covariance = True
        self.set_noise_estimate()
        
    def _get_finite_difference_evaluation_points(self):
        return self._finite_difference.compute_gradient_evaluation_points()

    def add_evaluation_set(self, model, objectives, data=None, 
                           states=None, data_conditioner_class=MaxAbsDataConditioner):
        super().add_evaluation_set(model, objectives, data, states, data_conditioner_class)
        for eval_set in self._evaluation_sets.values():
            for obj_set in eval_set.objective_sets:
                for obj_name in obj_set.objectives:
                    obj = obj_set.objectives[obj_name]
                    more_than_one_qoi = self._check_obj_qois_for_more_than_one_qoi(obj_set, 
                                                                                   obj_name)
                    objs_invalid = more_than_one_qoi and not obj.has_independent_field()
                    self._raise_error_if_objs_invalid(objs_invalid)

    def set_noise_estimate(self, noise_estimate=0.0):
        """
        Set the estimate for the noise in the data. 
        Currently only a single value is accepted for all data.
        This is the expected standard deviation of the noise.

        :param noise_estimate: value for the noise estimate
        :type noise_estimate: float
        """
        check_value_is_positive_real(noise_estimate, "noise_estimate", 
                                     f"{self.study_class}.set_noise_estimate")
        self._noise_variance=noise_estimate**2

    def set_calibrate_covariance(self, calibrate_covaraince=True):
        """
        By default, the laplace study will attempt to improve the 
        covariance through a calibraiton. Optionally, turn this off or back on.

        :param calibrate_covariance: flag to turn the covariance calibration process off or on
        :type calibrate_covariance: bool
        """
        self._calibrate_covariance = calibrate_covaraince

    def update_laplace_estimate(self, noise_estimate):
        """Update the laplace study covariance estimate after with an 
        updated noise estimate."""
        if self._results is None:
            raise RuntimeError("Study has not been run yet. Use the \'launch\' method " +
                "for the first study run. ")
        self.set_noise_estimate(noise_estimate)
        self._study_specific_postprocessing()
        return self._results

    def _log_total_sensitivity_information(self):
        super()._log_total_sensitivity_information()
        if self._calibrate_covariance:
            logger.info("Calibrated parameter covariance:")
            fit_sigma = (self._results.outcome["fitted_parameter_covariance"])
            logger.info(str(repr(fit_sigma)))
            logger.info("\n")

    def _check_obj_qois_for_more_than_one_qoi(self, obj_set, obj_name):
        more_than_one_qoi = False
        conditioned_exp_qois = obj_set.conditioned_experiment_qoi_collection[obj_name]
        for state in conditioned_exp_qois:
            for data in conditioned_exp_qois[state]:
                if data.length > 1:
                    more_than_one_qoi = True
        return more_than_one_qoi

    def _raise_error_if_objs_invalid(self, objs_invalid):
        if objs_invalid:
            raise ValueError(f"The {LaplaceStudy.study_class}Study"
                            " only accepts residuals/objectives of length 1 or" 
                            " objectives with independent fields variables"
                            " so that repeat data can be compared at common" 
                            " independent variable locations.")

    def _study_specific_postprocessing(self):
        total_eval_residual_vecs = self._extract_residual_information_for_processing()
        center_resids = total_eval_residual_vecs[self._get_center_eval_index()].T
        residual_gradients = self._calculate_residual_sensitivities(total_eval_residual_vecs)
        covariance_estimates = self._calculate_covariance(center_resids, residual_gradients)
        output = self._get_parameter_specific_results("residuals_gradient")
        output.update(self._get_overall_results(covariance_estimates))
        self._results._set_outcome(output)
        self._log_total_sensitivity_information()
    
    def _get_overall_results(self, covariance_estimates):
        results = OrderedDict()
        results["parameter_order"] = self._parameter_collection.get_item_names()
        results.update(covariance_estimates)
        return results

    def _calculate_residual_sensitivities(self, total_eval_residuals):
        self._finite_difference.set_function_values(total_eval_residuals)
        residual_sensitivities = np.atleast_3d(self._gradient())
        # grab first one from repeats, this is the most populated and since this is 
        # the derivative of the residuals where the third index is the repeat #, 
        # the first one is good enough for derivative of the model w.r.t. the parameters. 
        residual_sensitivities = residual_sensitivities.T[0, :, :]
        return residual_sensitivities

    def _calculate_covariance(self, center_resids, residual_sensitivities):
        estimated_covariance = _estimate_parameter_covariance(center_resids, 
                                                             residual_sensitivities, 
                                                             self._noise_variance) 
        covariance_results = OrderedDict()
        covariance_results["estimated_parameter_covariance"] = estimated_covariance
        if self._calibrate_covariance:
            fitted_posterior = _fit_posterior(center_resids, residual_sensitivities, 
                                              estimated_covariance, 
                                         self._noise_variance)
            covariance_results["fitted_parameter_covariance"] = fitted_posterior
        else:
            logger.info("Skipping covariance calibration by user request.\n")

        return covariance_results
    
    def _extract_residual_information_for_processing(self):
        residual_matrices=[]
        for eval_index in range(self._results.number_of_evaluations):
            eval_sub_residual_matrices = []
            for model, eval_set in self._evaluation_sets.items():
                for obj_set in eval_set.objective_sets:
                    for obj_name in obj_set.objectives:
                        resids_dc = self._get_raw_residuals(model.name, obj_name, eval_index)
                        if obj_set.objectives[obj_name].has_independent_field():
                            indep_field = obj_set.objectives[obj_name].independent_field
                            exp_dc = obj_set.data_collection
                            new_resids = self._get_interpolated_responses(resids_dc, exp_dc, 
                                                                          indep_field)
                        else:
                            new_resids = self._get_single_response_set(resids_dc)
                        eval_sub_residual_matrices += new_resids
            _combine_array_method = _combine_array_list_into_zero_padded_single_array
            total_residual_matrix = _combine_array_method(eval_sub_residual_matrices)
            residual_matrices.append(np.atleast_2d(total_residual_matrix))
        return residual_matrices
    
    def _get_interpolated_responses(self, residuals_dc, exp_dc, indep_field):
        data_stats = DataCollectionStatistics()
        combined_interpolated_residuals = []
        for state in residuals_dc:
            register_data_method = data_stats._interpolate_state_data_to_common_independent_variable
            interpolated_resids = register_data_method(indep_field, residuals_dc, state, exp_dc)
            interpolated_resids.pop(indep_field)
            for field in interpolated_resids:
                combined_interpolated_residuals.append(np.atleast_2d(interpolated_resids[field]).T)

        return combined_interpolated_residuals

    def _get_single_response_set(self, response_dc):
        combined_responses = []
        for state in response_dc:
            for field in response_dc.state_common_field_names(state):
                combined_resids_current_field = []
                for data in response_dc[state]:
                    if data.length > 1:
                        raise RuntimeError(f"Error in {LaplaceStudy.study_class}Study."
                                            " Contact MatCal support")
                    combined_resids_current_field.append(data[field][0])
                combined_responses.append(np.atleast_2d(combined_resids_current_field))
        return combined_responses
   

def sample_multivariate_normal(nsamples, mean, covariance_matrix, seed=None, param_names=None):
    """
    Sample the multivariate normal distributions for the study parameters 
    using the mean and covariance matrix provided by a LaplaceStudy or other UQ method. 
    
    :param nsamples: the number of samples to return from the distribution
    :type nsamples: int

    :param mean: the mean value for the parameters. This would be the calibrated 
        value for most MatCal studies.
    :type mean: Array-like

    :param covariance_matrix: parameter covariance matrix from which to generate
        samples from.
    :type covariance_matrix: Array-like
    
    :param seed: an optional seed for the random number generator performing the sampling
    :type seed: int

    :param param_names: optionally provide a list with the parameter names in the correct order.
        so that the resulting samples will be returned in a dictionary format where each parameter 
        name key will have a list of parameter values associated with it with length nsamples.
    :type param_names: list(str)
    
    :return: a dictionary for the generated samples where the keys are the parameter 
        names (if provided) and the values are arrays storing the sampled values.
        If parameter names are not provided a name is generated of the form "parameter_#".
    :rtype: dict(str, list(float))
    
    """
    _check_sample_covariance_mat_inputs(nsamples, mean, covariance_matrix, seed, param_names)
    samples = _get_multivariate_normal_samples(mean, covariance_matrix, nsamples, seed)
    samples_dict = _create_samples_dict_from_samples_array(samples, param_names)   
    
    return samples_dict

def _check_sample_covariance_mat_inputs(nsamples, mean, covariance_matrix, seed, param_names):
    check_value_is_positive_integer(nsamples, "nsamples", "sample_covariance_matrix")
    check_value_is_array_like_of_reals(mean, "mean", 
                                       "sample_covariance_matrix")
    check_value_is_array_like_of_reals(covariance_matrix, "covariance_matrix", 
                                "sample_covariance_matrix")
    if (len(mean)!=covariance_matrix.shape[0] or
       covariance_matrix.shape[0]!=covariance_matrix.shape[1]):
       raise ValueError("The mean and covariance matrix passed to \"sample_covariance_matrix\" "
                        "are invalid. Their sizes must match appropriately.")

    if seed is not None:
        check_value_is_positive_integer(seed, "seed", f"sample_covariance_matrix")
   
    if param_names is not None and len(param_names) != covariance_matrix.shape[0]:
        raise ValueError("The length of the parameter names list must equal the number of "
                         "rows and columns in the provided covariance matrix.")

def _get_multivariate_normal_samples(mean, sigma, nsamples, seed):
    try: # modern python
        if seed is not None:
            rng = np.random.default_rng(seed=seed)
        samples = rng.multivariate_normal(mean, sigma, nsamples).T
    except: # old python e.g. 3.7
        if seed is not None:
            np.random.seed(seed)
        samples = np.random.multivariate_normal(mean, sigma, nsamples).T
    return samples

def _create_samples_dict_from_samples_array( samples, param_names=None):
    samples_dict = OrderedDict()    
    for param_index, value in enumerate(samples[:, 0]):
        if param_names is not None:
            parameter_name = param_names[param_index]
        else:
            parameter_name = f"parameter_{param_index}"
        samples_dict[parameter_name] = samples[param_index, :]
    return samples_dict


class ClassicLaplaceStudy(_LaplaceStudyBase):
    """
    Use the MatCal :class:`~matcal.core.parameter_studies.ClassicLaplaceStudy` 
    to evaluate the Hessian (and gradient)
    at an optimal point in parameter space. The Hessian can then 
    be used to form the Laplace 
    approximation to the parameter covariance matrix for use in uncertainty 
    quantification. We perform this
    assuming uncertainty is due to noise in the data alone for the classical 
    approach the Laplace Approximation. 
    """
    study_class = "ClassicLaplaceStudy"
    _laplace_results_key = "laplace results"

    def _get_finite_difference_evaluation_points(self):
        return self._finite_difference.compute_hessian_evaluation_points()
    
    def _study_specific_postprocessing(self):
        results = self._extract_objective_information_for_processing()
        total_SSE_objectives, total_residual_vecs = results
        self._finite_difference.set_function_values(total_SSE_objectives)
        output = self._get_parameter_specific_results("objective_gradient")
        output.update(self._get_overall_results(total_residual_vecs))
        self._results._set_outcome(output)
        self._log_total_sensitivity_information()

    def _get_overall_results(self, total_residual_vecs):
        results = OrderedDict()
        results["hessian"]    = self._hessian()
        results["parameter_order"] = self._parameter_collection.get_item_names()

        total_noise_estimate = np.std(total_residual_vecs[self._get_center_eval_index()])
        results["standard_deviation"] = total_noise_estimate
        param_covariance = _get_total_scaled_covariance(self._inverse_hessian(), 
                                                        total_noise_estimate)
        results["estimated_parameter_covariance"] = param_covariance
        return results

    def _extract_objective_information_for_processing(self):
        SSE_objectives=[]
        flattened_resids = []
        for eval_index in range(self._results.number_of_evaluations):
            eval_flattened_residuals = np.array([])
            for model, eval_set in self._evaluation_sets.items():
                for obj_set in eval_set.objective_sets:
                    for obj_name in obj_set.objectives:
                        get_resid_method = self._get_normalized_weighted_conditioned_residuals
                        this_flattened_resids = get_resid_method(model.name, obj_name, 
                            eval_index, flatten=True)
                        eval_flattened_residuals = np.append(eval_flattened_residuals, 
                                                            np.atleast_1d(this_flattened_resids))
            flattened_resids.append(eval_flattened_residuals)
            objective = self._get_sum_of_squares_objective(eval_flattened_residuals)
            SSE_objectives.append(objective)
        return SSE_objectives, flattened_resids
   
    def _get_sum_of_squares_objective(self, residuals):
        return np.dot(residuals, residuals)

    def _hessian(self):
        H = self._finite_difference.hessian()
        return H
    
    def _inverse_hessian(self):
        H = self._finite_difference.hessian()
        try:
            C = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            logger.warning("Could not invert the hessian for this LaplaceStudy." 
                          " Error estimation due to external noise is invalid.")
            C = np.ones(H.shape)
        return C
  

def _get_total_scaled_covariance(inverse_hessian, std_dev_estimate):
    cov = inverse_hessian
    scale = 2*std_dev_estimate*std_dev_estimate
    return scale*cov


def _package_parameter_specific_results(param_collect, sens_info):
    out = OrderedDict()
    for sens_key, sens_val in sens_info.items():
        for param_i, param_key in enumerate(param_collect.keys()):
            if isinstance(sens_val, (list, np.ndarray)):
                out_name = f"{sens_key}:{param_key}"
                out[out_name] = sens_val[param_i]
    return out


def _combine_array_list_into_zero_padded_single_array(arrays):
    max_shape = [0,0]
    num_resids = 0
    for array in arrays:
        current_shape = array.shape
        max_shape[0] = np.max((max_shape[0], current_shape[0]))
        max_shape[1] = np.max((max_shape[1], current_shape[1]))
        num_resids += current_shape[0]
    
    combined_array = np.zeros((num_resids, max_shape[1]))
    current_eval_set_row = 0
    from copy import deepcopy
    for array in arrays:
        start_row = current_eval_set_row
        end_row = current_eval_set_row+array.shape[0]
        end_col = array.shape[1]
        combined_array[start_row:end_row, 0:end_col] = deepcopy(array)
        current_eval_set_row = end_row
    return combined_array


class VoronoiBatchStudy(ParameterStudy):
    def __init__(self, model, bounds,  X_test, y_test, surr_model_type='GPR', voronoi_type='full', 
                 finite_only=False, iterative_updates=True, rng=None):
        """Initialize the VoronoiBatchSamplingStudy

        :param model: Model to be evaluated, which has a 'fit' and 'predict' method 
        :type model:
        
        :param bounds: upper and lower bound of each parameter. Defines bounds of sampling region.
        :type bounds: list

        :param X_test: Parameter samples reserved for testing.
        :type X_test: nd_array
        
        :param y_test: Model output corresponding to X_test reserved for testing.
        :type y_test: nd_array
        
        :param surr_model_type: Type of surrogate to be used. Options are 'GPR' (default) and 'SVR'.
        :type surr_model_type: str
        
        :param voronoi_type: Defines shich variation of voronoi sampling to use. Options are 'full' (default), 'local' and
                             'sampling'. If voronoi_type == 'full', then the full voronoi tesselation is made. If
                             voronoi_type == 'local', then a voronoi tesselelation is only made for nearby points as
                             determined through k-nearest neighbors. This option, may reduce computational demand in
                             high dimensions. If voronoi_type == 'sampling', then new sample locations are determined
                             through a sampling algorithm instead of choosing the furthest point in the voronoi region.
                             This approach may also reduce computational demand in high dimensions.
        :type voronoi_type: str
        
        :param finite: With finite=True, only vertices which reside inside convex hull defined by boundary points
                       are considered as candidate locations for new samples. With finite=False (default), all vertices are
                       consider as candidate locations for new samples. In this case, vertices which fall outisde 
                       the parameter bounds are snipped to the convex hull defined by boundary points, which requires
                       more computational resources, especially in high dimensions.
        :type finite_only: bool
        
        :param iterative_updates: If iterative=True (default), the voronoi tessellation is remade after each new sample is added. 
                          This promotes a design that is more space filling. If iterative=False, the voronoi tesselation 
                          is updated once per batch after all the samples are chosen. Setting iterative=False can be faster,
                          especially in high dimensions but can result in sample clustering.
        :type iterative: bool

        :param rng: Pseudorandom number generator state. When rng is None, a new generator is created using entropy
                    from the operating system.
        :type rng: int

        :return:
        :type rtn:
        
        """

        self._initialize_attributes(model, bounds, X_test, y_test, surr_model_type, voronoi_type, finite_only,
                                    iterative_updates, rng)
        
    def _initialize_attributes(self, model, bounds, X_test, y_test, surr_model_type, voronoi_type, finite_only,
                                iterative_updates, rng):
        self.surr_model = model
        self.surr_model_type = surr_model_type
        self.bounds = bounds
        self.X_test = X_test
        self.y_test = y_test
        self.voronoi_type = voronoi_type
        self.finite_only = finite_only
        self.iterative_updates = iterative_updates

        self.dim = len(bounds)
        self.boundary_points = self.make_nd_grid(bounds, 2)

        # lists for tracking error as design evolves
        self._nbatch_samples = []
        self.mse = []
        self.mape    self._
        self.mae = []
        self.smape = []
        self.surrogate_loss = []
        
        # convergence check epsilon
        eps = 1e-5

    def launch(self, nsplits=8, nmax_folds=3, nmax_loo=25, cv_scale=None, cv_metric='sum_abs_error',
                 group_kfold=False, thin=None, random_selection=None, nbatches=20):
        """ Perform Voronoi batch sampling
        
        :param nsplits: The number of splits to use in k-fold cross validation. Default is 8.
        :type nsplits: int
    
        :param nmax_folds: The number of points with the greatest k-fold CV error to keep as candidates for new
                           sample locations. Default is 3.
        :type nmax_folds: int

        :param nmax_loo: The number of points with the greatest LOOCV error to keep as candidates for new sample locations.
                          Default is 25.
        :type nmax_loo: int
        
        :param cv_scale:
        :tpye cv_scale:
        
        :param cv_metric: Determines which metric to use when calculating errors during cross validation. Options are
                          'sum_abs_error' (default), 'mse', 'mape', 'rmse', 'sum_abs_perc_error'. 
        :type cv_metric: str
        
        :param group_kfolds: If set to True, then groups in KFold CV are pre-determined through k-means clustering
                             so that samples close together are always in the same fold during cross validation. If
                             set to False (default), then groups are randomly assinged by the KFold algorithm.
        :type group_kfolds: bool

        :param thin: If defined, then every nth candidate sample location is chosen as a new sample location. This helps
                    to reduce computational demand in high dimensions. Default is None.
        :type thin: int or None
        
        :param random_selection: If defined, then this defines the number of new samples that are randomly chosen
                                 from candidate sample locations. This helps to reduce computational demand in
                                 high dimensions. Default is None.
        :type random_selection: int or None
        
        :param nbatches: The number of sampling batches to perform. Default 20.
        :type nbatches: int
        """
        
        if random_selection is not None and thin is not None:
            raise ValueError("Only one of 'thin' or 'random_selection' can be activated. Not both.")
        if nmax_loo == 'all' and thin is None and random_selection is None:
            print("Samples will be drawn for all regions in nmax_folds since none of LOOCV, thinning, or  random selection are activated")

        # calculate initial surrogate error
        self._calculate_errors()
        self.calculate_surrogate_loss()

        self._nbatch_samples.append(X.shape[0])
    #    for batch_number in range(20):  # Specify the number of new samples to draw in a batch
        batch_number = 0
        while True:
            print(f"Sampling batch {batch_number}. Currently {X.shape[0]} samples.")
            print("................................................................")
            X = self._perform_voronoi_batch_sampling(X, y, model=model,
                nmax_folds=nmax_folds, nmax_loo=nmax_loo, iter_=batch_number,
                n_splits=nsplits, cv_metric=cv_metric, group_kfold=group_kfold, thin=thin,
                random_selection=random_selection, cv_scale=cv_scale)

            y = fun(X)
            model_list = Parallel(n_jobs=-1)(delayed(fit_model)(model, X, y) for _ in range(1))
            model = model_list[0]
            self._calculate_errors()
            self._calculate_surrogate_loss()
            
            self._nbatch_samples.append(X.shape[0])
            # convergence check
            if np.abs(surrogate_loss[batch_number+1] - surrogate_loss[batch_number]) <= eps:
                print(f"BREAKING: Convergence from surrogate loss.")
                print(surrogate_loss)
                break
            elif np.abs(voronoi_mse[batch_number+1] - voronoi_mse[batch_number]) <= eps:
                print(f"BREAKING: Convergence from surrogate loss.")
                print(surrogate_loss)
                break
            elif np.abs(voronoi_mse[batch_number+1] - voronoi_mse[batch_number]) <= eps:
                print(f"BREAKING: Convergence from MSE.")
                print(voronoi_mse)
                break
            else:
                print("Surrogate not converged yet.")
            batch_number += 1

        #return super().launch()

    def _calculate_errors(self):
        y_pred = self.surr_model.predict(self.X_test)
        pred_error = np.abs(y_pred - self.y_test)
        ntest = len(self.X_test)
        self.mse.append(1/(ntest) * sum(pred_error ** 2))
        self.mape.append(1/(ntest) * sum((pred_error/np.abs(self.y_test)) * 100))
        self.mae.append(pred_error.mean())
        self.smape.append(2/(ntest) * sum(pred_error / (np.abs(self.y_test) + np.abs(y_pred)) ) * 100)


    def _calculate_surrogate_loss(self):
        # convergence based on marginal log likelihood for GPR
        if self.surr_model_type == 'GPR':
            self.surrogate_loss.append(self.surr_model.surrogate.log_marginal_likelihood(\
                model.surrogate.kernel_.theta))
        if self.surr_model_type == 'SVR':
            epsilon = self.surr_model.surrogate.epsilon
            self.surrogate_loss.append(np.maximum(0, pred_error - epsilon).mean())


    def _perform_voronoi_batch_sampling(self, X, y, model, bounds, boundary_points, nmax_folds=1,
         nmax_loo=1, iter_=None, iterative_updates=True, figdir=None,
         plot_figs=False, finite_only=False, voronoi_type='full', n_splits=5,
         cv_metric='sum_abs_error', group_kfold=False, thin=None,
         random_selection=None, cv_scale=None):
        """
        Perform Voronoi batch sampling based on the specified algorithm.

        Parameters:
        X: np.ndarray
            Feature matrix (training samples): nsamples x feature dimension
        y: np.ndarray
            Target values (ground truth): 
        nmax_loo: int
            Retain the nmax_loo samples with max error from LOOCV.
        nmax_folds: int
            Retain the nmax_folds folds with max error from KFold CV.
        model: object
            A machine learning model that has fit and predict methods.
        bounds: list
            Bounds of feature space
        boundary_points: array
            Boundary points defining bounds

        Returns:
        list: New samples selected.
        """

        # Step 1: Randomly sort existing samples into K-folds and perform KFold Cross Validation
        ndim = X.shape[1]
        X_orig = X.copy()

        if n_splits > 0:
            print("Performing kfold cross-validation...")
            kf_start = time.time()
            kfcv = KFoldCrossValidation(model, n_splits=n_splits, group_kfold=group_kfold, scale=cv_scale)
            groups = None

            if group_kfold:
                kmeans = KMeans(n_clusters=n_splits, random_state=42)
                groups = kmeans.fit_predict(X)
                if True:
                    # Plot the results
                    plt.figure(figsize=(10, 6))
                    xdf = pd.DataFrame(X)
                    xdf['label'] = groups
                    plt.figure()
                    sns.pairplot(xdf, hue='label', palette='husl', plot_kws={'s':10})
                    plt.savefig(f"{figpath}/kmeans_groups_iter_{iter_}.png")
                    plt.close()

            kf = kfcv.perform_kfold_cv(X, y, metric=cv_metric, groups=groups)

            # Step 2: Select the fold(s) with the n largest K-fold CV error(s)
            print("Finding max kfold error...")
            max_folds = find_indices_of_n_largest_kf_errors(kf, nmax_folds)
            max_fold_indices = np.concatenate(list(max_folds.values())) 
            kf_end = time.time()
            #print(f"kfold operations: {kf_end - kf_start} sec, {(kf_end - kf_start)/60} min.")
            if plot_figs and ndim == 2:
                fig, ax = plot_voronoi(voronoi_tessellation, iter_, highest_kf_error=max_fold_indices, figdir=figdir)
                plt.close()

            # Step 3: Use LOOCV to evaluate each sample within the selected fold(s)
            print("Finding worst sample locations")
            ws_start = time.time()
            if nmax_loo == 'all':
                worst_sample_locations = X[max_fold_indices]
            else:
                loocv = LeaveOneOutCrossValidation(model, scale=cv_scale)
                errors = loocv.perform_loocv(X, y, max_fold_indices, metric=cv_metric)

                # Step 4: Identify the n sample(s) with the highest LOOCV error(s)
                max_loo_indices = find_indices_of_n_largest_errors(errors, nmax_loo)
                #worst_sample_indices = max_fold_indices[max_loo_indices]
                #worst_sample_locations = X[worst_sample_indices]
                worst_sample_locations = X[max_loo_indices]

                if plot_figs and ndim == 2:
                    fig, ax = plot_voronoi(voronoi_tessellation, iter_, highest_loo_error=worst_sample_indices, figdir=figdir)
                    plt.close()

            ws_end = time.time()
            #print(f"Time to find worst sample: {ws_end - ws_start} sec, {(ws_end - ws_start)/60} min.")

        else:
            # do not perform kfold CV. New samples drawn for all X regions.
            worst_sample_locations = X

        if thin is not None:
            worst_sample_locations = worst_sample_locations[::thin, ...]
        elif random_selection is not None:
            draw_n = np.min((int(0.5 * worst_sample_locations.shape[0]), random_selection))
            random_rows = np.random.choice(worst_sample_locations.shape[0], size=draw_n, replace=False)
            worst_sample_locations = worst_sample_locations[random_rows, ...]

        print(f"Initializing voronoi/tree for batch {iter_}")
        in_start = time.time()

        if voronoi_type == 'full':
            # Initialize Voronoi tessellation
            voronoi_tessellation = VoronoiTessellation(X, bounds, boundary_points, finite_only=finite_only)

        elif voronoi_type == 'local':
            all_points = X.copy()
            tree = KDTree(all_points)

        elif voronoi_type == 'sampling':
            clip_method = 'np_clip'
            if clip_method == "boundary_hull_clip":
                boundary_hull = ConvexHull(boundary_points)
            else:
                boundary_hull = None
            all_points = X.copy()
            tree = KDTree(all_points)
            lb = np.array(bounds)[:, 0]
            ub = np.array(bounds)[:, 1]
            factor = 500
            while True:
                num_initial = factor * ndim
                initial_samples = np.random.uniform(lb, ub, size=(num_initial, ndim))
                initial_samples = handle_points_outside_bounds(boundary_hull, bounds, ndim, initial_samples, method=clip_method, centroid=None)
                initial_nn = tree.query(initial_samples, k=1)
                nn_loc = all_points[initial_nn[1]]
                intersect = np.intersect1d(nn_loc, worst_sample_locations) 
                if len(np.unique(intersect)) < len(worst_sample_locations):
                    factor += 100
                    continue
                else:
                    break

        in_end = time.time()
        #print(f"Time to initiate voronoi/tree: {in_end - in_start} sec, {(in_end - in_start)/60} min.")

        if plot_figs and ndim == 2 and voronoi_type == 'full':
            fig, ax = plot_voronoi(voronoi_tessellation, iter_, figdir=figdir)
            plt.close()

        new_points = []
        print("Finding new sample locations...")
        v_start = time.time()
        for loc_idx, location in enumerate(worst_sample_locations):
            if np.mod(loc_idx, 100) == 0:
                print(f"Drawing new sample from region index {loc_idx} of {len(worst_sample_locations)}.")

            if voronoi_type == 'full':
                # identify corresponding voronoi cell
                region_index = voronoi_tessellation.get_voronoi_region(location)[0]

                # Step 5: Select the point within this sample’s Voronoi cell that is furthest from existing samples
                region_vertices, furthest_vertex_index = voronoi_tessellation.find_furthest_vertex(region_index)
                if region_vertices is None:
                    continue
                furthest_vertex = region_vertices[furthest_vertex_index]

                # Step 6: Add the new point and update Voronoi tessellation
                if iterative_updates:
                    voronoi_tessellation.add_points(furthest_vertex)
                if plot_figs and ndim == 2 and voronoi_type == 'full':
                    fig, ax = plot_voronoi(voronoi_tessellation, iter_, updated=True,
                        added_point=furthest_vertex, location_idx=loc_idx, figdir=figdir)
                # Step 7: Update X and y
                new_points.append(furthest_vertex)

            elif voronoi_type == 'local':
                nearest_neighbors = tree.query(location, k=10*X.shape[1])
                nn_points = all_points[nearest_neighbors[1].squeeze()]
                nn_vor = VoronoiTessellation(nn_points, bounds, boundary_points, finite_only=finite_only)
                nn_region = nn_vor.get_voronoi_region(location)[0]
                try: # i think there is an issue where identical points are showing up
                    nn_vert, nn_fvi = nn_vor.find_furthest_vertex(nn_region)
                except:
                    continue

                if nn_vert is None:
                    continue
                furthest_vertex = nn_vert[nn_fvi]
                new_points.append(furthest_vertex)
                if iterative_updates:
                    all_points = np.vstack((all_points, furthest_vertex))
                    tree = KDTree(all_points)

            elif voronoi_type == 'sampling':
                try:
                    furthest_vertex = farthest_point_adpative_sampling_var(\
                        initial_samples, initial_nn, tree, boundary_hull, all_points, location, bounds,\
                        num_initial=1000*ndim, num_refined=500*ndim, iterations=2,\
                        sigma_0=0.75**ndim, alpha=1.0, k=10, nn_sigma=True, clip_method=clip_method)
                    if furthest_vertex is None:
                        continue
                    new_points.append(furthest_vertex)
                    if iterative_updates:
                        all_points = np.vstack((all_points, furthest_vertex))
                        tree = KDTree(all_points)
                        #unique_X = set(tuple(row) for row in X)
                except ZeroDivisionError:
                    continue

            if plot_figs and ndim == 2 and voronoi_type == "full":
                fig, ax = plot_voronoi(voronoi_tessellation, iter_, sample_location=location,
                    location_idx=loc_idx, figdir=figdir)
                ax.plot(region_vertices[..., 0], region_vertices[..., 1], '.', markersize=20, color='m', label='region vertices')
                plt.legend()
                plt.savefig(f'{figdir}/new_sample_location_{loc_idx}_vertices_iter_{iter_}.png')
                ax.plot(furthest_vertex[..., 0], furthest_vertex[..., 1], '.', markersize=20, color='lime', label='furthest vertex')
                plt.legend()
                plt.savefig(f'{figdir}/new_sample_location_{loc_idx}_furthest_vertex_iter_{iter_}.png')

        new_points = np.asarray(new_points)
        nnew = new_points.shape[0]
        unique_points = set(tuple(row) for row in new_points)
        new_points = np.asarray([list(row) for row in unique_points])
        nnew_unique = new_points.shape[0]
        print(f"{nnew_unique} of the {nnew} new points are unique.")

        distances = np.linalg.norm(X - new_points[:, np.newaxis, :], axis=2)
        tree = KDTree(X)
        new_points_nn = tree.query(new_points, k=1)
        nn_loc = X[new_points_nn[1]]
        nn_distances = np.linalg.norm(nn_loc - new_points, axis=1)
        print(f"New points min distance to nn: {nn_distances.min()}")
        if True:
            xdf = pd.DataFrame(X)
            xdf['label'] = 'Current'
            ndf = pd.DataFrame(new_points)
            ndf['label'] = 'New'
            data = pd.concat([xdf, ndf])
            palette = {'Current' : 'blue', 'New': 'red'}
            plt.figure()
            sns.pairplot(data, hue='label', palette=palette, markers=['o', 's'], plot_kws={'s':10})
            plt.savefig(f"{figpath}/new_and_old_points_iter_{iter_}.png")

            plt.figure()
            plt.hist(nn_distances)
            plt.savefig(f"{figpath}/new_point_distance_to_nn_iter_{iter_}.png")
            plt.close("all")

        X = np.concatenate((X_orig, new_points))
        return X


    def _find_indices_of_n_largest_kf_errors(kf, n):

        # Create a list of (key, error, sample_index) tuples
        items = [(key, value[0], value[1]) for key, value in kf.items()]

        # Sort the items based on the error in descending order
        sorted_items = sorted(items, key=lambda x: x[1], reverse=True)

        # Get the top n items
        top_n_items = sorted_items[:n]

        # Extract the arrays associated with the top n largest floats
        result_arrays = {key: array for key, _, array in top_n_items}

        return result_arrays


    def _find_indices_of_n_largest_errors(loo, n, sort=False):
        """
        Find the indices of the n largest values in an array of errors.

        Parameters:
        errors (np.ndarray or list): An array or list of error values.
        n (int): The number of largest values to find.

        Returns:
        np.ndarray: An array of indices corresponding to the n largest values.
        """

        if n < 1:
            # treat as ratio of indices to keep
            nkeep = int(n * len(loo))
        else:
            nkeep = int(n)

        # Create a list of (key, error, sample_index) tuples
        items = [(key, value[0], value[1]) for key, value in loo.items()]

        # Sort the items based on the error in descending order
        sorted_items = sorted(items, key=lambda x: x[1], reverse=True)

        # Get the top n items
        top_n_items = sorted_items[:nkeep]

        # Extract the indices associated with the top n largest floats
        indices = [item[2] for item in top_n_items]

        if False:
            # Convert the input to a NumPy array if it's not already
            errors_array = np.concatenate(errors)

            # Get the indices of the n largest values
            indices = np.argsort(errors_array)[-n:]  # Get the last n indices from the sorted array

            if sort:
                # Sort the indices to return them in ascending order
                return np.sort(indices)
            else:
                # Return the indices in the order of descending error
                return np.flip(indices)

        return np.array(indices)


    def _make_nd_grid(bounds, npts_along_dim):

        ndim = len(bounds)
        grid_pts = []
        for dim in np.arange(ndim):
            grid_pts.append(np.linspace(bounds[dim][0], bounds[dim][1], npts_along_dim))
        #grid_tuple = tuple(grid_pts)
        #coords = np.meshgrid(*grid_tuple)
        coords = np.meshgrid(*grid_pts)
        coords_ravel = [np.asarray(coords[i]).ravel() for i in np.arange(ndim)]
        return np.vstack(tuple(coords_ravel)).T


    def _generate_test_data(fun, bounds, ngrid_pts=25, grid=True, npts=None):

        if grid:
            pts = make_nd_grid(bounds, ngrid_pts)
        else:
            assert npts is not None
            test_sampler = Halton(d=len(bounds), seed=42)
            pts_unscaled = test_sampler.random(n=npts)
            pts = qmc.scale(pts_unscaled, np.array(bounds)[:, 0], np.array(bounds)[:, 1])
        y_true = fun(pts)

        return pts, y_true

    def _fit_model(model, X, y):
        model.fit(X, y)
        return model

    def _find_boundary_hull_ray_crossings(boundary_hull, U, z):
        """
        Find where a ray crosses the convex hull of the boundary.

        Parameters:
        U (np.ndarray): Ray direction.
        z (np.ndarray): Ray origin.

        Returns:
        list: List of intersection points with the convex hull.
        """

        eq = boundary_hull.equations # (nfacet, ndim + 1)
        V, b = eq[:, :-1], eq[:, -1] # normal, offset
        crossing = np.zeros(U.shape)
        for ss in range(U.shape[0]):
            denom = np.dot(V, U[ss])
            num = -(b + np.dot(V, z.squeeze()))
            alpha = num[denom!=0] / denom[denom!=0]
            crossing[ss] = np.min(alpha[alpha >0]) * U[ss] + z
        return crossing

    def _clip_points(boundary_hull, samples, centroid):
        ray_direction = samples - centroid
        norm_ray_direction = ray_direction / np.linalg.norm(ray_direction)
        new_point = find_boundary_hull_ray_crossings(boundary_hull, norm_ray_direction, centroid)
        return new_point

    def _handle_points_outside_bounds(boundary_hull, bounds, ndim, samples, method='np_clip', centroid=None):
        lb = np.array(bounds)[:, 0]
        ub = np.array(bounds)[:, 1]
        outside_mask = (samples < lb).any(axis=1) | (samples > ub).any(axis=1)

        # Get the indices of vertices that are outside the bounds
        sample_outside = samples[outside_mask]
        if sample_outside.any():
            if method == 'boundary_hull_clip':
                assert centroid is not None
                clipped_samples = clip_points(boundary_hull, sample_outside, centroid)
            if method == 'np_clip':
                clipped_samples = np.clip(sample_outside, lb, ub)
            samples[outside_mask] = clipped_samples
        return samples

    def _find_farthest_sample_from_point(samples, point):
        sample_distances_from_p_i = np.linalg.norm(samples - point, axis=1)
        farthest_idx = np.argmax(sample_distances_from_p_i)
        farthest_candidate = samples[farthest_idx]
        farthest_distance = sample_distances_from_p_i[farthest_idx]
        return farthest_candidate, farthest_distance, sample_distances_from_p_i

    def _farthest_point_adpative_sampling_var(initial_samples, initial_nn, tree, boundary_hull, P, p_i, bounds, num_initial=100, num_refined=500,
        iterations=5, sigma_0=1.0, alpha=0.5, region_index=None, runID=None, k=15, nn_sigma=False, clip_method='np_clip'):

        nsamples, ndim = P.shape
        x_farthest = None
        max_dist = 0
        nearest_neighbors = tree.query(p_i, k=ndim*3)
        nn_distances_from_p_i = nearest_neighbors[0][..., 1:]
        nn_points = P[nearest_neighbors[1][..., 1:].squeeze()]
        point_index = np.argwhere((P == p_i).all(axis=1)).squeeze()
        if nn_sigma:
            sigma_0 = nn_distances_from_p_i.var() ** 0.5
        lb = np.array(bounds)[:, 0]
        ub = np.array(bounds)[:, 1]

        # initial random sampling
        valid_samples = initial_samples[initial_nn[1] == point_index]
        random_vector = multivariate_normal(np.zeros(ndim), np.ones(ndim)*sigma_0**2).rvs(size=num_initial//len(valid_samples))
        unclipped_new_samples = np.vstack(valid_samples[:, np.newaxis, :] + random_vector)
        samples = handle_points_outside_bounds(boundary_hull, bounds, ndim, unclipped_new_samples, method=clip_method, centroid=p_i)

        for t in range(iterations):

            sigma_t = sigma_0 * np.exp(-alpha * t) # dynamic variance reduction
            sample_nn = tree.query(samples, k=1)
            valid_samples = samples[sample_nn[1] == point_index]

            if valid_samples.shape[0] > 0:
                farthest_candidate, farthest_distance, sample_distances_from_p_i =\
                    find_farthest_sample_from_point(valid_samples, p_i)
                if farthest_distance > max_dist:
                    max_dist = farthest_distance
                    x_farthest = farthest_candidate

                if t < iterations - 1:
                    # focused re-sampling near best candidates
                    if t == 0:
                        top_k = valid_samples[np.argsort(-sample_distances_from_p_i)[:k]]
                        curr_top_k = top_k.copy()
                        curr_top_k_distances = sample_distances_from_p_i[np.argsort(-sample_distances_from_p_i)[:k]]
                    elif t > 0:
                        valid_samples = np.vstack((curr_top_k, valid_samples))
                        sample_distances = np.concatenate((curr_top_k_distances, sample_distances_from_p_i))
                        top_k = valid_samples[np.argsort(-sample_distances)[:k]]
                        curr_top_k = top_k.copy()
                        curr_top_k_distances = sample_distances[np.argsort(-sample_distances)[:k]]

                    random_vector = multivariate_normal(np.zeros(ndim), np.ones(ndim)*sigma_t**2).rvs(size=num_refined//len(top_k))
                    unclipped_new_samples = np.vstack(top_k[:, np.newaxis, :] + random_vector)
                    samples = handle_points_outside_bounds(boundary_hull, bounds, ndim, unclipped_new_samples, method=clip_method, centroid= p_i)
            else:
                print(f"no valid samples at iter {t}")

        return x_farthest
    

class VoronoiTessellation:
    def __init__(self, points, bounds,
                 incremental=False, finite_only=False):
        """Initialize the VoronoiBatchSamplingStudy

        Initialize the Voronoi tessellation with given points and bounds.

        Parameters:
        points: np.ndarray
            Array of points for Voronoi tessellation.
        bounds: list of tuples
            Bounds for the region, e.g., [(xmin, xmax), (ymin, ymax)] for 2D.
        """
        from scipy.spatial import Voronoi, Delaunay, voronoi_plot_2d, ConvexHull
        import pandas as pd
        import copy
        from mpl_toolkits.mplot3d import Axes3D

        self.points = np.array(points)
        self.ndim = self.points.shape[1]
        self.bounds = bounds
        self.boundary_points = self.make_nd_grid(npts_along_dim=2)
        if not finite_only:
            self.boundary_hull = ConvexHull(self.boundary_points)
            self.boundary_hull_eq = self.boundary_hull.equations # (nfacet, ndim + 1)
            self.boundary_hull_V, self.boundary_hull_b = self.boundary_hull_eq[:, :-1], self.boundary_hull_eq[:, -1] # normal, offset
            self.bhullD = Delaunay(self.boundary_points)
        else:
            self.boundary_hull = None
            self.bhullD = None
        self.create_ghost_points()
        self._all_points = np.vstack([self.points, self._ghost_points])
        self.vor = Voronoi(self._all_points, incremental=incremental)
        self.ghost_busters()
        self.finite_only = finite_only
        self.boundary_regions = self.get_voronoi_region(self.boundary_points) # may need to update

    def make_nd_grid(self, npts_along_dim):
        grid_pts = []
        for dim in np.arange(self.ndim):
            grid_pts.append(np.linspace(self.bounds[dim][0], self.bounds[dim][1], npts_along_dim))
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
        self._ghost_points = np.vstack([self._ghost_points, boundary_centroid + centCoef * max_dist * np.eye(self.points.shape[1])])
        self._ghost_points = np.vstack([self._ghost_points, boundary_centroid - centCoef * max_dist * np.eye(self.points.shape[1])])

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
            print(f"WARNING: infinite vertice in Region {region_index}")
        
        if identify_outside_vertices:
            updated_region = self.identify_vertices_outside_bounds(region)
            if not -2 in updated_region and len(updated_region) > 0:
                region_vertices = self.vor.vertices[region]
            elif -2 in updated_region:
                if self.finite_only:
                    if max(updated_region) < 0:
                        region_vertices = None
                    else: 
                        region_vertices = np.asarray([self.vor.vertices[i] for i in updated_region if i > 0])
                else:
                    region_tuple_list = list(zip(region, updated_region))
                    region_vertices = self.replace_unbounded_vertices(updated_region, region_index, region_tuple_list)
            if region_vertices is not None:
                if not self.finite_only:
                    boundary_in_region = [i for i in np.arange(len(self.boundary_regions)) if self.boundary_regions[i][0] == region_index]
                    if boundary_in_region:
                        boundary_vertices = self.boundary_points[boundary_in_region] 
                        region_vertices = np.concatenate((region_vertices, boundary_vertices))
                        unique_vertices = set(tuple(row) for row in region_vertices)
                        return np.asarray([list(row) for row in unique_vertices])
                    else:
                        return region_vertices
            else:
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
                print(f"WARNING: infinite vertice in Region {i}")

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
                        boundary_in_region = [ii for ii in np.arange(len(self.boundary_regions)) if self.boundary_regions[ii][0] == i]
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

        Parameters:
        region (list): A list of the voronoi regions. Each list contains indices of the voronoi vertices forming each Voronoi region.

        Returns:
        list: A new list of voronoi regions. With vertices outside the boudnign region replaced with -1.
        """

        #outside = lambda lb, ub, x: (x < lb) + (x > ub)
        # Create a boolean mask for vertices outside the bounds
        region = np.array(region)
        region_vertices = self.vor.vertices[region]
        outside_mask = np.zeros(region_vertices.shape, dtype=bool)

        for col_index in range(self.ndim):
            lb, ub = self.bounds[col_index]
            #vert_outside, = np.where(outside(lb, ub, region_vertices[:, col_index]))
            outside_mask[:, col_index] |= (region_vertices[:, col_index] < lb) | (region_vertices[:, col_index] > ub)

        # Get the indices of vertices that are outside the bounds
        vert_outside = np.where(outside_mask.any(axis=1))[0]
        if len(vert_outside) > 0:
            outside_vert_index = [region[i] for i in vert_outside]
            region[vert_outside] = -2
        return region.tolist()

    def replace_unbounded_vertices(self, region, region_index, region_tuple):
        """
        Replace the infinite vertices in a Voronoi region with new vertices on the edge of the bounding box.
        ** vertices that sit outside the bounding region are considered infinite here

        Parameters:
        region (list): A list of the voronoi regions. Each list contains indices of the Voronoi vertices forming each Voronoi region.
        region_index (int): Region index

        Returns:
        list: A new list of voronoi regions with infinite vertices replaced.
        """
        region_point_index, = np.argwhere(self.vor.point_region == region_index)
        region_vertices = []

        if -2 in region:
            finite_indices = [v for v in region if v >= 0]
            #if len(finite_indices) == 0:
            #    return None
            #self.raise_if_no_finite_vertices(finite_indices, region_index)
            finite_vertices = self.vor.vertices[finite_indices]
            new_vertices = self.snip_ridge_vertices(\
                region_index, region_point_index, region_tuple)

            # Replace the infinite vertex
            region_vertices = np.concatenate((finite_vertices, new_vertices))

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
                if urv[u] == -2: # and urv[v] > 0: # only one vertice is out of bounds - snip one end to the boundary hull
                    ray_end = self.vor.vertices[rv[u]]
                    ray_origin = self.vor.vertices[rv[v]]
                    ray_direction = ray_end - ray_origin
                    norm_ray_direction = ray_direction / np.linalg.norm(ray_direction)
                    new_vertice = self.find_boundary_hull_ray_crossings(norm_ray_direction, ray_origin)
                    if region_index in self.get_voronoi_region(new_vertice)[0]:
                        if self.bhullD.find_simplex(new_vertice) >= 0:
                            new_vertices.append(new_vertice)
                    #new_vertices.append(new_vertice)
                if urv[v] == -2: # both vertices are out of bounds - snip both ends to the boundary hull
                    ray_end = self.vor.vertices[rv[v]]
                    ray_origin = self.vor.vertices[rv[u]]
                    ray_direction = ray_end - ray_origin
                    norm_ray_direction = ray_direction / np.linalg.norm(ray_direction)
                    new_vertice = self.find_boundary_hull_ray_crossings(norm_ray_direction, ray_origin)
                    if region_index in self.get_voronoi_region(new_vertice)[0]:
                        if self.bhullD.find_simplex(new_vertice) >= 0:
                            new_vertices.append(new_vertice)

            elif len(urv) > 2: #3D + Voronoi region
                nunbounded_vert = urv.count(-1) + urv.count(-2)
                if nunbounded_vert > 0 and nunbounded_vert < len(urv):

                    edges = [[rv[i], rv[(i+1) % len(rv)]] for i in range(len(rv))]
                    updated_edges = [[urv[i], urv[(i+1) % len(urv)]] for i in range(len(urv))]
                    unbounded_edges = [[i, edge] for i, edge in enumerate(updated_edges) if -2 in edge]
                    for i, ev in unbounded_edges:
                        u, v = np.argsort(ev)
                        if ev[u] == -2 and ev[v] > 0:
                            ray_end = self.vor.vertices[edges[i][u]]
                            ray_origin = self.vor.vertices[edges[i][v]]
                            ray_direction = ray_end - ray_origin
                            norm_ray_direction = ray_direction / np.linalg.norm(ray_direction)
                            new_vertice = self.find_boundary_hull_ray_crossings(norm_ray_direction, ray_origin)
                            new_vertices.append(new_vertice)

        return np.asarray(new_vertices)

    def find_boundary_hull_ray_crossings(self, U, z):
        """
        Find where a ray crosses the convex hull of the boundary.

        Parameters:
        U (np.ndarray): Ray direction.
        z (np.ndarray): Ray origin.

        Returns:
        list: List of intersection points with the convex hull.
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
        vertices = self.get_region_vertices(region_index, identify_outside_vertices=identify_outside_vertices)
        if vertices is not None:
            centroid = self.get_region_seed(region_index)
            distances = np.linalg.norm(vertices - centroid, axis=1)
            furthest_vertex_index = np.argmax(distances)
        else:
            furthest_vertex_index = None
        return vertices, furthest_vertex_index

    def recalculate_with_new_seed(self, new_seed):
        """Recalculate the Voronoi tessellation with the addition of a new seed."""
        self.points = np.vstack([self.points, new_seed])
        self.vor = Voronoi(self.points)

    def get_region_seed(self, region_index):
        """
        Given a region_index, return the seed of the Voronoi tesselation that
        belongs to the region.

        Parameters:
        region_index (integer): Region index.

        Returns:
        array: The Voronoi seed that belongs to the indexed region.
        """

        # Find the index of the point
        point_index, = np.where(self.vor.point_region == region_index)
        return np.atleast_2d(self.vor.points[point_index[0]])

    def get_voronoi_region(self, point_array):
        """
        Given an array of points, return the region of the Voronoi tesselation that the
        points belongs to. If a point lies on a ridge or vertice, multiple regions are 
        returned for that point.

        Parameters:
        point (array-like): an array of points to find the region of.

        Returns:
        list: The Voronoi region(s) that contains the point.
        """
        point_array = np.atleast_2d(point_array)
        region_index = []
        for point in point_array:
            if point in self.vor.points:
                seed_index, = np.where(np.all(self.vor.points == point, axis=1))
            else:
                seed_index = self.get_closest_seed(point)
            
            # Get the region index for the point
            region_index.append(self.vor.point_region[seed_index].tolist())
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

    def add_points(self, points):
        """ process a set of additional points""" 
        points = np.atleast_2d(points)
        try:
            # Qhull error in dim>2 with incremental=True and restart=False
            # Must set Incremental=True to use add_points(), but very slow
            # May be faster to rebuild manually

            #self.vor.add_points(points, restart=True)
            self._all_points = np.vstack((self._all_points, points))
            self.vor = Voronoi(self._all_points)
            #self.vor.updated_ridge_vertices= [inner_list[:] for inner_list in self.vor.ridge_vertices]
        except:
            if np.any(np.all(self.vor.points == points, axis=1)):
                print(f'Point {point} already a seed')
            if np.any(np.isnan(points)) or np.any(np.isinf(points)):
                raise ValueError("Input points contain NaN or Inf.")
            print("exception raised in add_points()")


    def raise_if_invalid_region_index(self, region_index):
        if region_index > len(self.vor.regions):
            raise ValueError('Invalid region index. Index must be in (0, nregions]')

    def raise_if_no_finite_vertices(self, finite_indices, region_index):
        if len(finite_indices) == 0:
            point = self.get_region_seed(region_index)
            raise ValueError(f"0 finite indices for region {region_index}, with seed {point}")

    def plot_voronoi_3d(self):

        fig, ax = plt.subplots(111, projection='3d')
        # Plot the Voronoi vertices
        ax.scatter(self.vor.vertices[:, 0], self.vor.vertices[:, 1], self.vor.vertices[:, 2], color='orange')

        # Plot the Voronoi ridges
        for ridge in vor.ridge_vertices:
            if -1 in ridge:
                continue  # Skip infinite ridges
            ax.plot3D(*zip(vor.vertices[ridge[0]], vor.vertices[ridge[1]]), color='blue')
        plt.savefig("voronoi_3d.png")


class KFoldCrossValidation:
    def __init__(self, model, n_splits=5, group_kfold=False, scale=None):
        """
        Initialize the K-Fold Cross-Validation with a given surrogate model.

        Parameters:
        model: A machine learning model that has fit and predict methods.
        n_splits: int
            The number of folds for K-Fold Cross-Validation.
        """
        self.model = model
        self.n_splits = n_splits
        self.group_kfold = group_kfold
        self.scale = scale

    def calculate_sum_abs_error(self, y_true, y_pred):
        return  np.sum(np.abs(y_true - y_pred))

    def calculate_abs_perc_error(self, y_true, y_pred):
        return np.abs((y_true - y_pred) / y_true) * 100

    def calculate_mean_abs_perc_error(self, y_true, y_pred):
        return np.mean(self.calculate_abs_perc_error(y_true, y_pred))

    def calculate_mse(self, y_true, y_pred):
        sq_error =  (y_true - y_pred) ** 2
        return np.mean(sq_error)

    def calculate_rmse(self, y_true, y_pred):
        return np.sqrt(self.calculate_mse(y_true, y_pred))

    def calculate_sum_abs_perc_error(self, y_true, y_pred):
        return np.sum(self.calculate_abs_perc_error(y_true, y_pred))

    def perform_kfold_cv(self, X, y, metric='sum_abs_error', groups=None):
        """
        Perform K-Fold Cross-Validation.

        Parameters:
        X: np.ndarray
            Feature matrix (training samples).
        y: np.ndarray
            Target values (ground truth).

        Returns:
        tuple: (index_of_max_error, max_error)
            The index of the sample with the greatest prediction error and the corresponding error value.
        """
        from sklearn.metrics import mean_squared_error
        from sklearn.model_selection import GroupKFold, KFold, cross_val_score
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        import matplotlib
        from joblib import Parallel, delayed

        X = np.atleast_2d(X)
        nsamples = X.shape[0]
        if self.group_kfold:
            assert groups is not None
            cv = GroupKFold(n_splits=self.n_splits)
            # Use joblib to parallelize the cross-validation folds
            kf_results = Parallel(n_jobs=-1)(
                delayed(self.cross_val_fold)(train_index, test_index, X, y, metric)
                for train_index, test_index in cv.split(X, y, groups)
            )
        else:
            cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=1)
            # Use joblib to parallelize the cross-validation folds
            kf_results = Parallel(n_jobs=-1)(
                delayed(self.cross_val_fold)(train_index, test_index, X, y, metric)
                for train_index, test_index in cv.split(X)
            )

        # Convert the results to a dictionary
        kf = {k_idx: result for k_idx, result in enumerate(kf_results)}
        return kf

    def cross_val_fold(self, train_index, test_index, X, y, metric):
        """Perform a single fold of cross-validation."""
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Fit the model on the training data
        self.model.fit(X_train, y_train)

        # Make predictions for the test set
        y_pred = self.model.predict(X_test)
        if self.scale == 'cbrt':
            y_pred = cbrt(y_pred)
            y_test = cbrt(y_test)
        elif self.scale == 'log':
            y_pred = np.log(y_pred)
            y_test = np.log(y_test)

        # Calculate the prediction errors for the test samples
        if metric == 'sum_abs_error':
            error = self.calculate_sum_abs_error(y_test, y_pred)
        elif metric == 'mape':
            error = self.calculate_mean_abs_perc_error(y_test, y_pred)
        elif metric == 'mse':
            error = self.calculate_mse(y_test, y_pred)
        elif metric == 'rmse':
            error = self.calculate_rmse(y_test, y_pred)
        elif metric == 'sum_abs_perc_error':
            error = self.calculate_sum_abs_perc_error(y_test, y_pred)
        else:
            print("Chosen metric for kfold cross validation not recognized. Reverting to the sum of absolute errors.")
            error = self.calculate_sum_abs_error(y_test, y_pred)

        return error, test_index

    def cbrt(y):
        return np.sign(y) * np.abs(y) ** (1/3)

    def plot_kfold(self, cv, X, y, ax, xlim_max=100):
        """
        Plots the indices for a cross-validation object.

        Parameters:
        cv: Cross-validation object
        X: Feature set
        y: Target variable
        ax: Matplotlib axis object
        xlim_max: Maximum limit for the x-axis
        """

        # Set color map for the plot
        cmap_cv = plt.cm.coolwarm
        cv_split = cv.split(X=X, y=y)

        for i_split, (train_idx, test_idx) in enumerate(cv_split):
            # Create an array of NaNs and fill in training/testing indices
            indices = np.full(len(X), np.nan)
            indices[test_idx], indices[train_idx] = 1, 0

            # Plot the training and testing indices
            ax_x = range(len(indices))
            ax_y = [i_split + 0.5] * len(indices)
            ax.scatter(ax_x, ax_y, c=indices, marker="_", 
                       lw=10, cmap=cmap_cv, vmin=-0.2, vmax=1.2)

        # Set y-ticks and labels
        y_ticks = np.arange(self.n_splits) + 0.5
        ax.set(yticks=y_ticks, yticklabels=range(self.n_splits),
               xlabel="X index", ylabel="Fold",
               ylim=[self.n_splits, -0.2], xlim=[0, xlim_max])

        # Set plot title and create legend
        ax.set_title("KFold", fontsize=14)
        legend_patches = [Patch(color=cmap_cv(0.8), label="Testing set"),
                          Patch(color=cmap_cv(0.02), label="Training set")]
        ax.legend(handles=legend_patches, loc=(1.03, 0.8))# Example usage


class LeaveOneOutCrossValidation:
    def __init__(self, model, scale=None):
        """
        Initialize the LOOCV with a given surrogate model.

        Parameters:
        model: A machine learning model that has fit and predict methods.
        """
        self.model = model
        self.scale = scale

    def calculate_sum_abs_error(self, y_true, y_pred):
        return  np.sum(np.abs(y_true - y_pred))

    def calculate_abs_perc_error(self, y_true, y_pred):
        return np.abs((y_true - y_pred) / y_true) * 100

    def calculate_mean_abs_perc_error(self, y_true, y_pred):
        return np.mean(self.calculate_abs_perc_error(y_true, y_pred))

    def calculate_mse(self, y_true, y_pred):
        sq_error =  (y_true - y_pred) ** 2
        return np.mean(sq_error)

    def calculate_rmse(self, y_true, y_pred):
        return np.sqrt(self.calculate_mse(y_true, y_pred))

    def calculate_sum_abs_perc_error(self, y_true, y_pred):
        return np.sum(self.calculate_abs_perc_error(y_true, y_pred))

    def loo_val(self, X, y, metric, i):
        # Leave one out: create training and test sets
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i, axis=0)
        X_test = X[i].reshape(1, -1)  # Reshape for a single sample
        y_test = y[i]

        # Fit the model on the training data
        self.model.fit(X_train, y_train)

        # Make a prediction for the left-out sample
        y_pred = self.model.predict(X_test)
        if self.scale == 'cbrt':
            y_pred = cbrt(y_pred)
            y_test = cbrt(y_test)
        elif self.scale == 'log':
            y_pred = np.log(y_pred)
            y_test = np.log(y_test)

        if metric == 'sum_abs_error':
            error = self.calculate_sum_abs_error(y_test, y_pred)
        elif metric == 'mape':
            error = self.calculate_mean_abs_perc_error(y_test, y_pred)
        elif metric == 'mse':
            error = self.calculate_mse(y_test, y_pred)
        elif metric == 'rmse':
            error = self.calculate_rmse(y_test, y_pred)
        elif metric == 'sum_abs_perc_error':
            error = self.calculate_sum_abs_perc_error(y_test, y_pred)
        else:
            print("Chosen metric for kfold cross validation not recognized. Reverting to the sum of absolute errors.")
            error = self.calculate_sum_abs_error(y_test, y_pred)

        return error, i


    def perform_loocv(self, X, y, indices, metric='sum_abs_error'):

        """
        Perform Leave-One-Out Cross-Validation.

        Parameters:
        X: np.ndarray
            Feature matrix (training samples).
        y: np.ndarray
            Target values (ground truth).

        Returns:
        tuple: (index_of_max_error, max_error)
            The index of the sample with the greatest prediction error and the corresponding error value.
        """

        from joblib import Parallel, delayed
        loo_results = Parallel(n_jobs=-1)(
            delayed(self.loo_val)(X, y, metric, i)
            for i in indices
        )

        loo = {loo_idx: result for loo_idx, result in enumerate(loo_results)}

        return loo