"""
This module contains pure MatCal implementations of parameter studies. 
These do not invoke external algorithm libraries. 
"""
from abc import abstractmethod
from collections import OrderedDict
import numpy as np
from scipy.stats import qmc
import traceback

from matcal.core.data import MaxAbsDataConditioner, DataCollectionStatistics
from matcal.core.logger import initialize_matcal_logger
from matcal.core.parameter_batch_evaluator import ParameterBatchEvaluator
from matcal.core.study_base import StudyBase
from matcal.core.utilities import (check_value_is_real_between_values, 
                                   check_value_is_positive_integer, 
                                   check_value_is_nonnegative_real, 
                                   check_value_is_array_like_of_reals, 
                                   check_value_is_bool
                                   )


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
                                               param, closed=True)
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
        err_msg=""
        try:
            self._batch_results = self._matcal_evaluate_parameter_sets_batch(param_sets)
        except Exception as e:
            success = False
            exit_status = -1
            err_msg = str(repr(e))
            logger.error(f"Error evaluating current parameter batch.\n{err_msg}")
        self._results._set_exit_information(success, exit_status, f"{err_msg}")
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

    def __init__(self, *parameters):
        super().__init__(*parameters)
        self.l_bounds = []
        self.u_bounds = []
        for idx, key in enumerate(self._parameter_collection):
            self.l_bounds.append(self._parameter_collection[key].get_lower_bound())
            self.u_bounds.append(self._parameter_collection[key].get_upper_bound())
        self._number_parameters = len(self._parameter_collection)
        self.HaltonSampler = qmc.Halton(d=self._number_parameters)
        self._seed = None
        self._scramble=None 
        self._number_of_samples=None
        self._skip=None
        self.set_Halton_scramble()
        self.set_number_of_samples()
        
    def set_seed(self, seed):
        """
        Set the random seed for the random generator that the study uses
        to generate the samples.
        The method should be called **before**
        :meth:`launch` to
        guarantee reproducibility.

        :param seed: Integer seed for the pseudo‑random number generator.
        :type seed: int
        """
        check_value_is_positive_integer(seed, "seed")
        self._seed = seed
    
    def set_Halton_scramble(self, scramble=True):
        """
        :param scramble: set the scramble keyword for the numpy Halton object.
        :type scramble: bool        
        """
        check_value_is_bool(scramble, "scramble")
        self._scramble=scramble

    def set_number_of_samples(self, nsamples=20, skip=None):
        """
        Set the number of samples for the study.

        :param nsamples: Number of parameter samples to generate from Halton sequence.
        :type nsamples: int
        
        :param skip: When continuing an existing design, the user may optionally skip ahead in the
                     Halton sequence by an amount determined by 'skip'.
        :type skip: int
        """
        check_value_is_positive_integer(nsamples, 'nsamples')
        self._number_of_samples = nsamples
        if skip is not None:
            check_value_is_positive_integer(skip, 'skip')
            self._skip=skip

    def launch(self):
        self._generate_samples(self._number_of_samples, self._skip)
        return super().launch()

    def _generate_samples(self, nsamples, skip):
        halton_sampler = qmc.Halton(d=self._number_parameters, 
                                       scramble=self._scramble, 
                                        seed=self._seed)
        if skip is not None:
            halton_sampler.fast_forward(skip)
        logger.debug(
            f"Halton sampler initialized with seed = {self._seed}, scramble = " +
            f"{self._scramble}, and "+
            f"skip = {skip}."
        )
        unscaled_samples = halton_sampler.random(n=nsamples)
        scaled_samples = self._scale_samples_to_bounds(unscaled_samples)
        self._populate_parameter_evaluations(scaled_samples)
        
    def _populate_parameter_evaluations(self, scaled_samples):
        param_order = self._parameter_collection.get_item_names() 
        self._parameter_sets_to_evaluate = []
        for sample in scaled_samples:
            ss = { key:sample[i] for i, key in enumerate(param_order) }
            self._add_parameter_evaluation(**ss)
        self._check_parameter_sets_populated()

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
        raise RuntimeError(f"Users cannot add parameter evaluations to a "+
                           f"{self.__class__.__name__}.")


class FiniteDifference:

    def __init__(self, center_point, relative_step_size=1.e-3, 
                 epsilon=np.sqrt(np.finfo(float).eps)):
        self._center_point = np.array(center_point, dtype=float)
        self._number_of_variables = len(self._center_point)
        self._relative_step_size = relative_step_size
        self._step_sizes = []
        for x in self._center_point:
#         dx = np.abs(x)*relative_step_size
          dx = relative_step_size * max(abs(x), 1.0)
          if dx < epsilon: 
            dx = epsilon
          self._step_sizes.append(dx)
        self._finite_difference_evaluation_points = None
        self._gradient_coefficients = None
        self._gradient_indices      = None
        self._function_values = None
        self._hessian_coefficients = None
        self._hessian_indices      = None

    def Xset_function_values(self,ys): 
        self._function_values = ys
        ndim = np.squeeze(ys).ndim
        self._function_shape = None
        if ndim > 1: 
          self._function_shape = ys[0].shape

    def set_function_values(self, ys): 
        ys = np.asarray(ys)
        self._function_values = ys
        self._function_shape = ys.shape[1:] if ys.ndim > 1 else None

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


def _estimate_parameter_covariance(residuals, sensitivities, noise_variance,
#                                  method="svd_inverse",
                                   method="svd_pinv",
                                   rcond=1e-10,
                                   regularization=0.0,
                                   project_to_psd=False,
                                   min_eigenvalue=_small):
    has_replicas = len(residuals.shape) > 1
    if has_replicas:
        Sigma_y = _get_residual_covariance(residuals)
        Sigma_guess = _solve_for_parameter_covariance(
            Sigma_y,
            sensitivities,
            noise_variance,
            method=method,
            rcond=rcond,
            regularization=regularization,
            project_to_psd=project_to_psd,
            min_eigenvalue=min_eigenvalue
        )
    else:
        raise RuntimeError(
            "The LaplaceStudy has no repeats. Repeat data are needed "
            "for the study."
        )

    return Sigma_guess


def _get_residual_covariance(residuals):
    Sigma_y = np.cov(residuals.T) 
    return np.atleast_2d(Sigma_y)


def _solve_for_parameter_covariance(output_covariance, 
                                    residual_sensitivities, 
                                    noise_variance=0.0,
#                                   method="svd_inverse",
#                                   method="svd_pinv",
                                    method="least_squares",
                                    rcond=1e-10,
                                    regularization=0.0,
                                    project_to_psd=False,
                                    min_eigenvalue=_small):
    """
    Estimate parameter covariance from an output covariance and residual
    sensitivities.

    The supported methods are:

    ``"svd_inverse"``
        Original embedded-Laplace initial estimator:

        Sigma_theta = (U.T A)^-1 D (U.T A)^-T

        using a direct inverse.

    ``"svd_pinv"``
        Same eigenspace estimator, but using a pseudo-inverse for U.T A.

    ``"least_squares"``
        Solve the linearized covariance equation

        Sigma_y - sigma^2 I ~= A Sigma_theta A.T

        directly in a least-squares sense, enforcing symmetry of Sigma_theta.

    :param output_covariance: observed/sample residual covariance, Sigma_y.
    :type output_covariance: np.ndarray

    :param residual_sensitivities: residual sensitivity matrix, A, with shape
        n_outputs x n_parameters.
    :type residual_sensitivities: np.ndarray

    :param noise_variance: measurement noise variance, sigma^2.
    :type noise_variance: float

    :param method: covariance estimator method. One of
        "svd_inverse", "svd_pinv", or "least_squares".
    :type method: str

    :param rcond: cutoff for pseudo-inverse or least-squares solve.
    :type rcond: float

    :param regularization: optional Tikhonov regularization for the least-squares
        method. This regularizes the upper-triangular entries of Sigma_theta.
    :type regularization: float

    :param project_to_psd: if True, symmetrize and project the estimated
        parameter covariance to positive semidefinite.
    :type project_to_psd: bool

    :param min_eigenvalue: minimum eigenvalue used in PSD projection.
    :type min_eigenvalue: float

    :return: estimated parameter covariance matrix.
    :rtype: np.ndarray
    """
    output_covariance = np.asarray(output_covariance, dtype=float)
    residual_sensitivities = np.asarray(residual_sensitivities, dtype=float)

    mineval, maxeval = _check_covariance(output_covariance)
    if np.abs(mineval) < _small:
        logger.warning("Residual covariance is nearly singular or not positive definite.")

    n_y = output_covariance.shape[0]

    # Important: do not mutate the caller's covariance matrix.
    shifted_output_covariance = (
        output_covariance - np.eye(n_y) * noise_variance
    )
    shifted_output_covariance = _symmetrize_matrix(shifted_output_covariance)

    method = method.lower()
    if method in ["svd_inverse", "svd_pinv"]:
        sigma_theta = _solve_for_parameter_covariance_svd_subspace(
            shifted_output_covariance,
            residual_sensitivities,
            method=method,
            rcond=rcond
        )
    elif method in ["least_squares", "ls"]:
        sigma_theta = _solve_for_parameter_covariance_least_squares(
            shifted_output_covariance,
            residual_sensitivities,
            rcond=rcond,
            regularization=regularization
        )
    else:
        raise ValueError(
            "Unknown parameter covariance estimation method "
            f"'{method}'. Valid options are 'svd_inverse', "
            "'svd_pinv', and 'least_squares'."
        )

    sigma_theta = _symmetrize_matrix(sigma_theta)

    if project_to_psd:
        sigma_theta = _project_symmetric_matrix_to_psd(
            sigma_theta,
            min_eigenvalue=min_eigenvalue
        )

    return sigma_theta


def _solve_for_parameter_covariance_svd_subspace(shifted_output_covariance,
                                                 residual_sensitivities,
#                                                method="svd_inverse",
                                                 method="svd_pinv",
                                                 rcond=1e-10):
    """
    Original embedded-Laplace eigenspace estimator, with either direct inverse
    or pseudo-inverse for U.T A.
    """
    n_p = residual_sensitivities.shape[1]

    U, d, Vt = np.linalg.svd(shifted_output_covariance)

    if len(d) < n_p:
        raise ValueError(
            "LaplaceStudy under determined. The output covariance has fewer "
            "singular values than there are parameters."
        )

    # The first n_p singular vectors/eigenvectors are assumed to span the
    # parameter-induced covariance subspace.
    Ubar = U[:, :n_p]
    Dbar = np.diag(d[:n_p])
    B = Ubar.T @ residual_sensitivities

    logger.debug(
        "Initial covariance SVD-subspace diagnostics:\n"
        f"  singular values used = {d[:n_p]}\n"
        f"  cond(A) = {np.linalg.cond(residual_sensitivities)}\n"
        f"  cond(U.T @ A) = {np.linalg.cond(B)}"
    )

    if method == "svd_inverse":
        # Original behavior, but fix the off-by-one rank check.
        if (d[:n_p] < _small).any():
            raise ValueError(
                "LaplaceStudy under determined. System may be under determined."
            )
        Binv = np.linalg.inv(B)
    elif method == "svd_pinv":
        Binv = np.linalg.pinv(B, rcond=rcond)
    else:
        raise ValueError(f"Unsupported SVD covariance method '{method}'.")

    sigma_theta = Binv @ Dbar @ Binv.T
    return sigma_theta


def _solve_for_parameter_covariance_least_squares(shifted_output_covariance,
                                                  residual_sensitivities,
                                                  rcond=1e-10,
                                                  regularization=0.0):
    """
    Solve

        C ~= A Sigma_theta A.T

    for symmetric Sigma_theta using least squares, where

        C = Sigma_y - sigma^2 I.

    The unknowns are the upper-triangular entries of Sigma_theta.

    This avoids relying on the dominant eigenspace of C and can be more stable
    when the sample covariance eigenspace is noisy.
    """
    C = np.asarray(shifted_output_covariance, dtype=float)
    A = np.asarray(residual_sensitivities, dtype=float)

    C = _symmetrize_matrix(C)

    n_y, n_p = A.shape

    param_tri_rows, param_tri_cols = np.triu_indices(n_p)
    output_tri_rows, output_tri_cols = np.triu_indices(n_y)

    n_unknowns = len(param_tri_rows)
    n_equations = len(output_tri_rows)

    design_matrix = np.empty((n_equations, n_unknowns), dtype=float)

    for col_index, (p_i, p_j) in enumerate(zip(param_tri_rows, param_tri_cols)):
        # Contribution of Sigma_theta[p_i, p_j].
        #
        # If p_i == p_j:
        #   C_ab contribution = A[a, p_i] A[b, p_i]
        #
        # If p_i != p_j and Sigma_theta is symmetric:
        #   C_ab contribution =
        #       A[a, p_i] A[b, p_j] + A[a, p_j] A[b, p_i]
        basis_pushforward = np.outer(A[:, p_i], A[:, p_j])
        if p_i != p_j:
            basis_pushforward += np.outer(A[:, p_j], A[:, p_i])

        design_matrix[:, col_index] = basis_pushforward[
            output_tri_rows, output_tri_cols
        ]

    rhs = C[output_tri_rows, output_tri_cols]

    if regularization is not None and regularization > 0.0:
        sqrt_reg = np.sqrt(regularization)
        design_matrix = np.vstack(
            [design_matrix, sqrt_reg * np.eye(n_unknowns)]
        )
        rhs = np.concatenate([rhs, np.zeros(n_unknowns)])

    coeffs, residuals, rank, singular_values = np.linalg.lstsq(
        design_matrix, rhs, rcond=rcond
    )

    logger.debug(
        "Initial covariance least-squares diagnostics:\n"
        f"  n_outputs = {n_y}\n"
        f"  n_parameters = {n_p}\n"
        f"  n_equations = {n_equations}\n"
        f"  n_unknowns = {n_unknowns}\n"
        f"  rank(design_matrix) = {rank}\n"
        f"  singular values = {singular_values}\n"
        f"  cond(design_matrix) = {np.linalg.cond(design_matrix)}"
    )

    sigma_theta = np.zeros((n_p, n_p), dtype=float)
    sigma_theta[param_tri_rows, param_tri_cols] = coeffs
    sigma_theta[param_tri_cols, param_tri_rows] = coeffs

    return _symmetrize_matrix(sigma_theta)


def _symmetrize_matrix(matrix):
    matrix = np.asarray(matrix, dtype=float)
    return 0.5 * (matrix + matrix.T)


def _project_symmetric_matrix_to_psd(matrix, min_eigenvalue=0.0):
    """
    Project a symmetric matrix to the positive-semidefinite cone by clipping
    eigenvalues.

    This is useful for forming a numerically safe initial covariance estimate
    for the subsequent fitted-posterior optimization.
    """
    matrix = _symmetrize_matrix(matrix)
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.maximum(eigvals, min_eigenvalue)
    projected = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return _symmetrize_matrix(projected)


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


def _log_posterior_predictive(theta, residual_sensitivities, residuals, noise_variance):
    if noise_variance == 0.0:
        noise_variance = _small
    variances, correlation_coefficients = _from_theta(theta, residual_sensitivities.shape[1])
    Sigma_y = _pushed_forward_variances(variances,correlation_coefficients,residual_sensitivities)
    Sigma_y = Sigma_y + noise_variance*np.eye(Sigma_y.shape[0])
    
    print(f"{Sigma_y}")
    with np.errstate(invalid='raise'):
        try:
           sign, logdet =  np.linalg.slogdet(Sigma_y)
           print(f"LOGDET {sign=}",flush=True)
           print(f"LOGDET {logdet=}",flush=True)
           logdetSigma_y = sign*logdet
        except FloatingPointError as e:
           print(f'Caught NumPy exception: {e}')
           traceback.print_exc()
    diagSig_y = np.diag(Sig_y)
    invSig= np.diag(1.0/diagSig_y)
#   invSigma = np.linalg.solve(Sigma_y, np.eye(Sigma_y.shape[0]))
    mse = np.einsum("ki,ij,kj",residuals,invSigma,residuals) 
    n_repeats = residuals.shape[0]
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

        self._initial_covariance_method = "least_squares"
        self._initial_covariance_rcond = 1e-10
        self._initial_covariance_regularization = 0.0
        self._initial_covariance_project_to_psd = False
        self._initial_covariance_min_eigenvalue = _small

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
        check_value_is_real_between_values(step_size, 0, 1, "step_size")
        self._step_size=step_size
        if self._finite_difference is not None:
            self._setup_finite_difference()

    def set_initial_covariance_estimator(self,
#                                        method="svd_inverse",
                                         method="svd_pinv",
                                         rcond=1e-10,
                                         regularization=0.0,
                                         project_to_psd=False,
                                         min_eigenvalue=_small):
        """
        Select the initial covariance estimator used by LaplaceStudy.

        Valid methods are:

        ``"svd_inverse"``
            Original eigenspace method using a direct inverse.

        ``"svd_pinv"``
            Eigenspace method using a pseudo-inverse for improved robustness
            when U.T @ A is ill-conditioned.

        ``"least_squares"``
            Direct least-squares solve of

                Sigma_y - sigma^2 I ~= A Sigma_theta A.T

            using the symmetric upper-triangular entries of Sigma_theta as
            unknowns.

        :param method: covariance estimator method.
        :type method: str

        :param rcond: cutoff used in pseudo-inverse or least-squares solve.
        :type rcond: float

        :param regularization: Tikhonov regularization used only by the
            least-squares method.
        :type regularization: float

        :param project_to_psd: project the result to positive semidefinite.
        :type project_to_psd: bool

        :param min_eigenvalue: minimum eigenvalue for PSD projection.
        :type min_eigenvalue: float
        """
        valid_methods = ["svd_inverse", "svd_pinv", "least_squares", "ls"]
        method = method.lower()

        if method not in valid_methods:
            raise ValueError(
                f"Unknown initial covariance estimator '{method}'. "
                "Valid options are 'svd_inverse', 'svd_pinv', and "
                "'least_squares'."
            )

        check_value_is_nonnegative_real(rcond, "rcond")
        check_value_is_nonnegative_real(regularization, "regularization")
        check_value_is_bool(project_to_psd, "project_to_psd")
        check_value_is_nonnegative_real(min_eigenvalue, "min_eigenvalue")

        if method == "ls":
            method = "least_squares"

        self._initial_covariance_method = method
        self._initial_covariance_rcond = rcond
        self._initial_covariance_regularization = regularization
        self._initial_covariance_project_to_psd = project_to_psd
        self._initial_covariance_min_eigenvalue = min_eigenvalue

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

        :param noise_estimate: value for the noise estimate, must be non-negative
        :type noise_estimate: float
        """
        check_value_is_nonnegative_real(noise_estimate, "noise_estimate")
        self._noise_variance=noise_estimate**2

    def set_calibrate_covariance(self, calibrate_covariance=True):
        """
        By default, the laplace study will attempt to improve the 
        covariance through a calibraiton. Optionally, turn this off or back on.

        :param calibrate_covariance: flag to turn the covariance calibration process off or on
        :type calibrate_covariance: bool
        """
        self._calibrate_covariance = calibrate_covariance

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
        output.update({"residuals":center_resids})
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
        np.save("AA.npy",residual_sensitivities)
## HERE
        # grab first one from repeats, this is the most populated and since this is 
        # the derivative of the residuals where the third index is the repeat #, 
        # the first one is good enough for derivative of the model w.r.t. the parameters. 
        residual_sensitivities = residual_sensitivities.T[0, :, :]
        np.save("A.npy",residual_sensititivities)
        return residual_sensitivities

    def _calculate_covariance(self, center_resids, residual_sensitivities):
        estimated_covariance = _estimate_parameter_covariance(
            center_resids,
            residual_sensitivities,
            self._noise_variance,
            method=self._initial_covariance_method,
            rcond=self._initial_covariance_rcond,
            regularization=self._initial_covariance_regularization,
            project_to_psd=self._initial_covariance_project_to_psd,
            min_eigenvalue=self._initial_covariance_min_eigenvalue
        )
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
    check_value_is_positive_integer(nsamples, "nsamples")
    check_value_is_array_like_of_reals(mean, "mean")
    check_value_is_array_like_of_reals(covariance_matrix, "covariance_matrix")
    if seed is not None:
        check_value_is_positive_integer(seed, "seed")
    _check_sample_covariance_mat_inputs(mean, covariance_matrix, param_names)
    samples = _get_multivariate_normal_samples(mean, covariance_matrix, nsamples, seed)
    samples_dict = _create_samples_dict_from_samples_array(samples, param_names)   
    
    return samples_dict

def _check_sample_covariance_mat_inputs(mean, covariance_matrix, param_names):
    if (len(mean)!=covariance_matrix.shape[0] or
       covariance_matrix.shape[0]!=covariance_matrix.shape[1]):
       raise ValueError("The mean and covariance matrix passed to \"sample_covariance_matrix\" "
                        "are invalid. Their sizes must match appropriately.")
   
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


