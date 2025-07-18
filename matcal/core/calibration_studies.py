
from abc import abstractmethod
from collections import OrderedDict
import numpy as np

from matcal.core.best_material_file_writer import MatcalFileWriterFactory
from matcal.core.logger import initialize_matcal_logger
from matcal.core.study_base import StudyBase
from matcal.core.utilities import (check_value_is_nonempty_str, 
                                   check_item_is_correct_type, 
                                   check_value_is_positive_real)

logger = initialize_matcal_logger(__name__)

    
class _ScipyCalibrationStudyBase(StudyBase):
    _best_material_filename = "best_material.inc"

    @property
    @abstractmethod
    def _algorithm_options(self):
        """"""

    @property
    @abstractmethod
    def _default_method(self):
        """"""

    def __init__(self, *parameters, method=None, **kwargs):
        """

        :param parameters: The parameters of interest for the study.
        :type parameters: list(:class:`~matcal.core.parameters.Parameter`) or
            :class:`~matcal.core.parameters.ParameterCollection`

        :param method: specify a specific method that is valid 
            for the Scipy `optimize` function used by the study
        :type method: str

        :param kwargs: pass valid keyword arguments for the chosen method. 
            The 'bounds' keyword argument is set 
            by MatCal and cannot be used.
        :type kwargs: dict(str, float or str or dict(str, float or str)) 
        """
        super().__init__(*parameters)
        if method is None:
            method = self._default_method
        check_value_is_nonempty_str(method, 'method', f'{self.study_class}.__init__')
        method = method.lower()
        self._check_valid_method(method)
        self._method = method
        self._current_gradient_value = None
        self._current_hessian_value = None
        self._step_size = None
        self.set_step_size()
        self._three_point_finite_difference = False
        self._check_kwargs(kwargs)
        self._kwargs = kwargs
        self._use_matcal_jac = False
        self._use_matcal_hess = False

    def _check_valid_method(self, method):
        if method not in self._algorithm_options.keys():
            raise ValueError(f"The method \'{method}\' "
                             f" is an invalid method for the {self.study_class}.")

    def _check_kwargs(self, kwargs):
        if "bounds" in kwargs:
            raise ValueError("The keyword argument 'bounds' is set by MatCal and is not" 
                             f" valid for the {self.study_class}.")
        if self._method_options.hessian == False and "hess" in kwargs:
            raise ValueError(f"Scipy optimize method \"{self._method}\" does not support "
                             f" the \"hess\" keyword argument")
        if self._method_options.gradient == False and "jac" in kwargs:
            raise ValueError(f"Scipy optimize method \"{self._method}\" does not support "
                             f"the 'jac' keyword argument")

    @property
    def _method_options(self):
        return self._algorithm_options[self._method]

    @property
    def _needs_finite_difference_gradient(self):
        meth_otps = self._method_options
        return (meth_otps.gradient == _AlgorithmOptions.finite_difference 
                and self._use_matcal_jac)
    
    @property
    def _needs_finite_difference_hessian(self):
        meth_otps = self._method_options
        return (meth_otps.hessian == _AlgorithmOptions.finite_difference and 
                self._use_matcal_hess)
 
    @property
    def _supports_bounds(self):
        meth_opts = self._method_options
        return meth_opts.bounds

    def use_three_point_finite_difference(self, use_three_point_finite_difference=True):
        """
        This method sets the finite difference stencil for gradients to a three point
        finite difference scheme.

        .. note::
            This only affects the gradients. Only two point finite 
            difference hessians are available. 
            However, if the method requires a finite difference hessian and gradient,
            a three point gradient is automatically used.

        :param use_three_point_finite_difference: an optional boolean that can 
            be passed as False to turn of using a three point finite difference 
            stencil for the gradient. By default it is True, so that this method 
            turns on the three point finite difference for the gradient.
        :type use_three_point_finite_difference: bool
        """
        check_item_is_correct_type(use_three_point_finite_difference, 
                                   bool, 
                                   self.study_class+".use_three_point_finite_difference", 
                                   "use_three_point_finite_difference")
        self._three_point_finite_difference = use_three_point_finite_difference

    def restart(self):
        """
        Restarts not supported with Scipy studies.
        """
        raise NotImplementedError

    def set_step_size(self, step_size=5e-5):
        """
        When a MatCal calculated finite difference gradient or hessian is used for 
        a method, this will set the finite difference relative step size. 

        .. warning:: 
            If using Scipy finite difference methods, use the appropriate keyword 
            argument for the method in the study ``__init__`` to set the step size, 
            not this method.
        
        :param step_size: the relative step size desired for finite difference 
            gradients and hessians.
        :type step_size: float
        """
        check_value_is_positive_real(step_size, "step_size", 
                                     f"{self.study_class}.set_step_size")

        self._step_size = step_size

    def _run_study(self):
        x0 = np.array(list(self._parameter_collection.get_current_value_dict().values()))
        jac = self._determine_jacobian_argument()
        self._kwargs = self._update_kwargs_with_hessian_argument(self._kwargs)
        scipy_results = self._scipy_function(self._matcal_evaluate_parameter_sets_batch, 
                          x0, method=self._method, 
                          jac=jac, 
                          bounds=self._get_bounds(), **self._kwargs)
        param_names = list(self._parameter_collection.get_current_value_dict().keys())
        parameter_results = _package_calibration_results(OrderedDict(zip(param_names, scipy_results.x)))
        self._results._set_outcome(parameter_results)
        self._results._initialize_exit_status(scipy_results.status, scipy_results.message)

        return self._results
    
    def _determine_jacobian_argument(self):
        jac = None
        if 'jac' in self._kwargs:
            jac = self._kwargs.pop('jac')
        elif self._method_options.gradient == _AlgorithmOptions.finite_difference:
            jac = self._get_current_gradient_value
            self._use_matcal_jac = True
        return jac

    def _update_kwargs_with_hessian_argument(self, kwargs):
        hess = self._method_options.hessian
        if "hess" in kwargs:
            return kwargs

        if hess == _AlgorithmOptions.finite_difference:
            hess = self._get_current_hessian_value
            self._use_matcal_hess = True

        if hess is not None and hess is not False:
            kwargs.update({"hess":hess})
        return kwargs

    def _matcal_evaluate_parameter_sets_batch(self, parameter_set):
        parameter_sets = [parameter_set]
        if self._needs_finite_difference_hessian or self._needs_finite_difference_gradient:
            finite_difference, results = self._evaluate_finite_difference(parameter_set)
            self._current_gradient_value = finite_difference.gradient()
            if self._needs_finite_difference_hessian:
                self._current_hessian_value = finite_difference.hessian()
            return results[0]
        else:
            results = super()._matcal_evaluate_parameter_sets_batch(parameter_sets)
            return results[0]

    def _evaluate_finite_difference(self, parameter_set):
        finite_difference, finite_diff_points =  self._prepare_finite_difference(parameter_set) 
        results = super()._matcal_evaluate_parameter_sets_batch(finite_diff_points, True)
        finite_difference.set_function_values(results)
        return finite_difference,results

    def _get_current_gradient_value(self, x):
        try:
            return self._current_gradient_value.T
        except AttributeError:
            return self._current_gradient_value
        
    def _get_current_hessian_value(self, x):
        return self._current_hessian_value

    def _prepare_finite_difference(self, center_point):
        from matcal.core.parameter_studies import FiniteDifference
        finite_diff = FiniteDifference(center_point, 
                                        relative_step_size=self._step_size)
        if self._needs_finite_difference_hessian:
            finite_diff_pts = finite_diff.compute_hessian_evaluation_points()
        else:
            three_point_finite_diff = self._three_point_finite_difference  
            finite_diff_pts = finite_diff.compute_gradient_evaluation_points(three_point_finite_diff)
        return finite_diff, finite_diff_pts

    def _format_parameter_batch_eval_results(self, batch_raw_objectives, 
                                             flattened_batch_results, 
                                             total_objs, parameter_sets, batch_qois):
        combined_objs, combined_resids, eval_dirs = flattened_batch_results
        return_values = None
        if self._needs_residuals:
            logger.debug(" Scipy method needs residual\n")
            return_values = combined_resids
        else:
            logger.debug(" Scipy method needs objective\n")
            return_values = np.array(list(total_objs.values()))
        return return_values

    def _format_parameters(self, parameter_set):
        param_names = self._parameter_collection.get_item_names()
        param_dict = OrderedDict()
        for idx, param_name in enumerate(param_names):
            param_dict[param_name] = parameter_set[idx]
        return param_dict

    def _study_specific_postprocessing(self):
        self._make_best_material_file()

    def _make_best_material_file(self):
        for eval_set in self._evaluation_sets.values():
            self._create_and_write_best_material_file(eval_set.model)

    def _create_and_write_best_material_file(self, model):
        file_writer = MatcalFileWriterFactory.create(model.executable, self._results.outcome)
        filename = self._make_best_filename(model)
        file_writer.write(filename)        

    def _make_best_filename(self, model):
        return "Best_Material_{}.inc".format(model.name)

    def launch(self):
        """
        Scipy calibration studies return calibration information in an 
        ``OptimizeResult`` object. This includes 
        the best fit parameter set, the final objective, and the Jacobian and/or Hessian 
        of the objective if available. It also includes other useful information such as  
        messages related to the success or failure of the chosen method, number of evaluations,
        and number of method iterations. For more information on what is returned in 
        an ``OptimizeResult`` object for a given method see the Scipy documentation.
        """
        return super().launch()


def _package_calibration_results(best_dict):
    out = OrderedDict()
    for name, value in best_dict.items():
        best_name = f"best:{name}"
        out[best_name] = value
        best_dict[name] = value
    return out


class _AlgorithmOptions:
    finite_difference = "finite_difference"
    central_finite_difference = "central_finite_difference"
    def __init__(self, grad, hessian, bounds, constraints):
        self._grad = grad
        self._hess = hessian
        self._bounds = bounds
        self._constraints = constraints
    @property
    def gradient(self):
        return self._grad
    @property
    def hessian(self):
        return self._hess
    @property
    def bounds(self):
        return self._bounds
    @property
    def constraints(self):
        return self._constraints


class _ScipyMinimizeAlgorithms():
    methods = OrderedDict()
    # unconstrained methods - trust krylov not supported - 
    #  won't pass simple single parameter production calibration test.
    methods["cg"] = _AlgorithmOptions(_AlgorithmOptions.finite_difference, 
                                     False, False, False)
    methods["bfgs"] = _AlgorithmOptions(_AlgorithmOptions.finite_difference, 
                                       False, False, False)
    methods["newton-cg"] = _AlgorithmOptions(_AlgorithmOptions.finite_difference, 
                                            _AlgorithmOptions.finite_difference, 
                                            False, False)
    methods["trust-ncg"] = _AlgorithmOptions(_AlgorithmOptions.finite_difference,
                                            _AlgorithmOptions.finite_difference, 
                                            False, False)
    methods["dogleg"] = _AlgorithmOptions(_AlgorithmOptions.finite_difference, 
                                         _AlgorithmOptions.finite_difference, 
                                         False, False)
    methods["trust-exact"] = _AlgorithmOptions(_AlgorithmOptions.finite_difference, 
                                              _AlgorithmOptions.finite_difference, 
                                              False, False)

    #bound constrained methods
    #Note: nelder-mead fails if starts or hits bounds
    methods["nelder-mead"] = _AlgorithmOptions(False, False, True, False) 
    methods["l-bfgs-b"] = _AlgorithmOptions(_AlgorithmOptions.finite_difference,
                                            False, True, False)
    methods["powell"] = _AlgorithmOptions(False, False, True, False)
    methods["tnc"] =  _AlgorithmOptions(_AlgorithmOptions.finite_difference,
                                        False, True, False)

    #bound constrained and linear/nonlinear constraints
    methods["cobyla"] = _AlgorithmOptions(False, False, True, True)
    methods["slsqp"] = _AlgorithmOptions(_AlgorithmOptions.finite_difference, 
                                        False, True, True)
    methods["trust-constr"] = _AlgorithmOptions(_AlgorithmOptions.finite_difference, 
                                               None, True, True)


class ScipyMinimizeStudy(_ScipyCalibrationStudyBase):
    """
    This study class is the MatCal interface to the Scipy ``minimize`` function. 
    It can be used to perform local calibrations to objective
    functions that are generally smooth and convex. It has access to both gradient 
    based methods and gradient free methods. We support all Scipy ``minimize`` methods
    except for the ``trust-krylov`` method. For methods that require Hessians 
    and/or gradients, we use an internal finite difference algorithm so that we can 
    take advantage of parallelism for expensive objective function evaluations. However, 
    if desired, the ``jac`` and ``hess`` keyword arguments can be used to override using
    the MatCal finite difference algorithm and use any valid Scipy option.
    
    .. note::
        MatCal's finite difference steps do not currently adhere to bounds or constraints.

    We default to the Scipy ``minimize`` ``l-bfgs-b`` method that is a gradient 
    method using only finite difference gradients (no Hessian) and enforces upper and lower bounds.
    """
    study_class = "ScipyMinimizeStudy"
    _algorithm_options = _ScipyMinimizeAlgorithms.methods
    _default_method = 'l-bfgs-b'

    @property
    def _needs_residuals(self):
        return False

    @property    
    def _scipy_function(self):
        from scipy.optimize import minimize
        return minimize

    def _get_bounds(self):
        if self._supports_bounds:
            bounds = []
            for param_name in self._parameter_collection:
                param = self._parameter_collection[param_name]
                param_bound = [param.get_lower_bound(), 
                            param.get_upper_bound()]
                bounds.append(param_bound)
        else:
            bounds = None
        return bounds       


class _ScipyLeastSquaresAlgorithms():
    methods = OrderedDict()
    methods["trf"] = _AlgorithmOptions(_AlgorithmOptions.finite_difference, 
                                     False, True, False)
    methods["lm"] = _AlgorithmOptions(_AlgorithmOptions.finite_difference, 
                                     False, False, False)
    methods["dogbox"] = _AlgorithmOptions(_AlgorithmOptions.finite_difference, 
                                     False, True, False)


class ScipyLeastSquaresStudy(_ScipyCalibrationStudyBase):
    """
    This study class is the MatCal interface to the Scipy ``least_squares`` function. 
    It can be used to perform local calibrations to objective
    functions that are smooth and convex. We support all Scipy ``least_squares`` methods. 
    All methods require calculation of the Jacobian, and 
    we use an internal finite difference algorithm so that we can 
    take advantage of parallelism for expensive objective function evaluations. 
    
    .. note::
        MatCal's finite difference steps do not currently adhere to bounds or constraints.

    We default to the Scipy ``least_squares`` ``trf`` method that 
    enforces upper and lower bounds.
    """
    _default_method = 'trf'
    _algorithm_options = _ScipyLeastSquaresAlgorithms.methods
    study_class = "ScipyLeastSquaresStudy"
    
    @property
    def _needs_residuals(self):
        return True

    @property    
    def _scipy_function(self):
        from scipy.optimize import least_squares
        return least_squares
    
    def _get_bounds(self):
        if self._supports_bounds:
            upper_bounds = []
            lower_bounds = []
            for param_name in self._parameter_collection:
                param = self._parameter_collection[param_name]
                lower_bounds.append(param.get_lower_bound())
                upper_bounds.append(param.get_upper_bound())
            return lower_bounds, upper_bounds
        else:
            lower_bounds = -np.inf
            upper_bounds = np.inf
        return np.array(lower_bounds), np.array(upper_bounds)
