"""
This module contains MatCal's interface to Dakota's local calibration methods.
"""
from numbers import Real

from matcal.core.logger import initialize_matcal_logger
from matcal.core.utilities import (check_item_is_correct_type, 
                                   check_value_is_positive_integer, 
                                   check_value_is_real_between_values)

from matcal.dakota.dakota_studies import DakotaCalibrationStudyBase
from matcal.dakota.input_file_writer import (DakotaCalibrationFile, GeneralGradientMethodType, 
                                             NongradientResponseBlock, DakMethodKeys, 
                                             dakota_response_identifier, GradientResponseBlock, 
                                             LeastSquaresResponseBlock, NumericalGradientBlock, 
                                             DakGradientKeys, GeneralNongradientMethodType, 
                                             method_type_defaults_identifier)

logger = initialize_matcal_logger(__name__)


class _LeastSquaresMethodKeys():
    nl2sol = "nl2sol"
    optpp_g_newton = "optpp_g_newton"
    nlssol_sqp = "nlssol_sqp"


dakota_response_identifier.register(_LeastSquaresMethodKeys.nl2sol, 
                                    LeastSquaresResponseBlock)
dakota_response_identifier.register(_LeastSquaresMethodKeys.optpp_g_newton, 
                                    LeastSquaresResponseBlock)
dakota_response_identifier.register(_LeastSquaresMethodKeys.nlssol_sqp, 
                                    LeastSquaresResponseBlock)


class _GradientMethodKeys():
    npsol_sqp = "npsol_sqp"
    dot_mmfd = "dot_mmfd"
    dot_slp = "dot_slp"
    dot_sqp = "dot_sqp"
    conmin_mfd = "conmin_mfd"
    optpp_q_newton = "optpp_q_newton"
    optpp_g_newton = "optpp_g_newton"
    optpp_fd_newton = "optpp_fd_newton"


dakota_response_identifier.register(_GradientMethodKeys.npsol_sqp, 
                                    GradientResponseBlock)
dakota_response_identifier.register(_GradientMethodKeys.dot_mmfd, 
                                    GradientResponseBlock)
dakota_response_identifier.register(_GradientMethodKeys.dot_slp, 
                                    GradientResponseBlock)
dakota_response_identifier.register(_GradientMethodKeys.dot_sqp, 
                                    GradientResponseBlock)
dakota_response_identifier.register(_GradientMethodKeys.conmin_mfd, 
                                    GradientResponseBlock)
dakota_response_identifier.register(_GradientMethodKeys.optpp_q_newton, 
                                    GradientResponseBlock)
dakota_response_identifier.register(_GradientMethodKeys.optpp_fd_newton, 
                                    GradientResponseBlock)


class _GradientDakotaFile(DakotaCalibrationFile):
    _method_class = GeneralGradientMethodType

    least_squares_methods = [_LeastSquaresMethodKeys.nl2sol, 
                             _LeastSquaresMethodKeys.optpp_g_newton,
                             _LeastSquaresMethodKeys.nlssol_sqp]
    
    gradient_methods = [_GradientMethodKeys.npsol_sqp,
                        _GradientMethodKeys.dot_mmfd,
                        _GradientMethodKeys.dot_slp, 
                        _GradientMethodKeys.dot_sqp,
                        _GradientMethodKeys.conmin_mfd, 
                        _GradientMethodKeys.optpp_q_newton,
                        _GradientMethodKeys.optpp_fd_newton]

    valid_methods = least_squares_methods+gradient_methods

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_method(_LeastSquaresMethodKeys.nl2sol)
    
    def set_method(self, method_name):
        """
        Sets the Dakota gradient based method to be used by the study. The default method is a least squares method 
        "nl2sol". Other least squares methods include "optpp_g_newton" and "nlssol_sqp". 

        Optionally, gradient based methods that operate on objective functions can be chosen which is recommended 
        for problems with large numbers of residuals such as some full-field interpolation objective. 
        These methods include "npsol_sqp", "dot_sqp" , "optpp_q_newton", and "conmin_mfd". 

        Some gradient methods that use an objective that are available but not recommended included 
        "dot_slp", "dot_mmfd", and  "optpp_fd_newton". These performed poorly on our production level tests, so 
        the other methods are recommended unless you have a specific case where these are required.

        .. note: Some methods treat the method options differently and may not behave the same for different options 
            such as "convergence_tolerance", "max_iterations" and "max_function_evaluations".

        """
        self._set_method(method_name)

    def get_gradient_block(self):
        response_block = self.get_response_block()
        return  response_block.get_subblock(NumericalGradientBlock.type)
    
    def set_step_size(self, value=
                      NumericalGradientBlock.default_values[DakGradientKeys.fd_step_size]):
        """
        Sets the finite difference step sizes for the gradient decent optimizations.
        Default step size is a relative step of 5e-5.
        
        :param step_size: the desired step_size
        :type step_size: float
        """
        check_value_is_real_between_values(value, 1e-9, 1e-1, 
                                           "step size", "set_step_size")
        grad_block = self.get_gradient_block()
        step_size_line  = grad_block.get_line(DakGradientKeys.fd_step_size) 
        step_size_line.set(value)


class GradientCalibrationStudy(_GradientDakotaFile, DakotaCalibrationStudyBase):

    """
    The Gradient Calibration algorithm is a local optimization method that requires the objective
    function be smooth and convex. This method can quickly find a nearby minimum if the
    objective function is smooth. It is useful if you are calibrating to a single model with only a
    couple states and you know a decent initial guess for the parameters. This is the MatCal implementation of the
    NL2SOL method from Dakota.
    """

    def __init__(self, *parameters):
        DakotaCalibrationStudyBase.__init__(self, *parameters)
        _GradientDakotaFile.__init__(self, )

class _NongradientMethodKeys():
    coliny_cobyla = "coliny_cobyla"
    coliny_pattern_search = "coliny_pattern_search"
    coliny_solis_wets = "coliny_solis_wets"
    mesh_adaptive_search = "mesh_adaptive_search"
    optpp_pds = "optpp_pds"


dakota_response_identifier.register(_NongradientMethodKeys.coliny_cobyla, 
                                    NongradientResponseBlock)
dakota_response_identifier.register(_NongradientMethodKeys.coliny_pattern_search, 
                                    NongradientResponseBlock)
dakota_response_identifier.register(_NongradientMethodKeys.coliny_solis_wets, 
                                    NongradientResponseBlock)
dakota_response_identifier.register(_NongradientMethodKeys.mesh_adaptive_search, 
                                    NongradientResponseBlock)
dakota_response_identifier.register(_NongradientMethodKeys.optpp_pds, 
                                    NongradientResponseBlock)


class _NongradientDakotaFile(DakotaCalibrationFile):
    _method_class = GeneralNongradientMethodType


class _NongradientDakotaFileWithVarTol(_NongradientDakotaFile):
    
    def set_variable_tolerance(self, value):
        """
        Set the variable convergence tolerance which limits the minimum step length for the algorithm and is a measure
        of algorithm convergence. A value greater than 0.0 and less than 1.0 is expected. For specifics,
        see the Dakota
        user manual.

        :param value: the desired variable tolerance
        :type value: float
        """
        check_value_is_real_between_values(value, 0, 1.0, "variable tolerance", 
                                           "set_variable_tolerance")
        self.set_method_type_block_line(DakMethodKeys.variable_tolerance,
                                        value)

    def get_variable_tolerance(self):
        """
        Returns the variable tolerance size for the study.
        Returns None if it has not been specified.
        :rtype: float or None
        """
        method_type_block = self.get_method_type_block()
        value = None
        if DakMethodKeys.variable_tolerance in method_type_block.lines:
            value = method_type_block.get_line_value(DakMethodKeys.variable_tolerance)
        return value


class _MeshAdaptiveDakotaFile(_NongradientDakotaFileWithVarTol):

    valid_methods = [_NongradientMethodKeys.mesh_adaptive_search]
    
    class Keywords(_NongradientDakotaFileWithVarTol.Keywords):
        variable_neighborhood_search = "variable_neighborhood_search"

    def set_variable_neighborhood_search(self, value):
        """
        Set the variable neighborhood search parameter for the algorithm. It increases the number of objective
        function evaluations each iteration to search more of the space in an attempt to escape local minimums. For
        specifics, see the Dakota user manual. A value of 0 to 1.0 is expected.

        :param value: the desired variable tolerance
        :type value: float
        """
        check_value_is_real_between_values(value, 0, 1.0, "variable neighborhood search", 
                                           "set_variable_neighborhood_search")
        self.set_method_type_block_line(self.Keywords.variable_neighborhood_search, value)

    def get_variable_neighborhood_search(self):
        """
        Returns the variable neighborhood search for the study.
        Returns None if a value is not user specified
        
        :rtype: float or None
        """
        method_type_block = self.get_method_type_block()
        value = None
        if self.Keywords.variable_neighborhood_search in method_type_block.lines:
            value = method_type_block.get_line_value(self.Keywords.variable_neighborhood_search)
        return value


class _MeshAdaptiveSearchDefaults:
    method_specific_default_values = {}
    default_values = dict(**GeneralNongradientMethodType.default_values, 
                          **method_specific_default_values)


method_type_defaults_identifier.register(_NongradientMethodKeys.mesh_adaptive_search, 
                                         _MeshAdaptiveSearchDefaults.default_values)


class MeshAdaptiveSearchCalibrationStudy(_MeshAdaptiveDakotaFile, DakotaCalibrationStudyBase):
    """
    The Mesh Adaptive Search algorithm is a local optimization method that exhibits some robustness to noisy objective
    functions. The true global minimum may not be found, but this method can find a nearby minimum even if the
    objective function is not smooth. This method can require more iterations than a gradient based method
    but is cheaper than global search methods. It is useful if you are calibrating to a couple models with only a
    couple states and you know a decent initial guess for the parameters. This is the MatCal implementation of the
    mesh_adaptive_search method from Dakota.
    """

    def __init__(self, *parameters):
        DakotaCalibrationStudyBase.__init__(self, *parameters)
        _MeshAdaptiveDakotaFile.__init__(self, )
        self._set_method(_NongradientMethodKeys.mesh_adaptive_search)


class _ColinyNongradientDakotaFile(_NongradientDakotaFileWithVarTol):
    valid_methods = [_NongradientMethodKeys.coliny_cobyla, 
                     _NongradientMethodKeys.coliny_solis_wets, 
                     _NongradientMethodKeys.coliny_pattern_search]   

    def set_solution_target(self, value):
        """
        Specifies a target value for the objective function. Once the objective function goes below or reaches this
        value, the calibration will stop.

        :param value: solution target value
        :type value: float
        """
        check_item_is_correct_type(value, Real, "set_solution_target", 
                                    "solution target")
        self.set_method_type_block_line(DakMethodKeys.solution_target, value)

    def get_solution_target(self):
        """
        Returns the solution target for the study. Returns 
        None if it has not been set.

        :rtype: float or None
        """
        method_type_block = self.get_method_type_block()
        value = None
        if DakMethodKeys.solution_target in method_type_block.lines:
            value = method_type_block.get_line_value(DakMethodKeys.solution_target)
        return value


class _ColinyCobylaDefaults:
    method_specific_default_values = {
                                      
                                      DakMethodKeys.convergence_tol:1e-3
                                      }
    default_values = dict(**GeneralNongradientMethodType.default_values, 
                          **method_specific_default_values)
    

method_type_defaults_identifier.register(_NongradientMethodKeys.coliny_cobyla, 
                                         _ColinyCobylaDefaults.default_values)


class CobylaCalibrationStudy(_ColinyNongradientDakotaFile, DakotaCalibrationStudyBase):
    """
    The Cobyla algorithm is a local optimization method that exhibits some robustness to noisy objective
    functions. The true global minimum may not be found, but this method can find a nearby minimum even if the
    objective function is not smooth. This method can require more iterations than a gradient based method
    but is cheaper than global search methods. It is useful if you are calibrating to a couple models with only a
    couple states and you know a decent initial guess for the parameters. This is the MatCal implementation of the
    coliny_cobyla method from Dakota.
    """
    def __init__(self, *parameters):
        DakotaCalibrationStudyBase.__init__(self, *parameters)
        _ColinyNongradientDakotaFile.__init__(self, )
        self._set_method(_NongradientMethodKeys.coliny_cobyla)


class _ColinySolisWetsDefaults:
    method_specific_default_values = {
                                      DakMethodKeys.convergence_tol:1e-3,
                                      }
    default_values = dict(**GeneralNongradientMethodType.default_values, 
                          **method_specific_default_values)
    
method_type_defaults_identifier.register(_NongradientMethodKeys.coliny_solis_wets,  
                                        _ColinySolisWetsDefaults.default_values)


class SolisWetsCalibrationStudy(_ColinyNongradientDakotaFile, DakotaCalibrationStudyBase):

    """
    The Solis Wets algorithm is a local optimization method that exhibits some robustness to noisy objective
    functions. The true global minimum may not be found, but this method can find a nearby minimum even if the
    objective function is not smooth. This method can require more iterations than a gradient based method
    but is cheaper than global search methods. It is useful if you are calibrating to a couple models with only a
    couple states and you know a decent initial guess for the parameters. This is the MatCal implementation of the
    coliny_solis_wets method from Dakota.
    """
    def __init__(self, *parameters):
        DakotaCalibrationStudyBase.__init__(self, *parameters)
        _ColinyNongradientDakotaFile.__init__(self, )
        self._set_method(_NongradientMethodKeys.coliny_solis_wets)

    
class _PatternSearchDakotaFile(_ColinyNongradientDakotaFile):

    valid_methods = [_NongradientMethodKeys.coliny_pattern_search]

    class Keywords(_ColinyNongradientDakotaFile.Keywords):
        exploratory_moves = "exploratory_moves"

    def set_exploratory_moves(self, value):
        """
        This can be used to set the way the pattern for the search is updated each iteration. Available options are
        "basic_pattern", "adaptive_pattern", and "multi_step". The default is currently "basic_pattern",
        which is subject to change. See Dakota manual for specifics.

        :param value: the exploratory move type to be used
        :type value: str
        """
        check_item_is_correct_type(value, str, "set_exploratory_moves", 
                                         "exploratory moves")

        valid_types = ["basic_pattern", "adaptive_pattern", "multi_step"]
        if value not in valid_types:
            valid_types_string = '\n'.join(x for x in valid_types)
            raise ValueError("{} is not a valid \"exploratory moves\" option. The following are valid "
                                            "options: {}".format(value, valid_types_string))

        self.set_method_type_block_line(self.Keywords.exploratory_moves, value)

    def get_exploratory_moves(self):
        """
        Returns the exploratory moves pattern for the study.
        Returns None if it is not user specified.

        :rtype: str or None
        """
        method_type_block = self.get_method_type_block()
        value =None
        if self.Keywords.exploratory_moves in method_type_block.lines:
            value = method_type_block.get_line_value(self.Keywords.exploratory_moves)
        return value
    

class _ColinyPatternSearchDefaults:
    method_specific_default_values = { 
                                      DakMethodKeys.convergence_tol:1e-3}
    default_values = dict(**GeneralNongradientMethodType.default_values, 
                          **method_specific_default_values)


method_type_defaults_identifier.register(_NongradientMethodKeys.coliny_pattern_search, 
                                         _ColinyPatternSearchDefaults.default_values)


class PatternSearchCalibrationStudy(_PatternSearchDakotaFile, DakotaCalibrationStudyBase):
    """
    The Pattern Search algorithm is a local optimization method that exhibits some robustness to noisy objective
    functions. The true global minimum may not be found, but this method can find a nearby minimum even if the
    objective function is not smooth. This method can require more iterations than a gradient based method
    but is cheaper than global search methods. It is useful if you are calibrating to a couple models with only a
    couple states and you know a decent initial guess for the parameters. This is the MatCal implementation of the
    coliny_pattern_search method from Dakota.
    """

    def __init__(self, *parameters):
        DakotaCalibrationStudyBase.__init__(self, *parameters)
        _PatternSearchDakotaFile.__init__(self, )
        self._set_method(_NongradientMethodKeys.coliny_pattern_search)
    

class _OptppPdsDakotaFile(DakotaCalibrationFile):
    _method_class = GeneralNongradientMethodType
    valid_methods = [_NongradientMethodKeys.optpp_pds]

    class Keywords(DakotaCalibrationFile.Keywords):
        search_scheme_size = "search_scheme_size"

    def set_search_scheme_size(self, value):
        """
        Set the number of samples used for creating the simplex each iteration. Dakota has a default of 32
        and MatCal has a default of 10. This should be at least N+1 where N is the number of parameters being
        calibrated.

        :param value: the search scheme size
        :type value: int
        """
        check_value_is_positive_integer(value, "set_search_scheme_size", 
                                        "search scheme size")
        self.set_method_type_block_line(self.Keywords.search_scheme_size, value)

    def get_search_scheme_size(self):
        """
        Returns the search scheme for the study.

        :rtype: int
        """
        method_type_block = self.get_method_type_block()
        value = method_type_block.get_line_value(self.Keywords.search_scheme_size)
        return value


class _OptppPdsDefaults:
    class _Keywords:
        search_scheme_size = "search_scheme_size"

    method_specific_default_values = {_Keywords.search_scheme_size:10}
    default_values = dict(**GeneralNongradientMethodType.default_values, 
                          **method_specific_default_values)
    

method_type_defaults_identifier.register(_NongradientMethodKeys.optpp_pds, 
                                         _OptppPdsDefaults.default_values)


class ParallelDirectSearchCalibrationStudy(_OptppPdsDakotaFile, DakotaCalibrationStudyBase):

    """
    The Parallel Direct Search algorithm is a local optimization method that exhibits some robustness to noisy
    objective functions. The true global minimum may not be found, but this method can find a nearby minimum even if the
    objective function is not smooth. This method can require more iterations than a gradient based method
    but is cheaper than global search methods. It is useful if you are calibrating to a couple models with only a
    couple states and you know a decent initial guess for the parameters. This is the MatCal implementation of the
    optpp_pds method from Dakota. Note that although the algorithm is designed for parallel execution, Dakota has not
    yet implemented it as parallel so it runs in serial.
    """

    def __init__(self, *parameters):
        DakotaCalibrationStudyBase.__init__(self, *parameters)
        _OptppPdsDakotaFile.__init__(self, )
        self._set_method(_NongradientMethodKeys.optpp_pds)

