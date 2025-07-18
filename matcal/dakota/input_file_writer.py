from abc import ABC, abstractmethod
from itertools import count

from matcal.core.input_file_writer import (InputFileBlock, 
                                           _BaseTypedInputFileBlock, 
                                           InputFileLine)
from matcal.core.object_factory import BasicIdentifier
from matcal.core.parameters import Parameter   
from matcal.core.utilities import (check_item_is_correct_type, 
                                   check_value_is_real_between_values,
                                   check_value_is_positive_integer, 
                                   check_value_is_nonempty_str)


class DakEnvKeys():
    output_file = "output_file"
    error_file = "error_file"
    tabular_data = "tabular_data"
    method_pointer = "method_pointer"
    write_restart = "write_restart"
    read_restart = "read_restart"


class DakotaEnvironment(_BaseTypedInputFileBlock):

    type = "environment"

    required_keys = [DakEnvKeys.output_file, 
                     DakEnvKeys.error_file, 
                     DakEnvKeys.tabular_data,
                     ]

    default_values = { DakEnvKeys.tabular_data:True,
                      DakEnvKeys.output_file:'"dakota.out"', 
                      DakEnvKeys.error_file:'"dakota.err"', 
                      }

    def set_read_restart_filename(self,filename):
        check_value_is_nonempty_str(filename, "filename", 
                                    "set_read_restart_filename")
        if DakEnvKeys.read_restart not in self._lines.keys():
            read_restart_line  = InputFileLine(DakEnvKeys.read_restart)
            self.add_line(read_restart_line)
        self.get_line(DakEnvKeys.read_restart).set(f"\"{filename}\"")

    def set_write_restart_filename(self,filename):
        check_value_is_nonempty_str(filename, "filename", 
                                    "set_write_restart_filename")
        if DakEnvKeys.write_restart not in self._lines.keys():
            write_restart_line  = InputFileLine(DakEnvKeys.write_restart)
            self.add_line(write_restart_line)
        self.get_line(DakEnvKeys.write_restart).set(f"\"{filename}\"")


class DakVarKeys():
    descriptor = "descriptor"
    initial_point = "initial_point"
    lower_bounds = "lower_bounds"
    upper_bounds = "upper_bounds"
    scale_types = "scale_types"
    active = "active"
    uncertain = "uncertain"


class BaseDakotaVariablesBlock(_BaseTypedInputFileBlock):

    required_keys = [DakVarKeys.descriptor, 
                     DakVarKeys.initial_point, 
                     DakVarKeys.lower_bounds, 
                     DakVarKeys.upper_bounds]

    default_values = {}

    def set_parameters(self, pc):
        num_param = pc.get_number_of_items()
        self.update_title(num_param)

        names = []
        ips = []
        lbs = []
        ubs = []

        for idx, n in enumerate(pc.get_item_names()):
            names.append( f"\"{n}\"")
            ips.append(pc[n].get_current_value()) 
            lbs.append(pc[n].get_lower_bound()) 
            ubs.append(pc[n].get_upper_bound()) 

        self.add_line(InputFileLine(DakVarKeys.descriptor, *names))
        self.add_line(InputFileLine(DakVarKeys.initial_point, *ips))
        self.add_line(InputFileLine(DakVarKeys.lower_bounds, *lbs))
        self.add_line(InputFileLine(DakVarKeys.upper_bounds, *ubs))

    def update_title(self, num_vars):
        self._title += " = {}".format(num_vars)


class ContinuousDesignBlock(BaseDakotaVariablesBlock):
    type = "continuous_design"
    block_specific_default_values = {DakVarKeys.scale_types:"\"auto\"", "active":"all"}
    default_values = dict(**BaseDakotaVariablesBlock.default_values, 
                          **block_specific_default_values)


class UniformUncertainBlock(BaseDakotaVariablesBlock):
    type = "uniform_uncertain"
    block_specific_default_values = {DakVarKeys.active:DakVarKeys.uncertain}
    default_values = dict(**BaseDakotaVariablesBlock.default_values, 
                          **block_specific_default_values)


class ResponseBlock(_BaseTypedInputFileBlock):
    type = "response"
    response_type_key = "objective_functions"
    required_keys = [response_type_key]

    def set_number_of_expected_responses(self, expected_responses):
        response_line = InputFileLine(self.response_type_key)
        response_line.set(expected_responses)
        self.add_line(response_line)

    def get_number_of_expected_responses(self):
        return self.get_line_value(self.response_type_key)

class DakGradientKeys():
    no_hessians = "no_hessians"
    method_source = "method_source"
    interval_type = "interval_type"
    fd_step_size = "fd_step_size"
    no_gradients = "no_gradients"


class NumericalGradientBlock(_BaseTypedInputFileBlock):
    type = "numerical_gradients"
    required_keys = []
    default_values = {
                    DakGradientKeys.method_source: "dakota", 
                    DakGradientKeys.interval_type:"forward", 
                    DakGradientKeys.fd_step_size:5e-5}

class GradientResponseBlock(ResponseBlock):
    default_values = {DakGradientKeys.no_hessians:True}

    def __init__(self):
        super().__init__()
        self.add_subblock(NumericalGradientBlock())


class LeastSquaresResponseBlock(GradientResponseBlock):
    response_type_key = "calibration_terms"
    required_keys = [response_type_key]


class NongradientResponseBlock(ResponseBlock):
    required_keys = [ResponseBlock.response_type_key]

    default_values = {DakGradientKeys.no_gradients:True, 
                      DakGradientKeys.no_hessians:True}
       

class NongradientResidualsResponseBlock(ResponseBlock):
    response_type_key = LeastSquaresResponseBlock.response_type_key
    required_keys = [LeastSquaresResponseBlock.response_type_key]

    default_values = {DakGradientKeys.no_gradients:True, 
                      DakGradientKeys.no_hessians:True}
    

class DakModelKeys:
    single = "single"


class DakotaModelBlock(_BaseTypedInputFileBlock):
    type = "model"
    required_keys = []
    default_values = {}

    def __init__(self):
        super().__init__()
        self.single_model_block = InputFileBlock(DakModelKeys.single)
        self.add_subblock(self.single_model_block)
       

class DakInterfaceKeys():
    python = "python"
    batch = "batch"
    analysis_driver = "analysis_driver = \"matcal_interface_batch\""
    deactivate = "deactivate"
    deactivate_cache_name = "deactivate_cache"
    eval_cache = "evaluation_cache"


class PythonInterfaceBlock(_BaseTypedInputFileBlock):
    type = "interface"
    required_keys = []
    default_values = {DakInterfaceKeys.batch:True}
    _python_interface_numbers = count(0)
    def __init__(self):
        super().__init__()
        # the following line is needed to run multiple dakota studies in a single 
        # python instance
        self._python_interface_number = next(self._python_interface_numbers)
        self.add_line(InputFileLine("id_interface", f"'python_{self._python_interface_number}_id'"))
        analysis_driver_subblock = InputFileBlock(DakInterfaceKeys.analysis_driver) 
        analysis_driver_subblock.add_line(InputFileLine(DakInterfaceKeys.python))
        self.add_subblock(analysis_driver_subblock)

    def do_not_save_evaluation_cache(self):
        deactivate_cache_line = InputFileLine(DakInterfaceKeys.deactivate, 
                                              name=DakInterfaceKeys.deactivate_cache_name)
        deactivate_cache_line.set(DakInterfaceKeys.eval_cache)
        deactivate_cache_line.suppress_symbol()
        self.add_line(deactivate_cache_line)


class DakMethodKeys():
    max_func_evals = "max_function_evaluations"
    max_iterations = "max_iterations"
    convergence_tol = "convergence_tolerance"
    scaling = "scaling"
    speculative = "speculative"
    output = "output"
    output_options = ["silent", "quiet", "normal", "verbose",  "debug"]    
    method = "method"
    variable_tolerance = "variable_tolerance"
    solution_target = "solution_target"
    seed = "seed"


class DakotaMethodBlock(_BaseTypedInputFileBlock):
    type = "method"
    required_keys = []
    default_values = {
        DakMethodKeys.output:"silent", 
                      }
    
    
class BaseMethodType(_BaseTypedInputFileBlock):
    required_keys = []
    
    def __init__(self):
        super().__init__()
        self.set_print_name()
        self.set_print_title(False)

    def set_method_name(self, method_name):
        self.set_name(method_name)


class GeneralNongradientMethodType(BaseMethodType):
    nongrad_default_values = {DakMethodKeys.scaling:True,
                              DakMethodKeys.max_func_evals:10000,
                         DakMethodKeys.max_iterations:1000}
    default_values = dict(**nongrad_default_values)
    type = "nongradient_method"


class GeneralGradientMethodType(BaseMethodType):
    grad_default_values = {DakMethodKeys.scaling:True, 
                         DakMethodKeys.max_func_evals:1000,
                        DakMethodKeys.max_iterations:100, 
                        DakMethodKeys.convergence_tol:1e-3, 
                        DakMethodKeys.speculative:True}
    default_values = dict(**grad_default_values)
    type = "gradient_method"

    
method_type_defaults_identifier = BasicIdentifier()
dakota_response_identifier = BasicIdentifier()


DEFAULT_FILENAME = "dakota.in"


class DakotaFileBase(_BaseTypedInputFileBlock):

    class Keywords:
        variables = "variables"

    @property
    @abstractmethod
    def _method_class(self):
        """"""

    @property
    @abstractmethod
    def valid_methods(self):
        """"""
   
    type = "dakota"
    required_keys = []
    default_values = {}

    def __init__(self, *args, **kwargs):
        super().__init__("#MatCal Dakota File")
        self.set_print_name(False)
        self.set_print_title(False)
        self.add_subblock(DakotaEnvironment())
        self.add_subblock(DakotaModelBlock())
        self.add_subblock(PythonInterfaceBlock())
        self.add_subblock(DakotaMethodBlock())
        self._add_method()

    def _add_method(self):
        method = self._method_class()
        self.get_subblock(DakotaMethodBlock.type).add_subblock(method)

    def populate_variables(self, parameter_collection):
        dists = Parameter.distributions
        var_block = InputFileBlock(self.Keywords.variables)
        if parameter_collection.get_distribution() == dists.continuous_design:
            var_subblock = ContinuousDesignBlock()
        else:
            var_subblock = UniformUncertainBlock()
        var_subblock.set_parameters(parameter_collection)
        var_block.add_subblock(var_subblock)
        self.add_subblock(var_block)

    def get_environment_block(self):
        """
        Returns the Dakota input file environement block. 
        Users can modify this input file block directly. 
        See :class:`matcal.core.input_file_writer.InputFileBlock`.

        :rtype: :class:`matcal.core.input_file_writer.InputFileBlock`
        """
        return self.get_subblock(DakotaEnvironment.type)

    def get_method_block(self):
        """
        Returns the Dakota input file method block. 
        Users can modify this input file block directly. 
        See :class:`matcal.core.input_file_writer.InputFileBlock`.

        :rtype: :class:`matcal.core.input_file_writer.InputFileBlock`
        """
        return self.get_subblock(DakotaMethodBlock.type)

    def get_variables_block(self):
        """
        Returns the Dakota input file variables block. 
        Users can modify this input file block directly. 
        See :class:`matcal.core.input_file_writer.InputFileBlock`.

        :rtype: :class:`matcal.core.input_file_writer.InputFileBlock`
        """
        return self.get_subblock(self.Keywords.variables)

    def get_response_block(self):
        """
        Returns the Dakota input file response block. 
        Users can modify this input file block directly. 
        See :class:`matcal.core.input_file_writer.InputFileBlock`.

        :rtype: :class:`matcal.core.input_file_writer.InputFileBlock`
        """
        response_block = None
        if self.is_response_block_added():
            response_block = self.get_subblock(ResponseBlock.type)
        return response_block
    
    def is_response_block_added(self):
        return ResponseBlock.type in self._subblocks

    def get_interface_block(self):
        """
        Returns the Dakota input file interface block. 
        Users can modify this input file block directly. 
        See :class:`matcal.core.input_file_writer.InputFileBlock`.

        :rtype: :class:`matcal.core.input_file_writer.InputFileBlock`
        """
        return self.get_subblock(PythonInterfaceBlock.type)

    def get_method_type_block(self):
        """
        Returns the Dakota input file method type block. 
        Users can modify this input file block directly. 
        See :class:`matcal.core.input_file_writer.InputFileBlock`.

        :rtype: :class:`matcal.core.input_file_writer.InputFileBlock`
        """
        method_block = self.get_method_block()
        method_type_block = method_block.get_subblock(self._method_class.type)
        return method_type_block

    def set_number_of_expected_responses(self, objective_terms, residual_terms=None):
        resp_block = self.get_response_block()
        expected_responses = objective_terms
        if self._needs_residuals and residual_terms is not None:
            expected_responses = residual_terms
        resp_block.set_number_of_expected_responses(expected_responses)

    def set_read_restart_filename(self, filename):
        """
        Change the filename for the restart file read by Dakota. 
        By default, Dakota always attempts to read a 
        restart file with the name "dakota.rst".

        :param filename: The restart filename to be used.
        :type filename: str
        """
        check_item_is_correct_type(filename, str, "set_read_restart_filename",
                                   "Dakota restart filename")
        self.get_subblock(DakotaEnvironment.type).set_read_restart_filename(filename)

    def set_restart_filename(self, filename):
        """
        Change the filename for the restart file written by Dakota. 
        By default, Dakota always outputs a 
        restart file with the name "dakota.rst".

        :param filename: The restart filename to be used.
        :type filename: str
        """
        check_item_is_correct_type(filename, str, "set_restart_filename",
                                   "Dakota restart filename")
        self.get_subblock(DakotaEnvironment.type).set_write_restart_filename(filename)

    def get_read_restart_filename(self):
        """
        Returns the restart filename that Dakota will read for restarting.
        :rtype: str
        """
        env_block = self.get_environment_block()
        return env_block.get_line_value(DakEnvKeys.read_restart).strip('"')

    def get_write_restart_filename(self):
        """
        Returns the restart filename that Dakota will write for restarting.
        :rtype: str
        """
        env_block = self.get_environment_block()
        return env_block.get_line_value(DakEnvKeys.write_restart).strip('"')

    def do_not_save_evaluation_cache(self):
        """
        Do not save the entire evaluation cache. This is recommended for
        studies with large data sets that may fill all available memory. 

        .. warning::
            This will make restarting fail. Only use this if you will
            likely not need restarts.
        """
        interface_block = self.get_interface_block()
        interface_block.do_not_save_evaluation_cache()

    def set_output_verbosity(self, output_verbosity=DakotaMethodBlock.default_values[DakMethodKeys.output]):
        """
        Change the Dakota output verbosity. By default, we set it to 
        \"silent\". 
        Other options are available. See Dakota's
        documentation for more information.

        :param output_verbosity: The output verbosity level
        :type output_verbosity: str
        """
        check_item_is_correct_type(output_verbosity, str, "set_output_verbosity", 
                                   "Dakota  output verbosity",
                                   TypeError) 
        if output_verbosity in DakMethodKeys.output_options:
            method_block = self.get_method_block()
            output_line = method_block.get_line(DakMethodKeys.output)
            output_line.set(output_verbosity)
        else:
            raise ValueError(f"Invalid option specified \"{output_verbosity}\" for Dakota output verbosity." 
            f'\n Valid options are: \n{DakMethodKeys.output_options}' )

    def _set_method(self, method_name):
        if (method_name in self.valid_methods):
            self._method_name = method_name
        else:
            valid_methods_string = "\n".join(self.valid_methods)
            raise ValueError("Incorrect Dakota method specified.\n"
                            f"Received \"{method_name}\".\n"
                            f"Valid methods are: \n{valid_methods_string}")
        method_type_block = self.get_method_type_block()
        self._method_name = method_name
        method_type_block.set_method_name(self._method_name)
        self._add_method_specific_defaults(method_type_block)
        self._add_response()

    def _add_method_specific_defaults(self, method_type_block):
        if method_type_block.name in method_type_defaults_identifier.keys:
            method_defaults = method_type_defaults_identifier.identify(method_type_block.name)
            method_type_block.add_lines_from_dictionary(method_defaults, replace=True)

    @property
    def _needs_residuals(self):
        resp_block = self.get_response_block()
        is_resid_response = (isinstance(resp_block, LeastSquaresResponseBlock) or 
                             isinstance(resp_block, NongradientResidualsResponseBlock))
        return is_resid_response
   
    def set_method_type_block_line(self, keyword, *values, suppress_symbol = False):
        """
        Add a new input file line to or change the value 
        of an existing lie in the method type block. This can be used 
        to add method specific options to the method type block. By default this 
        will add a line with the following format `keyword = value_1 value_2 ... value_n`
        to the method type block. 

        :param keyword: the keyword for the method type option being added.
        :type keyword: str

        :param values: optional unpacked list of values that are associated with the keyword.
        :type values: str or float

        :param suppress_symbol: optionally remove the "=" symbol from the line. 
        :type suppress_symbol: bool

        """
        method_type_block = self.get_method_type_block()
        if keyword not in method_type_block.lines:
            new_line = InputFileLine(keyword, *values)
            method_type_block.add_line(new_line)
        else:
            line = method_type_block.get_line(keyword)
            line.set(*values)

    def _add_response(self):
        method_type_block = self.get_method_type_block()
        response_block_class = dakota_response_identifier.identify(method_type_block.name)
        response_block = response_block_class()
        self._replace_response(response_block)

    def _replace_response(self, new_response):
        existing_response_options = self._get_reponse_block_user_options()
        for line in existing_response_options:
            new_response.add_line(line, replace=True)
        self._remove_existing_response_block()
        self.add_subblock(new_response)

    def _get_reponse_block_user_options(self):
        exisiting_options = []
        response_block = self.get_response_block()
        if response_block:
            for line_key in response_block.lines:
                exisiting_options.append(response_block.get_line(line_key))
        return exisiting_options
    
    def _remove_existing_response_block(self):
        if self.is_response_block_added():
            self.remove_subblock(ResponseBlock.type)

    def write_input_file(self, filename=DEFAULT_FILENAME):
        return super().write_input_to_file(filename)
    

class DakotaCalibrationFile(DakotaFileBase, ABC):

    def set_convergence_tolerance(self, value):
        """
        Sets the convergence tolerance for the calibration algorithm. See the Dakota manual for specifics. A value
        greater than 0.0 and less than 1.0 is expected.

        :param value: The desired convergence tolerance
        :type value: float
        """
        check_value_is_real_between_values(value, 0, 1.0, "convergence tolerance", 
                                           "set_convergence_tolerance")
        self.set_method_type_block_line(DakMethodKeys.convergence_tol,
                                        value)
        
    def set_max_iterations(self, max_iterations):
        """
        Sets the maximum iterations the calibration can complete.

        :param max_iterations: The desired maximum iterations
        :type max_iterations: int
        """
        check_value_is_positive_integer(max_iterations, "max iterations", 
                                        "set_max_iterations")
        self.set_method_type_block_line(DakMethodKeys.max_iterations,
                                        max_iterations)

    def set_max_function_evaluations(self, value):
        """
        Sets the maximum function evaluation the calibration algorithm can run.

        :param value: The desired maximum iterations
        :type value: int
        """
        check_value_is_positive_integer(value, "max function evaluations", 
                                        "set_max_function_evaluations")
        self.set_method_type_block_line(DakMethodKeys.max_func_evals,value)
        

def check_seed_value(seed):
    check_value_is_positive_integer(seed, "seed", "set_seed")


def check_number_of_samples(number_of_samples):
    check_value_is_positive_integer(number_of_samples, "number_of_samples", 
                            "set_number_of_samples")
    

class DakotaFileWithSeed(DakotaFileBase):

    def set_seed(self, seed):
        """
        Set the seed for study. 

        :param seed: seed to be used for the study
        :type seed: int 
        """
        check_seed_value(seed)        
        self.set_method_type_block_line(DakMethodKeys.seed, seed)

    def set_random_seed(self,seed):
        """
        See :meth:`~matcal.dakota.sensitivity_studies.LhsSensitivityStudy.set_seed`.
        """
        self.set_seed(seed)

    def get_seed(self):
        """
        Returns the seed for the random samples
        if specified by the user. Otherwise returns None.

        :rtype: int, None
        """
        method_type_block = self.get_method_type_block()
        seed = None
        if DakMethodKeys.seed in method_type_block.lines:
            seed = method_type_block.get_line_value(DakMethodKeys.seed)
        return seed