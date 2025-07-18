
"""
This module contains MatCal's interface to Dakota's global calibration methods.
"""
from matcal.core.utilities import (check_item_is_correct_type, 
                                   check_value_is_positive_integer, 
                                   check_value_is_real_between_values)

from matcal.dakota.input_file_writer import (DakotaCalibrationFile, 
                                             DakMethodKeys, GeneralNongradientMethodType, 
                                             method_type_defaults_identifier, 
                                             dakota_response_identifier, 
                                             NongradientResponseBlock, 
                                             check_seed_value)
from matcal.dakota.dakota_studies import DakotaCalibrationStudyBase

class _JegaKeywords():
    population_size = "population_size"
    crossover_type = "crossover_type"
    crossover_rate = "crossover_rate"
    mutation_type = "mutation_type"
    mutation_rate = "mutation_rate"
    moga = "moga"
    soga = "soga"


class _JegaMethodsDefaults:
    method_specific_default_values = {_JegaKeywords.population_size:50, 
                                     _JegaKeywords.crossover_type:"shuffle_random", 
                                    _JegaKeywords.crossover_rate:0.7, 
                                    _JegaKeywords.mutation_type:"offset_normal", 
                                    _JegaKeywords.mutation_rate:0.2, 
                                    DakMethodKeys.convergence_tol:1e-3}
    
    default_values = dict(**GeneralNongradientMethodType.default_values, 
                          **method_specific_default_values)

method_type_defaults_identifier.register(_JegaKeywords.soga, 
                                         _JegaMethodsDefaults.default_values)
method_type_defaults_identifier.register(_JegaKeywords.moga, 
                                         _JegaMethodsDefaults.default_values)

class _JegaCalibrationDakotaFile(DakotaCalibrationFile):
    _method_class = GeneralNongradientMethodType
    valid_methods = [_JegaKeywords.soga, _JegaKeywords.moga]

    def set_population_size(self, population_size):
        """
        Sets the population size for the genetic algorithm calibration.

        :param population_size: The desired population size
        :type population_size: int
        """
        check_value_is_positive_integer(population_size, "set_population_size",
                                   "population size")
        self.set_method_type_block_line(_JegaKeywords.population_size, population_size)

    def get_population_size(self):
        """
        Returns the population size for the study.
        :rtype: int
        """
        method_type_block = self.get_method_type_block()
        population_size = method_type_block.get_line_value(_JegaKeywords.population_size)
        return population_size

    def set_seed(self, seed):
        """
        Specify the seed used to generate the initial population.

        :param seed: the seed value
        :type seed: int
        """
        check_seed_value(seed)
        self.set_method_type_block_line(DakMethodKeys.seed, seed)

    def set_random_seed(self,seed):
        """
        See :meth:`~matcal.dakota.global_calibration_studies.SingleObjectiveGACalibrationStudy.set_seed`.
        """
        self.set_seed(seed)

    def get_seed(self):
        """
        Returns the seed for the random initial population
        if specified by the user. Otherwise returns None.

        :rtype: int, None
        """
        method_type_block = self.get_method_type_block()
        seed = None
        if DakMethodKeys.seed in method_type_block.lines:
            seed = method_type_block.get_line_value(DakMethodKeys.seed)
        return seed

    def set_crossover_rate(self, crossover_rate):
        """
        Specify the crossover rate for the genetic algorithm calibration. 
        It must be between zero and 1.

        :param crossover_rate: crossover rate value
        :type crossover_rate: float
        """
        check_value_is_real_between_values(crossover_rate, 0, 1.0, "crossover rate", 
                                           "set_crossover_rate")
        self.set_method_type_block_line(_JegaKeywords.crossover_rate, crossover_rate)

    def get_crossover_rate(self):
        """
        Returns the crossover rate value for the study.
        :rtype: float
        """
        method_type_block = self.get_method_type_block()
        crossover_rate = method_type_block.get_line_value(_JegaKeywords.crossover_rate)
        return crossover_rate

    def set_crossover_type(self, crossover_type):
        """
        Set the crossover type for the genetic algorithm.

        :param crossover_type: the type of crossover to be used. 
            See manuals at dakota.sandia.gov.
        :type crossover_type: str
        """
        check_item_is_correct_type(crossover_type, str, "set_crossover_type", 
                                   "crossover type")
        crossover_types = ['multi_point_binary', 
                           'multi_point_parameterized_binary', 'multi_point_real',
                           'shuffle_random']
        if crossover_type not in crossover_types:
            raise ValueError("'{}' is not a valid crossover type."
                                  " Please choose one of the following:"
                                  "\n{}".format(crossover_type, 
                                                crossover_types))

        self.set_method_type_block_line(_JegaKeywords.crossover_type, crossover_type)

    def get_crossover_type(self):
        """
        Returns the crossover type for the study.
        :rtype: str
        """
        method_type_block = self.get_method_type_block()
        crossover_type = method_type_block.get_line_value(_JegaKeywords.crossover_type)
        return crossover_type

    def set_mutation_rate(self, mutation_rate):
        """
        Specify the mutation rate for the genetic algorithm calibration. It must be between zero and 1.

        :param mutation_rate: mutation rate value
        :type mutation_rate: float
        """
        check_value_is_real_between_values(mutation_rate, 0, 1.0, "mutation rate", 
                                           "set_mutation_rate")

        self.set_method_type_block_line(_JegaKeywords.mutation_rate,  mutation_rate)

    def get_mutation_rate(self):
        """
        Returns the mutation rate value for the study.
        :rtype: float
        """
        method_type_block = self.get_method_type_block()
        mutation_rate = method_type_block.get_line_value(_JegaKeywords.mutation_rate)
        return mutation_rate

    def set_mutation_type(self, mutation_type):
        """
        Set the mutation type for the genetic algorithm.

        :param mutation_type: the type of mutation to be used. 
            See manuals at dakota.sandia.gov.
        :type mutation_type: str
        """
        check_item_is_correct_type(mutation_type, str, "set_mutation_type", 
                                   "mutation type")
        mutation_types = ['bit_random', 'replace_uniform', 
                          'offset_normal', 'offset_cauchy', 'offset_uniform']
        if mutation_type not in mutation_types:
            raise ValueError("'{}' is not a valid mutation type. "
                                  "Please choose one of the following:"
                                  "\n{}".format(mutation_type, mutation_types))
        self.set_method_type_block_line(_JegaKeywords.mutation_type, mutation_type)

    def get_mutation_type(self):
        """
        Returns the mutation type for the study.
        :rtype: str
        """
        method_type_block = self.get_method_type_block()
        mutation_type = method_type_block.get_line_value(_JegaKeywords.mutation_type)
        return mutation_type


dakota_response_identifier.register(_JegaKeywords.moga, 
                                    NongradientResponseBlock)
dakota_response_identifier.register(_JegaKeywords.soga, 
                                    NongradientResponseBlock)


class MultiObjectiveGACalibrationStudy(_JegaCalibrationDakotaFile, DakotaCalibrationStudyBase):
    """
    The Multi-objective Genetic Algorithm is a global optimization method 
    that does Pareto optimization for
    multiple objectives. This method is a robust calibration 
    method that can handle noisy objective functions. It also
    is good for calibrations where multiple models and states 
    are involved and weighting each contribution to the
    objective is unclear. A downside is that this method 
    requires many objective function evaluations and is slow to
    converge. This is the MatCal implementation of the MOGA Dakota method.
    """

    def __init__(self, *parameters):
            DakotaCalibrationStudyBase.__init__(self, *parameters)
            _JegaCalibrationDakotaFile.__init__(self, )
            self._set_method(_JegaKeywords.moga)


class SingleObjectiveGACalibrationStudy(_JegaCalibrationDakotaFile, DakotaCalibrationStudyBase):
    """
    The Single-objective Genetic Algorithm is a global 
    optimization method that is robust and can
    handle noisy objective functions. A downside is that this method requires
    many objective function evaluations and is 
    slow to converge. This is the MatCal 
    implementation of the SOGA method
    from Dakota.
    """

    def __init__(self, *parameters):
            DakotaCalibrationStudyBase.__init__(self, *parameters)
            _JegaCalibrationDakotaFile.__init__(self, )
            self._set_method(_JegaKeywords.soga)


