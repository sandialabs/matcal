"""
This module contains MatCal's interface to Dakota's uncertainty quantification methods.
"""
from numbers import Real

from matcal.core.object_factory import BasicIdentifier
from matcal.core.utilities import (check_value_is_positive_integer, 
                                   check_value_is_positive_real)       

from matcal.dakota.dakota_constants import DAKOTA_MCMC_CHAIN_FILE
from matcal.dakota.dakota_studies import DakotaStudyBase
from matcal.dakota.input_file_writer import (DakotaFileWithSeed, BaseMethodType,
                                             check_number_of_samples, 
                                             dakota_response_identifier, 
                                             NongradientResidualsResponseBlock, 
                                             NongradientResponseBlock)


non_garident_bayes_type = "bayes_calibration"


class _AdaptiveMetropolisMethod(BaseMethodType):
    type = non_garident_bayes_type

    class Keywords():
        adaptive_metroplis = "adaptive_metropolis"
        proposal_covariance = "proposal_covariance"
        burn_in_samples = "burn_in_samples"
        export_chain_points_file = "export_chain_points_file"
        chain_samples = "chain_samples"
        queso = "queso"

    required_keys = [Keywords.chain_samples, Keywords.proposal_covariance]

    default_values = {Keywords.queso:True,
                      Keywords.chain_samples:100, 
                      Keywords.burn_in_samples:10,
                      Keywords.adaptive_metroplis:True,
                      Keywords.export_chain_points_file:f'"{DAKOTA_MCMC_CHAIN_FILE}"'
                      }


class _DramMethodQueso(BaseMethodType):
    type = non_garident_bayes_type

    class Keywords():
        dram = "dram"
        proposal_covariance = "proposal_covariance"
        burn_in_samples = "burn_in_samples"
        export_chain_points_file = "export_chain_points_file"
        chain_samples = "chain_samples"
        queso = "queso"

    required_keys = [Keywords.chain_samples, Keywords.proposal_covariance]

    default_values = {Keywords.queso:True,
                      Keywords.chain_samples:100, 
                      Keywords.burn_in_samples:10,
                      Keywords.dram:True,
                      Keywords.export_chain_points_file:f'"{DAKOTA_MCMC_CHAIN_FILE}"'
                      }


class _DramMethodMuq(BaseMethodType):
    type = non_garident_bayes_type

    class Keywords():
        dram = "dram"
        proposal_covariance = "proposal_covariance"
        burn_in_samples = "burn_in_samples"
        chain_samples = "chain_samples"
        muq = "muq"

    required_keys = [Keywords.chain_samples, Keywords.proposal_covariance]

    default_values = {Keywords.muq:True,
                      Keywords.chain_samples:100, 
                      Keywords.burn_in_samples:10,
                      Keywords.dram:True
                      }


dakota_response_identifier.register(non_garident_bayes_type, 
                                    NongradientResidualsResponseBlock)


class DakotaBayesFileBase(DakotaFileWithSeed):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._set_method(self._method_class.type)
    
    def set_number_of_samples(self, number_of_samples):
        """
        Set the number of samples.

        :param number_of_samples: number of samples desired for the study
        :type number_of_samples: int
        """
        check_number_of_samples(number_of_samples)
        self.set_method_type_block_line(self._method_class.Keywords.chain_samples,
                                         number_of_samples)

    def get_number_of_samples(self):
        """
        Returns the number of samples for the study.

        :rtype: int
        """
        method_type_block = self.get_method_type_block()
        value = method_type_block.get_line_value(self._method_class.Keywords.chain_samples)
        return value

    def set_number_of_burnin_samples(self, value):
        """
        Set the total number of burn-in samples for the study.

        :param value: number of burn-in samples to run
        :type value: int
        """
        check_value_is_positive_integer(value, "number of burn on samples", 
                                        "set_number_of_burn_in_samples")
        self.set_method_type_block_line(self._method_class.Keywords.burn_in_samples, value)

    def get_number_of_burnin_samples(self):
        """
        Returns the number of burn-in samples for the study.

        :rtype: int
        """
        method_type_block = self.get_method_type_block()
        value = method_type_block.get_line_value(self._method_class.Keywords.burn_in_samples)
        return value

    def set_proposal_covariance(self, *proposal_covariance):
        """
        Set the proposal covariance for the parameters.
        This can be set as the same value for all parameters
        or as a specific value for each parameter. It can
        also
        accept any custom string 
        format that a Dakota input deck can accept.

        :param proposal_covariance: the proposal covariance value/values:
            a single float; a comma separated list or 
            unpacked list of floats of length N, 
            where N is the number of study parameters; 
            a custom string valid for covariance specification 
            in a Dakota input deck.
        :type proposal_covariance: str or float
        """
        if len(proposal_covariance) == 1:
            proposal_covariance = proposal_covariance[0]
            if isinstance(proposal_covariance, str):  # custom format compatible with dakotas
                proposal_covariance = proposal_covariance
            elif isinstance(proposal_covariance, Real):
                check_value_is_positive_real(proposal_covariance, "proposal covariance", 
                                     "set_proposal_covariance")
                proposal_covariance = self._get_uniform_diagonals(proposal_covariance)
        else:
            proposal_covariance = self._get_general_diagonals(proposal_covariance)
        self.set_method_type_block_line(self._method_class.Keywords.proposal_covariance, 
                                        proposal_covariance)

    def _get_general_diagonals(self, proposal_covariances):
        for proposal_covariance in proposal_covariances:
            check_value_is_positive_real(proposal_covariance, "proposal covariance", 
                                         "set_proposal_covariance")
        return "diagonal values " + " ".join(map(str, proposal_covariances))

    def _get_uniform_diagonals(self, proposal_covariance):
        return "diagonal values " + str(proposal_covariance)

    def get_proposal_covariance(self):
        """
        Returns the processed proposal covariance for the study.
        Returns None if not user specified.
        :rtype: str or None
        """
        method_type_block = self.get_method_type_block()
        value = None
        if self._method_class.Keywords.proposal_covariance in method_type_block.lines:
            proposal_covar_kw = self._method_class.Keywords.proposal_covariance
            value = method_type_block.get_line_value(proposal_covar_kw)
        return value


class _DakotaQuesoFileAM(DakotaBayesFileBase):
    _method_class = _AdaptiveMetropolisMethod
    valid_methods = [_AdaptiveMetropolisMethod.type]


class _DakotaMuqFileDram(DakotaBayesFileBase):
    _method_class = _DramMethodMuq
    valid_methods = [_DramMethodMuq.type]


class _DakotaQuesoFileDram(DakotaBayesFileBase):
    _method_class = _DramMethodQueso
    valid_methods = [_DramMethodQueso.type]


class AdaptiveMetropolisBayesianCalibrationStudy(_DakotaQuesoFileAM, DakotaStudyBase):
    """
    Runs a Bayesian calibration study for a given parameter 
    collection and set of evaluation sets. The current 
    values for the parameters sent to this study should be 
    from a traditional calibration method that located an 
    objective minimum.
    """
    study_class = "Bayes"

    def __init__(self, *parameters):
        DakotaStudyBase.__init__(self, *parameters)
        _DakotaQuesoFileAM.__init__(self, )

    def _study_specific_postprocessing(self):
        """"""

    def _package_results(self, dakota_results):
        return dakota_results

    def _return_output_information(self, output_filename):
        return self._dakota_reader(output_filename).parse_bayes()
    
    
class QuesoDramBayesianCalibrationStudy(_DakotaQuesoFileDram, DakotaStudyBase):
    
    study_class = "Bayes"
    
    def __init__(self, *parameters):
        DakotaStudyBase.__init__(self, *parameters)
        _DakotaQuesoFileDram.__init__(self, )

    def _study_specific_postprocessing(self):
        """"""

    def _package_results(self, dakota_results):
        return dakota_results

    def _return_output_information(self, output_filename):
        return self._dakota_reader(output_filename).parse_bayes()
    

class MuqDramBayesianCalibrationStudy(_DakotaMuqFileDram, DakotaStudyBase):
    
    study_class = "Bayes"
    
    def __init__(self, *parameters):
        DakotaStudyBase.__init__(self, *parameters)
        _DakotaMuqFileDram.__init__(self, )

    def _study_specific_postprocessing(self):
        """"""

    def _package_results(self, dakota_results):
        return dakota_results

    def _return_output_information(self, output_filename):
        return self._dakota_reader(output_filename).parse_bayes()
    

MatCalDramSelector = BasicIdentifier()
MatCalDramSelector.register('queso', QuesoDramBayesianCalibrationStudy)
MatCalDramSelector.register('muq', MuqDramBayesianCalibrationStudy)
    
    
def DramBayesianCalibrationStudy(*parameters, library='queso'):
    """
    Runs a Bayesian calibration study for a given parameter 
    collection and set of evaluation sets. The current 
    values for the parameters sent to this study should be 
    from a traditional calibration method that located an 
    objective minimum.
    
    DRAM stands for Delayed Rejection Adaptive Metropolis. Delayed Rejection
    means that there is some memory in the method that the sampling of points 
    which may improve the efficiency of the chain. The adaptive part of the 
    name means the calibration will perform adaptive adjustments 
    to their proposal covariance. This puts less weight on the initial
    guess used to define the proposal covariance. But the better the user submitted
    values, the more efficient the early chain will be. 
    
    There are two library options for using DRAM, QUESO and MUQ. Currently, 
    only QUESO is fully supported with MatCal. MUQ is a newer library and can 
    be run with MatCal, however it is on the user to gather the Dakota output.
    The Dakota output can be found in the 'dakota.out' file.  
    There is work with the Dakota team to add the necessary features to make 
    MUQ fully supported by MatCal.
    
    :param parameters: The parameters of interest for the study.
    :type parameters: list(:class:`~matcal.core.parameters.Parameter`) or
        :class:`~matcal.core.parameters.ParameterCollection`
    
    :param library: Which library to use for the back end Bayesian calculations. 
        Currently there are two options queso and muq. queso is the only fully 
        supported option at the moment. work is underway to fully support muq. 
        muq will run a study, but the user will need to read the 'dakota.out' file
        to get the results of their study, and MatCal will raise an error when 
        the study completes because it can not parse the results correctly. 
    """
    return MatCalDramSelector.identify(library)(*parameters)