"""
This module contains MatCal's interface to Dakota's sensitivity study methods.
"""
from matcal.dakota.dakota_studies import DakotaStudyBase
from matcal.dakota.input_file_writer import (DakotaFileWithSeed, 
                                             BaseMethodType, 
                                             check_number_of_samples, 
                                             dakota_response_identifier, 
                                             NongradientResidualsResponseBlock, 
                                             NongradientResponseBlock, 
                                             )


class _SampleMethod(BaseMethodType):
    type = "sampling"

    class Keywords():
        sample_type = "sample_type"
        samples = "samples"
        lhs = "lhs"
        variance_based_decomp = "variance_based_decomp"
        
    required_keys = []
    method_specific_default_values = {Keywords.sample_type:Keywords.lhs, 
                                      Keywords.samples:10}
    default_values = dict(**method_specific_default_values)   

dakota_response_identifier.register(_SampleMethod.type, 
                                    NongradientResidualsResponseBlock)


class _DakotaSensitivityFile(DakotaFileWithSeed):
    _method_class = _SampleMethod
    valid_methods = [_SampleMethod.type]
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._set_method(_SampleMethod.type)

    def set_number_of_samples(self, number_of_samples):
        """
        Set the number of samples for the sensitivity study.

        :param number_of_samples: number of samples desired for the study
        :type number_of_samples: int
        """
        check_number_of_samples(number_of_samples)
        self.set_method_type_block_line(_SampleMethod.Keywords.samples, number_of_samples)

    def get_number_of_samples(self):
        """
        Returns the number of samples for the study.

        :rtype: int, None
        """
        method_type_block = self.get_method_type_block()
        value = method_type_block.get_line_value(_SampleMethod.Keywords.samples)
        return value

    def use_overall_objective(self):
        """
        Use the overall objective value as the target of the sensitivity study. 
        If this method is not called, the sensitivities will be calculated for each 
        experimental data point.
        """
        self._replace_response(NongradientResponseBlock())


class LhsSensitivityStudy(_DakotaSensitivityFile, DakotaStudyBase):
    """
    Perform LHS sensitivity studies for a parameter collection. This can provide 
    parameter Pearson correlations and Sobol indices for a set of evaluation sets 
    over the specified parameter range depending on options. The default behavior is
    to find the sensitivities at each experimental data point. For large data sets this 
    may require down sampling. To only compare to the overall objective value, use the
    use_overall_objective method.  
    """
    study_class = "Sensitivity"

    def __init__(self, *parameters):
        DakotaStudyBase.__init__(self, *parameters)
        _DakotaSensitivityFile.__init__(self, )
        self._reader = self._parse_pearson
        
    def _study_specific_postprocessing(self):
        """"""
    
    def _package_results(self, dakota_results):
        return dakota_results

    def make_sobol_index_study(self):
        """
        Runs the study so that it will output the Sobol indices. 

        .. warning:: Due to our adoption of the Dakota interface, this will 
            run more samples than expected. It will run :math:`N(M+2)` samples 
            where :math:`N` is the number of requested user samples and :math:`M` 
            is the number of study parameters being investigated.

        """
        self.set_method_type_block_line(_SampleMethod.Keywords.variance_based_decomp)
        self._reader = self._parse_sobol

    def _return_output_information(self, output_filename):
        return self._reader(output_filename)

    def _parse_sobol(self, output_filename):  
        return self._dakota_reader(output_filename).parse_sobol()

    def _parse_pearson(self, output_filename):
        return self._dakota_reader(output_filename).parse_pearson()

    def launch(self):
        """
        The Dakota LhsSensitivityStudy returns sensitivity information which 
        varies depending on the options used for the study.

        By default the study will run and calculate the Pearson correlations 
        between the parameters and the study objectives. This should give an 
        approximation of linear correlations between the model study parameters
        and the objectives of concern for the study. These results are 
        output as a dictionary with the keys being the study parameter names 
        and the values being the correlation of those parameters to 
        the objective values of interest.

        If the :meth:`~matcal.dakota.sensitivity_studies.LhsSensitivityStudy.make_sobol_index_study`
        has been chosen the study will return the Sobol indices for each parameter. 

        See the Dakota documentation for more information on the output :cite:p:`dakota`.

        :return: dictionary of parameter correlations to objectives
        :rtype: dict(str, float)
        """
        return super().launch()
