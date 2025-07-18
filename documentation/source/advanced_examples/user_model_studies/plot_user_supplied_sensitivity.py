"""
Latin Hypercube Sampling to Obtain Local Material Sensitivities
---------------------------------------------------------------
In this section, we will cover how to run a Latin Hypercube Sampling study using
a model from an external 
physics modeling software. MatCal is written to be modular, so any tools taught 
here can be mixed and matched with tools presented in other examples. 
This example builds on the work done in the previous section, :ref:`Calibration 
of Two Different Material Conductivities`. The points discussed in 
this example will be used to elucidate the differences in setup between a 
calibration study and a sensitivity study. 

If you want more details about particular types of studies please see 
:ref:`MatCal Studies`. 

After performing our calibration in the previous example, we want to know 
how important our 
conductivities are to our prediction. To do this, we want to 
get a measure of our solution's 
sensitivity to our parameters. Fortunately for us, we have done 
much of the hard work already 
for the calibration, and we can just change a few lines in our 
MatCal script to perform a Latin Hypercube Sampling study 
and get Pearson correlation values. Pearson correlations tell 
us how correlated our parameters are with our quantities of interest. 
For this study, we can begin by adapting our MatCal input from 
the previous example. 
"""
from matcal import *
import numpy as np

cond_1 = Parameter("K_foam", .1 , .2, distribution="uniform_uncertain")
cond_2 = Parameter("K_steel", 30, 60, distribution="uniform_uncertain")

#%% 
# We start our MatCal input the same way we did with the calibration, 
# by importing MatCal and defining our 
# parameters. However, we are only attempting to asses how sensitive our model 
# is to the parameters over the range of interest. As a result, data do 
# not need to be supplied. We just need to define what fields we are interested
# in from the model results and at what independent values and states we want
# these data. To that end, we create the 
# state of interest and independent values of interest below.
low_flux_state = State("LowFlux", exp_flux=1e3)
import numpy as np
independent_field_values = np.linspace(0,20,11)

#%%
# The objective for this study will be a 
# :class:`~matcal.core.objective.SimulationResultsSynchronizer`.
# Since we do not need to compare to data for this study, 
# this is will synchronize simulation results to common 
# independent field values that are user specified for comparison.
# For MatCal, the synchronizer behaves like an objective and 
# is comparing the simulation 
# results to a vector of zeros for each dependent field. 
# As a result, there is no data conditioning, normalization or weighting
# applied to the simulation results.
objective = SimulationResultsSynchronizer("time", independent_field_values,
                                          "T_middle", "T_bottom")
#%%
# Next, we set up our model and objective just as we 
# did in the calibration example. 
user_file = "two_material_square.i"
geo_file = "two_material_square.g"
sim_results_file = "two_material_results.csv"
model = UserDefinedSierraModel('aria', user_file, geo_file)
model.set_results_filename(sim_results_file)

#%%
# The last difference between the calibration study and 
# this sensitivity study is our choice of 
# study. In this example we are using a 
# :class:`~matcal.dakota.sensitivity_studies.LhsSensitivityStudy`. 
# We initialize it and add our evaluation set
# to it as is common MatCal study procedure.
# In this study we are just looking at one state. 
# However,  multiple states can be run if desired through 
# the ``states`` keyword argument. 
sens = LhsSensitivityStudy(cond_1, cond_2)
sens.add_evaluation_set(model, objective, states=low_flux_state)
sens.set_core_limit(56)

#%%
# The last input needed is how many samples to take in the LHS study. 
# The study needs a certain number of samples to 
# produce a converged solution; however, that number  
# is problem dependent. We will likely have to run our study a 
# few times to confirm we have a converged solution. 
# A decent starting guess is ten times the number of parameters 
# you are studying. As a result,
# we set the number of samples to 20. 
sens.set_number_of_samples(20)

#%%
# Now all that is left to do is to launch the study and wait for our results. 
results = sens.launch()
print(results)
make_standard_plots('time')

#%% 
# We now repeat the study, but request Sobol Indices to 
# be output. 
sens = LhsSensitivityStudy(cond_1, cond_2)
sens.set_random_seed(1702)
sens.add_evaluation_set(model, objective, states=low_flux_state)
sens.set_core_limit(56)
sens.set_number_of_samples(20)
sens.make_sobol_index_study()
results = sens.launch()

#%%
# Notice that much more examples are 
# now run. For a study producing Sobol indices,
# Dakota will run :math:`N*(M+2)` samples 
# where :math:`N` is the number of requested samples 
# and :math:`M` is the number of study parameters in the study.
# As a result, for this study a total of 80 samples are run.
print(results)

#%%
# As can bee seen above, there are some unexpected results. The 
# method provides the sensitivity indices for main effects and the 
# total effects for each parameter, respectively. The main 
# effects are representative of the contribution of each 
# study parameter the variance in the model response.
# While the total effects represent the contribution 
# of each parameter in combination with all 
# the other parameters to the variance in the model response.
#
# For both cases the result should be positive. However, 
# in these results the ``K_steel`` parameter has 
# some negative values. This is most likely 
# due to the sampling size being too small and the fact 
# that the index values are small. As a result, 
# numerical errors cause the values to become negative.
# To investigate this issue we 
# re-run the study with more samples. This time 
# we choose a sample size of 200, which will run 
# a total of 800 models. Dakota's documentation 
# recommends hundreds to thousands of samples for a
# sampling study producing Sobol indices. 
sens = LhsSensitivityStudy(cond_1, cond_2)
sens.add_evaluation_set(model, objective, states=low_flux_state)
sens.set_core_limit(56)
sens.set_random_seed(1702)
sens.set_number_of_samples(200)
sens.make_sobol_index_study()
results = sens.launch()
print(results)

#%%
# In these results, we see that all of the indices have changed
# significantly. 
# This indicates that the Sobol indices are likely not converged.
# For a real problem, users should continue running studies with
# increasing samples 
# until the indices converge. Also, regarding the negative values 
# from the 20 sample study, some values are still negative. 
# This is likely due to them being near zero and within 
# expected numerical errors with the number of samples.
# If we were to do a proper sample size convergence study, 
# these would continue decreasing in magnitude but may 
# never turn all positive.  Although potentially not converged, 
# we can plot the current results and make conclusions about the 
# influence of the parameters on our QoIs.
make_standard_plots('time')

#%% 
# In these results, we see that across all time the conductivity of
# the foam has a strong correlation with both of our
# temperature values of interest,
# while the conductivity of steel has very little. 
# This indicates that this experimental series is likely a good 
# approach for determining a foam conductivity, 
# however, is less useful in determining the steel conductivity. 
# It would be useful to find
# another set of data to help us study the steel. 
