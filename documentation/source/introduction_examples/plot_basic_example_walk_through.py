"""
Linear Python Model Example
==================================

This section applies many of the topics introduced in the previous chapter 
to a simple calibration of 
the equation of a line (:math:`y=\\text{m}x+\\text{b}`) to data.
While the example is simple it contains all of the fundamental topics necessary 
for more advanced calibrations. 

At the top of all Python files is typically the location where all of the
modules are imported. Here 
we import all of MatCal's tools. 
"""

from matcal import *

#%%
# We use the ``from <> import *`` format to import all of the user facing 
# objects and functions into Python from MatCal. 
# The first thing to do in a MatCal study is to define what are our 
# parameters of interest. 
# For this example, a line is being optimized to fit a set of data; therefore, 
# there are two parameters of interest: the slope and the y intercept.

#%%
slope = Parameter('slope', 0, 10, 9.9)
#%% 
# In the previous line, a :class:`~matcal.core.parameters.Parameter` object 
# is created and assigned to ``slope``.
# The :class:`~matcal.core.parameters.Parameter` class is used by MatCal to 
# designate and characterize the parameters of interest. 
# The parameter  class has three mandatory ordered arguments: the parameter 
# name, a lower bound, and an upper bound. 
# The parameter name is the name that the model will use to look up the 
# parameter value. 
# The lower and upper bounds bracket the parameter investigation range. 

y_int = Parameter('y0', -10, 10)
#%%
# Here another :class:`~matcal.core.parameters.Parameter` object is 
# created for the y-intercept. 
# There are a couple differences in this object instantiation versus 
# the previous.
# First, the parameter name and the name of the variable storing the 
# parameter object do not have the same name. 
# In practice, it is useful to make sure that those names are similar, 
# because it makes the code more readable. 
# Second, the parameter object is taking in four inputs instead of three. 
# The fourth input is optional and will dictate the current parameter value. 
# If no current value is passed as an argument, the current value is assigned 
# to be the midpoint between the lower and upper bound.
#
# Next the parameters are bundled together into
# a :class:`~matcal.core.parameters.ParameterCollection`.
my_parameters = ParameterCollection('line parameters', slope, y_int)
#%%
# This is an object that holds the parameters and makes it easier to 
# move the related set of parameters around. There are other types of 
# Collections in MatCal that behave in a similar manner. 
# To create the collection a collection name is passed into the constructor 
# followed by all items to become part of the collection separated by commas.
#
# Now that the parameters of interest have been defined, we can create a 
# MatCal study to act on these parameters. 
my_calibration = GradientCalibrationStudy(my_parameters)
my_calibration.set_results_storage_options(results_save_frequency=len(my_parameters)+1)

#%%
# The central MatCal tool is a study. As discussed in :ref:`MatCal Studies`, 
# a study will assess how the parameters 
# affect a model's response when compared to a set of data using an objective. 
# In this case, the goal is to calibrate the slope and y-intercept of a line 
# to match a set of data. Therefore, 
# a :class:`~matcal.dakota.local_calibration_studies.GradientCalibrationStudy`
# is created. The :class:`~matcal.dakota.local_calibration_studies.GradientCalibrationStudy`
# will use gradient descent to calibrate the parameters it is given to a set of 
# data when a model and objective is provided. 
# 
# A MatCal study evaluates the effects of its parameters on evaluation sets that 
# have been added to the study. 
# An evaluation set is a collection of three concepts: a model, instructions on how 
# to compare the model results to some reference data, 
# and the reference data. 
# To add an evaluation set first each of these components need to be defined.
#
# The first of this triad that will be defined is the reference data:

my_data = FileData('my_data_file.csv')

#%%
# MatCal typically expects reference data to be contained in external data files. 
# MatCal can read several common types of data files using the 
# :func:`~matcal.core.data_importer.FileData`, such as csv and npy.
# In the above line, MatCal reads in 'my_data_file.csv' and converts it 
# into an object it can use.  
# In this case there is only one set of data of interest located in the 
# file 'my_data_file.csv'.
#
# The next component to create is the model. 

def my_function(**parameters):
  import numpy as np
  time = np.linspace(0,100)
  slope = parameters['slope']
  y0 = parameters['y0']
  values = y0 + slope * time
  results = {'time':time, 'Y': values}
  return results

my_model = PythonModel(my_function)

#%%
# Above we define a simple Python function that takes in a dictionary of the 
# parameters of interest and returns a dictionary containing processed values.
# This function will be the model used by MatCal to calibrate the line 
# parameters to the data. 
# The function ``my_function`` unpacks the parameter dictionary using 
# Python's unpacking feature ``**``.
# The keywords that MatCal assigns the different parameters are the same as the 
# name supplied by the user in the creation of the parameters. 
# MatCal can use this function as a model using the 
# :class:`~matcal.core.models.PythonModel` class, where the 
# only input is our ``my_function`` Python function. 
#
# The last of the evaluation set triad is the instructions on how to 
# compare the model results to the reference data. 
# In MatCal, these instructions are conveyed to the study by creating an 
# objective. 

my_objective = CurveBasedInterpolatedObjective('time','Y')

#%%
# In the above line, a 
# :class:`~matcal.core.objective.CurveBasedInterpolatedObjective` is created. 
# This objective is an l2-norm of the differences between two curves in two dimensions 
# and is the most commonly used objective in MatCal studies. 
# :class:`~matcal.core.objective.CurveBasedInterpolatedObjective` takes 
# in at least two string inputs. 
# The first input string is the independent variable that will be used 
# to parameterize the comparison of the dependent variables (the X-Axis). 
# The second input string, and beyond, are the dependent variables that will 
# be compared. In this example we use 'time' as our independent variable 
# and 'Y' as our dependent variable. 
# For our objective to work correctly, the data that we read in from 
# 'my_data_file.csv' needs to have 'time' and 'Y' fields. In addition, 
# our model needs to also produce 
# 'time' and 'Y' value results. 
#
# To add the evaluation set to the study the line below is called.
my_calibration.add_evaluation_set(my_model, my_objective, my_data)

#%%
# Now the study has the means to calibrate the parameters. It is possible for a 
# study to have more than one evaluation set if applicable, but that will 
# be discussed in a later example. 
# To run the study the following line is used. 
results = my_calibration.launch()

# %%
# The :meth:`~matcal.dakota.local_calibration_studies.GradientCalibrationStudy.launch` 
# method begins the study, and will run until either the calibration completes, 
# or it encounters an error. 
# If the study completes successfully, it will return a dictionary of results 
# including the calibrated parameters, and it will also create a file 
# saving the best parameter values.
# MatCal's progress is recorded in "matcal.log", and, if the study encounters 
# an error, details of the error will be stored there. More information 
# on MatCal's output and results 
# can be found at :ref:`Results and Plotting`.
make_standard_plots('time')
print(f"Print best as results attribute:\n{results.best}")
print(f"\nAccess and print y0 as attribute from best:\n{results.best.y0}")
print(f"\nAccess and print best as dict:\n{results.best.to_dict()}")
print(f"\nAccess and print best as list:\n{results.best.to_list()}")
