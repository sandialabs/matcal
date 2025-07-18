"""
Data Noise Issue - Low and High Noise Example
=============================================
In this case the data comes from the same model as the one we are calibrating. 
The model is a bilinear function similar to elastic-plastic 
response with the parameter "Y" controlling where change from one linear 
trend to the other takes place and the parameter "H" controlling the slope of the second trend. 
The slope of the initial trend is fixed, as if it were known (as the elastic modulus typically would be).

The fits for the low and high noise cases look similar; however, 
if the error as a function of the parameters is examined you can see: 
(a) the optimum has shifted and (b) the bowl is flatter for the high noise case. 
The noise induces bias in the optimal parameters and 
makes the problem harder to solve. With enough noise 
the calibration becomes useless. Also, it is apparent from 
the shape of the error contours the two parameters Y and H 
have correlated effects on the error, i.e., combinations of 
high Y and low H have the same error as low Y and high H. 
This is common in physics models and presents a tradeoff in 
calibration that in its extreme becomes another issue 
(which will be explored in another example  targeting "identifiability").
"""
# sphinx_gallery_thumbnail_number = 4

from matcal import *
import numpy as np

#%%
# We create a python function to be used as a MatCal PythonModel
# for this calibration. The python model uses Numpy in the function
# and requires that Numpy be imported within the function.
# As stated above, the data will be fitted to a model with a 
# bilinear response similar to an elastic-plastic model.
# The initial slope (E) is assumed to be known. 
# The parameter "Y" determines where the model changes to 
# the second linear trend and the parameters "H" determines the second slope.

def bilinear_model(**parameters):
  import numpy as np
  max_strain = 1.0
  npoints = 100
  strain = np.linspace(0,max_strain,npoints)
  E = 1.5
  Y = parameters['Y']
  H = parameters['H']
  eY = Y/E
  stress = np.where(strain < eY, E*strain, Y + H*(strain-eY))
  response = {'strain':strain, 'stress': stress}
  return response

#%%
# With the function for the model defined above,
# we create the parameters for our MatCal 
# calibration study and the MatCal PythonModel
# from the function.

Y = Parameter('Y',0.0,1.0, 0.501)
H = Parameter('H',0.0,1.0, 0.501)
parameters = ParameterCollection('parameters', Y, H)

model = PythonModel(bilinear_model)
model.set_name("bilinear")
#%%
# Start with low noise fit.
# A simple gradient-based calibration with a 
# mean squared error objective is enough to illustrate the point. 
# We load the data, create the calibration based on the model parameters, 
# and define an objective of fitting the model response to the data.
#
data_low_noise = FileData('bilinear_lownoise.csv')
calibration = GradientCalibrationStudy(parameters)
objective = CurveBasedInterpolatedObjective('strain','stress')
calibration.add_evaluation_set(model, objective, data_low_noise)
#%%
# After running the calibration, we 
# load the optimal parameters and the best fit.
#
results = calibration.launch()
best_parameters_low_noise = results.best.to_dict()
best_response = results.best_simulation_data(model, 'matcal_default_state')

#%%
# We can then run the calibration and load the optimal parameters and the best fit.
#
data_strain_low_noise = data_low_noise['strain']
data_stress_low_noise = data_low_noise['stress']
model_strain = best_response['strain']
model_stress = best_response['stress']

#%%
# First let us compare the response curves.
# We plot the calibrated model with lines and 
# the data with points. Generally, there are about as 
# many points above the fitted line as below. 
# This is due to using a mean squared error objective.
#

import matplotlib.pyplot as plt
plt.plot(model_strain, model_stress,'b',label="fit")
plt.scatter(data_strain_low_noise,data_stress_low_noise,2,'r',label="data")
plt.xlabel("STRAIN")
plt.ylabel("STRESS")
plt.show()

#%%
# Second, examine how the error changes in the vicinity of the best fit.
# This a helper function to evaluate the 
# error on a grid of parameter values for plotting. 
# MatCal's ParameterStudy can also do this.
#
def sample_error(model, data, ranges = {"Y":[0.0,0.2],"H":[0.0,0.2]} ):
  Ys, Hs = np.mgrid[ ranges["Y"][0]:ranges["Y"][1]:100j, ranges["H"][0]:ranges["H"][1]:100j]
  Zs = np.empty_like(Ys)
  for i in range(Ys.shape[0]):
    for j in range(Ys.shape[1]):
      parameters = {"Y":Ys[i,j],"H":Hs[i,j]}
      response = model(**parameters)
      residual = response['stress']-data['stress']
      error = np.sum(residual**2)
      Zs[i,j] = error
  return Ys,Hs,Zs
#%% 
# The contour plot depicts the change in error in the vicinity of the optimum.
# Clearly it rises smoothly from the minimum at the optimum.
# If viewed from the side the error vs. parameters would 
# look like a parabola with an elliptical cross-section.
#
Ys,Hs,Zs = sample_error(bilinear_model,data_low_noise)
plt.contourf(Ys,Hs,np.log(Zs),20)
plt.grid(True)
plt.xlabel("Y")
plt.ylabel("H")
plt.colorbar(label="log error")
plt.show()

#%%
# Now do the high noise fit.
# Load data from the same model as 
# the one to be fitted with added uncorrelated 
# noise but now with more (higher variance) noise.
#
data_high_noise = FileData('bilinear_highnoise.csv')

#%%
# We again create the parameters, but this 
# time slightly change the initial point to 
# get around a known issue with Dakota. 
# See the note below. 
Y = Parameter('Y',-10.0,2.0, 0.4999) # note jiggle initial value to get around dakota issue
H = Parameter('H',-1.0,2.0, 0.4999) # note jiggle initial value to get around dakota issue

#%%
#
# .. include:: ../multiple_dakota_studies_in_python_instance_warning.rst
#

parameters = ParameterCollection('parameters', Y, H)

#%%
# Again, a simple gradient-based calibration 
# with a mean squared error objective is used to perform the calibration.
#
calibration = GradientCalibrationStudy(parameters)
calibration.add_evaluation_set(model, objective, data_high_noise)
results = calibration.launch()
best_parameters_high_noise = results.best.to_dict()
best_response = results.best_simulation_data(model, 'matcal_default_state')

#%%
# Grab the true/experimental response and the model/fitted response
#
data_strain_high_noise = data_high_noise['strain']
data_stress_high_noise = data_high_noise['stress']
model_strain = best_response['strain']
model_stress = best_response['stress']

import matplotlib.pyplot as plt
#%%
# Compare the response curves. 
# We plot the calibrated model with lines and 
# the data with points. Again, there are about as many 
# points above the fitted line as below, but you can see 
# if the spread in the data becomes larger the slope and 
# the change-over point will become less well defined.
#
plt.plot(model_strain,model_stress,'b',label="fit")
plt.scatter(data_strain_low_noise,data_stress_high_noise,2,'r',label="data")
plt.xlabel("STRAIN")
plt.ylabel("STRESS")
plt.show()

#%%
# Once again, examine how the error changes in the vicinity of the best fit.
# The bottom of the error bowl, depicted by the contours, is now not centered 
# on the true parameter values Y=0.1, H=0.1. The contours are still elliptical, 
# but the slope of the bowl is shallower. In the limit the bowl can become 
# so shallow that the descent direction may be hard to determine.
#
Ys,Hs,Zs = sample_error(bilinear_model,data_high_noise)
plt.contourf(Ys,Hs,np.log(Zs),20)
plt.grid(True)
plt.xlabel("Y")
plt.ylabel("H")
plt.colorbar(label="log error")
plt.show()
