Parameter Types and Specification
=================================
In the most general sense, MatCal provides a 
clean interface to study the effect model input 
parameters have on a model's responses or a set 
of objectives. We can use this interface to perform
calibrations, sensitivity studies and parameter
uncertainty quantification. Here we cover how to
specify model input parameters in MatCal, and how 
they will be used and passed to models.

In MatCal we have three types of parameters. In order
of precedence from a model perspective they are:

#. Study parameters - These are the primary input to the 
   MatCal study constructor and are defined by the 
   :class:`~matcal.core.parameters.Parameter` class. 
   These should be required by one or more of the models
   that the study will run and should affect the output 
   from the models. These take the highest precedence, meaning
   if they are defined using another type of parameter as described below, 
   the study parameter value overwrites the other type of parameter 
   value, and the study parameter value is used.

#. Model constant parameters - These are model parameters 
   that will be passed to models before they are launched but 
   will be kept constant throughout a study. They can be assigned to the model
   for all states using :meth:`~matcal.core.models.PythonModel.add_constants`
   or specifically for a single state using 
   :meth:`~matcal.core.models.PythonModel.add_state_constants`. 
   These will overwrite a state parameter described below, if the 
   same parameter name is used for both a model constant and a state parameter.
  
#. State parameters - As covered in :ref:`Data Object Creation and Storage`
   state parameters are meant to represent experimental conditions that 
   may vary over a given test type, but are held constant 
   for a given experiment for that test type. These parameters may or may not be
   required to model the experiments correctly. Some common examples of 
   state parameters are the initial temperature for a test, the 
   extraction location of a specimen for a test, or a rate relevant to the test. 

.. note:: Although study parameters will take precedent over model constants, some 
   model parameters cannot be affected in a study for MatCal 
   generated SIERRA models by design for model and solution robustness.
   See :ref:`Overriding model parameters during a study` 
   in :ref:`MatCal Generated SIERRA Standard Models`. 

These three parameter types are meant to provide 
flexibility to the user in the studies they perform and the 
way they setup their models. If used effectively, they can allow
for a variety of study types to be performed from calibrations 
to sensitivity studies. A few useful examples 
showing the different types of parameters in use for 
different study types are:

#. State and study parameters for calibration:
   :ref:`304L stainless steel viscoplastic calibration`
#. State parameters, model constants and study parameters for 
   a mesh convergence study:
   :ref:`304L stainless steel mesh and time step convergence`
#. State parameter, model constants and study parameters 
   for verification of our SIERRA/SM thermal modeling options:
   :ref:`Comparing iterative, staggered and adiabatic coupling solutions`

Model Constants and State Parameters
------------------------------------
Model constants and state parameters perform similar functions 
in MatCal. They both are required to be constant for the test or 
simulation but are important inputs to the test or model. In MatCal, 
they both are defined using 
keyword/value pairs when passed to a class or method 
that creates them. In addition, they both can take numeric or 
string values which gives them more flexibility than 
study parameters. 

Model constants can only be created with the two 
:meth:`~matcal.core.models.PythonModel.add_constants`
and 
:meth:`~matcal.core.models.PythonModel.add_state_constants`
methods referenced above. As a result, they are specific to 
the model from which they are called. To add them to all 
models, call these methods on each model. In this case, 
good practice would be to define them in a dictionary
and pass the unpacked dictionary to the methods for each model. 

State parameters can be created several ways. They 
can be created using the :class:`~matcal.core.state.State`
class where they are passed as keyword/value arguments to the 
constructor. They can also be added to an existing state using 
the :meth:`~matcal.core.state.State.update` and 
:meth:`~matcal.core.state.State.update_state_variable` methods. 
Finally, they can be assigned based on information inside files 
from which data are imported as discussed in 
:ref:`CSV file data importing details`. Sets of state 
classes and their associated parameters can be stored in a :class:`~matcal.core.state.StateCollection`.
A state collection can be used to select a subset of valid data states for 
comparison in a study when calling 
:meth:`~matcal.core.study_base.StudyBase.add_evaluation_set` on a study object.

Once defined, these parameters will be sent to the model upon 
the launch of the model simulation. These are passed directly into 
the python function as keyword arguments for any
:class:`~matcal.core.models.PythonModel`. For models 
that rely on external executables, they are passed to the model through 
files on the disk. Currently all state parameters and model constants
are passed in one file and the design parameters are passed in another.

Study Parameters
----------------
Study parameters have more complexity 
than state or model parameters. Besides 
their current value and name, these parameters
have several other attributes
that could be used based on the study being performed.
Study parameters require that a name, 
lower bound and upper bound be specified when 
created using the :class:`~matcal.core.parameters.Parameter`
class in MatCal. When initialized, a value will automatically
be assigned to the current value as the average of the bounds. 
Other optional attributes include a distribution for the parameter
and units for the parameters. Distributions are 
used for sensitivity and uncertainty quantification studies. Units 
are only used for GRANTA uploads. 

As with state and model constants, these parameters 
are passed either directly into a :class:`~matcal.core.models.PythonModel`
function as keyword arguments, or are passed to external executable 
models through files on the disk. 

Study parameters can be 
stored in a :class:`~matcal.core.parameters.ParameterCollection`. 
This is required when sending them to some methods and functions 
such as :meth:`~matcal.core.models.ModelBase.run` for running 
models outside of a study. However, it is also convenient 
when working with a group of related parameters that might 
change value over a set of MatCal studies. For example,
you can run a calibration, take the resulting calibrated parameters and pass 
them to a study :class:`~matcal.core.parameter_studies.ParameterStudy`
to perform mesh or time step convergence on the calibrated 
models.

   
