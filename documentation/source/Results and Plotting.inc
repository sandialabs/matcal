Results and Plotting
====================
MatCal has a few tools to help users manage, view and
manipulate all 
the model results created through a MatCal study. 

Plotting
--------
The primary tool for results viewing is the "plot_matcal" plotting 
utility. It should be available 
for command line execution after loading a 
"matcal" module on a computing resource where MatCal 
is installed.
It currently creates three types of plots when called:

#. The simulation QoIs along with the experimental
   data QoIs. 
#. The objective value as a function of the 
   evaluation number. 
#. The objective value as a function of each 
   parameter. 

By default, the objective plots are created only for 
the total objective which is a combination of all 
evaluation set objectives. If you want to plot each objective independently, 
use the "--plot_model_objectives"/"-pmo" option. The QoI plots 
are created for all model and experiment QoI sets added as evaluation sets. 
In addition, for the simulation and experiment QoIs, 
a plot will be created 
for each state, since the results for each state may have
much different maximum and minimum bounds. By default,
all images are displayed after the plotting function 
has been called. To run the plotting script from the 
command line using the following::

   plot_matcal -i "independent_field_1,independent_field_2,...,independent_field_n"

where ``independent_field_1`` through
``independent_field_n`` are valid
independent field names for the data sets contained in the model objectives.
These independent fields specify all fields that will be used as independent 
fields on the horizontal axes of the QoI plots. All remaining fields available in the objectives
will be plotted as dependent fields on the vertical axes of the plots.
This command must be run in the directory where MatCal is running and 
reads results from the \*.joblib files created by MatCal
which stores study results as the studies run.

.. warning:: For large studies,
    this can result in many figures. 
    A ``--no-show`` will be available in future releases
    to suppress showing images.
   
Since model and experiment QoIs from all objectives
will be plotted when using this command, 
a valid field name must be given for each objective when specifying desired 
independent fields or the plotting 
script will fail. Another option is to pass no independent fields. In this case, 
all fields will be plotted as both the dependent and independent field for the QoI 
plot, creating an NxN table of subplots where N is the number of fields in the 
results data. 

The :func:`~matcal.core.plotting.make_standard_plots` can 
be called from a python interactive shell or a python script to 
make the same plots. See its documentation for use.

Example showing results created with the plotting script are 
:ref:`304L stainless steel viscoplastic calibration` and
:ref:`Successful Calibration`.

Results Data and Output
-----------------------
Results output from MatCal studies currently comes in several forms 
with some dependence on the type of study being performed. For 
all MatCal studies, there are at least three different ways to access
study results.

#. The results object returned from the 
   :meth:`~matcal.core.study_base.StudyBase.launch` method.
#. The "in_progress_results.joblib" and "final_results.joblib"
   files store the entire evaluation history 
   from all completed parameter batches. The in progress results 
   stores results up to and including that last evaluated parameter set batch. It 
   is replaced by the final results file at the end of the study which includes 
   the entire parameter evaluation history. Both files store a 
   :class:`~matcal.core.study_base.StudyResults` object.
#. The MatCal log file which contains objective information 
   for all evaluated parameters and any errors or output 
   from the underlying study.
#. Depending on the type of study, a best material 
   parameter file may also be written. It will contain
   Aprepro :cite:p:`aprepro` style variables or a dictionary with the values
   taken from the evaluation with the lowest objective
   from all parameter sets that were evaluated during 
   the study. Output is dependent on how MatCal is setup for your platform.

For Dakota based studies, additional output includes:

#. The "dakota.out", "dakota.err" and "dakota_tabular.dat" files written 
   by Dakota. The verbosity of output written to "dakota.out" can be controlled
   by the :meth:`~matcal.dakota.local_calibration_studies.GradientCalibrationStudy.set_output_verbosity`
   method called on a Dakota study.
#. The "dakota.rst" files used to provide restart information for a 
   Dakota based study.
#. Any additional output files produced by Dakota's libraries. 

MatCal is designed such that users should interact with the 
files output by MatCal and the results object returned from 
the study. 

.. note::  
   The evaluation name is also 
   the folder name that the models for the evaluation are run in 
   if their simulations require an external executable. 

The results object contains significant data related to the objective 
calculation which includes the simulation/experiment QoIs, the 
residuals and the objective values calculated from the residuals for each 
evaluation.
It also includes the entire parameter history and the names 
for each evaluation in the history. To access the results from the files written 
to disk use the :func:`~matcal.core.serializer_wrapper.matcal_load` function 
with the appropriate results filename passed to it. 

.. warning:: 
   The specifics stored in the results files may change over time. 
   We are still working to determine what should be stored in these objects;
   however, we will attempt to keep the objects backward compatible.

Always refer to the API documentation for 
:func:`~matcal.core.serializer_wrapper.matcal_load`
and :class:`~matcal.core.study_base.StudyResults` to get the most 
up-to-date structure for the files returned from the functions. 

The results output from the study :meth:`~matcal.core.study_base.StudyBase.launch` method
will return a the same :class:`~matcal.core.study_base.StudyResults`
as is saved to disk. The results saved from different studies 
will likely have different attributes specific to that study. 
See the API documentation for specifics on the returns from different studies. 
For Dakota calibrations, the most relevant returned result is the calibrated parameter sets.
In the results dictionary returned from these studies these are accessed with the 
under the "best" attribute using "results.best". 

The examples are also a good reference for how to access and use results data 
for post-study analysis, results plotting or follow-on studies. The :ref:`Introduction Examples`
include most study types, and demonstrate how to manipulate the output data from some of these studies. 
