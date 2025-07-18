MatCal Objective Calculations
=============================

As discussed in the :ref:`General MatCal Overview`, the objective is the key value used to 
drive parameter selection in MatCal's study methods. 
In the simplest case, this calculation simply requires subtracting a vector of 
simulation predictions from a vector of reference data, and then taking its Euclidean norm. 
For material model calibration activities, we often define the objective function the same way as we do 
in idealized cases but need to perform more processing on our simulation data 
and reference data to perform the calculation or improve study performance. 
In this section we, will 
cover the process MatCal takes to calculate an objective. 

In calculating the objective for an evaluation, MatCal executes the 
following processes on the data:

#. Conditioning: Scale the data to bring all quantities 
   of interest approximately to the unit scale.
#. Mapping: Map the simulation data to the same locations/space as 
   the experimental data so that they can be meaningfully compared.
#. Differencing: Find the residual by subtracting the experimental data 
   from the mapped simulation data. This
   results in negative residuals when the simulation is lower than the experimental data and 
   positive residuals when the simulation data higher.
#. Weighting: Scale sections of the residual to emphasize certain 
   aspects of the data over others. 
#. Normalizing: Scale the residual by the number of data points 
   present in the data set to remove bias towards larger data sets.
#. Measuring: Calculate a norm on the weighted residual to ascertain 
   an overall measure of how closely the simulation predictions align with the experimental data.

Conditioning
------------
Data used to for calibration needs to be conditioned to bring all quantities of 
interest to the same scale of value. Failure to do so will bias the calibration towards 
quantities of interest that have measurements farther away from zero. For example, 
if you were performing a calculation based on a single temperature and a single pressure measurement, 
you would likely get different calibration results if you reported your pressure 
data in MPa versus Pa. For the Pa case, you would likely get better agreement with the pressure data,
while getting worse agreement with the temperature data. While in the MPa case, 
the fit to the temperature data would improve and the fit to the pressure data would 
become worse. 

This change in calibration behavior is because small relative 
differences in the pressure are reported as values on the order 
of :math:`10^5` in the Pa case and :math:`10^{-1}` for the MPa case, 
while small relative differences in temperature are reported 
on the order of :math:`1`. These differences in scale translate 
to the amount of emphasis each measurement 
contributes to the objective, and measurements of larger scales 
will get more 'attention' from the calibration process 
while measurements at smaller scales may be ignored.
Calibration results should be independent of the choice of 
units used in the experimental data and the model predictions.

For MatCal to ensure unit independence of its studies, 
all the experimental data and simulation data are 
rescaled to be approximately in the range of negative one to one. 
This helps ensure that relative differences in measurement 
for all quantities of interest are represented on the same value scale.
MatCal conditions data by analyzing the experimental data 
and defining a linear scaling by state that brings the 
data into the desired range for all data by state.
For more information on what is meant by state 
see :ref:`Data Object Creation and Storage` 
and :class:`~matcal.core.state.State`.
This is done by using equation :eq:`eq_matcal_conditioning` as follows: 

.. math::
  :label: eq_matcal_conditioning

  \underline{\bar{u}} = \frac{\underline{u} - o}{s}

where :math:`\underline{\bar{u}}` are the conditioned values, 
:math:`\underline{u}` are the unconditioned values, and 
:math:`\underline{u}_{exp}` represents the 
combined data for all data sets in a single state. The 
:math:`o` and :math:`s` values are offset and scaling values 
that are applied in the conditioner and calculated 
from :math:`\underline{u}_{exp}`. How these are calculated is dependent on the 
specific conditioner 
used. MatCal's data conditioners can be found in the :mod:`~matcal.core.data` module. 
The default conditioner is the :class:`~matcal.core.data.MaxAbsDataConditioner`.

The simulation predictions are also 
conditioned using :eq:`eq_matcal_conditioning` with the
scaling values defined by the experimental data they are being
compared against. 
This means that while the experimental data is likely to 
be close to the desired range, 
the simulation data may exceed 
these boundaries because the model may predict 
values that are beyond the range of the experimental data.

Mapping
-------
Not all calibrations require a mapping step, 
but in most cases, we are interested in comparing data 
at specific points in time, space, or state. 
Unless very special care is taken in setting up the 
experimental data and the model prediction results, 
every data set in a typical material calibration 
is evaluated at a slightly 
different set of points. Even if all data sets have 
the same number of points, direct comparison of 
the points is likely to be meaningless as they will 
not be comparing the same measured quantities.

To address this MatCal maps the model 
data points to the experimental data 
points via linear interpolation. It is assumed 
that the simulation data are well suited 
to linear interpolation because they have no noise and are dense. 
Interpolating on the smooth and denser data set minimizes 
the potential error incurred in the mapping process. 
This linear interpolation mapping is done by default when using 
the :class:`~matcal.core.objective.CurveBasedInterpolatedObjective`
for data comparison. If this mapping is not 
acceptable, you can use the :class:`~matcal.core.objective.Objective`
for comparison and assign your own mapping algorithm using the 
:meth:`~matcal.core.objective.Objective.set_qoi_extractors` method. 

A quantity of interest (QoI) extractor is what performs the mapping 
for the objective. In the :class:`~matcal.core.objective.CurveBasedInterpolatedObjective`, 
the QoI extractor is NumPy ``interp`` and interpolates the simulation
data to the experimental data points. However, the  :class:`~matcal.core.objective.Objective`
class has no default assigned QoI extractor, and expects the data to be comparable 
when called or requires the user to set its QoI extractor. MatCal's 
predefined QoI extractors are in :mod:`~matcal.core.qoi_extractor`. If 
the predefined extractors available do not suit your needs, 
the :class:`~matcal.core.qoi_extractor.UserDefinedExtractor`
can be used to define your own that operates on the data before 
compared using a Python function. 

.. note::
    Note that the same QoI extractor can be applied 
    to both the simulation and experimental data, or 
    different QoI extractors can be applied to the 
    different data. See :meth:`~matcal.core.objective.Objective.set_qoi_extractors`, 
    :meth:`~matcal.core.objective.Objective.set_experiment_qoi_extractor`, and 
    :meth:`~matcal.core.objective.Objective.set_simulation_qoi_extractor`.

Differencing
------------
With the data conditioned and collocated from the previous steps, 
a residual can be formed from the different experimental-model data sets.
The residual is formed by subtracting each experimental data 
from their respective model prediction data sets:

.. math::
  :label: eq_matcal_residual

  \underline{r}_i = \underline{\bar{u}}_{sim} - \underline{\bar{u}}_{exp}

where :math:`\underline{\bar{u}}_{sim}` are the simulation QoIs. 
MatCal supports calibrating a material model using data from 
different models, experimental states, and repeats of experimental data. 
This additional complexity requires that MatCal loop through 
all models, states, and repeats of the data to generate all 
of the correct residuals for each objective. For the collection of 
residuals to become an objective, they must be concatenated 
to form a single residual. However, there is additional processing
required before the residual can be concatenated. 
Thus, for the time being MatCal holds on to each residual in isolation.

Weighting
---------
Due to a wide variety of reasons (data sampling locations, density of data points, 
speed of transition, model form, etc.) certain features in a data set may be ignored in a calibration
for the calibration to focus on capturing other more dominant features. 
The root source of this neglect can at times be difficult to determine, 
and potentially even impossible to fix. 
For example, it is not possible to add more data points 
to an experimental data set, if there is no budget to run more experiments.
To help address this issue, MatCal provides its users with 
the option of conditionally weighting their residuals to emphasize different regions of their data. 
Weighting may help force the calibration to better capture an event 
in the data that only exists over a small fraction of the total data points in a data set. 

By default, there are no automatic weighting functions in MatCal. 
To use weighting, a weighting function needs to be supplied by the user. 
An example of how to add a weighting function to an objective is shown in 
example :ref:`sphx_glr_introduction_examples_sierra_material_point_examples_plot_sierra_material_point_calibration.py`.
But to cover it in brief here, the user specifies an independent and dependent field 
to be passed to their weighting function along with the current residual. 
Inside their function, the user may adjust the values of the residual however they would like.
The values of the fields passed into the function along with the residual are 
index the same, this means that the data present in position :math:`i` in each array 
corresponds to the same data measurement. 
This allows for easy weighting of the residual based off  
the field values requested by the user.

.. note:: It is recommended that users test their weighting
          functions with a residual set of all ones.
          Plotting these results will allow the user to 
          confirm the function they've passed to MatCal 
          is behaving the way they intend it to.

This weighting is applied to the residual results just before they are normalized in the next step.

Normalizing
-----------
An example parameter set evaluation tree structure is shown below. 
We can see in this problem, we have two models, with model one 
having one experimental state and model two having two experimental states. 
This is great for our calibration, because we are hopefully examining 
a wider range of material states than we would with a single experiment, 
providing us with a more robust model.

.. figure:: evaluationstructure.png
    :scale: 35 %

    File structure present in a given parameter set evaluation. 
    Each model is evaluated at all states included in the data present
    in its evaluation set. The results of the model are found at the bottom of the tree. 

However, in the calibration process we need to be careful 
when we have different counts of data from different sources. 
Having a larger number of data points in one set of data will bias 
the results to this data set.
This is because that data set represents a larger contingency 
of the numbers that will be used to calculate the objective.
To address this issue, MatCal normalizes all its residuals by 
the number of entries in the residuals.

This normalization means that if there is a ten percent error 
in a model prediction with 100 data points, then its evaluation generates 
the same magnitude of objective difference as if it had 1000 data points. 
To achieve this effect, the residuals are scaled by their data set size, 
the number of data sets in their state, and the number of states in the model.
How these values are applied changes depending upon which objective definition is used.
Most calibrations use something similar to an L2-norm, 
which requires that the residual be scaled by the square 
root of the terms listed above. This is shown in equation :eq:`eq_matcal_norm`. 

.. math::
  :label: eq_matcal_norm

  \underline{r}_{norm} = \frac{1}{\sqrt{n_{states}n_{data}len(\underline{r})}} \underline{r}


In our example if each data set had 16 entries, 
the residual from results 1A would be scaled by
:math:`\frac{1}{\sqrt{2(1)16}} = \frac{1}{4\sqrt{2}}`
and the residual from results 
2B and 2C would be scaled by :math:`\frac{1}{\sqrt{2(2)16}} = \frac{1}{8}`.

Measuring
---------
The last stage of the objective calculation in MatCal is 
to convert the collection of weighted, normalized residuals 
into a single objective. 
The objective serves as an overall measurement of 
the total disagreement between the experimental data and the model predictions. 
For most calibrations in MatCal this is done using the Euclidean norm (L2-norm) 
which is the default norm for most objectives. 
Several norms are available including the L1-norm and all vector norms 
supported by NumPy's ``linalg.norm`` function. See
:class:`~matcal.core.objective.NormMetricFunction`, 
:class:`~matcal.core.objective.L1NormMetricFunction`, and 
:class:`~matcal.core.objective.L2NormMetricFunction`

For this measurement, all the residuals from a given objective 
set are concatenated 
on top of each other to form a monolithic residual vector. 
An objective set includes all residuals calculated for a study 
as a result of adding an evaluation set to the study with 
:meth:`~matcal.dakota.local_calibration_studies.GradientCalibrationStudy.add_evaluation_set`.

.. note::
    Due to how we handle objectives in objective sets, 
    you will get a different number of objectives if you add multiple 
    objectives using a single call of 
    :meth:`~matcal.dakota.local_calibration_studies.GradientCalibrationStudy.add_evaluation_set`
    than you will if you add each on individually 
    with separate calls of 
    :meth:`~matcal.dakota.local_calibration_studies.GradientCalibrationStudy.add_evaluation_set`.
    This will primarily affect algorithms that explicitly handle multiple objectives, but 
    will produce slightly different objectives for the same problem.

With the concatenated residual(:math:`\underline{R}`), 
the objective(:math:`\Pi`) is calculated by taking the appropriate norm 
for the objectives. For the typical case of the L2-norm, this is the inner product of the 
concatenated residual with itself, and then taking the square root of it(equation :eq:`eq_eu_norm`).

.. math::
  :label: eq_eu_norm

  \Pi = \sqrt{\underline{R} \cdot \underline{R}} = \sqrt{\sum_i R_i^2}

For problems with multiple objectives, the normalized values 
are summed to form a single objective. 
This is the value that most calibrations attempt to minimize. 
Some calibration methods will operate directly on the residual, 
in which the normalized, weighted residual is used by 
the calibration method to calibrate the material parameters. 
Even if the objective is not used for the calibration method directly, 
it is still calculated by MatCal and 
can be output using MatCal's internal plotting tools to 
easily observe the convergence behavior of a particular calibration. 
