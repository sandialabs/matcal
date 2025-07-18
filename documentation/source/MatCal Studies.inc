MatCal Studies
==============

MatCal's primary interface for material model calibration and related
work is the MatCal study class. Several types of 
studies currently exist in MatCal, however, most
are calibration studies that use Dakota's optimization 
algorithms :cite:p:`dakota`. We also support some of Dakota's 
other algorithms. Our :class:`~matcal.dakota.sensitivity_studies.LhsSensitivityStudy` is our interface 
to their LHS sensitivity analysis algorithm and our 
:class:`~matcal.dakota.uncertainty_quantification_studies.AdaptiveMetropolisBayesianCalibrationStudy` 
and 
:class:`~matcal.dakota.uncertainty_quantification_studies.DramBayesianCalibrationStudy` are our interfaces 
to two of their
Bayesian calibration methods. 

.. warning:: The :class:`~matcal.dakota.uncertainty_quantification_studies.AdaptiveMetropolisBayesianCalibrationStudy`
    is still being developed and not quite ready for production calibrations.

In addition to our interfaces to Dakota algorithms, we have a few pure MatCal studies 
that use our own algorithms. Currently, we have 
implementations of parameter studies that evaluate the objective values
and finite difference approximations of the Hessian and gradient of the 
objectives at user supplied parameter values. These studies include 
the :class:`~matcal.core.parameter_studies.ParameterStudy`
and the :class:`~matcal.core.parameter_studies.LaplaceStudy`. The 
:class:`~matcal.core.parameter_studies.ParameterStudy` is meant to 
provide the user with a simple interface to evaluate their models 
and objectives at user specified points. This can be used 
for manual calibration, creating surrogate model training data, or 
any similar task. The :class:`~matcal.core.parameter_studies.LaplaceStudy`
is an uncertainty quantification study based on the Laplace approximation.
It provides a simple method for parameter uncertainty quantification. 

.. warning::
    The :class:`~matcal.core.parameter_studies.LaplaceStudy` is currently
    designed to be used with simple calibrations where there are few data sets and a single model.
    It is also a meant to provide a rough tool for parameter uncertainty estimation, but simple verification 
    can be performed as shown in :ref:`Parameter uncertainty example - external noise and internal variability`. 

As with most tools in MatCal, studies are python objects that 
user are meant to interact with using the objects' methods, attributes and properties.
To initialize these objects, their constructors take MatCal :class:`~matcal.core.parameters.Parameter`
objects in a comma separated list or a single :class:`~matcal.core.parameters.ParameterCollection`
object. These are the parameters that will be the primary focus of the study and are reffered to 
as study parameters. The study will be used to assess the effects of the study 
parameters on the study objectives. Objectives are added to a study 
through the study :meth:`~matcal.core.study_base.StudyBase.add_evaluation_set` 
method. An evaluation set is a set of components needed to evaluate 
an objective for the study and consists of a model, the objective used 
to compare the model results to calibration data, and the data that the model is being 
calibrated to.  Multiple evaluation 
sets can be added to a study for more complex studies
requiring comparisons between multiple experiments each requiring different models, 
different objectives for comparing different data sets or QoIs to results generated from a single model, or 
different combinations of states, QoI extractors and weights for a given data set and model. 
The ability to add multiple evaluation sets provides great flexibility for specifying  a final 
combined objective, so if you are unsure if MatCal can support your needs please contact us.
After initializing the study and adding evaluation sets, most studies are ran using the 
:meth:`~matcal.core.study_base.StudyBase.launch` method which then returns the 
results. The results can be stored and manipulated as its own either in the remaining 
lines of the MatCal input or in a separate python file where the results are imported and 
manipulated.

.. warning::
    We currently return different results objects based on the specific study being
    performed. We plan on updating this in the future for consistency, but backward
    compatibility is not guaranteed. The "\*.serialized" files are backward compatible, 
    and those results should always be accessible. See :ref:`Results Data and Output`
    for more information on study results.

Common attributes and methods for all studies can be found in the :ref:`User API Documentation`
under :class:`~matcal.core.study_base.StudyBase`. Besides the essential methods listed above 
and the aforementioned common features, each study may have their own 
specific methods and attributes. These should be accessed in the :ref:`User API Documentation`
under the specific study being used. Currently, the studies available in MatCal can be found
in the following module documentation pages:

#. Pure MatCal studies: :mod:`~matcal.core.parameter_studies`
#. Dakota local calibration studies: :mod:`~matcal.dakota.local_calibration_studies`
#. Dakota global calibration studies: :mod:`~matcal.dakota.global_calibration_studies`
#. Dakota sensitivity studies: :mod:`~matcal.dakota.sensitivity_studies`
#. Dakota uncertainty quantification studies: :mod:`~matcal.dakota.uncertainty_quantification_studies`







Currently, MatCal performs calibrations using 
interfaces to optimization algorithms
provided by Dakota :cite:`dakota` and SciPy :cite:`scipy`.
Using Dakota requires an installation with links to the python
you will use for MatCal. Using SciPy requires a standard install of the 
SciPy package for the python version you will use for MatCal.

We have 
interfaces to three classes of optimization algorithms through 
Dakota:

#. Gradient-based local optimization algorithm (:class:`~matcal.dakota.local_calibration_studies.GradientCalibrationStudy`)
#. Gradient-free local optimization algorithms (:class:`~matcal.dakota.local_calibration_studies.CobylaCalibrationStudy` 
   , :class:`~matcal.dakota.local_calibration_studies.MeshAdaptiveSearchCalibrationStudy` 
   , :class:`~matcal.dakota.local_calibration_studies.ParallelDirectSearchCalibrationStudy` 
   , :class:`~matcal.dakota.local_calibration_studies.PatternSearchCalibrationStudy` 
   , :class:`~matcal.dakota.local_calibration_studies.SolisWetsCalibrationStudy`)
#. Gradient-free global optimization algorithms (:class:`~matcal.dakota.global_calibration_studies.SingleObjectiveGACalibrationStudy` 
   , :class:`~matcal.dakota.global_calibration_studies.MultiObjectiveGACalibrationStudy`)

Generally, the 
:class:`~matcal.dakota.local_calibration_studies.GradientCalibrationStudy` 
is best used for simple calibrations such as 
calibrations with a single model and a low number of states. If the calibration
is well suited for a gradient-based algorithm, the 
:class:`~matcal.dakota.local_calibration_studies.GradientCalibrationStudy` 
will work quickly and be the least 
computationally expensive requiring usually 
fewer than a coupled hundred objective 
function evaluations. It also takes advantage 
of parallelism and concurrent function evaluations
for the finite difference derivatives that it 
takes of the objectives. If the calibration gets more 
complicated with the addition of more models or 
inclusion of several states, consider using a gradient-free algorithm for
the calibration. The gradient-free calibrations 
tend to be more robust for complex objective functions with 
several local minima. When calibrating to 
multiple models and states, there tend to be 
as many local minima as there are models and 
states since model form error makes
fitting all data sets perfectly with a single 
parameter set unlikely. The local gradient-free algorithms  
tend to take longer than the 
:class:`~matcal.dakota.local_calibration_studies.GradientCalibrationStudy` and 
require a moderate number of objective function evaluations which can be on 
the order of a few hundred to a few thousand depending 
on the calibration. These algorithms may or may not allow for evaluation 
concurrency due to the algorithm itself or limitations 
in Dakota's interface to them. See the 
`Dakota documentation <https://dakota.sandia.gov/documentation.html>`_ 
for more detail. However, they are generally more robust
for complex objectives than the 
:class:`~matcal.dakota.local_calibration_studies.GradientCalibrationStudy`. 
The most robust algorithms are the gradient-free global optimization algorithms. 
However, they 
are also the most expensive requiring thousands to tens of thousands of 
objective function evaluations. These global
algorithms do allow for evaluation concurrency and can run dozens 
of objective function evaluations concurrently depending
on algorithm settings. In closing, the choice of optimization algorithm 
depends greatly on the calibration specifics. 
Look through MatCal's examples to find a calibration similar to yours to 
help decide what algorithm and objectives 
you should use.

Below we have highlighted MatCal features and examples
that may be of use and help you get started:

#. Python models (:class:`~matcal.core.models.PythonModel`) 
   - these models use user defined python functions to act as the simulation.
#. Uniaxial loading material point model 
   (:class:`~matcal.sierra.models.UniaxialLoadingMaterialPointModel`) -
   MatCal generated SIERRA/SM material point model for uniaxial compression and tension.
#. Round uniaxial tension model 
   (:class:`~matcal.sierra.models.RoundUniaxialTensionModel`) - 
   MatCal generated SIERRA/SM round tension model.
#. Rectangular uniaxial tension model 
   (:class:`~matcal.sierra.models.RectangularUniaxialTensionModel`) - 
   MatCal generated SIERRA/SM rectangular tension model.
#. A brief introduction to object-oriented python which 
   is important for understanding MatCal input: 
   :doc:`introduction_examples/a_python_primer`
#. A simple python model calibration introduction example: 
   :doc:`introduction_examples/plot_basic_example_walk_through`
#. An in-depth SIERRA/SM material point model calibration 
   introduction example: :ref:`SIERRA/SM Material Point Model Practical Examples`
#. A simple user defined model calibration/sensitivity 
   study introduction example: :ref:`SIERRA User Defined Model Studies`