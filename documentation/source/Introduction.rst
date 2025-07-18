************
Introduction
************

Any continuum mechanics model will require three components: 
(1) a discretized geometry of the boundary value problem being studied,
(2) the partial differential equations to be solved, and 
(3) the initial conditions and boundary conditions for the problem. 
To describe material behavior in these computational models,
material models contribute to (2) the underlying 
equations and, occasionally, to (3) the initial conditions for the simulation. 
These material models can exhibit a mathematical form that 
is empirically based, based on first principles, or
developed from both empirical observations and known physics. 
In general, these models are meant to represent a class of materials
with well understood behavior. As a result, material models have 
parameters that must be tuned or calibrated so that the model response matches 
characterization 
data available for the specific material it is intended to represent when used to 
simulate a specific system. For simple models, such
as isotropic, linear elastic materials in solid mechanics, this calibration 
process can be a simple 
analytical calculation directly extracting the parameters from experimental 
measurements. For complex models
that have many inputs and require many characterization datasets to adequately 
identify the material behavior, the model calibration process can require an inverse
problem approach where an optimization is performed to tune the model parameters to 
the available data. For a complete calibration, the initial deterministic calibration 
of the parameters should be followed by an uncertainty quantification study to 
quantify the uncertainty of the calibrated model parameters. Also, sensitivity studies  
may aid in model form choice before calibration or understanding calibrated model 
behavior post calibration. These uncertainty quantification and sensitivity study 
steps are considered part of the model calibration process.

General MatCal Overview
=======================

MatCal, short for "material calibration", is designed to simplify calibrations and 
provide a standardized and verified tool for calibrating material models using the inverse 
problem approach. A simplified overview of optimization for the inverse problem 
approach for calibration can be found at :ref:`Optimization Overview` in :ref:`Calibration`.
It would also be good to review books on the subject 
such as :cite:p:`luenberger1984linear` and :cite:p:`sun_model_cal` for those 
interested in more details and theory.
In practice, the material calibration process usually consists of 5 steps:

#. Performing a literature survey to understand the material to be simulated and 
   the phenomena it will exhibit.
#. Acquiring material data either through literature surveys or by performing material 
   characterization experiments designed to measure quantities 
   relevant to the phenomena identified in the previous step.
#. Selecting a material model or multiple model forms to represent the material
   responses measured in the previous step.
#. Preparing and executing the inverse problem to determine the nominal 
   parameter values and their uncertainty.
#. Validation of the calibrated material model and parameters to a separate 
   experiment or set of experiments.

MatCal is meant to assist in (3) and (4) by providing a clean and verified interface
to algorithms that study the relationship between a model's input parameters and a 
quantitative metric calculated from model results. In MatCal, these algorithms 
are wrapped by our MatCal "study" classes, and we provided different studies 
for sensitivity analysis, calibration and uncertainty quantification. See :ref:`MatCal Studies`
for a general overview of MatCal's studies. Our studies support three different 
types of input parameters which are explained in :ref:`Parameter Types and Specification`.
All of these studies require a quantitative metric to be returned from the model
for a given set of input parameters. Generally, 
we refer to this metric as an objective, and objectives are intended to quantitatively 
compare the model 
response to some reference data. See :ref:`MatCal Objectives` and :ref:`Full-field Data Tools`
for more information on objectives that MatCal can calculate for our studies.
Another way that MatCal simplifies the calibration process is by 
providing tools for objective function 
formulation and normalization. See :ref:`MatCal Objective Calculations`. 
Models used in MatCal studies are treated as black-box models.  
We support several model interfaces including our :class:`~matcal.core.models.PythonModel`, 
:class:`~matcal.core.models.UserExecutableModel`, :class:`~matcal.core.models.MatCalSurrogateModel` 
and :ref:`MatCal Generated SIERRA Standard Models`. 
See :ref:`MatCal Models`. 
Finally for results visualization, we provide plotting utilities so that users can 
visualize the calibrated model 
results against the characterization data, and plot objective results 
against input parameter values and/or 
the evaluation number. See :ref:`Results and Plotting`

MatCal provides some specific features that make it 
particularly useful for complex material model calibrations. 
This includes the ability to calibrate a single material 
model to multiple characterization tests with 
different experimental conditions, referred 
to as states (:mod:`~matcal.core.state.State`), concurrently.
For example, if calibrating an anisotropic solid 
mechanics material model that is rate and 
temperature dependent, the user can calibrate the 
material model to tension and shear tests at 
different rates, temperatures and loading directions 
relative to a material direction. In this example, 
the user would specify two models (tension and shear) 
with multiple states, where the states would account for the experimental
conditions of rate, temperature and loading direction. MatCal 
will run the simulations in parallel based on computational 
limitations, and then collate, scale and normalize the 
residuals and objectives appropriately to provide a calibration 
that matches all data as best as possible. It does 
this by attempting to make all objective entries equal 
weight regardless of data length, magnitude or number 
of repeats at a given state. The specifics of this process
can be found in :ref:`MatCal Objective Calculations`.
The user also has the option to influence the 
calibration by specifying different 
quantity of interest extractors (QoI extractors, 
:mod:`~matcal.core.qoi_extractor`) for each objective. 
The QoI extractors operate on the simulation 
and/or the experimental data and 
produce the specific quantities that will be 
compared to provide a residual. Furthermore, the user 
can weight certain datasets or features in datasets 
more than others using MatCal's 
weighting features. See the weighting objects in 
:mod:`~matcal.core.residuals`. All of these features provide the user with 
the tools needed to appropriately specify the objective 
needed for their calibration goals.

With the ability to calibrate to multiple models concurrently, 
MatCal also provides job management for the simulations 
to be used in the calibration. These tools allow for parallel execution of models, 
with studies that support it, to speed calibration activities without 
over-prescribing computational resources.
See :ref:`Computer Resources and Job Management`.

The chapter :ref:`MatCal Features and Objects` will go over the primary
MatCal features and objects
that are necessary to perform calibration activities. This includes:

#. A basic overview of :ref:`MatCal Studies` types and interfaces. In depth overviews can 
   be found in subsequent chapters in this documentation. 
#. :ref:`Parameter Types and Specification` covers the three parameter types we support, 
   how to use them and how they are passed to models.
#. :ref:`MatCal Objective Calculations` covers how we calculate objective values 
   including normalization and scaling we automatically perform.
#. :ref:`MatCal Objective Tools` covers the basic objective tools that are available and 
   how to use them to customize objective functions.
#. :ref:`MatCal Models` covers the different model interfaces we support in MatCal 
   and how to use them to run your models.
#. :ref:`Data Importing and Manipulation` covers the data format MatCal can 
   import for both experimental data and model results 
   from external executable or python models.
#. :ref:`Results and Plotting` discusses how results are output and how to 
   do basic plotting to monitor calibration progress or visualize study 
   results once they finish.

MatCal as a Python Package
==========================
MatCal is written as a python package. As a result, 
MatCal's "input files" are just python code files. 
This was done to avoid writing a parser, allow users to 
use python to manipulate MatCal tools with python tools 
and allow users to easily mix MatCal tools with other calibration tools
readily available from other python packages such as SciPy :cite:`scipy`
and Scikit-Learn :cite:`scikit-learn`. 
We attempted to make our user facing tools easy to use at a basic level
so that they could be usable by users not well versed in python. However, 
since it is based on python and written using object oriented code a basic 
understanding of python and object oriented programming is needed. 


INCLUDE PYTHON PRIMER EXAMPLE