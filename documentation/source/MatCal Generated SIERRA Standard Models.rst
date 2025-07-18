MatCal SIERRA Solid Mechanics Standard Models
---------------------------------------------
All available MatCal SIERRA/SM models are listed here
with links to their specific documentation. 

.. toctree::
   :maxdepth: 1

   UniaxialLoadingMaterialPointModel
   UniaxialTensionModels
   RoundNotchedTensionModel
   SolidBarTorsionModel
   TopHatShearModel
   VFMUniaxialTensionModels
   
Several features are common to the MatCal SIERRA/SM standard models. 
These include:

#. Full model generation including geometry, mesh and input deck creation. 
   A suite of options control what the discretized geometry
   and simulation input deck include.
#. Simple thermomechanical coupling through iterative 
   or staggered coupling in SIERRA/Arpeggio or through 
   adiabatic heating in SIERRA/SM.
#. Support for user specified exodus element and 
   nodal variable output.
#. Support for element death according to user specified 
   death criterion. Nonlocal failure can also be used in 
   conjunction with element death.
#. User specified time step and start/end time controls
   to override defaults.
#. Support for both implicit quasi-statics and implicit 
   dynamics simulations. 
#. Support both the SIERRA/SM default element 
   (under-integrated 8 node hexahedral with hourglass control)
   and the total lagrange 8 node hexahedral element. MatCal's defalt behavior is to use the total lagrange element.
#. The ability to generate the model files for a given state
   without running the model using the 
   :meth:`~matcal.sierra.models.UniaxialLoadingMaterialPointModel.preprocess`
   method. This is useful to verify the simulation files created, if desired, and for 
   model archiving.
#. The ability to run the model for a given state and 
   a valid parameter collection using the 
   :meth:`~matcal.sierra.models.UniaxialLoadingMaterialPointModel.run`
   method. 

Standard Model Geometry and Input Generation
--------------------------------------------
The model generation for MatCal standard models uses 
a similar procedure for all models. This process includes
standard model object construction/instantiation, adding data to the 
model from which to extract boundary conditions, and  
setting model options. 

Model object instantiation
^^^^^^^^^^^^^^^^^^^^^^^^^^
To begin, you instantiate the desired
class for the characterization experiment you need to simulate for 
calibration. The instantiation of all standard 
model classes takes a :class:`~matcal.sandia.material.Material`
object and a list of geometric parameters. All required geometric 
parameters for a standard model must be provided upon
instantiation. Any missing parameters will result in an error 
that reports the missing parameter. It is easiest to build
a dictionary of the geometry parameters and pass that 
into the standard model constructor using Python's
keyword argument unpacking feature for dictionaries. 

Boundary condition generation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
After the standard model class has been constructed, 
the boundary condition data can be passed to the model 
using the 
:meth:`~matcal.sierra.models.UniaxialLoadingMaterialPointModel.add_boundary_condition_data`
method. This method takes a :class:`~matcal.core.data.DataCollection` or
:class:`~matcal.core.data.Data` object with the fields required for boundary
condition generation specific to the model being built.
The model will use this data to determine its boundary 
conditions by state since maximum deformation, 
material behavior and experiment setup can vary significantly over different states.
The boundary condition isn't calculated until needed, so the 
:meth:`~matcal.sierra.models.UniaxialLoadingMaterialPointModel.add_boundary_condition_data`
method can be called multiple times if needed. In general,
the model will select the data from all added data sets that would 
result in the largest deformation in the model for a given state. For example, 
if multiple repeat data sets were given for a tension specimen
at a given state, the model would select the data set with the largest
engineering strain for that state and attempt to deform the specimen 
until its gauge section experiences that level of engineering strain. 

A model option exists to scale this automatically determined boundary 
condition if so desired. The 
:meth:`~matcal.sierra.models.UniaxialLoadingMaterialPointModel.set_boundary_condition_scale_factor`
will scale the maximum boundary condition field by the specified scale factor, 
and increase the maximum deformation applied to the model. 
This can be useful in cases where you want the model to
produce data that goes beyond and fully encapsulates the 
experimental data for the calibration. Such a case could be calibrating failure parameters 
using a tension specimen model where one calibration QoI would be the failure displacement. 
In this case, you would want the model
to run further than the largest displacement provided in the experimental data. 
This is desired because you want the calibration to identify when the model fails with 
too large of a displacement. In order to accommodate this, the model must run to some factor beyond the 
provided experimental data maximum displacement.  
It is also useful when the deformation 
in a model is measured across some subsection of the model 
away from where the boundary condition is applied. 
If significant deformation occurs outside of the subsection where deformation is measured, 
then the scale factor can be used to compensate for errors in the boundary condition 
generation.

Available simulation features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Several methods exist for all standard models to control 
simulation features and behavior.
All MatCal generated SIERRA/SM standard models 
use the same base input decks, so the options generally 
implement the same behavior across all models except for a 
few cases where the options are not valid. The only input deck differences
for different MatCal SIERRA/SM standard models are boundary conditions, 
model specific output
and the finite element model geometry.

.. note::
    Some of the following methods may be modified by specific MatCal SIERRA/SM
    standard model classes  
    to have slightly different behavior. Always check the method 
    documentation for the model being used. 

#. The :meth:`~matcal.sierra.models.RoundUniaxialTensionModel.activate_element_death`
   method can be used to activate element death with or without nonlocal averaging. It accepts
   four input arguments depending on the desired behavior: (1) *death_variable*, (2) *critical_value*, 
   (3) *nonlocal_radius*, and (4) *initial_value*.
#. The :meth:`~matcal.sierra.models.RoundUniaxialTensionModel.activate_thermal_coupling`
   method activates adiabatic heating or staggered thermomechanical coupling based on the passed 
   arguments to the method. For more detail on this feature 
   see :ref:`Staggered and iterative coupling`.
#. The :meth:`~matcal.sierra.models.RoundUniaxialTensionModel.use_iterative_coupling`
   method changes the thermomechanical coupling from staggered coupling to iterative coupling.
   Note that this method is only valid after staggered coupling has been activated. Once again, see 
   :ref:`Staggered and iterative coupling` for more details.
#. The :meth:`~matcal.sierra.models.RoundUniaxialTensionModel.set_allowable_load_drop_factor`
   method allows the user activate solution termination for the simulation after the load field 
   for the model has decreased by 
   the specified fraction. This will allow the model to terminate before the predetermined simulation 
   end time after a desired amount of unloading has occurred. Usually for solid mechanics,
   the model only needs to unload a certain amount for all the desired data to be obtained 
   for a study. Using this method allows all necessary data to be output from the model without wasted computational 
   expense. 
#. The :meth:`~matcal.sierra.models.RoundUniaxialTensionModel.set_start_time`, 
   :meth:`~matcal.sierra.models.RoundUniaxialTensionModel.set_end_time`, and
   :meth:`~matcal.sierra.models.RoundUniaxialTensionModel.set_number_of_time_steps`
   methods are used to set model time control settings. See :ref:`Simulation time control settings`.
#. The :meth:`~matcal.sierra.models.RoundUniaxialTensionModel.use_under_integrated_element`
   changes the model element type from the total lagrange element with volume averaged pressure 
   to the SIERRA/SM default under integrated hexahedral 8 node element with hourglass control.
#. The :meth:`~matcal.sierra.models.RoundUniaxialTensionModel.activate_implicit_dynamics`
   allows the simulation to be run with implicit dynamics instead of implicit quasi-statics.
#. The :meth:`~matcal.sierra.models.RoundUniaxialTensionModel.activate_exodus_output`, 
   :meth:`~matcal.sierra.models.RoundUniaxialTensionModel.add_element_output_variable`, and
   :meth:`~matcal.sierra.models.RoundUniaxialTensionModel.add_nodal_output_variable`
   methods control the simulation exodus output. See :ref:`Simulation exodus output and settings`.

Simulation Coordinate Systems Available to the Material Model
-------------------------------------------------------------
Anisotropic material models generally require that the 
material directions be defined relative to a reference coordinate system.
MatCal currently supports two reference coordinate systems for all standard models.
There is a rectangular coordinate system that is aligned with the 
global Cartesian coordinate system that is named *rectangular_coordinate_system* in 
the SIERRA namespace. There is also a cylindrical coordinate system with the 
cylindrical axis aligned with the global Y direction that is named *cylindrical_coordinate_system*
in the SIERRA namespace. The local cylindrical coordinate system 
is shown in :numref:`cylindrical_coordinate_system`.

.. _cylindrical_coordinate_system:
.. figure:: figures/cylindrical_coordinate_system.*
   :scale: 15%

   The local *cylindrical_coordinate_system* is shown at two different points 
   on a cylinder. The axis is aligned with the global Y direction. The radial 
   direction is defined from the global Cartesion origin through the point of interest
   which is typically a node or inegration point. The tangent direction is 
   defined as the cross product of the radial and axis directions. Each local 
   coordinate system is created through points of 
   interest in the model's deformed configuration.


Note that the :ref:`Uniaxial Tension Models`, 
:ref:`Round Notched Tension Model`, :ref:`Top Hat Shear Model`, 
and :ref:`Solid Bar Torsion Model` are loaded such that the 
reaction force and displacements are positive in the
Y direction. For the :ref:`Solid Bar Torsion Model`, this is a rotation 
and torque about the Y axis. The :ref:`Uniaxial Loading Material Point Model`
is loaded such that the 
reaction force and displacements are positive in the Z direction.

Simulation time control settings
--------------------------------
Several options exist for controlling time behavior of the simulations. 
The :meth:`~matcal.sierra.models.RoundUniaxialTensionModel.set_number_of_time_steps`
method sets the target number of time steps for the simulation over 
the entire anticipated time history. However, for simulation 
robustness, adaptive time stepping is always active. Also, the simulation may end early 
due to one of the other simulation features such as 
:meth:`~matcal.sierra.models.RoundUniaxialTensionModel.set_allowable_load_drop_factor`.
As a result, this is viewed as the target time step, but usually the model will not 
take the exact number of time steps specified. The default target number 
of time steps is 300 steps.

By default, the simulation start and end time are determined depending on 
the data passed to the model for boundary condition generation. If no "time" 
field is present, the start time is zero and the end time is either one 
or the time needed to apply the desired deformation with the appropriate 
deformation rate. If the "time" field is present, the start time is 
the first time value available in the data, and the end time is the last time 
value available in the data. The simulation then targets the specified 
number of time steps for the model. These start and end times 
can be modified with the 
:meth:`~matcal.sierra.models.RoundUniaxialTensionModel.set_start_time` and 
:meth:`~matcal.sierra.models.RoundUniaxialTensionModel.set_end_time`
methods. 

.. note::
   The :meth:`~matcal.sierra.models.RoundUniaxialTensionModel.set_start_time` and 
   :meth:`~matcal.sierra.models.RoundUniaxialTensionModel.set_end_time`
   methods only affect the simulation time and will not change the maximum target deformation for the model. 
   However, if the end time is increased
   past the calculated end time required to reach the target deformation based on the supplied
   boundary condition data, then the final displacement is held constant for the rest of the 
   simulation. This can be useful if attempting to simulate stress relaxation. Additionally, if a complex loading
   function is supplied to the model as the boundary condition data, these methods can be used to run 
   a subset of that loading history.

Simulation global output and access
-----------------------------------
Most global results output from a model are specific to 
the model being run; however, all models output the "time" field. 
This global data can be accessed as a :class:`~matcal.core.data.Data`
or :class:`~matcal.core.data.DataCollection` depending on how
the model is run. If the model is run using the 
:meth:`~matcal.sierra.models.RoundUniaxialTensionModel.run` method,
a :class:`~matcal.core.simulators.SimulatorResults` class is returned. 
The model results :class:`~matcal.core.data.Data` object is then accessed through the 
SimulatorResults.results_data attribute. The SimulatorResults output also 
contains the standard output, standard error and the exit code for the simulation. 
If the results are obtained from a study, they will be 
stored in a :class:`~matcal.core.data.DataCollection` that must be 
extracted from the study results return as a dictionary of 
:class:`~matcal.core.objective_results.ObjectiveResults` objects.

Simulation exodus output and settings
-------------------------------------
By default, the MatCal SIERRA/SM standard models
do not write exodus output in order to save disk space. 
However, the user can activate user output with the
:meth:`~matcal.sierra.models.RoundUniaxialTensionModel.activate_exodus_output`
method. By default, the models will output exodus data every 20th 
time step. This interval can be adjusted by passing the desired interval 
to the method. Element output by default only includes the following:

#. avg_pressure
#. avg_vm
#. avg_log_strain
#. avg_temperature (if temperature is initialized and used in the model
   through any of the MatCal generate model temperature options.)

where the "avg\_" prefix indicates it has been averaged for the element so 
that individual integration point values are not output. Additional 
element variables are output based on chosen model options. 
If failure is activated the following variables are added to the element 
output:

#. death_status
#. avg\_\ *death_variable_name*

where *death_variable_name* is the user specified death criterion variable name.
For loosely coupled models, the user specified *plastic_work_variable* is 
also output. 

Nodal output includes "displacement" which is the displacement vector of each node and 
the "temperature" which is the nodal temperature that is only output
for models with conduction. 

Exodus output is found in the model/state directory where the model is run. 
With a study the path to these results is "matcal_workdir.#/model_name/state_name/results/results.e".
When using :meth:`~matcal.sierra.models.RoundUniaxialTensionModel.run`, the results will be in the 
current directory or the target directory instead of the "matcal_workdir.#" directory. 
MatCal SIERRA/SM standard models do not combine parallel results.

Additional exodus output can be added with the 
:meth:`~matcal.sierra.models.RoundUniaxialTensionModel.add_element_output_variable` and
:meth:`~matcal.sierra.models.RoundUniaxialTensionModel.add_nodal_output_variable`
methods where the passed arguments are a comma separated list of desired output variable names.
No checks are performed on these arguments to ensure they are valid in SIERRA/SM, so 
model errors could result from incorrectly passed variable names. Also, currently 
these variables are not volume averaged and will be output for each integration point.

Staggered and iterative coupling
--------------------------------
We have included that ability to easily activate staggered or iterative 
thermomechanical coupling with all MatCal generated SIERRA/SM models. For 
large deformation at intermediate rates, thermomechanical coupling
provides the only way to accurate predict structural softening 
due to the heating associated with plastic work. The following boundary 
conditions, initial conditions and assumptions are used:

#. We assume the experiment was run in such a way that convection is negligible, and 
   temperatures are low enough that radiation is also negligible. As a result,
   only conduction through the body is important.
#. The grips are considered an infinite heat sink with perfect conductivity at the interface 
   between the specimen and the load frame grips. The specifics of this interface depends on 
   the specific model being used, and the boundary condition is simulated by applying
   a Dirichlet boundary condition at this interface. The temperature of the nodes at this interface
   are set to the value of the :class:`~matcal.core.state.State` parameter 
   "temperature" if provided. The "temperature" :class:`~matcal.core.state.State` parameter 
   *should* always be set by the user. However, this is not strictly enforced, 
   and these models have a default value of 100 set for the temperature. If 
   a :class:`~matcal.core.state.State` parameter temperature is not set by the user, a warning is sent 
   to MatCal output stating the default value of 100 is being used. 
#. All other surfaces have no thermal boundary conditions, and, as a result, are perfectly insulated.
#. The initial temperature of the body is set to the 
   "temperature" :class:`~matcal.core.state.State` parameter value.

Thermal coupling is activated by the 
:meth:`~matcal.sierra.models.RoundUniaxialTensionModel.activate_thermal_coupling`
method. The behavior of the method is the same for all MatCal generated standard models except the 
:class:`~matcal.sierra.models.UniaxialLoadingMaterialPointModel` model which only supports 
adiabatic heating. When the method is called, the user must supply the required thermal properties, 
which include density, thermal conductivity, and specific heat. They must also 
supply the name of the element variable where the plastic work rate is stored. If none of these 
are supplied to 
:meth:`~matcal.sierra.models.RoundUniaxialTensionModel.activate_thermal_coupling`,
then the model is assumed to be adiabatic and the command only changes the output from the model.
If a subset of the thermal properties is provided, MatCal will exit with an error.

.. note::
   The Taylor-Quinney coefficient is not needed in the thermal model 
   and should be in the solid mechanics material 
   input deck.

By default, MatCal uses staggered coupling if thermomechanical coupling with conduction 
is requested through the :meth:`~matcal.sierra.models.RoundUniaxialTensionModel.activate_thermal_coupling`
method. Staggered coupling behaves as follows:

#. After initialization, the solid mechanics region is advanced for the current timestep.
#. Next, the displacements, death status variable (if applicable), and volumetric plastic work 
   rate variable is copied to the thermal model mesh.
#. The thermal model is then advanced to calculate the temperature field in the body
   due to the plastic work converted to heat.
#. Finally, the updated temperature is transferred to the solid mechanics region before
   continuing to the next time step.

No iteration occurs to reduce a residual for staggered coupling. If desired, 
iterative coupling, which repeats the sequence above until the initial 
residual for the thermal region is below 1e-8, can be activated using
:meth:`~matcal.sierra.models.RoundUniaxialTensionModel.use_iterative_coupling`. 
The need to use iterative coupling over staggered is problem dependent. However, 
staggered coupling is generally sufficient for most material calibrations for metals. 

We have verified MatCal's use of SIERRA's Arpeggio thermomechanical coupling 
against SIERRA/SM's adiabatic heating capability. 
See :ref:`sphx_glr_matcal_model_v_and_v_plot_coupling_verification.py`.
To verify the coupled simulations, results from coupled
simulations where the thermal conductivity is set to zero are compared to 
results produced using a SIERRA/SM adiabatic simulation. The temperature rise calculated 
in SIERRA/SM has been verified as shown in the Lame manual :cite:p:`lame_manual` and is expected to 
be accurate.
Currently, the results 
converge with decreasing simulation time step size and match the adiabatic 
case with relative low error with sufficient temporal resolution.

.. warning::
   We are actively working with the SIERRA code teams to 
   improve the coupling capability and reduce its computational cost. 
   This may result in small changes to the models and solutions across
   MatCal versions.

More work is planned for verification and 
validation of these features which will be completed and referenced 
in future releases of the MatCal documentation. 
Additionally, validation has been performed on the coupling method 
for a 304L calibration for the Ductile Failure project. Predicted temperatures
for several tension specimens agreed well with 
experimentally measured temperatures. 

Overriding model parameters during a study
------------------------------------------

A useful feature for these SIERRA/SM standard models
is that some of the optional parameters can 
be set using  
:class:`~matcal.core.state.State` parameters
or model constants. The following parameters can be overridden:

#. Geometry parameters can be overridden during study initialization 
   using :class:`~matcal.core.state.State` parameters or model 
   constants. 
#. The model initial temperature and grip Dirichlet boundary 
   condition temperature can be set using the model constant or 
   :class:`~matcal.core.state.State` parameter named "temperature".
   We recommend using a state parameter for temperature.

.. note::
   Model constants can be set for all states using 
   :meth:`~matcal.sierra.models.RoundUniaxialTensionModel.add_constants`
   or by state using 
   :meth:`~matcal.sierra.models.RoundUniaxialTensionModel.add_state_constants`.
   The latter of which is useful when changing simulations model options by state, such 
   as time stepping, coupling, or even specific options within your custom material 
   model file.

.. include:: matcal_model_v_and_v/index.rst
   :start-after: :orphan:   

