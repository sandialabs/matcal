****************************************
Simple Shear Material Point Model
****************************************
MatCal's :class:`~matcal.sierra.models.SimpleShearMaterialPointModel`
is meant to be used in calibrations that can use a material point 
model subject to simple shear loading as the simulation of the experiment. 
This can be a valid model for experiments with simple shear loading conditions,
such as hat-section shear tests or direct shear tests, where the deformation 
is dominated by shear and localization effects are not being characterized.
The :class:`~matcal.sierra.models.SimpleShearMaterialPointModel` model has most of the MatCal standard 
model features as described in :ref:`MatCal SIERRA Solid Mechanics Standard Models`. Due to 
the local nature of the model, only adiabatic thermomechanical loading is supported, 
and implicit dynamics is not supported. 
In this section, we will provide more information about how the geometry is generated, 
specifics on simulation boundary conditions, and what is output from the model.

.. note::
   This model follows the same patterns and examples as the
   :ref:`Uniaxial Loading Material Point Model`. Additional examples can be found at
   :ref:`SIERRA/SM Material Point Model Practical Examples`

Material point geometry and mesh generation
===========================================
The MatCal generated material point model is simulated as a unit cube
and modeled using a single hexahedral element. No user input parameters 
are supported or required for geometry generation and discretization. 

Simple shear material point boundary conditions
===============================================
The simple shear material point model has boundary conditions
applied in order to load the hexahedral element in a simple shear state.
The boundary conditions are shown graphically in :numref:`simple_shear_material_point_bcs`.

.. _simple_shear_material_point_bcs:
.. figure:: figures/simple_shear_material_point/simple_shear_material_point_model_bcs.*
   :scale: 15%

   The single element model and boundary conditions for the
   simple shear material point model.

In summary, they include:

#.  Fixed displacement boundary conditions on the bottom surface (ns_negative_z) in all directions (X, Y, Z).
#.  Fixed displacement boundary conditions on the top surface (ns_positive_z) in the Y and Z directions,
    preventing motion perpendicular to the shear direction.
#.  An applied displacement function in the X direction that varies with time to the nodes on the 
    top surface of the element. The function is determined based on the information in 
    :ref:`Simple shear displacement function determination`.

The resulting deformation imposes a simple shear state where the top surface displaces in the 
X direction while the bottom surface remains fixed, creating a shear deformation in the XZ plane.

Simple shear displacement function determination
------------------------------------------------
The applied displacement function is calculated for the model based on data supplied to the 
:meth:`~matcal.sierra.models.SimpleShearMaterialPointModel.add_boundary_condition_data`
method. 
This method must be supplied a :class:`~matcal.core.data.Data` or 
:class:`~matcal.core.data.DataCollection` class that contains 
an "engineering_strain" field for the 
states of interest for the model. The "engineering_strain" field represents the 
engineering shear strain (γ), which for a material point with unit height equals the 
displacement in the shear direction. The data can also optionally include 
a "time" field. The 
:meth:`~matcal.sierra.models.SimpleShearMaterialPointModel.add_boundary_condition_data` 
method determines the boundary condition function to be applied 
to the model according to the following 
algorithm:

#. Determine the boundary condition by state since maximum deformation, 
   material behavior and experiment setup can vary significantly over different states.
#. For each state, find the data set with the largest shear strain and use it for 
   boundary condition generation.
#. The engineering shear strain data is used directly for the boundary condition generation.
   Since the mesh is a unit cube with unit height, the engineering shear strain (γ = displacement/height)
   can be applied directly to the model as a displacement boundary condition in the X direction
   to achieve the correct shear deformation.
#. If the data does not contain a "time" field and there is *not* a :class:`~matcal.core.state.State`
   parameter named "strain_rate", then apply a linear displacement function from 
   zero to the maximum engineering shear strain found in the data over one second.
#. If the data does not contain a "time" field and there *is* a :class:`~matcal.core.state.State`
   parameter named "strain_rate", then apply a linear displacement function from 
   zero to the maximum engineering shear strain found in the data. This is done over a time period
   beginning at zero seconds and ending at a time calculated by dividing 
   the maximum engineering shear strain by the "strain_rate" :class:`~matcal.core.state.State`
   parameter.
#. If the data does contain a "time" field, use the function directly as provided for  
   the "engineering_strain" field.

.. note::
    Cyclical shear loading can be modeled with the simple shear 
    material point model by supplying strain/time data to the 
    :meth:`~matcal.sierra.models.SimpleShearMaterialPointModel.add_boundary_condition_data` 
    method. This can be useful when modeling shear stress relaxation and reloading, or
    hysteresis under shear. Note that when using the :class:`~matcal.core.objective.CurveBasedInterpolatedObjective`
    for complex loading cycles, you may need to use "time" as the independent 
    variable and "engineering_stress" or "true_stress" as the dependent variables because
    it requires monotonically increasing independent variables for interpolation.

.. warning::
    The simple shear material point model requires that stress and strain values
    be negative for reversed shear (negative displacement in X direction) and positive 
    for forward shear tests. Not abiding by this general rule may result in invalid 
    studies even if the models run and studies complete.

Material point thermal model boundary conditions
------------------------------------------------
For a material point, 
only adiabatic heating is supported using the 
:meth:`~matcal.sierra.models.SimpleShearMaterialPointModel.activate_thermal_coupling` method.
When using adiabatic heating, the entire body 
of the model is prescribed an initial temperature of  
:class:`~matcal.core.state.State` parameter 
"temperature". For uncoupled simulations, the model is given a prescribed
temperature of :class:`~matcal.core.state.State` parameter 
"temperature" if provided.

Simple shear material point model specific output
=================================================
By default, the simple shear material point model includes the following global 
output fields: 

#. time
#. displacement - measured in the X direction (shear direction) at the nodes with the applied displacement function
#. load - measured at the applied boundary condition nodes in the X direction (shear force)
#. engineering_strain - engineering shear strain, same as displacement for unit height
#. engineering_stress - engineering shear stress, same as load for unit area
#. true_strain - the log strain of the element in the XZ direction (primary shear component)
#. true_stress - the Cauchy stress of the element in the XZ direction (primary shear component)
#. temperature - the element temperature (if thermal coupling is active)
#. log_strain_xx/yy/zz - the log strain of the element in the normal directions
#. cauchy_stress_xx/yy/zz - the Cauchy stress of the element in the normal directions

All element values are output from the average of all values at element integration points. 
For the simple shear case, these should be equal, but averaging the values simplifies 
output for the different element types supported by MatCal's SIERRA/SM generated models.

The simple shear model outputs both the primary shear components (XZ) and all normal 
components (XX, YY, ZZ) to allow full characterization of the stress and strain state 
during shear deformation. This is particularly useful when calibrating complex material 
models that exhibit coupling between shear and normal stress/strain components.
