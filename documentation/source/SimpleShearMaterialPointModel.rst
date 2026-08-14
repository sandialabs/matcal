****************************************
Simple Shear Material Point Model
****************************************

MatCal's :class:`~matcal.sierra.models.SimpleShearMaterialPointModel`
is meant to be used in calibrations that can use a material point 
model subject to simple shear loading as the simulation of the experiment. 
This can be a valid model for experiments with simple shear loading conditions
where the deformation 
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

Simple shear kinematics
========================
The simple shear deformation imposed by this model has interesting kinematic properties
that are important to understand when interpreting results. While the deformation is 
predominantly shear, it is inherently **isochoric** (volume-preserving) with :math:`J = \det(\mathbf{F}) = 1`,
and produces small but non-zero normal strains in the X and Z directions at large deformations.

Deformation gradient
--------------------
For simple shear in the XZ plane, where the top surface displaces in the X direction
while the bottom surface remains fixed, the deformation gradient takes the form:

.. math::

   \mathbf{F} = \begin{bmatrix}
   1 & 0 & \gamma \\
   0 & 1 & 0 \\
   0 & 0 & 1
   \end{bmatrix}

where :math:`\gamma` is the engineering shear strain, defined as the ratio of the 
X-direction displacement of the top surface to the height of the element:

.. math::

   \gamma = \frac{u_x}{h}

For the unit cube geometry used in this model (:math:`h = 1`), the engineering shear strain
:math:`\gamma` equals the X-direction displacement directly.

Isochoric deformation
---------------------
The determinant of the deformation gradient is always unity:

.. math::

   J = \det(\mathbf{F}) = 1

This confirms that simple shear is an isochoric (volume-preserving) deformation for all
values of :math:`\gamma`. This property is important for material models that treat 
volumetric and deviatoric responses differently.

Finite strain measures
----------------------
While the deformation gradient appears to have only shear components, finite strain measures
reveal the development of normal strains at large deformations. The right Cauchy-Green 
tensor is:

.. math::

   \mathbf{C} = \mathbf{F}^T \mathbf{F} = \begin{bmatrix}
   1 & 0 & \gamma \\
   0 & 1 & 0 \\
   \gamma & 0 & 1 + \gamma^2
   \end{bmatrix}

The Green-Lagrange strain tensor is:

.. math::

   \mathbf{E} = \frac{1}{2}(\mathbf{C} - \mathbf{I}) = \begin{bmatrix}
   0 & 0 & \gamma/2 \\
   0 & 0 & 0 \\
   \gamma/2 & 0 & \gamma^2/2
   \end{bmatrix}

Note that :math:`E_{zz} = \gamma^2/2` is a quadratic function of the shear strain, while
:math:`E_{xx} = E_{yy} = 0` exactly. This shows that the Green-Lagrange measure predicts
normal strain only in the Z direction.

The logarithmic (Hencky or true) strain tensor, computed as :math:`\boldsymbol{\varepsilon} = \frac{1}{2}\ln(\mathbf{C})`,
reveals a more complete picture. The right Cauchy-Green tensor :math:`\mathbf{C}` has 
three eigenvalues:

.. math::

   \lambda_+ &= \frac{2 + \gamma^2 + \gamma\sqrt{4 + \gamma^2}}{2} \\
   \lambda_- &= \frac{2 + \gamma^2 - \gamma\sqrt{4 + \gamma^2}}{2} \\
   \lambda_y &= 1

The principal logarithmic strains are then:

.. math::

   \varepsilon_+ &= \frac{1}{2}\ln(\lambda_+) \\
   \varepsilon_- &= \frac{1}{2}\ln(\lambda_-) \\
   \varepsilon_y &= 0

where :math:`\varepsilon_+` and :math:`\varepsilon_-` lie in the XZ plane but are rotated 
from the X and Z coordinate axes. When expressed in the XYZ coordinate system, the 
logarithmic strain tensor has the general form:

.. math::

   \boldsymbol{\varepsilon}(\gamma) = \begin{bmatrix}
   \varepsilon_{xx}(\gamma) & 0 & \varepsilon_{xz}(\gamma) \\
   0 & 0 & 0 \\
   \varepsilon_{xz}(\gamma) & 0 & \varepsilon_{zz}(\gamma)
   \end{bmatrix}

where :math:`\varepsilon_{xx}(\gamma) < 0`, :math:`\varepsilon_{zz}(\gamma) > 0`, and 
:math:`\varepsilon_{xz}(\gamma) > 0` for positive shear. The exact functional forms are 
obtained through the eigenvalue decomposition and coordinate transformation shown above.

Several important observations:

#. **Isochoric constraint**: :math:`\varepsilon_{xx} + \varepsilon_{yy} + \varepsilon_{zz} = 0`, 
   consistent with :math:`J = 1`
#. **Equal and opposite normal strains**: :math:`\varepsilon_{xx} = -\varepsilon_{zz}`, representing 
   contraction in X and extension in Z
#. **No Y-direction strain**: :math:`\varepsilon_{yy} = 0`, as expected for simple shear in the XZ plane
#. **Magnitude of normal strains**: The ratio :math:`|\varepsilon_{xx}|/|\varepsilon_{xz}|` 
   increases with shear strain. For small strains, normal strains are quadratic in :math:`\gamma`
   (approximately :math:`\varepsilon_{xx} \approx -\gamma^2/4`), but at :math:`\gamma = 1.0`, 
   the normal strain magnitude is approximately 50% of the shear strain magnitude.

Implications for material calibration
--------------------------------------
The development of normal strains in simple shear has several implications:

#. **Material models with Poisson effects**: For elastic-plastic materials, the normal stresses
   that develop due to these normal strains will depend on the material's Poisson ratio and 
   plastic flow rule. This makes simple shear tests valuable for calibrating materials that 
   exhibit coupling between shear and normal stress components.

#. **Large deformation effects**: At engineering shear strains above :math:`\gamma \approx 0.5`,
   the normal strain components become significant (>20% of shear strain). This is important
   when calibrating material models at large strains, as the material response will be influenced
   by the three-dimensional stress state, not just pure shear.

#. **Output interpretation**: The model outputs both shear components (XZ) and normal components
   (XX, YY, ZZ) of stress and logarithmic strain, allowing full characterization of this 
   three-dimensional stress state. Users should expect to see non-zero :math:`\sigma_{xx}` and 
   :math:`\sigma_{zz}` stresses even though the applied boundary conditions involve only 
   X-direction displacement.

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
   :scale: 25%

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
displacement in the shear direction. Also, the simple shear material point model
will load the sample in the positive x-direction with positive engineering strains
and report positive engineering stresses.
Negative engineering strains will load the sample in the negative x-direction and 
report negative engineering stresses. The boundary data can also optionally include 
a "time" field to apply time dependent boundary conditions. The 
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
