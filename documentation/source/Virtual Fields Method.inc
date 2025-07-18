Virtual Fields Method 
======================
The Virtual Fields Method (VFM) uses the principle of virtual work 
to balance the internal virtual work within a specimen with the external virtual work  
on the same specimen for material model calibration. When combined 
with full-field data measurement techniques, such as digital image correlation, 
we can use the measured displacement field to determine the 
material stress state under a plane stress assumption 
which allows for rapid model evaluation and efficient material model 
parameter identification :cite:p:`pierron_vfm`.

While the formulation of VFM is agnostic to the application, in practice it is difficult to 
apply it out side of solid mechanics. This is because it is difficult to accurately represent the boundary conditions well enough to use in many other physics. 

VFM Theory Overview
-------------------

For the case of finite deformation where inertial and 
body forces are ignored, 
the principle of virtual power is used 

.. math::
    :label: eq_general_virt_power

    \int_{V_0} \mathbf{\Pi} : \dot{\mathbf{F}}^*dV = \int_{\partial V_0} \left(\mathbf{\Pi} \cdot \mathbf{n}_0 \right) \cdot \mathbf{v}^* dS

where :math:`V_0` is the body in the reference configuration, 
:math:`\Pi` is the first Piola-Kirchhoff stress, 
:math:`\dot{\mathbf{F}}^* = \frac{\partial \mathbf{v}^*}{\partial \mathbf{X}}`, 
:math:`\mathbf{v}^*` are the virtual velocity fields,
and :math:`n_0` is the surface normal in the reference
configuration :cite:p:`large_def_VFM`. The surface traction 
on the boundary in the external 
power term cannot be easily measured, however, 
the total force on the surface is easily measured 
with a load cell. In order to simplify the external 
power term, a virtual velocity function is chosen 
such that the function is constant along the boundary
of loading. As a result, the virtual power balance simplifies to

.. math::
    :label: eq_simple_virt_power

    \int_{V_0} \mathbf{\Pi} : \dot{\mathbf{F}}^*dV = \mathbf{f} \cdot  \mathbf{v}^* 

where :math:`\mathbf{f}` is the measured external load along the
boundary.
If desired, the virtual power can be integrated 
to obtain the virtual work. For our purposes, 
we will work with the virtual power to 
allow more flexibility within MatCal's optimization framework.

With the virtual power balance, a residual 
can be formed where inputs are the external load on the test 
specimen, a set of kinematically admissible virtual velocity 
functions, and the first Piola-Kirchhoff stress at any 
point in the body.  To calculate the first Piola-Kirchhoff 
stresses in the body, the measured displacement 
field is mapped to a finite element mesh. Adopting 
a plane stress assumption, this mesh 
is used to determine the plane stress kinematics 
for the supplied displacement field and provide the 
appropriate strain measures needed to determine 
the stress state in each element. Given a set of 
material parameters, a material point 
solver can then be used to return the correct stresses 
from the calculated kinematics. Then the internal 
power integral can be calculated easily 
on the finite element mesh.

In a practical application, full-field 
data is represented as several discrete points in time, 
these timestamps then define the individual points where an individual residual is calculated.
Lastly, a total objective 
is calculated using an objective 
metric to combine the virtual work residuals
over the time frame of interest or the residual 
for each time step can be used in a nonlinear 
least squares solver. 

In summary, the VFM method reformulates the calibration problem, such that it 
takes in experimental DIC data, boundary loading data, and a set of parameters to 
directly produce a residual, and skip the conventional finite element solve required for most other calibration methods. 

For a given set of experimental data, 
producing a mesh of the specimen of interest can be accomplished
with traditional meshing software. The virtual velocity 
functions can be any kinematically admissible function 
which means they must be constant at the loading boundary 
so that the simplified virtual power equation in :eq:`eq_simple_virt_power` 
is valid. As a result, they are easily defined. 
Difficulties in using VFM arise in few places:

#. Data manipulation and importing.
#. Mapping the data to an appropriate finite element mesh. 
#. Solving the element kinematics and material model.
#. Interfacing with an optimization routine.

Each of these steps require specialized software tools to 
make the process efficient and with an approachable 
user interface.
MatCal aims to address these issues in our production VFM capability. 

MatCal's VFM Implementation
---------------------------

.. include:: vfm_model_notes_and_warnings.rst

MatCal's VFM implementation provides the software tools
needed to simplify using VFM for engineering applications. 
This work was inspired by previous research efforts at Sandia
related to VFM 
:cite:p:`VFM_LDRD_Kramer,VFM_LDRD_Jones`. These efforts 
highlighted potential benefits for using VFM 
for material parameter identification, but did not 
result in tools that the general analyst could use in 
production setting. Building off these previous works, 
we implemented VFM tools that could be easily distributed, 
easily modified and used 
for research and applied engineering problems.
Specifically, we address the issues discussed at the end of 
:ref:`Virtual Fields Method`, addressing data importing (see :ref:`Full-field Data Specific Features`), 
full-field data mapping, the finite element kinematics, 
material modeling and interfacing with an optimization algorithm.
We do this through the development of our own tools and relying 
on several external libraries and codes. The tools we have 
available and their limitations are discussed or linked to 
in this section.

Full-field Data Mapping for VFM Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We currently use MatCal's interface to PyCompadre
for GMLS based data mapping for full-field data
interpolation and extrapolation. The mapping 
is performed automatically for the user. However, 
the user must ensure that the point cloud data being 
used as the boundary condition data in the VFM model
is appropriately aligned with the mesh they are 
using for their VFM model. With a properly aligned 
point cloud and mesh, the VFM model will map 
the in-plane displacements and, if necessary, full-field 
temperatures onto the VFM model mesh.

See :ref:`Full-field Interpolation and Extrapolation`
and :ref:`Full-field Interpolation Verification`
for more details on the GMLS interpolation methods.

Element Kinematics and Material Modeling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
MatCal currently does not support any 
element kinematics calculations or material 
model solvers. We rely solely on Sierra
to perform these functions for us. However, 
due to restrictions on available interfaces to 
Sierra algorithms, we do so all through a full
Sierra finite element (FE) model that is solved 
using implicit quasi-statics for the field data provided.
We currently have two VFM models available for users. 
Both use hexahedral elements and are a single element 
thick with all in-plane displacements prescribed
from the provided full-field data that is mapped 
to the mesh nodes. These models are valid for thin sheet
problems exhibiting plane stress for which 
VFM was originally derived. Since SierraSM does 
not fully support plane stress 
elements, some through-the-thickness nodal degrees of freedom 
must still be solved for in the model.
Although a full FE simulation is still performed, 
the significantly reduced 
degrees of freedom due to the fully prescribed 
in-plane boundary conditions and reduced density of the stiffness matrix 
reduces the computational cost of the model solve considerably.

The largest benefit of VFM is that it the FEA solver is now solving a series of independent material model 
instead of one large coupled FE model solve.
Even though our implementation does not fully realize this, we do realize 
several other benefits that out weigh this single draw back. 

#.  We are able to access all SierraSM material models and all 
    the testing, verification and validation that comes along with them. Otherwise, 
    users or developers would have to make their own versions compatible with 
    python and our classes and interfaces. This task is nontrivial.
#.  We can benefit from all FE model functionality that may be of interest 
    to users and is included in Sierra. Most importantly 
    from these features, we support adiabatic heating
    with material model temperature dependence. For one of the two 
    VFM models provided by MatCal, we can solve heating due to plastic work 
    and any conduction that may occur throughout the body.
#.  We get some computational cost benefit 
    from any speed related feature that is built into SierraSM.
    These include parallelism, any speed 
    improvements implemented in the material models themselves 
    and the eventual adoption of GPUs.

As alluded to previously, we have two VFM models 
available for MatCal studies. Both are valid for only 
uniaxial loading and use hexahedral meshes.
The :class:`~matcal.sierra.models.VFMUniaxialTensionHexModel` 
is the recommended model to use for most cases.
It has the advantages of being 
the least computationally expensive and most robust of the two models.
Its single disadvantage is that it cannot model heat conduction. For 
intermediate loading rates where heating and conduction is expected 
temperatures must be measured during the experiments.
For cases where heating due to plastic work and 
conduction are important, the 
:class:`~matcal.sierra.models.VFMUniaxialTensionConnectedHexModel`
is available, but will be more computationally expensive 
and potentially less reliable.
This model has the advantage of being able to solve the 
conduction problem with the disadvantages of less 
robustness and increased cost. Another advantage is better 
representation of out-of-plane deformation due to the continuity enforcement across elements
and the the use of a through-the-thickness
symmetry boundary condition on one of the out-of-plane surfaces.
As a result, this model effectively has two elements through the 
thickness of the thin specimen being simulated.

For more information about these models and their specific features, 
see :ref:`Virtual Fields Method Uniaxial Tension Models`.

.. Warning::
    Several reminders of constraints and potential issues for the VFM models
    are included here once again:

    #.  The out-of-plane stresses are currently ignored. However, we have
        plans to include them in future releases an effort to attempt 
        using VFM past peak load for specimens that plastically localize
        and exhibit non-negligible out-of-plane stresses.
    #.  Currently, the VFM models can only be used to simulate
        test specimens with loading along a single axis.
    #.  Careful mesh alignment with the source field data for 
        displacement data is critical to obtain correct results.


VFM Interfacing to MatCal Studies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using MatCal's VFM models in MatCal studies such as 
calibration studies or uncertainty quantification studies
requires only the use of one of the MatCal's 
VFM models, the creation of a 
:class:`~matcal.full_field.objective.MechanicalVFMObjective` 
object, and global load-time data for the 
external virtual work calculation. 

The :class:`~matcal.full_field.objective.MechanicalVFMObjective` 
performs the calculation of the virtual power residuals.
It has two optional input parameters of the field names in 
the supplied experimental data for time and external global 
load. By default, it assumes the fields are named
"time" and "load", respectively, 
for these two data fields, but the user can change 
the names if needed. We currently only support the 
virtual velocity functions that were presented in 
:cite:p:`VFM_LDRD_Jones`.
They take the form

.. math::
    :label: eq_matcal_virtual_velocity_func_x

    \mathbf{v}^*_X=\cos\frac{\pi\bar{Y}}{h}

.. math::
    :label: eq_matcal_virtual_velocity_func_y

    \mathbf{v}^*_Y=\frac{2\bar{Y}+h}{2h}

where :math:`Y` is the direction of loading, 
:math:`\bar{Y}` is the centered position 
of the current point of interest 
in the reference configuration, 
and :math:`h` is the total height of 
the data. The data points :math:`Y`
are centered using

.. math::
    :label: eq_matcal_virtual_velocity_centering

    \bar{Y}=Y-\left(\frac{h}{2}+\min\left(\mathbf{Y}\right)\right)

where :math:`\mathbf{Y}` is a vector of 
all the Y direction coordinates from 
the points of interest. As implemented in 
MatCal, :math:`\mathbf{Y}` is used from 
the VFM model not the experimental data
since the experimental 
data may not extend fully to the edges
of the gauge section. Research has shown  
that the virtual velocity functions can 
be chosen to improve objective 
sensitivity to the input parameters 
:cite:p:`sensitivity_virtual_fields_1,sensitivity_virtual_fields_2,sensitivity_virtual_fields_3`. 
These techniques have yet to be implemented 
in MatCal but could be implemented in 
future releases.

After creating a VFM model and a 
:class:`~matcal.full_field.objective.MechanicalVFMObjective`,
any available MatCal study can be performed 
using them by adding them to the study with
:meth:`~matcal.core.study_base.StudyBase.add_evaluation_set`.
Since the model is the only component in the evaluation set 
that requires full-field data, this method takes data in the form of 
:class:`matcal.core.data.Data`, :class:`matcal.full_field.data.FieldData`
or :class:`matcal.core.data.DataCollection` objects. 
The study will then compare the internal virtual power from the model
to the external virtual power from all applicable experiments 
to form a residual for the evaluation set. If desired, additional evaluation sets can 
be added to the study with any supported MatCal model and data type. 
This would result in a hybrid VFM model and traditional FE model 
study and may be required if full-field data could 
not be taken for all material characterization tests or 
the plane stress assumption does not apply to some of the 
characterization tests.

VFM next steps
^^^^^^^^^^^^^^
With the base features in place, extensions and 
additions can be explored as future work.
Potential features and research activities include:

#.  A general VFMPythonModel will
    fully realize the primary benefit of VFM and is of use to organizations 
    and groups that do not use Sierra. We will provide a few 
    standard material models and a interface for users to implement 
    their own.
#.  Inclusion of out-of-plane stresses into the internal virtual work calculation.
#.  Optional addition of extra element layers in the 
    :meth:`~matcal.sierra.models.VFMUniaxialTensionConnectedHexModel`.
#.  General extension of the VFM models and tools to three-dimensions. 
#.  Implementation of sensitivity-based virtual velocity functions.

