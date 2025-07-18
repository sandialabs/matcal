*********************************************
Virtual Fields Method Uniaxial Tension Models
*********************************************

.. include:: vfm_model_notes_and_warnings.rst

MatCal currently provides two models developed in support 
of Virtual Fields Method capabilities. These models are
MatCal's :class:`~matcal.sierra.models.VFMUniaxialTensionHexModel` 
and :class:`~matcal.sierra.models.VFMUniaxialTensionConnectedHexModel`. 
They can be used to reduce the computational cost of models 
for MatCal studies when 
the following are true:

#. The loading and specimen geometry from a material characterization 
   test allow for a plane stress
   assumption to be valid for the specimen's stress state 
   over the specimen gauge section.
#. Full-field, in-plane displacement data over the entire specimen 
   gauge section are obtained.
#. Global load-displacement data are measured.

These models have most of the MatCal standard 
model features as described in :ref:`MatCal SIERRA Solid Mechanics Standard Models`. 
The following features are disabled for the 
:class:`~matcal.sierra.models.VFMUniaxialTensionHexModel`:

#. :meth:`~matcal.sierra.models.VFMUniaxialTensionHexModel.activate_thermal_coupling` 
   for conduction problems
#. :meth:`~matcal.sierra.models.VFMUniaxialTensionHexModel.use_iterative_coupling`

These features are disabled for all VFM models:

#. :meth:`~matcal.sierra.models.VFMUniaxialTensionHexModel.activate_element_death`
#. :meth:`~matcal.sierra.models.VFMUniaxialTensionHexModel.set_allowable_load_drop_factor`
#. :meth:`~matcal.sierra.models.VFMUniaxialTensionHexModel.activate_self_contact`
#. :meth:`~matcal.sierra.models.VFMUniaxialTensionHexModel.set_boundary_condition_scale_factor`

In this section, we will provide more information
about how the meshes are generated, 
specifics on simulation boundary conditions, 
and what is output from these models.

For more specifics about the VFM method theory and 
MatCal implementation details see :ref:`Virtual Fields Method`.

VFM model geometry and mesh generation
======================================
Both of these models require only two 
geometric parameters for their geometry
to be built and meshed,
and they are meshed in very similar ways. These 
two
parameters are passed to the models' 
constructors and include a surface
mesh filename and the thickness 
of the specimen being simulated.
In the constructor, these parameters
have the names:

#. surface_mesh_filename
#. thickness

The provided surface mesh should only include
the portion of the test specimen
where full-field data is collected. To be valid 
for VFM, this should include all of the 
visible surface of the tested component between the grips
which we will refer to as the gauge section. 
Since most DIC software is unable to calculate 
displacements near the free-edges of the specimen, 
the GMLS tool we employ
for mapping field data will extrapolate
the displacements linearly over the small regions 
near free-edges where DIC data is not provided. 

.. note::
   This user provided 
   mesh must be in the X-Y plane with the Y axis as the 
   loading axis.
 
The only difference between the two models is
the connectivity of the mesh elements and
the portion of the volume meshed. 
The :class:`~matcal.sierra.models.VFMUniaxialTensionHexModel` 
most closely emulates classical VFM by taking 
the user provided quadrilateral mesh of the desired surface 
and creates a hexahedral element from each surface element that 
has no connectivity. The hexahedral elements are created 
by extruding the surface element in the Z direction. 
The total thickness of 
each hexahedral element created is the thickness of the tested specimen. 
The resulting disconnected hexahedral mesh most 
closely emulates classical VFM because each 
element effectively operates as a material 
point simulator.
The :class:`~matcal.sierra.models.VFMUniaxialTensionConnectedHexModel` 
creates a hexahedral mesh from the user provided
quadrilateral surface mesh that 
maintains the correct connectivity such that it is 
the three-dimensional equivalent of the quadrilateral mesh.
The thickness of the created mesh
is half the thickness of the tested specimen. 
This mesh is created by extruding the entire 
surface mesh in the Z direction and updating 
the mesh with the correct connectivity.

VFM model boundary conditions
=============================
The boundary conditions for the models are described
in this section. Since these models 
can support thermomechanical coupling, the boundary condition
descriptions have been separated into the following two subsections
associated with the solid mechanics and thermal models.

VFM solid mechanics boundary conditions
---------------------------------------
The solid mechanics boundary 
conditions for these models are applied 
to best approximate the plane stress assumption 
required by VFM.
Both of these models have their in-plane 
displacements fully prescribed. These 
in-plane displacement boundary conditions are
mapped onto the meshes using MatCal's
interface to the PyCompadre's GMLS
algorithm. See :ref:`Full-field Interpolation and Extrapolation`
for more details. Both meshes are only one 
element thick and both in-plane 
faces for each element have their 
in-plane nodal displacements prescribed
by two-dimensional interpolation and extrapolation 
from the experimental data.
For both models, the out-of-plane displacements 
for the nodes on one of the in-plane faces of each 
element is fixed. The other out-of-plane displacements for 
the remaining nodes are free and must be calculated using 
a finite element solve. For the 
:class:`~matcal.sierra.models.VFMUniaxialTensionHexModel`, 
these boundary conditions closely emulate plane stress deformation. 
For the 
:class:`~matcal.sierra.models.VFMUniaxialTensionConnectedHexModel`, 
these boundary conditions loosely represent plane stress deformation, while 
more accurately capturing through thickness stresses and maintaining 
a continuous displacement field on the free surface. Since only half the 
specimen is simulated for this model with an out-of-plane symmetry boundary 
condition, two elements are used through the thickness which should 
provide better estimate of the through thickness stress. 

.. note:: 
   We acknowledge that the boundary conditions are imperfect, however, 
   believer they are the best available given limitations in what 
   is available in Sierra and experimentally. In verification problems 
   on synthetic data, these models produce similar accuracy and are able to reproduce
   material model parameters that were used to generate the synthetic 
   data within 1%. 

The in-plane displacements are determined from full-field 
data supplied by the user. The data are passed to the model 
through the :meth:`~matcal.sierra.models.VFMUniaxialTensionHexModel.add_boundary_condition_data`
method. Currently, only a single :class:`matcal.full_field.data.FieldData` object can be passed 
for each state to the :meth:`~matcal.sierra.models.VFMUniaxialTensionHexModel.add_boundary_condition_data`
method. If an averaged set of displacements is desired for repeats from a single state, 
combine the repeat data sets before hand. If the points from each data set are collocated
in space and time, 
use NumPy average to create an average field.  If not, use the 
:func:`~matcal.full_field.field_mappers.meshless_remapping` 
function to map all data to a common mesh and common time steps. 
User must make sure the data are properly aligned 
in both space and time before doing so.

VFM thermal model boundary conditions
-------------------------------------
There are no thermal boundary conditions current supported 
for the VFM models. All models are fully insulated. 
As previously mentioned, 
the :class:`~matcal.sierra.models.VFMUniaxialTensionHexModel`
only supports adiabatic heating with the 
volumetric heat source being the heating due to plastic work.
The :class:`~matcal.sierra.models.VFMUniaxialTensionConnectedHexModel`
supports the addition of conduction with temperature independent 
thermal properties.

VFM model specific output
=========================
By default, the VFM models require the following 
exodus element output: 

#. global time
#. element first_pk_stress - The volume weighted average of 
   the element integration point first Piola-Kirchhoff stress 
#. element centroid - element centroid location for determining
   the appropriate virtual velocity gradient values to use 
   for the virtual internal work calculation
#. element volume - the current element volume, only the 
   volume from the first time step is used as the 
   reference configuration element volume.

If coupling is activated or the temperature is 
prescribed from the data or state, the following
temperature output is in the exodus output: 

#. element avg_temperature - The volume weighted average of 
   the element integration point temperature (always)
#. nodal temperature - temperature at the nodes
   (only for models with conduction)