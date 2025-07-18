***********************
Uniaxial Tension Models
***********************
MatCal's :class:`~matcal.sierra.models.RoundUniaxialTensionModel`
and :class:`~matcal.sierra.models.RectangularUniaxialTensionModel`
are meant to be used in calibrations requiring the simulation of a 
uniaxial tension test. These models have all the MatCal standard 
model features as described in :ref:`MatCal SIERRA Solid Mechanics Standard Models`. 
In this section, we will provide more information about how the geometries are generated, 
specifics on simulation boundary conditions, 
and what is output from these models.

.. note::
   Some examples and V&V studies that include these models are:

   #. :ref:`304L annealed bar viscoplastic calibrations`
   #. :ref:`6061T6 aluminum plate calibrations`
   #. :ref:`MatCal Generated Models V&V Activities`
   
Uniaxial tension geometry and mesh generation
=============================================
Both of these models require very similar geometry parameters 
and are meshed in very similar ways. For their geometry to be 
accurately built, the models require
the following keyword arguments be provided to their constructor.
    
#. total_length
#. gauge_length 
#. extensometer_length
#. fillet_radius  
#. taper - Specifies the taper from the target gauge width/radius at the center
   of the specimen gauge section to the ends of the gauge length where the gauge width/diameter 
   increase by this value.
#. grip_contact_length - distance that the grips contact the specimen in the grip section.
#. necking_region - the fraction of the extensometer length where necking is expected 
   to occur. This is used for output and mesh refinement for some of the *mesh_methods*.
#. element_size - target element edge length 
#. mesh_method - user specified meshing method options

Keyword arguments specific to the :class:`~matcal.sierra.models.RoundUniaxialTensionModel` 
include those shown below.

#. gauge_radius 
#. grip_radius

These parameters provide information related to geometry, discretization sizing
and output and boundary condition mesh entities such as blocks and node sets.
The geometric parameters for the :class:`~matcal.sierra.models.RoundUniaxialTensionModel` 
are shown in :numref:`round_tension_model_geo`.

.. _round_tension_model_geo:
.. figure:: figures/uniaxial_tension/round_tension_bcs_and_dimensions.*
   :scale: 20%

   The geometric dimensions and boundary conditions for the
   round cross-section model.

Similarly, arguments specific to the :class:`~matcal.sierra.models.RectangularUniaxialTensionModel` 
are as follows.

#. grip_width
#. gauge_width 
#. thickness

The geometric parameters for the :class:`~matcal.sierra.models.RectangularUniaxialTensionModel` 
are shown in :numref:`rectangular_tension_model_geo`. 

.. _rectangular_tension_model_geo:
.. figure:: figures/uniaxial_tension/rectangular_tension_bcs_and_dimensions.*
   :scale: 20%

   The geometric dimensions and boundary conditions for the
   rectangular cross-section model.

The keyword 
*element_size* is used to specify the approximate element edge length 
that Cubit will target in the mesh. Depending on the *meshing_method* chosen, 
this could be the entire model or just a subregion of the model. The *meshing_method*
parameter allows the user to change how the geometry is meshed. Differences in 
the 5 *meshing_method* available are shown in :numref:`round_meshing_method` 
and :numref:`rectangular_meshing_method`. In general, low *meshing_method* parameters are 
intended for coarser meshes and result in higher element counts. In contrast,
high *meshing_method* parameters are intended for 
finer meshes, result in lower element counts and begin to 
use Cubit numsplits for *meshing_method* >= 4. Note that higher number *mesh_methods*
can result in lower resolution of geometry away from the gauge section of the 
specimen if the target mesh size is too coarse.

.. _round_meshing_method:
.. figure:: figures/uniaxial_tension/round_tension_mesh_method.*
   :scale: 26%

   The resulting meshes for different *mesh_method* options for the
   round cross-section model.

.. _rectangular_meshing_method:
.. figure:: figures/uniaxial_tension/rectangular_tension_mesh_method.*
   :scale: 15%

   The resulting meshes for different *mesh_method* options for the
   rectangular cross-section model.

Currently, the entire geometry is meshed in order to support thermomechanical
coupling. Since conduction into the grips and load frame may be non-negligible, 
the entire specimen is important to model. We have found the extra computational
cost associated with including the grips to be small.  

Uniaxial tension boundary conditions
====================================
The tension models currently only support :math:`\frac{1}{8}^{\text{th}}` symmetry geometry,
and, as a result, have boundary conditions that reflect that. The boundary 
conditions are shown graphically in :numref:`round_tension_model_geo` 
and :numref:`rectangular_tension_model_geo`. Since these models 
can easily be coupled with thermal modeling, the boundary condition
descriptions have been separated into the following two subsections
associated with the solid mechanics and thermal models.

Uniaxial tension solid mechanics boundary conditions
----------------------------------------------------
The tensile loading is caused by a displacement function
applied to the outer surface 
of the grip section block in the axial direction
away from the specimen center. 
This function acts on the surface of the specimen 
where the grips would contact it,
and includes nodes from the top of the specimen 
down by the *grip_contact_length* dimension.
For the round specimen, this includes all nodes on the outside radius 
of the grip section. For the rectangular specimen, 
this includes the nodes on the wide grip surface but not on the side grip 
surface that have a width equal to *thickness*. These node sets are shown for the 
two tension specimens in :numref:`round_tension_model_geo` and 
:numref:`rectangular_tension_model_geo`.

The applied function is determined using the 
:meth:`~matcal.sierra.models.RoundUniaxialTensionModel.add_boundary_condition_data`. 
This method must be supplied a :class:`~matcal.core.data.Data` or 
:class:`~matcal.core.data.DataCollection` class that contains 
either "engineering_strain" or "displacement" fields for the 
states of interest for the model. They can also optionally include 
a "time" field. The 
:meth:`~matcal.sierra.models.RoundUniaxialTensionModel.add_boundary_condition_data` 
method determines the boundary condition function to be applied 
to the tension specimens according to the following 
algorithm:

#. Determine the boundary condition by state since maximum deformation, 
   material behavior and experiment setup can vary significantly over different states.
#. For each state, find the data set with the largest displacement/strain and use it for 
   boundary condition generation.
#. If displacement data is provided, select it for the boundary condition generation
   and ignore engineering strain data. MatCal automatically scales the displacement by 
   the *gauge_length*/*extensometer_length*   
   This assumes that the strain is uniform throughout the gauge section for 
   most of the load history and is intended to 
   ensure that enough displacement is applied to the grips to achieve the desired displacement
   across the extensometer length.
#. If displacement data is *not* provided, use engineering strain for boundary
   condition generation and convert it to displacement using the geometry *gauge_length*.
   Once again this is to ensure enough displacement is applied to the specimen to achieve
   the correct deformation across the extensometer length.
#. If the data does not contain a "time" field and there is *not* a :class:`~matcal.core.state.State`
   parameter named "displacement_rate", then apply a linear displacement function from 
   zero to the maximum displacement found in the data over one second.
#. If the data does not contain a "time" field and there *is* a :class:`~matcal.core.state.State`
   parameter named "displacement_rate" or "engineering_strain_rate", then apply a linear displacement function from 
   zero to the maximum displacement found in the data. For "displacement_rate", this is done over a time period
   beginning at zero seconds and ending at a time calculated by dividing 
   the maximum displacement at the extensometer by the "displacement_rate" :class:`~matcal.core.state.State`
   parameter. For "engineering_strain_rate", this is done over a time period calculated by dividing the 
   maximum engineering strain by  "engineering_strain_rate" and the specimen 
   is displaced to a maximum displacement of the maximum engineering strain times *gauge_length*.
#. If the data does contain a "time" field, use the function directly as provided after scaling 
   the "displacement" or "engineering_strain" fields such that they 
   account for deformation in the *gauge_length* outside of the 
   *extensometer_length*.

.. note::
   This algorithm assumes that negligible deformation occurs in the regions
   outside of the specimen gauge length. If this is known or suspected to be 
   an invalid assumption, an additional scale factor can be applied to increase 
   the displacement applied to the grips. Use the 
   :meth:`~matcal.sierra.models.RoundUniaxialTensionModel.set_boundary_condition_scale_factor`
   method to add a scale factor to scale the displacement function. It must be between 1 and 10 
   and it directly multiplies the displacement determined from the boundary condition generation 
   algorithm.
   
The remaining solid mechanics boundary conditions only include the symmetry boundary conditions 
where displacements normal to the symmetry surfaces are set to zero.

Uniaxial tension thermal model boundary conditions
--------------------------------------------------
Since MatCal SIERRA/SM standard models only allow 
heat flux out of the specimen through the grips, 
only the grip boundary condition is 
described here. As discussed in the previous section, 
the boundary condition for the grip-to-specimen interface
includes the nodes between the ends of the tension 
specimen model and *grip_contact_length* away from the ends of the specimen. 
As described in :ref:`Staggered and iterative coupling`, 
the temperature at the nodes is fixed to the value of the :class:`~matcal.core.state.State` parameter 
"temperature". The entire body 
of the model is prescribed an initial temperate of  
:class:`~matcal.core.state.State` parameter 
"temperature" for 
all simulations regardless of coupling specification (uncoupled, staggered coupling, 
iterative coupling or adiabatic).  For uncoupled simulations, this is only done
if a temperature state variable is provided.

Uniaxial tension model specific output
======================================
By default, the tension models include the following global 
output fields: 

#. time
#. displacement - measured across extensometer length in the loading direction
#. load - measured at the applied boundary condition node set in the loading direction.
#. engineering_strain - displacement divided by *extensometer_length*
#. engineering_stress - load divided by the gauge section center
   cross-sectional area.

If coupling is activated, the following global
temperature output is provided: 

#. low_temperature
#. med_temperature
#. high_temperature

and how they are calculated is dependent on the type of coupling. For 
adiabatic simulations, they are the minimum, average and maximum 
element temperatures in the gauge section of the model.
For coupled simulations, the same quantities are provided by 
acting on the nodal temperatures instead of the element temperatures.
