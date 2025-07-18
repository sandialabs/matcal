***************************
Round Notched Tension Model
***************************
MatCal's :class:`~matcal.sierra.models.RoundNotchedTensionModel`
is meant to be used in calibrations requiring the simulation of a 
round notched tension test where higher stress triaxialities are 
required for parameter calibration. This model has all the MatCal standard 
model features as described in :ref:`MatCal SIERRA Solid Mechanics Standard Models`. 
In this section, we will provide more information about how the geometry is generated, 
specifics on simulation boundary conditions, 
and what is output from the model.

.. note::
   Some examples and V&V studies that include these models are:

   #. :ref:`sphx_glr_matcal_model_v_and_v_plot_coupling_verification.py`

Notched tension geometry and mesh generation
============================================
This model has geometry parameters similar to those in the 
:class:`~matcal.sierra.models.RoundUniaxialTensionModel`. 
The only differences are the addition of parameters regarding 
the notch geometry,  *notch_radius* and *notch_gauge_radius*, and 
the removal of the inapplicable *taper* parameter.
For the geometry to be 
accurately built, the model requires
the following keyword arguments be provided to its constructor.
    
#. total_length
#. gauge_length 
#. extensometer_length
#. fillet_radius  
#. grip_contact_length - distance that the grips contact the specimen in the grip section.
#. necking_region - the fraction of the extensometer length where necking is expected 
   to occur. This is used for output and mesh refinement for some of the *mesh_methods*.
#. element_size - target element edge length 
#. mesh_method - user specified meshing method options
#. gauge_radius 
#. grip_radius
#. notch_radius
#. notch_gauge_radius

These parameters provide information related to geometry, discretization sizing
and output and boundary condition mesh entities such as blocks and node sets.
The geometric parameters are shown in :numref:`round_notched_tension_model_geo`.

.. _round_notched_tension_model_geo:
.. figure:: figures/round_notched_tension/round_notched_tension_bcs_and_dimensions.*
   :scale: 20%

   The geometric dimensions and boundary conditions for the
   round notched tension model.

The keyword 
*element_size* is used to specify the approximate element edge length 
that Cubit will target in the mesh. Depending on the *meshing_method* chosen, 
this could be the entire model or just a subregion of the model. The *meshing_method*
parameter allows the user to change how the geometry is meshed. Differences in 
the 5 *meshing_methods* available are shown in :numref:`round_notched_tension_meshing_method_element_sizes`
and :numref:`round_notched_tension_meshing_method_with_mesh`.
In general, low *meshing_method* parameters are 
intended for coarser meshes and result in higher element counts. In contrast,
high *meshing_method* parameters are intended for 
finer meshes, result in lower element counts and begin to 
use Cubit numsplits for *meshing_method* >= 4. Note that higher number *mesh_methods*
can result in lower resolution of geometry away from the gauge section of the 
specimen if the target mesh size is too coarse.

.. _round_notched_tension_meshing_method_element_sizes:
.. figure:: figures/round_notched_tension/mesh_method_element_sizes.*
   :scale: 26%

   The target element sizes for different *mesh_method* options for
   different regions of the
   round notched tension model.

.. _round_notched_tension_meshing_method_with_mesh:
.. figure:: figures/round_notched_tension/mesh_method_composite.*
   :scale: 26%

   The resulting meshes for different *mesh_method* options for the
   round notched tension model.

Currently, the entire geometry is meshed in order to support thermomechanical
coupling. Since conduction into the grips and load frame may be non-negligible, 
the entire specimen is important to model. We have found the extra computational
cost associated with including the grips to be small.  

Notched tension boundary conditions
===================================
This model currently only supports :math:`\frac{1}{8}^{\text{th}}` symmetry geometry,
and, as a result, have boundary conditions that reflect that. The boundary 
conditions are shown graphically in :numref:`round_notched_tension_model_geo`.
Since this model can easily be coupled with thermal modeling, the boundary condition
descriptions have been separated into the following two subsections
associated with the solid mechanics and thermal models.

Notched tension solid mechanics boundary conditions
---------------------------------------------------
The tensile loading is caused by a displacement function
applied to the outer surface 
of the grip section block in the axial direction
away from the specimen center. 
This function acts on the surface of the specimen 
where the grips would contact it,
and includes nodes from the top of the specimen 
down by the *grip_contact_length* dimension.
This includes all nodes on the outside radius 
of the grip section. These node sets are shown for the 
two tension specimens in :numref:`round_notched_tension_model_geo`.

The applied function is determined using the 
:meth:`~matcal.sierra.models.RoundNotchedTensionModel.add_boundary_condition_data`. 
This method must be supplied a :class:`~matcal.core.data.Data` or 
:class:`~matcal.core.data.DataCollection` class that contains 
at a "displacement" field for the 
states of interest for the model. They can also optionally include 
a "time" field. The 
:meth:`~matcal.sierra.models.RoundNotchedTensionModel.add_boundary_condition_data` 
method determines the boundary condition function to be applied 
to the specimen according to the following 
algorithm:

#. Determine the boundary condition by state since maximum deformation, 
   material behavior and experiment setup can vary significantly over different states.
#. For each state, find the data set with the largest displacement and use it for 
   boundary condition generation.
#. Perform no scaling on the displacement. This assumes 
   that the strain is primarily localized to the notched 
   region of the specimen. 
#. If the data does not contain a "time" field and there is *not* a :class:`~matcal.core.state.State`
   parameter named "displacement_rate", then apply a linear displacement function from 
   zero to the maximum displacement found in the data over one second.
#. If the data does not contain a "time" field and there *is* a :class:`~matcal.core.state.State`
   parameter named "displacement_rate", then apply a linear displacement function from 
   zero to the maximum displacement found in the data. This is done over a time period
   beginning at zero seconds and ending at a time calculated by dividing 
   the maximum displacement at the extensometer by the "displacement_rate" :class:`~matcal.core.state.State`
   parameter.
#. If the data does contain a "time" field, use the displacement function directly as provided.

.. note:
   This algorithm assumes that negligible deformation occurs in the regions
   outside of the notched region of the geometry. If this is known or suspected to be 
   an invalid assumption, an additional scale factor can be applied to increase 
   the displacement applied to the grips. Use the 
   :meth:`~matcal.sierra.models.RoundUniaxialTensionModel.set_boundary_condition_scale_factor`
   method to add a scale factor to scale the displacement function. It must be between 1 and 10 
   and it directly multiplies the displacement determined from the boundary condition generation 
   algorithm.
   
The remaining solid mechanics boundary conditions only include the symmetry boundary conditions 
where displacements normal to the symmetry surfaces are set to zero.

Notched tension thermal model boundary conditions
-------------------------------------------------
Since MatCal SIERRA/SM standard models only allow 
heat flux out of the specimen through the grips, 
only the grip boundary condition is 
described here. As discussed in the previous section, 
the boundary condition for the grip-to-specimen interface
includes the nodes between the ends of the model geometry
and *grip_contact_length* away from the ends of the specimen. 
As described in :ref:`Staggered and iterative coupling`, 
the temperature at the nodes is fixed to the value of the :class:`~matcal.core.state.State` parameter 
"temperature". The entire body 
of the model is prescribed an initial temperate of  
:class:`~matcal.core.state.State` parameter 
"temperature" for 
all simulations regardless of coupling specification (uncoupled, staggered coupling, 
iterative coupling or adiabatic). For uncoupled simulations, this is only done
if a temperature state variable is provided.

Notched tension model specific output
=====================================
By default, the round notched tension model includes the following global 
output fields: 

#. time
#. displacement - measured across extensometer length in the loading direction
#. load - measured at the applied boundary condition node set in the loading direction.

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
