Full-field Interpolation and Extrapolation
==========================================
For full-field interpolation and extrapolation, we use the python implementation 
of the Compadre Toolkit :cite:p:`compadre_toolkit,pycompadre_sand`. From the 
Compadre Toolkit, we use the generalized moving least squares method (GMLS)
to interpolate and extrapolate from one point cloud to another.
Usually, we are using this tool to map fields from the experiment 
data cloud to a computational discretization. This workflow was chosen as 
our default operation in order 
to conserve memory and decrease computational cost. The 
computational cost is reduced in terms of both cpu time and memory for 
the following reasons:

#. When we interpolate on the experimental data to the simulation 
   discretization, we can store the mapped experimental data 
   at the simulation discretization locations and 
   compare directly to simulation output. As a result, we would 
   only perform the mapping upon initialization and not every time
   the model is evaluated. The mapping process requires non-negligible computational 
   cost in terms of cpu time and memory, so this reduces the overall computational cost 
   for each evaluation. 
#. For Digital Image Correlation (DIC), which measures full-field 
   displacements over a domain, the experiment point cloud is frequently 
   denser than the simulation discretization. During a calibration, if we had to 
   interpolate from a coarser discretization to a finer one, we would 
   be up-sampling the data and increasing our memory foot print 
   for each simulation result evaluated without 
   affecting the results for a given simulation discretization.
   In calibrations problems, 
   this additional memory expense can become significant if many models and time steps 
   are needed for every objective function evaluation.

Generalized Moving Least Squares Theory
---------------------------------------
A basic overview of the Compadre GMLS theory is presented here. For our
purposes of data interpolation and extrapolation, we use the Compadre GMLS solver
to solve a traditional moving least squares problem. As a result, 
we will describe the theory for the moving least square problem. For more details 
see :cite:p:`GMLS_book_ref`. Moving least squares solves a local, weighted
least squares problem to approximately reproduce a function :math:`f\left(x\right)`
from a discrete set of values for :math:`f` known
at locations :math:`X = \left\{x_1, ..., x_n\right\}`.
The approximation of :math:`f`, :math:`f^h`,  is determined
by the solution of 

.. math::
    :label: eq_mls_min

    \min\left\{\sum\left[f\left(x_i\right)-p\left(x_i\right)\right]w\left(x,x_i\right)\right\}

where :math:`p` is a polynomial of a pre-specified order, and :math:`w` is a weighting 
function specific to the current point of interest :math:`x_i`.

In order to reduce the cost of the least squares, the weights for the 
least squares problem are designed so that only points within some distance
away from the point of interest are included making it a local 
least squares problem. This is done using a 
weight function with local compact support such that it is one near the point 
of interest and drops to zero some distance away from the point of interest.
The weighting function used by the python interface to Compadre is a power weighting function 
of the form 

.. math::
    :label: eq_gmls_weighting_func
    
    w\left(r,n,p,h\right) = \left(1-\left|\frac{r}{h}\right|^n\right)^p

where :math:`n` and :math:`p` are parameters governing the weighting function 
shape, :math:`h` is the radius of compact support for the function and :math:`r` is the Euclidean 
distance from the current point of interest. By default, the weighting function exponents 
are set such that :math:`p=2` and :math:`n=1`. 
By setting up the least squares problem in this way, the system to be solved
is sparse and the fitted function can be calculated using efficient methods.

After solving the minimization problem defined in :eq:`eq_mls_min`, we 
have an analytical function, :math:`f^h`, that can be used to estimate the value of 
:math:`f` at any point :math:`x`.
Note that :math:`x` should be within or close to
the domain of the function space where :math:`f` is measured :math:`x_i` 
to avoid extrapolating. 

.. warning::
    MatCal will use :math:`f^h` to extrapolate. This is necessary to support 
    VFM calibrations using DIC data that is not defined near the test specimen 
    edged. Care must be taken to ensure the extrapolation values are valid.
    We choose parameters for the GMLS algorithm in an effort to make the method robust
    and reliable, however, these parameters may need to be changed depending on the 
    your data and modeling choices.  See :ref:`Full-field Interpolation Verification`
    for more information on how these parameters were chosen.

MatCal's GMLS Implementation
----------------------------
MatCal interpolates using Compadre's python implementation pycompadre. 
Compadre leverages the advantages of the Kokkos library to perform the 
calculations necessary for GMLS in a scalable parallel fashion.  
MatCal wraps the various tools used by pycompadre, so that there is a 
limited need to understand the implementation details of Compadre by the rest of the code. 
The main tools of interest that use GMLS are the function 
:func:`~matcal.full_field.field_mappers.meshless_remapping` 
and :class:`~matcal.full_field.objective.InterpolatedFullFieldObjective`. 

The :func:`~matcal.full_field.field_mappers.meshless_remapping` function is used throughout MatCal to perform 
the unstructured interpolation from
one point cloud to another point cloud. 
This function is also available with the default import of MatCal, 
providing users easy access to the 
ability to interpolate between two unstructured point clouds. 
The function stores no information between calls, 
therefore it is best to submit as much information in one large batch call as possible. Large batch calls avoid the need for the recreation 
of the interpolation weights which occurs each time the function is called. 
For more details on how to use :func:`~matcal.full_field.field_mappers.meshless_remapping` 
please see the function's API documentation. 

The :class:`~matcal.full_field.objective.InterpolatedFullFieldObjective` is the way that MatCal generates an objective based on interpolating 
an experimental point cloud data to node locations from a simulation. MatCal expects nodes from a single surface of a simulation, 
but does not require any connectivity information about the surface.
Therefore simulation results from finite element like simulations and meshless simulations can both be calibrated.
the :class:`~matcal.full_field.objective.InterpolatedFullFieldObjective` uses the same underlying
wrapping tools that :func:`~matcal.full_field.field_mappers.meshless_remapping` uses.
In this objective the experimental full-field data is interpolated onto
the simulation nodes in space and the simulation values are 
interpolated in time to the experimental time stamps (:numref:`tab_interp_dir`). 

.. _tab_interp_dir:

.. list-table:: Interpolation Direction
    :header-rows: 1
    :widths: 25 25 25
    :align: center
    
    * - 
      - Space
      - Time
    * - Experiment
      - From
      - To
    * - Simulation 
      - To
      - From

These directions were chosen because they interpolate from the more dense data set to the less dense data set.
Full-field experimental data points are often much more dense than the computational nodes, and experimental data sets
can often efficiently be down-sampled in time without losing important information to be less dense than computational time step requirements.
Interpolating this way minimized the data requirements per evaluation, and helps ensure that there is enough computational support to 
meaningfully interpolate. 

In addition to the point cloud data supplied to these tools, the key values that the user needs to supply to the methods is the 
order of polynomial to use and the search radius multiplier. The first one is the polynomial order used to approximate the field
and the other effectively changes the number of reference points used in the local polynomial fit.
More details about these terms can be found in section :ref:`Full-field Methods Verification`. 

MatCal uses GMLS for extrapolation in space. MatCal does not separate its interpolation and extrapolation workflows.
Keeping the two together means that they share the same polynomial order and search radius. Extrapolation often 
performs best with lower polynomial orders, while interpolation performs better with higher. MatCal has a default of 
polynomial order of 1 and search radius multiplier of 2.75, as these values seem to perform reasonably well in both 
extrapolation and interpolation for data sets we have seen. 