
.. DO NOT EDIT.
.. THIS FILE WAS AUTOMATICALLY GENERATED BY SPHINX-GALLERY.
.. TO MAKE CHANGES, EDIT THE SOURCE PYTHON FILE:
.. "advanced_examples/6061T6_anisotropic_calibration/plot_6061T6_c_anisotropy_calibration_cluster.py"
.. LINE NUMBERS ARE GIVEN BELOW.

.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        :ref:`Go to the end <sphx_glr_download_advanced_examples_6061T6_anisotropic_calibration_plot_6061T6_c_anisotropy_calibration_cluster.py>`
        to download the full example code.

.. rst-class:: sphx-glr-example-title

.. _sphx_glr_advanced_examples_6061T6_anisotropic_calibration_plot_6061T6_c_anisotropy_calibration_cluster.py:


6061T6 aluminum calibration with anisotropic yield
--------------------------------------------------
With the material model choice justified (See :ref:`6061T6 aluminum data analysis`)
and an initial point determined 
(See :ref:`6061T6 aluminum anisotropy calibration initial point estimation`), 
we can set up the calibration for this material. 
The items needed for the calibration include the data, the 
models for the tests, the objectives for the calibration, and the MatCal
calibration study object with the parameters that will be calibrated.

.. note::
    Useful Documentation links:

    #. :ref:`Uniaxial Tension Models`
    #. :class:`~matcal.sierra.models.RoundUniaxialTensionModel`
    #. :class:`~matcal.dakota.local_calibration_studies.GradientCalibrationStudy`
    #. :class:`~matcal.core.residuals.UserFunctionWeighting`

First, we import the tools that will be used 
for this example and setup our 
preferred plotting options.   

.. GENERATED FROM PYTHON SOURCE LINES 24-36

.. code-block:: Python

    import numpy as np
    from matcal import *
    from site_matcal.sandia.computing_platforms import is_sandia_cluster, get_sandia_computing_platform
    from site_matcal.sandia.tests.utilities import MATCAL_WCID

    import matplotlib.pyplot as plt

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('font', size=12)
    figsize = (4,3)








.. GENERATED FROM PYTHON SOURCE LINES 37-47

Next, we import the data
we will calibrate to. This includes 
the uniaxial tension data and top hat shear data. 
Like in the preceding examples, we
use MatCal's :class:`~matcal.core.data_importer.BatchDataImporter`
to perform the import and categorize the data according to states.
See :ref:`Data Importing and Manipulation` and 
:ref:`6061T6 aluminum data analysis` for more information 
about how these data files were setup to be imported 
correctly by the data importer.

.. GENERATED FROM PYTHON SOURCE LINES 47-54

.. code-block:: Python

    tension_data_collection = BatchDataImporter("aluminum_6061_data/" 
                                                  "uniaxial_tension/processed_data/"
                                                  "cleaned_[CANM]*.csv",).batch
    top_hat_data_collection = BatchDataImporter("aluminum_6061_data/" 
                                                  "top_hat_shear/processed_data/cleaned_*.csv").batch









.. GENERATED FROM PYTHON SOURCE LINES 55-60

We now modify the data to fit our calibration 
needs. For the tension data, 
we convert the engineering stress from
ksi units to psi units using the 
:func:`~matcal.core.data.scale_data_collection` function.

.. GENERATED FROM PYTHON SOURCE LINES 60-63

.. code-block:: Python

    tension_data_collection = scale_data_collection(tension_data_collection, 
                                                      "engineering_stress", 1000)








.. GENERATED FROM PYTHON SOURCE LINES 64-83

The top hat data needs more specialized 
modifications. Since some of these 
tests were not run to complete failure, 
we must remove the data after peak load. 
We do this by removing the time steps 
in the data after peak load. This will 
successfully remove unloading data from 
specimens that were not loaded until failure.
Also, since this calibration is calibrating a
plasticity model, we remove data after a displacement 
of 0.02". This is required because cracks can 
initiate well before peak load for these specimens 
and such cracks are likely not present before this displacement. 
Since most specimens have reached a region of linear 
load-displacement behavior by 0.02", the data up to this point should 
be sufficient for our calibration. 
We use NumPy array slicing to perform
the data modification for each data set 
in each state.

.. GENERATED FROM PYTHON SOURCE LINES 83-94

.. code-block:: Python

    for state, state_data_list in top_hat_data_collection.items():
        for index, data in enumerate(state_data_list):
            max_load_arg = np.argmax(data["load"])
            # This slicing procedure removes the data after peak load.
            data = data[data["time"] < data["time"][max_load_arg]]
            # This one removes the data after a displacement of 0.02"
            # and reassigns the modified data to the 
            # DataCollection
            top_hat_data_collection[state][index] = data[data["displacement"] < 0.02]
    top_hat_data_collection.remove_field("time")








.. GENERATED FROM PYTHON SOURCE LINES 95-97

We now plot the data to verify that 
we have modified it as desired for the calibration.

.. GENERATED FROM PYTHON SOURCE LINES 97-129

.. code-block:: Python

    tension_fig = plt.figure(figsize=figsize, constrained_layout=True)
    tension_data_collection.plot("engineering_strain", "engineering_stress", 
                                 state="temperature_5.330700e+02_direction_R22", 
                                 show=False, labels="$R_{22}$", figure=tension_fig, 
                                 color='tab:red')
    tension_data_collection.plot("engineering_strain", "engineering_stress", 
                                 state="temperature_5.330700e+02_direction_R11", 
                                 show=False, labels="$R_{11}$", figure=tension_fig,
                                 color='tab:blue')
    tension_data_collection.plot("engineering_strain", "engineering_stress", 
                                 state="temperature_5.330700e+02_direction_R33", 
                                 labels="$R_{33}$", figure=tension_fig, 
                                 color='tab:green')
    plt.xlabel("engineering strain (.)")
    plt.ylabel("engineering stress (psi)")

    tension_data_collection.remove_field("time")

    top_hat_fig = plt.figure(figsize=figsize, constrained_layout=True)
    top_hat_data_collection.plot("displacement", "load", show=False,
                                 state="direction_R12", labels="$R_{12}$",
                                 figure=top_hat_fig, color='tab:cyan')
    top_hat_data_collection.plot("displacement", "load", show=False,
                                 state="direction_R23", labels="$R_{23}$",
                                 figure=top_hat_fig, color='tab:orange')
    top_hat_data_collection.plot("displacement", "load",
                                 state="direction_R31", labels="$R_{31}$", 
                                 figure=top_hat_fig, color='tab:purple')
    plt.xlabel("displacement (in)")
    plt.ylabel("displacement (lbs)")





.. rst-class:: sphx-glr-horizontal


    *

      .. image-sg:: /advanced_examples/6061T6_anisotropic_calibration/images/sphx_glr_plot_6061T6_c_anisotropy_calibration_cluster_001.png
         :alt: plot 6061T6 c anisotropy calibration cluster
         :srcset: /advanced_examples/6061T6_anisotropic_calibration/images/sphx_glr_plot_6061T6_c_anisotropy_calibration_cluster_001.png
         :class: sphx-glr-multi-img

    *

      .. image-sg:: /advanced_examples/6061T6_anisotropic_calibration/images/sphx_glr_plot_6061T6_c_anisotropy_calibration_cluster_002.png
         :alt: plot 6061T6 c anisotropy calibration cluster
         :srcset: /advanced_examples/6061T6_anisotropic_calibration/images/sphx_glr_plot_6061T6_c_anisotropy_calibration_cluster_002.png
         :class: sphx-glr-multi-img


.. rst-class:: sphx-glr-script-out

 .. code-block:: none


    Text(20.771400166044003, 0.5, 'displacement (lbs)')



.. GENERATED FROM PYTHON SOURCE LINES 130-143

With the data prepared, we move on to 
building the models. 
The first step is to prepare the material model 
input deck file that is required by SIERRA/SM.
We do this within python because the 
file is relatively short and simple. It also 
makes it easy to ensure naming is consistent 
in the SIERRA/SM input deck files and our 
MatCal objects. We create a string 
with the material model syntax that SIERRA/SM 
expects and the Aprepro variables 
that MatCal will populate with study and 
state parameters when running a study. 

.. GENERATED FROM PYTHON SOURCE LINES 143-177

.. code-block:: Python

    material_name = "6061T6_anisotropic_yield"
    material_string = f"""
      begin material {material_name}
        density = 0.00026
        begin parameters for model hill_plasticity
          youngs modulus                = 10e6
          poissons ratio                = 0.33
          yield stress                  = {{yield_stress*1e3}}

          hardening model = voce
          hardening modulus = {{hardening*1e3}}
          exponential coefficient = {{b}}

          r11                           =   1
          r22                           =   {{R22}}
          r33                           =   {{R33}}
          r12                           =   {{R12}}
          r23                           =   {{R23}}
          r31                           =   {{R31}}
          coordinate system             =   rectangular_coordinate_system
      
          {{if(direction=="R11")}}
          direction for rotation        = 3
          alpha                         = 90.0
          {{elseif((direction=="R33") || (direction=="R31"))}}
          direction for rotation        = 1
          alpha                         = -90.0
          {{elseif(direction=="R23")}}
          direction for rotation        = 2
          alpha                         = 90.0
          {{endif}}
        end
      end
    """







.. GENERATED FROM PYTHON SOURCE LINES 178-182

We save that string to a file, so 
MatCal can add it to the model files 
that we generate for the tension and top hat 
shear test models.

.. GENERATED FROM PYTHON SOURCE LINES 182-186

.. code-block:: Python

    material_filename = "hill_plasticity.inc"
    with open(material_filename, 'w') as fn:
        fn.write(material_string)








.. GENERATED FROM PYTHON SOURCE LINES 187-191

MatCal communicates all required material 
model information to its MatCal generated
finite element models through a :class:`~matcal.sierra.material.Material`
object, so we create the required object.

.. GENERATED FROM PYTHON SOURCE LINES 191-193

.. code-block:: Python

    material = Material(material_name, material_filename, "hill_plasticity")








.. GENERATED FROM PYTHON SOURCE LINES 194-200

Now we create our tension model 
which requires the specimen geometry and model 
discretization options.
We create a dictionary with all the 
required key words for 
creating the tension model mesh.

.. GENERATED FROM PYTHON SOURCE LINES 200-212

.. code-block:: Python

    tension_geo_params = {"extensometer_length": 1.0,
                "gauge_length": 1.25,
                "gauge_radius": 0.125,
                "grip_radius": 0.25,
                "total_length": 4,
                "fillet_radius": 0.188,
                "taper": 0.0015,
                "necking_region":0.375,
                "element_size": 0.0125,
                "mesh_method":3,
                "grip_contact_length":1}








.. GENERATED FROM PYTHON SOURCE LINES 213-215

Then we create a :class:`~matcal.sierra.models.RoundUniaxialTensionModel`
that takes the material and geometry as input.

.. GENERATED FROM PYTHON SOURCE LINES 215-217

.. code-block:: Python

    ASTME8_tension_model = RoundUniaxialTensionModel(material, **tension_geo_params) 








.. GENERATED FROM PYTHON SOURCE LINES 218-223

A name is specified so that 
results information can be easily accessed 
and associated with this model. MatCal will 
generate a name for the model, but it may 
be convenient to supply your own.

.. GENERATED FROM PYTHON SOURCE LINES 223-225

.. code-block:: Python

    ASTME8_tension_model.set_name('tension_specimen')








.. GENERATED FROM PYTHON SOURCE LINES 226-232

To ensure the model does not run longer than required for our 
calibration, we use the
:meth:`~matcal.sierra.models.RoundUniaxialTensionModel.set_allowable_load_drop_factor`
method. 
This will end the simulation when the load in the simulation 
has decreased by 25% from peak load.

.. GENERATED FROM PYTHON SOURCE LINES 232-234

.. code-block:: Python

    ASTME8_tension_model.set_allowable_load_drop_factor(0.25)








.. GENERATED FROM PYTHON SOURCE LINES 235-240

To complete the model, MatCal needs boundary condition 
information so that the model is deformed appropriately 
for each data set that is of interest to the calibration. 
We pass the uniaxial tension data collection to the model,
so that it can form the correct boundary conditions for each state.

.. GENERATED FROM PYTHON SOURCE LINES 240-242

.. code-block:: Python

    ASTME8_tension_model.add_boundary_condition_data(tension_data_collection)








.. GENERATED FROM PYTHON SOURCE LINES 243-247

Next, we set optional platform options. 
Since we will run this calibration on either an HPC cluster
or a local machine, we setup the model 
with the appropriate platform specific options. 

.. GENERATED FROM PYTHON SOURCE LINES 247-256

.. code-block:: Python

    if is_sandia_cluster():
        ASTME8_tension_model.run_in_queue(MATCAL_WCID, 0.25)
        ASTME8_tension_model.continue_when_simulation_fails()
        platform = get_sandia_computing_platform()
        num_cores = platform.get_processors_per_node()
    else:
        num_cores = 8
    ASTME8_tension_model.set_number_of_cores(num_cores)








.. GENERATED FROM PYTHON SOURCE LINES 257-262

The model for the top hat shear test 
is built next. The same inputs 
are required for this model. 
First, we build a dictionary 
with all the needed geometry and discretization parameters.

.. GENERATED FROM PYTHON SOURCE LINES 262-277

.. code-block:: Python

    top_hat_geo_params = {"total_height":1.25,
            "base_height":0.75,
            "trapezoid_angle": 10.0,
            "top_width": 0.417*2,
            "base_width": 1.625, 
            "base_bottom_height": (0.75-0.425),
            "thickness":0.375, 
            "external_radius": 0.05,
            "internal_radius": 0.05,
            "hole_height": 0.3,
            "lower_radius_center_width":0.390*2,
            "localization_region_scale":0.0,
            "element_size":0.005, 
            "numsplits":1}








.. GENERATED FROM PYTHON SOURCE LINES 278-280

Next, we create the :class:`~matcal.sierra.models.TopHatShearModel`
and give it a name.

.. GENERATED FROM PYTHON SOURCE LINES 280-283

.. code-block:: Python

    top_hat_model = TopHatShearModel(material, **top_hat_geo_params)
    top_hat_model.set_name('top_hat_shear')








.. GENERATED FROM PYTHON SOURCE LINES 284-286

We set its allowable load drop factor 
and provide boundary condition data. 

.. GENERATED FROM PYTHON SOURCE LINES 286-289

.. code-block:: Python

    top_hat_model.set_allowable_load_drop_factor(0.05)
    top_hat_model.add_boundary_condition_data(top_hat_data_collection)








.. GENERATED FROM PYTHON SOURCE LINES 290-292

Lastly, we setup the platform information 
for running the model. 

.. GENERATED FROM PYTHON SOURCE LINES 292-297

.. code-block:: Python

    top_hat_model.set_number_of_cores(num_cores*2)
    if is_sandia_cluster():
      top_hat_model.run_in_queue(MATCAL_WCID, 30.0/60)
      top_hat_model.continue_when_simulation_fails()








.. GENERATED FROM PYTHON SOURCE LINES 298-304

We now create the objectives for the 
calibration. 
Both models are compared to the data 
using a :class:`~matcal.core.objective.CurveBasedInterpolatedObjective`. 
The tension specimen is calibrated to the engineering stress/strain data
and the top hat specimen is calibrated to the load-displacement data.

.. GENERATED FROM PYTHON SOURCE LINES 304-307

.. code-block:: Python

    tension_objective = CurveBasedInterpolatedObjective("engineering_strain", "engineering_stress")
    top_hat_objective = CurveBasedInterpolatedObjective("displacement", "load")








.. GENERATED FROM PYTHON SOURCE LINES 308-316

With the objectives ready, 
we create :class:`~matcal.core.residuals.UserFunctionWeighting`
objects that will remove data points from the data sets 
that we do not want included in the calibration objective. 
For the tension data, we remove the data in the elastic regime 
and data near failure. 
The following function does this by setting the residuals 
that correspond to these features in the data to zero.

.. GENERATED FROM PYTHON SOURCE LINES 316-326

.. code-block:: Python

    def remove_failure_points_from_residual(eng_strains, eng_stresses, residuals):
        import numpy as np
        weights = np.ones(len(residuals))
        peak_index = np.argmax(eng_stresses)
        peak_strain = eng_strains[peak_index]
        peak_stress = eng_stresses[peak_index]
        weights[(eng_strains > peak_strain) & (eng_stresses < 0.89*peak_stress)  ] = 0
        weights[(eng_strains < 0.005) ] = 0
        return weights*residuals








.. GENERATED FROM PYTHON SOURCE LINES 327-331

The preceding function is used to create 
the :class:`~matcal.core.residuals.UserFunctionWeighting` object
for the tension objective and then added to the 
objective as a weight.

.. GENERATED FROM PYTHON SOURCE LINES 331-336

.. code-block:: Python

    tension_residual_weights = UserFunctionWeighting("engineering_strain", 
                                                     "engineering_stress", 
                                                     remove_failure_points_from_residual)
    tension_objective.set_field_weights(tension_residual_weights)








.. GENERATED FROM PYTHON SOURCE LINES 337-341

A similar modification is required for the top hat data. 
Since the data in the failure region has been removed 
from the data itself, we only remove the data in 
the elastic region with the following function.

.. GENERATED FROM PYTHON SOURCE LINES 341-347

.. code-block:: Python

    def remove_elastic_region_from_top_hat(displacements, loads, residuals):
        import numpy as np
        weights = np.ones(len(residuals))
        weights[(displacements < 0.005) ] = 0
        return weights*residuals








.. GENERATED FROM PYTHON SOURCE LINES 348-351

Then we create our 
:class:`~matcal.core.residuals.UserFunctionWeighting` object
and apply it to the top hat objective.

.. GENERATED FROM PYTHON SOURCE LINES 351-355

.. code-block:: Python

    top_hat_residual_weights = UserFunctionWeighting("displacement", "load", 
                                                     remove_elastic_region_from_top_hat)
    top_hat_objective.set_field_weights(top_hat_residual_weights)








.. GENERATED FROM PYTHON SOURCE LINES 356-362

Now we create the study parameters that 
will be calibrated. We provide
reasonable bounds and assign their 
current value to be the initial point
that we determined in :ref:`6061T6 aluminum anisotropy 
calibration initial point estimation`.

.. GENERATED FROM PYTHON SOURCE LINES 362-371

.. code-block:: Python

    yield_stress = Parameter("yield_stress", 15, 50, 42)
    hardening = Parameter("hardening", 0, 60, 10.1)
    b = Parameter("b", 10, 40, 35.5)
    R22 = Parameter("R22", 0.8, 1.15, 1.05)
    R33 = Parameter("R33", 0.8, 1.15, 0.95)
    R12 = Parameter("R12", 0.8, 1.15, 1.0)
    R23 = Parameter("R23", 0.8, 1.15, 0.97)
    R31 = Parameter("R31", 0.8, 1.15, 0.94)








.. GENERATED FROM PYTHON SOURCE LINES 372-375

Finally, we can create our study. For
This calibration we use a 
:class:`~matcal.dakota.local_calibration_studies.GradientCalibrationStudy`.

.. GENERATED FROM PYTHON SOURCE LINES 375-377

.. code-block:: Python

    study = GradientCalibrationStudy(yield_stress, hardening, b, R22, R33, R12, R23, R31)
    study.set_results_storage_options(results_save_frequency=9)







.. GENERATED FROM PYTHON SOURCE LINES 378-380

We run the study in a subdirectory named ``6061T6_anisotropy``
to keep the current directory cleaner.

.. GENERATED FROM PYTHON SOURCE LINES 380-382

.. code-block:: Python

    study.set_working_directory("6061T6_anisotropy", remove_existing=True)








.. GENERATED FROM PYTHON SOURCE LINES 383-390

We set the core limit so that it runs all model concurrently. 
MatCal knows if the models will be run in a queue on a remote node and will only 
assign one core to each model that is run in a queue. 
Since there are two models with three states and eight 
parameters we need to run a maximum of 54 concurrent models. On a cluster, 
we ensure that we can run all concurrently. On a local platform, we allow MatCal
to use all processors that are available.

.. GENERATED FROM PYTHON SOURCE LINES 390-396

.. code-block:: Python

    if is_sandia_cluster():
      study.set_core_limit(6*9+1)
    else:
      study.set_core_limit(60)









.. GENERATED FROM PYTHON SOURCE LINES 397-399

We add evaluation sets for each model and data set and 
set the output verbosity to the desired level. 

.. GENERATED FROM PYTHON SOURCE LINES 399-403

.. code-block:: Python

    study.add_evaluation_set(ASTME8_tension_model, tension_objective, tension_data_collection)
    study.add_evaluation_set(top_hat_model, top_hat_objective, top_hat_data_collection)
    study.set_output_verbosity("normal")








.. GENERATED FROM PYTHON SOURCE LINES 404-407

The study is then launched and the 
best fit parameters will be printed 
and written to a file after it finished. 

.. GENERATED FROM PYTHON SOURCE LINES 407-411

.. code-block:: Python

    results = study.launch()
    print(results.best.to_dict())
    matcal_save("anisotropy_parameters.serialized", results.best.to_dict())





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    OrderedDict([('yield_stress', 43.466890299), ('hardening', 11.540764702), ('b', 12.397622148), ('R22', 1.0168755292), ('R33', 0.97813235629), ('R12', 0.96795111031), ('R23', 0.92103510217), ('R31', 0.91096387184)])




.. GENERATED FROM PYTHON SOURCE LINES 412-415

We use MatCal's plotting features to 
plot the results and verify a satisfactory 
calibration has been achieved.

.. GENERATED FROM PYTHON SOURCE LINES 415-422

.. code-block:: Python

    import os
    init_dir = os.getcwd()
    os.chdir("6061T6_anisotropy")
    make_standard_plots("displacement", "engineering_strain")
    os.chdir(init_dir)





.. rst-class:: sphx-glr-horizontal


    *

      .. image-sg:: /advanced_examples/6061T6_anisotropic_calibration/images/sphx_glr_plot_6061T6_c_anisotropy_calibration_cluster_003.png
         :alt: plot 6061T6 c anisotropy calibration cluster
         :srcset: /advanced_examples/6061T6_anisotropic_calibration/images/sphx_glr_plot_6061T6_c_anisotropy_calibration_cluster_003.png
         :class: sphx-glr-multi-img

    *

      .. image-sg:: /advanced_examples/6061T6_anisotropic_calibration/images/sphx_glr_plot_6061T6_c_anisotropy_calibration_cluster_004.png
         :alt: plot 6061T6 c anisotropy calibration cluster
         :srcset: /advanced_examples/6061T6_anisotropic_calibration/images/sphx_glr_plot_6061T6_c_anisotropy_calibration_cluster_004.png
         :class: sphx-glr-multi-img

    *

      .. image-sg:: /advanced_examples/6061T6_anisotropic_calibration/images/sphx_glr_plot_6061T6_c_anisotropy_calibration_cluster_005.png
         :alt: plot 6061T6 c anisotropy calibration cluster
         :srcset: /advanced_examples/6061T6_anisotropic_calibration/images/sphx_glr_plot_6061T6_c_anisotropy_calibration_cluster_005.png
         :class: sphx-glr-multi-img

    *

      .. image-sg:: /advanced_examples/6061T6_anisotropic_calibration/images/sphx_glr_plot_6061T6_c_anisotropy_calibration_cluster_006.png
         :alt: plot 6061T6 c anisotropy calibration cluster
         :srcset: /advanced_examples/6061T6_anisotropic_calibration/images/sphx_glr_plot_6061T6_c_anisotropy_calibration_cluster_006.png
         :class: sphx-glr-multi-img

    *

      .. image-sg:: /advanced_examples/6061T6_anisotropic_calibration/images/sphx_glr_plot_6061T6_c_anisotropy_calibration_cluster_007.png
         :alt: plot 6061T6 c anisotropy calibration cluster
         :srcset: /advanced_examples/6061T6_anisotropic_calibration/images/sphx_glr_plot_6061T6_c_anisotropy_calibration_cluster_007.png
         :class: sphx-glr-multi-img

    *

      .. image-sg:: /advanced_examples/6061T6_anisotropic_calibration/images/sphx_glr_plot_6061T6_c_anisotropy_calibration_cluster_008.png
         :alt: plot 6061T6 c anisotropy calibration cluster
         :srcset: /advanced_examples/6061T6_anisotropic_calibration/images/sphx_glr_plot_6061T6_c_anisotropy_calibration_cluster_008.png
         :class: sphx-glr-multi-img

    *

      .. image-sg:: /advanced_examples/6061T6_anisotropic_calibration/images/sphx_glr_plot_6061T6_c_anisotropy_calibration_cluster_009.png
         :alt: plot 6061T6 c anisotropy calibration cluster
         :srcset: /advanced_examples/6061T6_anisotropic_calibration/images/sphx_glr_plot_6061T6_c_anisotropy_calibration_cluster_009.png
         :class: sphx-glr-multi-img

    *

      .. image-sg:: /advanced_examples/6061T6_anisotropic_calibration/images/sphx_glr_plot_6061T6_c_anisotropy_calibration_cluster_010.png
         :alt: plot 6061T6 c anisotropy calibration cluster
         :srcset: /advanced_examples/6061T6_anisotropic_calibration/images/sphx_glr_plot_6061T6_c_anisotropy_calibration_cluster_010.png
         :class: sphx-glr-multi-img






.. rst-class:: sphx-glr-timing

   **Total running time of the script:** (116 minutes 38.494 seconds)


.. _sphx_glr_download_advanced_examples_6061T6_anisotropic_calibration_plot_6061T6_c_anisotropy_calibration_cluster.py:

.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-example

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download Jupyter notebook: plot_6061T6_c_anisotropy_calibration_cluster.ipynb <plot_6061T6_c_anisotropy_calibration_cluster.ipynb>`

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download Python source code: plot_6061T6_c_anisotropy_calibration_cluster.py <plot_6061T6_c_anisotropy_calibration_cluster.py>`

    .. container:: sphx-glr-download sphx-glr-download-zip

      :download:`Download zipped: plot_6061T6_c_anisotropy_calibration_cluster.zip <plot_6061T6_c_anisotropy_calibration_cluster.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
