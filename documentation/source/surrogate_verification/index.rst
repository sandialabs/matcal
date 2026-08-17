:orphan:

Verification Examples
---------------------
Each example produces a convergence plot and diagnostic plots showing training
sample locations over the true Peaks function and the corresponding surrogate
absolute-error field.

``plot_a1_peaks_fixed_sample_random_gp.py``
    Builds fixed-sample Gaussian-process surrogates from nested random training
    sets.

``plot_a2_peaks_fixed_sample_random_rbf.py``
    Builds fixed-sample RBF surrogates from nested random training sets.

``plot_a3_peaks_fixed_sample_random_random_forest.py``
    Builds fixed-sample random-forest surrogates from nested random training sets.

``plot_a4_peaks_adaptive_voronoi_gp.py``
    Builds an adaptive Gaussian-process surrogate using KFCV-Voronoi sampling.

``plot_a5_peaks_adaptive_voronoi_rbf.py``
    Builds an adaptive RBF surrogate using KFCV-Voronoi sampling.

``plot_a6_peaks_adaptive_voronoi_random_forest.py``
    Builds an adaptive random-forest surrogate using KFCV-Voronoi sampling.

``plot_a7_peaks_adaptive_sparse_grid.py``
    Builds an adaptive sparse-grid surrogate using PyApprox through MatCal's
    sparse-grid adaptive surrogate study.




.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example verifies a fixed-sample Gaussian process surrogate on the 2D Peaks benchmark function used by voronoi_adaptive_surrogates. This example is  part of the verification example set for surrogates. This set of examples is meant to demonstrate features and behavior of the surrogates in MatCal.  They are not meant to be used as templates for user MatCal files.">

.. only:: html

  .. image:: /surrogate_verification/images/thumb/sphx_glr_plot_a1_peaks_fixed_sample_random_gp_thumb.png
    :alt:

  :ref:`sphx_glr_surrogate_verification_plot_a1_peaks_fixed_sample_random_gp.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Paper Peaks Verification: Fixed-sample Random-Sampling Gaussian Process Surrogate</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example verifies a fixed-sample radial basis function (RBF) surrogate on the 2D Peaks benchmark function used by voronoi_adaptive_surrogates. This example is  part of the verification example set for surrogates. This set of examples is meant to demonstrate features and behavior of the surrogates in MatCal.  They are not meant to be used as templates for user MatCal files.">

.. only:: html

  .. image:: /surrogate_verification/images/thumb/sphx_glr_plot_a2_peaks_fixed_sample_random_rbf_thumb.png
    :alt:

  :ref:`sphx_glr_surrogate_verification_plot_a2_peaks_fixed_sample_random_rbf.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Paper Peaks Verification: Fixed-sample Random-Sampling RBF Surrogate</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example verifies a fixed-sample random forest surrogate on the 2D Peaks benchmark function used by voronoi_adaptive_surrogates. This example is  part of the verification example set for surrogates. This set of examples is meant to demonstrate features and behavior of the surrogates in MatCal.  These examples are not meant to be used as templates for user MatCal files.">

.. only:: html

  .. image:: /surrogate_verification/images/thumb/sphx_glr_plot_a3_peaks_fixed_sample_random_random_forest_thumb.png
    :alt:

  :ref:`sphx_glr_surrogate_verification_plot_a3_peaks_fixed_sample_random_random_forest.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Paper Peaks Verification: Fixed-sample Random-Sampling Random Forest Surrogate</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example verifies MatCal&#x27;s KFCV-Voronoi adaptive surrogate workflow on the 2D Peaks benchmark function from voronoi_adaptive_surrogates.  It builds a Gaussian process surrogate using the adaptive sampling. This example is  part of the verification example set for surrogates. This set of examples is meant to demonstrate features and behavior of the surrogates in MatCal.  They are not meant to be used as templates for user MatCal files.">

.. only:: html

  .. image:: /surrogate_verification/images/thumb/sphx_glr_plot_a4_peaks_adaptive_voronoi_gp_thumb.png
    :alt:

  :ref:`sphx_glr_surrogate_verification_plot_a4_peaks_adaptive_voronoi_gp.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Paper Peaks Verification: Adaptive Voronoi Gaussian Process Surrogate</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example verifies MatCal&#x27;s Voronoi adaptive surrogate workflow on the 2D Peaks benchmark function from voronoi_adaptive_surrogates.  It builds a radial basis function (RBF) surrogate using the adaptive sampling.  This example is part of the verification example set for surrogates.  This set of examples is meant to demonstrate features and  behavior of the surrogates in MatCal. They are not meant to be used as templates for user MatCal files.">

.. only:: html

  .. image:: /surrogate_verification/images/thumb/sphx_glr_plot_a5_peaks_adaptive_voronoi_rbf_thumb.png
    :alt:

  :ref:`sphx_glr_surrogate_verification_plot_a5_peaks_adaptive_voronoi_rbf.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Paper Peaks Verification: Adaptive Voronoi RBF Surrogate</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example verifies MatCal&#x27;s KFCV-Voronoi adaptive surrogate workflow on the 2D Peaks benchmark function from voronoi_adaptive_surrogates.  It builds a random forest surrogate using the adaptive sampling. This example is  part of the verification example set for surrogates. This set of examples is meant to demonstrate features and behavior of the surrogates in MatCal.  These examples are not meant to be used as templates for user MatCal files.">

.. only:: html

  .. image:: /surrogate_verification/images/thumb/sphx_glr_plot_a6_peaks_adaptive_voronoi_random_forest_thumb.png
    :alt:

  :ref:`sphx_glr_surrogate_verification_plot_a6_peaks_adaptive_voronoi_random_forest.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Paper Peaks Verification: Adaptive Voronoi Random Forest Surrogate</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example verifies MatCal&#x27;s adaptive sparse-grid surrogate workflow on the 2D Peaks benchmark function from voronoi_adaptive_surrogates. This example is  part of the verification example set for surrogates. This set of examples is meant to demonstrate features and behavior of the surrogates in MatCal.  These examples are not meant to be used as templates for user MatCal files.">

.. only:: html

  .. image:: /surrogate_verification/images/thumb/sphx_glr_plot_a7_peaks_adaptive_sparse_grid_thumb.png
    :alt:

  :ref:`sphx_glr_surrogate_verification_plot_a7_peaks_adaptive_sparse_grid.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Paper Peaks Verification: Adaptive Sparse-Grid Surrogate</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /surrogate_verification/plot_a1_peaks_fixed_sample_random_gp
   /surrogate_verification/plot_a2_peaks_fixed_sample_random_rbf
   /surrogate_verification/plot_a3_peaks_fixed_sample_random_random_forest
   /surrogate_verification/plot_a4_peaks_adaptive_voronoi_gp
   /surrogate_verification/plot_a5_peaks_adaptive_voronoi_rbf
   /surrogate_verification/plot_a6_peaks_adaptive_voronoi_random_forest
   /surrogate_verification/plot_a7_peaks_adaptive_sparse_grid


.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-gallery

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download all examples in Python source code: surrogate_verification_python.zip </surrogate_verification/surrogate_verification_python.zip>`

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download all examples in Jupyter notebooks: surrogate_verification_jupyter.zip </surrogate_verification/surrogate_verification_jupyter.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
