:orphan:

Study Examples
--------------
Each example produces a convergence plot, diagnostic plots showing training
sample locations over parameter space and the worst surrogate predictions and errors 
produced at the end of training. The surrogates are 
evaluated against a common-test set so that qualitative comparisons can be made.

``plot_generate_surrogate_a_standard.py``
    Builds a standard PCA-based surrogate from a Latin-hypercube training study.
    This example predicts both ``TC_top`` and ``TC_bottom``. Near the end, it
    scores the surrogate on the common Halton test set used by the adaptive
    examples.

``plot_generate_surrogate_b1_adaptive_voronoi_GP.py``
    Builds a Voronoi adaptive surrogate for ``TC_bottom`` using a Gaussian
    Process backend. The adaptive sampling loop adds training points in regions
    where cross-validation indicates the surrogate needs improvement.

``plot_generate_surrogate_b2_adaptive_voronoi_RBF.py``
    Builds a Voronoi adaptive surrogate for ``TC_bottom`` using an RBF backend
    instead of a Gaussian Process backend. This example is useful when users want
    a deterministic local interpolator without Gaussian-process predictive
    uncertainty.

``plot_generate_surrogate_c_adaptive_sparse_grid.py``
    Builds a PyApprox sparse-grid adaptive surrogate for ``TC_bottom``. This
    example demonstrates an alternative adaptive strategy that can be effective
    for smooth responses in moderate-dimensional parameter spaces.




.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to generate a basic surrogate from a MatCal study. This example will cover:">

.. only:: html

  .. image:: /surrogate_examples/images/thumb/sphx_glr_plot_generate_surrogate_a_standard_thumb.png
    :alt:

  :ref:`sphx_glr_surrogate_examples_plot_generate_surrogate_a_standard.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Surrogate Generation Example</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to generate a surrogate using a MatCal study that performs adaptive sampling for training the surrogate. This study is a follow-on example to  Surrogate Generation Example and uses the same boundary value problem from that example. The primary difference is that a Matcal adaptive surrogate  study is used for surrogate training.  In this example we use a VoronoiAdaptiveSurrogateStudy to create the surrogate. ">

.. only:: html

  .. image:: /surrogate_examples/images/thumb/sphx_glr_plot_generate_surrogate_b1_adaptive_voronoi_GP_thumb.png
    :alt:

  :ref:`sphx_glr_surrogate_examples_plot_generate_surrogate_b1_adaptive_voronoi_GP.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Voronoi Adaptive Surrogate Example</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to generate a Voronoi adaptive surrogate using MatCal with an RBF backend surrogate regressor.">

.. only:: html

  .. image:: /surrogate_examples/images/thumb/sphx_glr_plot_generate_surrogate_b2_adaptive_voronoi_RBF_thumb.png
    :alt:

  :ref:`sphx_glr_surrogate_examples_plot_generate_surrogate_b2_adaptive_voronoi_RBF.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Voronoi Adaptive Surrogate Example with RBF Backend</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to generate a surrogate using a MatCal study that performs adaptive sampling for training the surrogate. This study is a follow-on example to  Surrogate Generation Example and uses the same boundary value problem from that example. The primary difference is that a Matcal adaptive surrogate  study is used for surrogate training.  In this example we use a SparseGridAdaptiveSurrogateStudy to create the surrogate. ">

.. only:: html

  .. image:: /surrogate_examples/images/thumb/sphx_glr_plot_generate_surrogate_c_adaptive_sparse_grid_thumb.png
    :alt:

  :ref:`sphx_glr_surrogate_examples_plot_generate_surrogate_c_adaptive_sparse_grid.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Sparse Grid Adaptive Surrogate Example</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /surrogate_examples/plot_generate_surrogate_a_standard
   /surrogate_examples/plot_generate_surrogate_b1_adaptive_voronoi_GP
   /surrogate_examples/plot_generate_surrogate_b2_adaptive_voronoi_RBF
   /surrogate_examples/plot_generate_surrogate_c_adaptive_sparse_grid


.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-gallery

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download all examples in Python source code: surrogate_examples_python.zip </surrogate_examples/surrogate_examples_python.zip>`

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download all examples in Jupyter notebooks: surrogate_examples_jupyter.zip </surrogate_examples/surrogate_examples_jupyter.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
