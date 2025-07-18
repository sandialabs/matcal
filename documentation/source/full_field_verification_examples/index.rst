:orphan:

Full-field Methods Verification
===============================
In this set of examples, we perform 
verification exercises for MatCal's 
full-field data methods. These examples
provide verification of some of the 
tools that contribute to full-field
data calibration. These verification 
examples are extensions to some of the 
tests that are required to pass 
for each MatCal update.

.. _alink:




.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="#.  We evaluate the function on a set of points over a     a domain that is 5% smaller then the domain we      will interpolate and extrapolate to. This will      be referred to as our measurement grid and is     meant to be representative of experimental data. #.  We add noise with a normal distribution to the      data generated in the previous step. The noise      has a maximum amplitude of 2.5% of the function      maximum value to represent the noise present      in measured data. #.  We create a separate domain with 75% of the points      from the measured grid that is 5% larger     in both the X and Y directions and evaluate the function      at these points without noise. This is      to be used as the truth      value of the function and this set of      points will be referred to as the simulation      grid. We will attempt to reproduce      these values with GMLS      interpolation and extrapolation. #.  We loop over different input options to the GMLS      algorithm and evaluate the accuracy of the method     against the truth data with three measures of error:     (1) the maximum percent error of the field produced      by the GMLS tool, (2) the a normalized L2 norm of this      field and (3) plots of the error field for all of the      input options studied for the GMLS algorithm.">

.. only:: html

  .. image:: /full_field_verification_examples/images/thumb/sphx_glr_plot_a_interpolation_methods_verification_thumb.png
    :alt:

  :ref:`sphx_glr_full_field_verification_examples_plot_a_interpolation_methods_verification.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Full-field Interpolation Verification</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="#.  We evaluate the function on a set of points over a     a predetermined domain. This will      be referred to as our measurement grid and is     meant to be representative of experimental data. #.  We add noise with a normal distribution to the      data generated in the previous step. The noise      has a maximum amplitude of 2.5% of the function      maximum value to represent the noise present      in measured data. #.  We create a separate domain with the same number of points      from the measured grid that is unstructured and evaluate the function      at these points without noise. This is      to be used as the truth      value of the function and this set of      points will be referred to as the simulation      cloud.  #.  We loop over different input options to the HWD      algorithm and evaluate the accuracy of the method     against the truth data with five measures of error:     (1) the normalized maximum percent error of the weights produced      by the HWD tool, (2) the a normalized L2 norm of these      weights, (3) the maximum percent error      of the function reconstructed on the simulation cloud      using the experimental grid HWD weights     HWD, (4) the normalized L2 norm      of this function and (5) plots of the reconstructed function     data error for a subset of the      input options studied for the HWD algorithm.">

.. only:: html

  .. image:: /full_field_verification_examples/images/thumb/sphx_glr_plot_hwd_methods_verification_not_collocated_thumb.png
    :alt:

  :ref:`sphx_glr_full_field_verification_examples_plot_hwd_methods_verification_not_collocated.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Polynomial HWD Verification - Analytical Function</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This test is performed  on the experimental data for one of the X specimens (XR4) and  the same data that has been mapped  to a simulation mesh surface using  MatCal&#x27;s meshless_remapping function.">

.. only:: html

  .. image:: /full_field_verification_examples/images/thumb/sphx_glr_plot_hwd_methods_verification_not_collocated_X_specimen_thumb.png
    :alt:

  :ref:`sphx_glr_full_field_verification_examples_plot_hwd_methods_verification_not_collocated_X_specimen.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Polynomial HWD Verification - X Specimen Data</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Once again, we will generate two instances of our full-field data with noise and strive to have their difference be as  small as possible. The same procedure is used here a was used in Polynomial HWD Verification - Analytical Function.">

.. only:: html

  .. image:: /full_field_verification_examples/images/thumb/sphx_glr_plot_j_hwd_methods_verification_thumb.png
    :alt:

  :ref:`sphx_glr_full_field_verification_examples_plot_j_hwd_methods_verification.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Polynomial HWD Verification with Colocated Points</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="    \sigma_{yy} =\frac{L}{W T}">

.. only:: html

  .. image:: /full_field_verification_examples/images/thumb/sphx_glr_plot_vfm_methods_verification_thumb.png
    :alt:

  :ref:`sphx_glr_full_field_verification_examples_plot_vfm_methods_verification.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Virtual Fields Method Verification</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /full_field_verification_examples/plot_a_interpolation_methods_verification
   /full_field_verification_examples/plot_hwd_methods_verification_not_collocated
   /full_field_verification_examples/plot_hwd_methods_verification_not_collocated_X_specimen
   /full_field_verification_examples/plot_j_hwd_methods_verification
   /full_field_verification_examples/plot_vfm_methods_verification


.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-gallery

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download all examples in Python source code: full_field_verification_examples_python.zip </full_field_verification_examples/full_field_verification_examples_python.zip>`

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download all examples in Jupyter notebooks: full_field_verification_examples_jupyter.zip </full_field_verification_examples/full_field_verification_examples_jupyter.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
