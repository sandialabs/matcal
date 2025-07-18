MatCal Models
=============
This section covers the interfaces we provide for 
different models...

more coming soon

Python Models
-------------
Go over issues, such as need to be global and work with global variables if 
not MatCal parameters for pickling

User Executable Models
----------------------

Add example - simple Python Model


Surrogate Models
----------------

Add example - building surrogate (might link)

MatCal Generated SIERRA Standard Models
---------------------------------------
Since material characterization tests are frequently standardized, we have
developed a library of validated and maintained models that simulate a 
subset of these tests. Currently, only standard models 
for SIERRA Solid Mechanics have been developed.

These models have been peer-reviewed and used in calibrations found in the
various MatCal examples. For code and model quality, these models are tested 
during each code update in MatCal's unit and regression test suite. During 
production code releases, these models are tested in a larger suite of production tests to 
further ensure their accuracy and quality.
Finally, the models are also validated against experimental data and, when possible, 
material models calibrated using these models are validated against 
validation experiments that typically use nonstandard test geometry.
This provides implicit validation of the models, since they can be 
used to provide calibrated material models that accurately predict 
responses of validation experiments. 

.. toctree::
   :maxdepth: 1

   MatCal Generated SIERRA Standard Models.rst

Deriving Custom Model Classes
-----------------------------