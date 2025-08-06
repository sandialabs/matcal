************
Setting Up MatCal
************


Simple Setup 
=======================
These instructions for setting up MatCal go over the basic steps to get the MatCal
repository cloned to your local machine, installing the key MatCal dependencies, 
creating an appropriate MatCal Python environment, and confirming that the 
setup is working correctly. 

This setup will allow users to use MatCal in a limited capacity. MatCal will be
able to interface with models that have a Python interface and be limited to optimization 
methods from scipy. To enable more complex models and optimization methods, following
the :ref:`Advanced Setup` will be required. 

Instructions for compiling the documentation will be added in the near future. 

Clone Repository
-------------
From the main page of the MatCal repository, copy your preferred remote URL (HTTPS or SSH).
Open up a terminal on your local computer, and navigate to the desired directory to contain
the MatCal repository. 

Clone the GitHub repo by executing the following command:

.. code-block:: bash
    
    git clone <URL>

Where <URL> is replaced with the GitHub URL. 
The repo will be cloned in to a directory called 'matcal'.


Set Python Environment
-------------
It is recommended that you create a new python environment to run MatCal in. 
Conda can be used to do this. Currently, MatCal has been developed with Python 
version 3.11.5. To create this python environment you can do: 

.. code-block:: bash

    conda create --name matcal python=3.11.5

The MatCal environment can now be activated with the command:

.. code-block:: bash

    conda activate matcal

After activating the MatCal environment, you can install the prerequisites by
running the following pip command:

.. code-block:: bash

    pip install -r matcal/requirements.txt

The last step before confirming the installation is to add MatCal to your computer's 
path, so it knows where to look for MatCal. This can be done a variety of ways. 
It is recommended that you add the MatCal path using conda in your MatCal conda environment.
This way you can be sure that your path is only altered when necessary. To do this you will
need conda-build which can be installed by executing:

.. code-block:: bash

    conda install conda-build

Then you add your path to the top level MatCal directory by

.. code-block:: bash

    conda develop </path/to/matcal>

If this approach does not work you can edit your path directly 

.. code-block:: bash 

    export PATH="</path/to/matcal>:$PATH"

If this command is executed in the terminal it will only work for this current
terminal session. To have it occur every time you open a new terminal, you will need
to add this command to one of the start-up files for your terminal (such as .bashrc).


Confirm Setup
-------------
In this section we will confirm that the core of MatCal is working correctly, by
running some of MatCal's unit and production tests. 

To run the unit tests go to the core module's unit test directory and run unittest.

.. code-block:: bash
    
    cd matcal/matcal/core/tests/unit
    python -m unittest

These tests confirm that the basic code that supports MatCal is executing correctly. 
If there are problems with MatCal these test help isolate where the problem originates from. 
After the tests run, you can confirm that the simple setup as been done correctly
if the only tests that fail report problems with loading 'site_matcal'.
(At time of writing only 1 test is failing for this reason.)
Setting up 'site_matcal' will be covered in a later section. 

If the unit tests pass you can further confirm that the simple setup has been done 
correctly if you run the core production tests.

.. code-block:: bash

    cd matcal/matcal/core/tests/production
    python -m unittest


This set of tests will run a series of problems that are similar to applied problems
you would want to solve using MatCal. They combine a wide range of the code sections
of MatCal to perform more complex analysis. If the simple setup has been done correctly 
then after running the production tests it should report no test failues other than 
those originating from problems with 'site_matcal'.

At this point you can now start using some of the basic features of MatCal. To enable 
more features, such as enabling a direct interface to finite element software or 
to use advanced optimizer and sampling tools from the Dakota software package 
please see the :ref:`Advanced Setup` section.



Advanced Setup 
=======================
To run the advanced setup, it is necessary to to have completed the :ref:`Simple Setup` instructions first.

Creating Platform Specifics
-------------
Before establishing platform specifics it is recommended that all desired optional 
installs are performed before establishing the side specific details.

A key concept in customizing MatCal to meet new needs or work on specific platforms
is the concept of a factory. MatCal has several factories that build objects during runtime,
while not requiring any new code to alter existing code. This patten adheres to the 
principal of "open to extension, close to modification". What this means is that 
users can extend the capabilities of MatCal, while not needing to alter the way the core of MatCal works. 

Factories have two distinct actions they perform. The first is registration, at initialization 
of the program all the factories are populated with the information they need to make the 
various objects they are in charge of. In MatCal this is done with a 'register' method. 
Creating code that looks like

.. code-block:: python

    MatCalMeshDecomposerIdentifier.register('e', ExodusMeshDecomposer)

Here MatCalMeshDecomposerIdentifier is the factory object, 'e' is the key the 
factory used to identify what kind of decomposer to use, and ExodusMeshDecomposer
is the decomposer that will be returned after initialization (if necessary). The 
second action factories perform is returning the desired object for the key passed 
to it. For MatCalMeshDecomposerIdentifier this method is identify

.. code-block:: python

    key = 'e'
    decomposer = MatCalMeshDecomposerIdentifier.identify(key)

For MatCalMeshDecomposerIdentifier, it behaves similar to a dictionary, but 
other factories are more complicated and can identify what to return based on 
function calls or other criteria. 


Registration Location
-------------
To get MatCal to register custom and site specific tools, MatCal looks for an 
"__init__.py" file inside of a "matcal/site_matcal". you will need to create this
directory and file. It is also recommended that any custom code development occur 
inside 'site_matcal' as much as possible. When MatCal is importing, it will 
use the "__init__.py" file to know what to expose inside the "site_matcal" directory. 

An simple example "__init__.py" file can look like:

.. code-block:: python 

    __all__ = []

    from . import mysite
    from .mysite import *
    __all__ += mysite.__all__

In this example the file 'mysite.py' is imported and all functions and classes within 
'mysite' is exposed. Files can also be run in these the __init__.py files. This
is how MatCal registers all of the various options for its factories. For example

.. code-block:: python 

    __all__ = []

    from . import mysite
    from .mysite import *
    __all__ += mysite.__all__

    import site_matcal.register_factories

modifies the initial example to run the file 'register_factories.py'. It is in 
this type of file that all of the factory registration is recommended to be done. 

What to Register
-------------
If you have access to a SIERRA distribution. The following factories will need to be registered to
link SIERRA to MatCal:

#. matcal_exodus_importer_identifier : if using Cubit is used for mesh generation, this will setup the correct pathing and environment.
#. matcal_mesh_decomposer_identifier : How to decompose a large mesh for parallel processing
#. matcal_mesh_composer_identifier : How to compose a mesh from its parallel decomposition to a single file. 
#. matcal_module_command_identifier
#. matcal_parameter_reporter_identifier : (optional)

If you wish to extend your SIERRA capabilities with MatCal standard models:

#. matcal_cubit_executable_path_identifier

If you have a queueing system for your computer systems:

#. matcal_permissions_checker_function_identifier
#. matcal_job_dispatch_delay_function_identifier
#. matcal_job_dispatch_delay_function_identifier
#. matcal_platform_environment_setup_identifier


For development and testing, details will be covered in future documentation:
#. matcal_test_platform_options_function_identifier
#. matcal_test_module_identifier


Installing Dakota
-------------
Dakota is an advanced optimization and uncertainty quantification(UQ) library
developed by Sandia National Labs. It contains an array of useful methods for 
the calibration and study of material models. 

Dakota must be downloaded and installed from the `Dakota website <https://dakota.sandia.gov/>`.
This process changes depending on what type of machine you are installing Dakota on. 
for simple Linux, Mac, and Windows configurations there are binaries that you can 
download. This reduces the install process to just adding the binary to an appropriate
directory and adding it to your path. Some configurations, such as Intel Mac, computers
do not have binary distributions of the most recent version of Dakota, and will require
either compiling recent versions locally or using an older version of Dakota. 

When you are done installing Dakota, you can check to make sure that your install works 
correctly by running the dakota tests in matcal/dakota/tests/unit.


Installing Cubit
-------------
TBD


Building Documentation
======================
TBD

