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



Building Documentation
======================
TBD

