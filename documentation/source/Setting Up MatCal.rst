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




Confirm Setup
-------------






Advanced Setup 
=======================
To run the advanced setup, it is necessary to to have completed the :ref:`Simple Setup` instructions first.



Building Documentation
======================
TBD

