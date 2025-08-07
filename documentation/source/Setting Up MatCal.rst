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
Please note that this section is under active development, with more sections sections and details 
to come in the near future. For any unaddressed questions please file an issue at the github or 
reach out to one of the main points of contact(listed at the github).  

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

    matcal_mesh_decomposer_identifier.register('e', ExodusMeshDecomposer)

Here matcal_mesh_decomposer_identifier is the factory object, 'e' is the key the 
factory used to identify what kind of decomposer to use, and ExodusMeshDecomposer
is the decomposer that will be returned after initialization (if necessary). The 
second action factories perform is returning the desired object for the key passed 
to it. For matcal_mesh_decomposer_identifier this method is identify

.. code-block:: python

    key = 'e'
    decomposer = matcal_mesh_decomposer_identifier.identify(key)

For matcal_mesh_decomposer_identifier, it behaves similar to a dictionary, but 
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

#. matcal_exodus_importer_identifier : How to read in exodus mesh files.
#. matcal_mesh_decomposer_identifier : How to decompose a large mesh for parallel processing
#. matcal_mesh_composer_identifier : How to compose a mesh from its parallel decomposition to a single file. 
#. matcal_module_command_identifier : How to issue commands to environment configuration tool 'module'
#. matcal_parameter_reporter_identifier : (Optional) how to process the results from a MatCal study

If you wish to extend your SIERRA capabilities with MatCal standard models:

#. matcal_cubit_executable_path_identifier : Path to Cubit, a mesh generation tool

If you have a queueing system for your computer systems:

#. matcal_permissions_checker_function_identifier
#. matcal_job_dispatch_delay_function_identifier
#. matcal_computing_platform_function_identifier
#. matcal_platform_environment_setup_identifier


For development and testing, details will be covered in future documentation:

#. matcal_test_platform_options_function_identifier
#. matcal_test_module_identifier


matcal_exodus_importer_identifier
-------------
This factory returns a function that correctly returns the exodus python module
for MatCal to read in exodus meshes. 
This factory is included because exodus installs may require specific environmental 
setups to work correctly. 
This factory expects two functions that do not
take in any external arguments. The first function, the identifier, returns a 
boolean, with True indicating that it is appropriate to return the second function, 
the exodus specifier. 

The exodus specifier function takes in no arguments and returns an exodus python 
module. If there are no complications with exodus on your system the specifier function 
can be as simple as:

.. code-block:: python

    def works_on_all_systems():
        return True

    def import_exodus():
        import exodus3 as exo
        return exo
    
    matcal_exodus_importer_identifier(works_on_all_systems, import_exodus)


matcal_mesh_decomposer_identifier and matcal_mesh_composer_identifier
-------------
These factories return a handle to a class that can properly decompose or compose a given mesh. 
The registration to these classes expects as string of the file extension for a given 
mesh type, and the class of the decomposer/composer. Custom decomposers/composers can be created by 
creating a derived class from `matcal.core.models.MeshDecomposer` or `matcal.core.models.MeshComposer`. 

An fake example that would act on a file of extsion 'mock' is:

.. code-block:: python 

    from matal.core.models import MeshDecomposer, MeshComposer
    from matcal.core.computing_platforms import local_computer
    file_extension = 'mock'

    class MockDecomposer(MeshDecomposer):
        def decompose_mesh(self, mesh_file, number_of_cores, output_directory='.',
                       computer=local_computer)
            print(f"I want decompose {mesh_file} into {number_of_cores} pieces")
        def _build_commands(self):
            pass

    class MockComposer(MeshComposer):
        def compose_mesh(self, mesh_file, number_of_cores, output_directory='.', 
                     computer=local_computer):
            print(f"I want to compose {number_of_cores} pieces of a mesh into one.")
        def _build_commands(self):
            pass

    matcal_mesh_decomposer_identifier.register(file_extension, MockDecomposer)
    matcal_mesh_composer_identifier.register(file_extension, MockComposer)

matcal_platform_environment_setup_identifier
-------------
This factory returns a class that can do specific environment setup if setup 
changes between platforms (e.g. running on a local computer versus a compute cluster).
Registration for this factory is a function to identify what computing platform 
MatCal is currently running on and an instance of your environmental setup class.
The environmental setup class should be derived from 
`matcal.core.external_executable.ExecutableEnvironmentSetupBase`.

A simple example of this is:

.. code-block:: python

    from matcal.core.external_executable import ExecutableEnvironmentSetupBase
    import os

    class MyEnvSetup(ExecutableEnvironmentSetupBase)

        def __init__(self):
            self._ref_value = None
            self._var_name = "MY_IMPORTANT_VARIABLE"

        def prepare(self):
            self._ref_value = os.environ[self._var_name]
            os.environ[self._var_name] = 1

        def reset(self):
            if self._ref_value is not None:
                self.environ[self._var_name] = self._ref_value
    
    def is_compute_cluster():
        return os.environ['IS_HPC_CLUSTER'] == 1

    matcal_platform_environment_setup_identifier.register(is_compute_cluster, MyEnvSetup())


matcal_parameter_reporter_identifier
-------------
This factory will change the way MatCal writes the results files for a calibration study. 
Altering the default behavior may be useful if there are common from the parameters 
are expected to be used or reported. To register with this factory it expects a 
argument less identifier function, and a function that takes in a string filename and 
a dictionary of parameters to record the results. 

.. code-block:: python

    def record_as_csv(filename:str, params:dict):
        keys = list(params.keys())
        key_line = ""
            val_line = ""
            for i_key, key in enumerate(keys):
                if i_key != 0:
                    key_line += ","
                    val_line += ","
                key_line += key
                val_line += str(params[key])
        with open(filename, 'w') as f:
            f.write(key_line+"\n")
            f.write(val_line+"\n")
    
    def make_csv_default():
        return True

    matcal_parameter_reporter_identifier.register(make_csv_default, record_as_csv)    

            


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


Running Local Software with MatCal:
======================
If you have local physics/engineering codes that you want to use with MatCal, the
easiest way to have matcal use these programs is by using the :class:`~matcal.core.models.UserExecutableModel`.
This model allows one to specify an external program to use with MatCal's studies. 
Currently, MatCal passes parameter information my modifying input decks for external
applications. Thus the external program will need to be able to take in text files as 
inputs. 

What follows is an example of linking to an external python executable, and running 
a python file. (This can be done more efficiently using a PythonModel, but this was 
chosen to avoid requiring the reader to download and setup any additional libraries.)

The problem chosen is a simple linear decay problem solved over a period of time. 
the details of the problem are contained in the file solve_decay.py. We will have
run this file previously having set the decay constant to 0.1. This will generate 
a reference data set that we will try to calibrate a decay constant for. Notice 
the value of the decay constant(k) is assigned using jinja in the file we will use for 
this case. 

.. code-block:: python
    # Content of solve_decay.py
    import numpy as np
    from scipy.integrate import odeint

    # Define the linear decay model
    def model(y, t, k):
        dydt = -k * y
        return dydt

    # Parameters
    k = {{ k }} # Decay constant goal: k = 0.1
    y0 = 10  # Initial quantity
    t = np.linspace(0, 50, 100)  # Time points

    # Solve ODE
    solution = odeint(model, y0, t, args=(k,))

    # Exporting the results to a CSV file
    results = pd.DataFrame({'time': t, 'y': solution.flatten()})
    results.to_csv('linear_decay_results.csv', index=False)


In our larger calibration file we will use a :class:`~matcal.core.models.UserExecutableModel`
 to run this file. 
The full calibration script will be included at the end of this section. 
To use our python script as our math model

.. code-block:: python 

    special_python_path = "my/path/to/python3"
    path_to_file = "path/to/solve_decay.py"
    model = UserExecutableModel(special_python_path, path_to_file, results_filename='linear_decay_results.csv')
    model.set_results_filename('linear_decay_results.csv')


More details about the user executable can be found in the API documentation. 
With this model definition we can now use it like any other model in MatCal and 
perform a parameter study. Note that depending on versioning, there may be an 
error thrown about spawning subprocesses. This is a known complication and is 
being addressed. 

.. code-block:: python

    import matcal as mc

    # What parameter are we trying to calibrate (name, lower bound, upper bound)
    decay = mc.Parameter("k", 0, .5) 

    # What data are we calibrating against
    ref_data = mc.FileData("linear_decay_results_reference.csv")

    # How are we comparing the model results and the reference data
    objective = mc.CurveBasedInterpolatedObjective('time', 'y')

    from matcal.core.external_executable import MatCalExecutableEnvironmentSetupFunctionIdentifier
    from matcal.core.file_modifications import use_jinja_preprocessor
    MatCalExecutableEnvironmentSetupFunctionIdentifier._registry={}
    use_jinja_preprocessor()

    # What is our model and how do we expect to find its predictions
    special_python_path = "my/path/to/python3"
    path_to_file = "path/to/solve_decay.py"
    model = mc.UserExecutableModel(special_python_path, path_to_file, results_filename='linear_decay_results.csv')
    model.set_results_filename('linear_decay_results.csv')
    
    # What type of parameter study are we doing
    convergence_tol = 1e-5
    study = mc.ScipyMinimizeStudy(decay, tol=convergence_tol)
    study.add_evaluation_set(model, objective, ref_data) # What will the study compare
    study.set_core_limit(2) # That is the max limit of cores this study can use

    # Run the calibration and print the results.
    results = study.launch()
    print(results.best)




