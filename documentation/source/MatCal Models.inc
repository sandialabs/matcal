MatCal Models
=============

MatCal provides a unified interface for integrating different types of models into
calibration, optimization, and uncertainty quantification studies. All MatCal model
classes share common functionality for parameter specification, file setup, and execution,
making it straightforward to switch between different model implementations or combine
multiple models in a single study.

Common Model Interface
----------------------

All MatCal models inherit from a common base class and provide consistent methods for:

* **Adding model constants**: Parameters that remain fixed during the study
* **State management**: Handling different experimental conditions or configurations  
* **File setup**: Preparing working directories and input files
* **Execution**: Running the model with study parameters
* **Result retrieval**: Collecting and returning model outputs

**Parameter Precedence**

All MatCal models follow a four-tier parameter precedence system when combining
parameters from different sources:

1. **State parameters** (from :class:`~matcal.core.state.State`) - Applied first, lowest precedence
2. **Model constants** (from :meth:`~matcal.core.models.ModelBase.add_constants`) - 
   Override state parameters for all states
3. **Model state-specific constants** (from :meth:`~matcal.core.models.ModelBase.add_state_constants`) - 
   Override both state parameters and model constants for a specific state
4. **Study/design parameters** (from the calibration or sampling study) - Highest precedence, 
   override all other parameter types

This precedence system ensures that parameters being actively explored in a study (design parameters)
always take priority over fixed configuration values. State-specific overrides allow fine-grained
control when different experimental conditions require different constant values.

For more details on parameter types and their usage, see :ref:`Parameter Types and Specification`.

**Example of Parameter Precedence:**

.. code-block:: python

    from matcal.core.state import State
    from matcal.core.parameters import Parameter
    from matcal.core.calibration_studies import ScipyMinimizeStudy
    
    # State defines temperature=300
    state = State("room_temp", temperature=300)
    
    # Model adds a constant for temperature=350
    model.add_constants(temperature=350)
    
    # Model adds a state-specific constant for high_temp state
    state_high_temp = State("high_temp", temperature=500)
    model.add_state_constants(state_high_temp, temperature=400)
    
    # Study explores temperature with bounds [200, 600] starting at 300
    temp_param = Parameter("temperature", 200, 600, 300)
    study = ScipyMinimizeStudy(temp_param)  # Parameters passed to constructor
    
    # Results:
    # - Study parameter active: uses study parameter value (highest precedence)
    # - No study parameter for room_temp: temperature = 350 (model constant overrides state)
    # - No study parameter for high_temp: temperature = 400 (state-specific overrides model constant and state)

**Common Methods**

All models provide these methods:

* :meth:`~matcal.core.models.ModelBase.add_constants` - Add parameters for all states
* :meth:`~matcal.core.models.ModelBase.add_state_constants` - Add parameters for specific state
* :meth:`~matcal.core.models.ModelBase.reset_constants` - Remove all added constants
* :meth:`~matcal.core.models.ModelBase.set_name` - Set the model name
* :meth:`~matcal.core.models.ModelBase.run` - Execute the model
* :meth:`~matcal.core.models.ModelBase.preprocess` - Prepare model files and directories

Python Models
-------------

:class:`~matcal.core.models.PythonModel` allows you to use Python functions directly as
models in MatCal studies. This is ideal for fast analytical models, reduced-order models,
or when prototyping before implementing a full external model.

**Basic Usage:**

.. code-block:: python

    from matcal.core.models import PythonModel
    import numpy as np
    
    def my_model(E, nu, stress_levels):
        """
        Simple analytical model.
        
        Parameters are passed as keyword arguments.
        Must return a dictionary with field names as keys.
        """
        strain = stress_levels / E
        return {
            'strain': np.array(strain),
            'poisson_strain': np.array(strain * nu)
        }
    
    # Create model from function
    model = PythonModel(my_model)
    model.set_name("analytical_model")
    
    # Add any fixed constants
    model.add_constants(stress_levels=[100, 200, 300, 400])
    
    # Use in study
    study.add_model(model)

**Loading from External File:**

.. code-block:: python

    # If function is defined in external_model.py
    model = PythonModel("my_model", filename="path/to/external_model.py")

**Parameter Passing Options:**

By default, all parameters are passed as individual keyword arguments. You can change
this behavior with ``pass_params_by_category=True``:

.. code-block:: python

    def categorized_model(model_parameters, model_constants, state_parameters):
        """Receives parameters grouped by category."""
        E = model_parameters['E']
        density = model_constants['density']
        temp = state_parameters['temperature']
        # ... model logic
        return {'stress': np.array([...])}
    
    model = PythonModel(categorized_model, pass_params_by_category=True)

.. warning::
    When ``pass_params_by_category=True``, MatCal's parameter precedence is **not applied**.
    You receive separate dictionaries and must handle conflicts manually.

**Additional Options:**

* ``field_coordinates``: Provide coordinate arrays for field data
* ``pass_evaluation_number``: Pass current evaluation number to function

**Requirements:**

* Function must be picklable (defined at module level, not nested)
* Must return a dictionary: ``{field_name: np.array([values])}``
* All study, state, and constant parameters passed as kwargs (unless using ``pass_params_by_category``)

User Executable Models
----------------------

:class:`~matcal.core.models.UserExecutableModel` runs external executables as models.
This is used when your model is implemented as a compiled program, script, or any
executable that can be called from the command line.

**Basic Usage:**

.. code-block:: python

    from matcal.core.models import UserExecutableModel
    
    # Create model from executable
    model = UserExecutableModel(
        "my_simulation",              # Executable name (must be in PATH)
        "-input", "config.json",      # Command line arguments
        "-output", "results.txt",
        results_filename="results.txt",
        results_file_type="json"      # How to read results
    )
    
    model.set_name("my_external_model")
    
    # Add any necessary files
    model.add_necessary_files("mesh.g", "material_data.txt", "settings/")
    
    # Add constants
    model.add_constants(num_threads=8, tolerance=1e-6)
    
    # Use in study  
    study.add_model(model)

**How It Works:**

1. MatCal creates a working directory for each evaluation
2. Copies necessary files to the working directory
3. Applies Jinja2 templating to all text files with parameters
4. Writes parameter include files for optional reference
5. Runs: ``executable arg1 arg2 ...`` in the working directory
6. Reads results from the specified results file

**Parameter Access: Jinja2 Templating**

MatCal provides parameters to your executable through **Jinja2 templating** of text files
in the working directory. Any text file you add with :meth:`~matcal.core.models.UserExecutableModel.add_necessary_files`
will be automatically processed as a Jinja2 template, with all parameters available for substitution.

All parameters (state parameters, model constants, and study/design parameters) are available
in templates, with MatCal's parameter precedence already applied (study parameters override
model constants, which override state parameters).

**Example Input Template:**

Create an input file template with Jinja2 placeholders:

.. code-block:: python

    # File: simulation_input.txt
    # This file uses Jinja2 syntax for parameter substitution
    
    SIMULATION PARAMETERS
    Young's Modulus: {{ E }}
    Poisson's Ratio: {{ nu }}
    Density: {{ density }}
    Temperature: {{ temperature }}
    
    MESH FILE: mesh.g
    OUTPUT FILE: results.exo

Then reference that template file as a necessary file for running your simulation

.. code-block:: python

    # MatCal script
    from matcal.core.models import UserExecutableModel
    from matcal.core.state import State
    
    model = UserExecutableModel(
        "my_solver",
        "-i", 
        "simulation_input.txt",  # Tells the model to add these to the my_solver aguments
        results_filename="results.exo",  # Exodus II output file
        results_file_type="exo"           # File extension for Exodus II
    )
    
    # Add template file to be coppied to the simulation directory - will be Jinja-processed during run
    model.add_necessary_files("simulation_input.txt")
    
    # Add model constants
    model.add_constants(density=7800)
    
    # Create state with parameters
    state = State("room_temp", temperature=300)
    
    # When the model runs with study parameters E=200e9, nu=0.3:
    # simulation_input.template → simulation_input.txt with values substituted:
    #   Young's Modulus: 200000000000.0
    #   Poisson's Ratio: 0.3
    #   Density: 7800
    #   Temperature: 300

**Advanced Jinja2 Features:**

You can use any Jinja2 feature in your templates:

.. code-block:: jinja

    # Conditionals
    {% if temperature > 500 %}
    USE_THERMAL_EXPANSION = true
    {% else %}
    USE_THERMAL_EXPANSION = false
    {% endif %}
    
    # Loops
    LOAD_STEPS = {% for step in load_steps %}{{ step }} {% endfor %}
    
    # Calculations
    BULK_MODULUS = {{ E / (3 * (1 - 2*nu)) }}

**Parameter Include Files (Optional):**

In addition to Jinja2 templating, MatCal writes two parameter files to the working
directory that your executable can optionally read:

* ``state_values.inc`` - Contains state parameters and model constants (written once during preprocessing)
* ``design_parameter_values.inc`` - Contains only the study/design parameters (written per evaluation)

These files are provided for executables that prefer to read parameters from files
rather than having them templated into input files. The format is ``parameter_name=value``
(one per line), and can be configured using MatCal's parameter reporter system.

**Important:** The ``.inc`` files are **excluded from Jinja2 templating** to prevent
double-processing. All other text files are automatically templated.

.. code-block:: python

    # Example: Executable reads from parameter files
    model = UserExecutableModel(
        "my_solver",
        "--state-file", "state_values.inc",
        "--design-file", "design_parameter_values.inc",
        results_filename="results.csv"
    )

**When to Use Each Approach:**

* **Jinja2 templating (recommended):** Best for creating complete input files with parameters
  embedded directly. Most flexible and requires no special parameter parsing in your executable.

* **Parameter include files:** Useful when your executable already has a parameter file
  reader, or when you want to distinguish between state/constants and design parameters.

* **Both:** You can use Jinja2 for the main input file structure and reference the
  ``.inc`` files from your template if needed.

**Supported Result File Types:**

* ``"csv"`` - CSV files
* ``"json"`` - JSON files  
* ``"npy"`` - NumPy binary files
* ``"mat"`` - MATLAB .mat files
* ``"exo"`` - Exodus II files (also ``"e"`` or ``"g"`` extensions)
* Custom file types can be registered


.. _MatCalSurrogateModel Section:

Surrogate Models
----------------

MatCal provides surrogate models as an efficient alternative to expensive physics
simulations. Once a surrogate is generated from training data, it can be wrapped
as a MatCal model using :class:`~matcal.core.models.MatCalSurrogateModel` and used
in calibration, optimization, or uncertainty quantification studies.

**Using Surrogates in Studies**

A surrogate model is used in MatCal studies by wrapping it with
:class:`~matcal.core.models.MatCalSurrogateModel`::

    from matcal import matcal_load
    from matcal.core.models import MatCalSurrogateModel
    
    # Load a previously generated surrogate
    surrogate = matcal_load("my_surrogate.joblib")
    
    # Wrap it as a MatCal model
    model = MatCalSurrogateModel(surrogate)
    
    # Use in any MatCal study
    study = GradientCalibrationStudy(...)
    study.add_evaluation_set(model, ...)
    # ... configure and run study

**Surrogate Types**

MatCal supports two categories of surrogates:

* **Fixed-sample surrogates**: Built from a predetermined set of model evaluations
  using :class:`~matcal.core.surrogates.SurrogateGenerator`
  
  * :class:`~matcal.core.surrogates.MatCalMultiModalPCASurrogate`
  * :class:`~matcal.core.surrogates.MatCalMonolithicPCASurrogate`

* **Adaptive surrogates**: Built iteratively with automatic sample selection
  from :mod:`~matcal.core.adaptive_surrogates`
  
  * :class:`~matcal.core.adaptive_surrogates.AdaptiveSurrogate`
  * :class:`~matcal.core.adaptive_surrogates.AdaptiveSparseGridSurrogate`

For complete details on generating, configuring, and using surrogates, see the
:ref:`Surrogates` documentation section.

.. toctree::
   :maxdepth: 1

   Surrogates.rst

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

You can derive custom model classes from :class:`~matcal.core.models.ModelBase` to
integrate other simulation tools or implement specialized model workflows. This is
useful when you need more control than :class:`~matcal.core.models.UserExecutableModel`
provides, such as custom file generation, complex preprocessing, or integration with
simulation frameworks.

**Required Methods**

A custom model class must implement four abstract methods:

.. code-block:: python

    from matcal.core.models import ModelBase
    from matcal.core.simulators import SimulatorBase
    from typing import ClassVar
    
    class MyCustomModel(ModelBase):
        """Custom model for my simulation tool."""
        
        # Required class attributes
        model_type: ClassVar[str] = "my_custom_model"
        _simulator_class = MyCustomSimulator  # Your simulator class
        _input_file = "input.xml"  # Input file name or None
        
        def __init__(self, config_file, **kwargs):
            """Initialize your model."""
            super().__init__(executable="my_simulator")
            self._config_file = config_file
            # ... additional initialization
        
        def _get_simulator_class_inputs(self, state):
            """
            Return arguments to create the simulator.
            
            Returns:
                tuple: (args, kwargs) for _simulator_class(*args, **kwargs)
            """
            args = [
                self.name, 
                self._simulation_information,
                self._results_information,
                state,
                self.get_model_constants(state)
            ]
            kwargs = {"config": self._config_file}
            return args, kwargs
        
        def _setup_state(self, state, state_template_dir=None):
            """
            Prepare files and directory for this state.
            
            Called before each model evaluation. Use this to:
            - Generate input files
            - Copy necessary files
            - Set up directory structure
            """
            # Example: Copy config file
            import shutil
            if state_template_dir:
                shutil.copy(self._config_file, state_template_dir)
            
            # Example: Generate custom input file
            self._write_input_file(state, state_template_dir)
        
        def _write_input_file(self, state, directory):
            """Custom method to write input file."""
            input_path = os.path.join(directory, self._input_file)
            # Write your simulation tool's input file format
            # Can access parameters via self.get_model_constants(state)
            # ... custom file writing logic

**Custom Simulator Class**

You also need a simulator class that executes your model:

.. code-block:: python

    from matcal.core.simulators import SimulatorBase
    
    class MyCustomSimulator(SimulatorBase):
        """Simulator that runs my custom model."""
        
        def __init__(self, model_name, sim_info, results_info, state, 
                     model_constants, config):
            super().__init__(model_name, sim_info, results_info)
            self._state = state
            self._constants = model_constants
            self._config = config
        
        def run(self, parameter_dict, working_dir=None):
            """
            Execute the model with given parameters.
            
            Args:
                parameter_dict: Dictionary of study parameters
                working_dir: Directory to run in
                
            Returns:
                SimulatorResults object with model outputs
            """
            # Combine parameters with precedence
            from matcal.core.parameters import _get_parameters_according_to_precedence
            all_params = _get_parameters_according_to_precedence(
                self._state, 
                self._constants, 
                parameter_dict
            )
            
            # Run your simulation tool
            # ... custom execution logic
            
            # Return results
            return self._create_results(output_data, working_dir)

**Example: Integrating a Custom FEA Tool**

.. code-block:: python

    from matcal.core.models import ModelBase
    from matcal.core.simulators import ExecutableSimulator
    from typing import ClassVar
    import os
    
    class MyFEAModel(ModelBase):
        """Model for custom FEA solver."""
        
        model_type: ClassVar[str] = "my_fea"
        _simulator_class = ExecutableSimulator
        _input_file = "model.inp"
        
        def __init__(self, mesh_file, material_model):
            super().__init__(executable="my_fea_solver")
            self._mesh_file = mesh_file
            self._material_model = material_model
            self._results_information.results_filename = "results.exo"
            self._results_information.file_type = "exo"
        
        def _get_simulator_class_inputs(self, state):
            args = [
                self.name,
                self._simulation_information,
                self._results_information,
                state,
                self.get_model_constants(state)
            ]
            kwargs = {"commands": [self.executable, self._input_file]}
            return args, kwargs
        
        def _setup_state(self, state, state_template_dir=None):
            """Generate FEA input file with current parameters."""
            if not state_template_dir:
                return
            
            # Copy mesh
            import shutil
            mesh_dest = os.path.join(state_template_dir, "mesh.g")
            shutil.copy(self._mesh_file, mesh_dest)
            
            # Generate input file
            input_path = os.path.join(state_template_dir, self._input_file)
            self._write_fea_input(input_path, state)
        
        def _write_fea_input(self, filepath, state):
            """Write FEA input file with material model."""
            constants = self.get_model_constants(state)
            
            with open(filepath, 'w') as f:
                f.write(f"MESH: mesh.g\n")
                f.write(f"MATERIAL: {self._material_model}\n")
                # Write material parameters from constants
                for param, value in constants.items():
                    f.write(f"  {param}: {value}\n")
                f.write(f"OUTPUT: results.exo\n")

**Usage:**

.. code-block:: python

    from matcal.core.parameters import Parameter
    from matcal.core.calibration_studies import ScipyMinimizeStudy
    
    # Create custom model
    model = MyFEAModel(
        mesh_file="tensile_specimen.g",
        material_model="elastic_plastic"
    )
    
    # Use like any other model
    model.add_constants(density=7800, poisson_ratio=0.3)
    
    # Create study parameters
    E_param = Parameter("youngs_modulus", 180e9, 220e9, 200e9)
    yield_param = Parameter("yield_stress", 250e6, 450e6, 350e6)
    
    # Create study with parameters
    study = ScipyMinimizeStudy(E_param, yield_param)
    study.add_evaluation_set(model, objectives, data)

**Key Points:**

* Inherit from :class:`~matcal.core.models.ModelBase`
* Implement the four required abstract methods
* Use :meth:`~matcal.core.models.ModelBase.get_model_constants` to access constants
* Parameter precedence is automatically handled if you use
  :func:`~matcal.core.parameters._get_parameters_according_to_precedence`
* Your simulator can inherit from :class:`~matcal.core.simulators.SimulatorBase`
  or :class:`~matcal.core.simulators.ExecutableSimulator` for common patterns
