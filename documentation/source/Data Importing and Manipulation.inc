Data Importing and Manipulation
===============================
All MatCal calibrations and studies require data to which the models
will be compared. MatCal has several custom 
tools for data importing and manipulation. In combination
with what is available in Python, data can be imported, 
processed, analyzed and viewed quickly and easily. 

Data Object Creation and Storage
--------------------------------
First, data must be generated or imported from some 
external source and stored in the appropriate 
object. For MatCal, a single data set must 
be stored in a :class:`~matcal.core.data.Data` object to
be used for most MatCal functions and tools. 
A single data set is defined as group of measured quantities from a single 
test or simulation. In MatCal, the measured quantities
are referred to as fields and each field has an associated field name.
The only requirement on the data is that the data fields must be the 
same length. When the :class:`~matcal.core.data.Data` is 
created in MatCal using its constructor or the
:func:`~matcal.core.data.convert_dictionary_to_data` function no
checks on the data are made. 

The :class:`~matcal.core.data.Data`
objects also have an optional attribute :class:`~matcal.core.state.State`.
The state for a data set is meant to store conditions or 
metadata about the test that is necessary for 
identification and needed for a simulation of the test.
Common state parameters include temperature and rate 
for solid mechanics simulations. States are not
required for a data set but are used throughout
to inform model conditions when necessary and 
pair model results appropriate with experimental 
data. If none is assigned, MatCal will assign 
its default empty state :class:`~matcal.core.state.SolitaryState`
to the data. 

Data importing from external sources is supported 
through valid MatCal data importers found 
in :mod:`~matcal.core.data_importer`. There are three primary tools 
for data importing: 

#. Importing data from single files 
   using :func:`~matcal.core.data_importer.FileData` function. 
#. Import data from several similar files with a filename 
   regular expression using the 
   :class:`~matcal.core.data_importer.BatchDataImporter` class.
#. Downloading data from Sandia's GRANTA database using 
   the :mod:`~matcal.granta` module and
   then importing that data using one of the methods 
   above.

.. warning:: 
    The GRANTA database and tools at Sandia are 
    maintained and developed outside of MatCal's control.
    The database is currently not well supported or 
    curated by a dedicated team
    and is somewhat unreliable. 

A useful class for storing several related 
:class:`~matcal.core.data.Data` objects is 
a MatCal :class:`~matcal.core.data.DataCollection`.
These data collections store data objects
by their states. For each state, the collection 
can store multiple data objects. The intent 
is that when storing experiment data, the collection 
can store multiple repeat experiments for a given state. 
The following code shows how to access the 4th 
repeat data set for state "state_1" in a previously 
created data collection "my_data_collection"::

    my_data = my_data_collection["state_1"][3]

For more data manipulation examples, see
:ref:`Data manipulation`.
   
Importing a single data set from a file
---------------------------------------
The :func:`~matcal.core.data_importer.FileData` 
function will load CSV text files, NumPy arrays saved 
using the NumPy array ``save`` method, or 
MATLAB ".mat" files. The importer ensures 
the data are integer or floating-point values and 
then checks that all data are finite. 
It will allow
string data if requested by the user with the 
"import_strings" argument set to *True*. The imported 
data are returned from the importer in the 
MatCal :class:`~matcal.core.data.Data` format.
The name for the :class:`~matcal.core.data.Data` object is 
set to the absolute file path of the file from 
which it was loaded. The name is intended to be a 
unique identifier for the data set which 
may be important for data traceability.

States can be assigned to the data in three ways:

#. They can be passed as an argument through 
   :func:`~matcal.core.data_importer.FileData`. 
#. It can be assigned to the :class:`~matcal.core.data.Data`
   object through its :meth:`~matcal.core.data.Data.set_state`
   method.
#. Alternatively, state parameters can be placed 
   inside the file as dictionary for CSV file importing.
   This feature is described in detail in 
   below.

CSV file data importing details
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For CSV files in their simplest standard format, 
the function expects 
the first row that it reads to be the 
field names and the following rows to 
be the field values. Each column is dedicated 
to a field and separated by a comma. It also 
expects the files to have a ".csv" file extension, 
although other file extensions can be read in if the file
type is specified as "csv" using the ``file_type`` argument. 
More complex 
CSV file formats are supported, and the importer 
can read in state information from the file
and ignore any comments. Additionally, the 
:func:`~matcal.core.data_importer.FileData` 
function is a wrapper 
for the NumPy "genfromtxt" function when used on CSV 
files and can use most  
valid "genfromtxt" keyword arguments passed to it.
Valid "genfromtxt" keyward arguments include:

#. comments
#. usecols
#. skip_footer
#. converters
#. missing_values
#. filling_values

When reading a CSV file, the function expects 
it to have the following structure::

    {"state_param_1":value_1, "state_param_2":value_2, "state_param_3":"str_value_1"}
    #
    # Optional comments...
    #
    field_name_1, field_name_2, ..., field_name_n
     value[1, 1],  value[1, 1], ...,  value[1, n]
    ...
    #
    # Optional comments dispersed through data
    #
    ...
     value[m, 1],  value[m, 1], ...,  value[m, n]
    #
    # More optional comments...
    #

The first line is an optional dictionary 
containing state information for the data. This dictionary
must have correct Python dictionary
syntax and valid MatCal state parameter names 
and values or the state import will fail. If there 
is no dictionary on the first line, the data 
are loaded with the :class:`~matcal.core.state.SolitaryState`
default MatCal state with no state parameters. 
After the dictionary, there can then be any 
number of comments dispersed throughout 
the file as long as the correct comment character 
has been specified using the keyword argument 
"comments". For the file above, 
the function call would look as follows::

    my_data = FileData("csv_filename.csv", comments="#")

which sets the comment character to "#".
The next line to be read after removing the comments is 
the data header with the data field names followed
by the field data values. All rows that are read in
must have the same number of 
columns. 

For the file above, the data will 
be imported with the N field names and will 
have three state parameters "state_param_1", 
"state_param_2" and "state_param_3" with the appropriate
values for its assigned :class:`~matcal.core.state.State`.
The name for the state is automatically assigned as 
"state_param_1_value_1_state_param_2_value_2_state_param_3_str_value_1".
The numeric state values are formatted in scientific notation 
with a precision equal to six when the name is generated.

NPY and MAT file data importing details
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For ".npy" files, the data importer 
expects that the data are stored 
as a structured array so that 
the data types have names which will become 
:class:`~matcal.core.data.Data` objects
the field names. The data associated with each field 
can be an array of any dimension. 

For MATLAB ".mat" files, the 
data structure variable names will become the field names for
the MatCal :class:`~matcal.core.data.Data` object
and all fields are expected to be one dimensional vectors.

Importing data from a list of filenames or a regular expression
---------------------------------------------------------------
The :class:`~matcal.core.data_importer.BatchDataImporter` class
imports multiple files of a common file type into 
a :class:`~matcal.core.data.DataCollection`. In its 
simplest use case, it imports all data and assigns them
all a common state
:class:`~matcal.core.state.State`. 
For an import with no state information, 
data are assigned
the MatCal default :class:`~matcal.core.state.SolitaryState` state. 
If the "fixed_states" argument is passed to the class
constructor, then the data are assigned
a state named "batch_fixed_state" with the dictionary keyword/value 
pairs from "fixed_states" assigned to them as state parameters. 
The import for each file in the batch 
is done using the :func:`~matcal.core.data_importer.FileData`
function and supports its file types and limitations.
As a result, if all files are CSV files and have states defined in 
the files using a dictionary as the first line,
then the batch importer will store them by those states in 
a :class:`~matcal.core.data.DataCollection` it builds and returns with its 
:meth:`~matcal.core.data_importer.BatchDataImporter.batch` method.

The batch importer can also combine similar states into
a single state if desired. Frequently experiments will have target states,
but the specific value of the state for a given test will 
fluctuate with some error. For example, an experiment 
may be nominally testing a material at 500K. However, 
over three repeats the initial temperatures were measured 
at 498K, 502K, and 505K. To treat these as individual 
states in a calibration or other study
would require running 
the model three separate times since MatCal
evaluates all data states for model when comparing it to that data.
If the difference is negligible, the added computational
cost of running each state may not be acceptable. 
It may be desirable and beneficial, to assign these 
the state value of 500K the target temperature for 
the test. This can be done by setting the 
"state_precision"=1 option for the batch importer
using the :class:`~matcal.core.data_importer.BatchDataImporter.set_options`
method.  Currently, "state_precision" is the only option
that can be specified. An example input for loading 
data with a batch import is below::

   my_batch_importer = BatchDataImporter("test_data*.txt", file_type="csv")
   my_batch_importer.set_options(state_precision=1)
   my_test_data_collection = my_batch_importer.batch

Data manipulation
-----------------
Data manipulation is a vital preprocessing step
when preparing raw experimental data for a 
calibration or other MatCal study. With this in mind, 
the :class:`~matcal.core.data.Data` class
is derived from NumPy arrays :cite:p:`harris2020array` and 
can be manipulated using most NumPy array tools. In addition, 
for both :class:`~matcal.core.data.Data` and 
:class:`~matcal.core.data.DataCollection` we added a few 
methods to aid in manipulation. 

The most useful features for data manipulation 
in MatCal are NumPy's slicing features. 
For a :class:`~matcal.core.data.Data` slicing 
can be used to down sample or select subsets of 
the data. For example, if data have the 
field names "time", "load", "displacement", "temperature" 
with 100,000 points in each field, you may want to
down sample the data for memory reason and 
select only the fields needed for the study.
We will start by removing the unnecessary fields. 
Let's assume that we do not want to use the "time"
field. It can be removed using slice as shown 
below::
 
  data = data[["load", "displacement", "temperature"]]

where the "time" field is not included in the requested
fields from the data. In this case the data overwrites itself
effectively removing the "time" field from the data object.
Another less manual way to remove fields is::

   data_fields = data.field_names
   data_fields.remove("time")
   data = data[data_fields]

The above is what is done when using the data
:meth:`~matcal.core.data.Data.remove_field` method.
Continuing this example, we wish to purge data 
from the data object with displacements less than zero 
and down sample it to around 1000 points. Once 
again, we use slicing to do so::

   data = data[data["displacement"] >= 0]
   desired_data_len = 1000
   interval = int(len(data)/desired_data_len)
   data = data[::interval]

which produces the desired result and 
stores it back in "data". 

Another useful set of
features that :class:`~matcal.core.data.Data` 
inherits from NumPy arrays are all the 
math operators that can act on arrays. 
An example of applying simple scaling and offsets to the data for
general data manipulation is below.::

   from copy import deepcopy
   data_scaled = deepcopy(data)
   data_scaled["displacement"] *= 25.4 # scale from inches to mm 
   data_scaled["load"] -= 100 #shift the load down by a value of 100.

To see more NumPy operators or features that may be of use, 
please visit the NumPy documentation on arrays. 

The last :class:`~matcal.core.data.Data` method
that can be particularly useful when importing data from
different sources is the :meth:`~matcal.core.data.Data.remove_field` method.
It can be used to give a field a new name. When 
using :ref:`MatCal Generated SIERRA Standard Models`, this 
may be required to ensure the correct fields exist in the 
data for boundary condition generation and objective function
calculation. An example of using this method is::

   data.rename_field("old_name", "new_name")
   field_data=data["new_name"]

and the previously named "old_name" field is now accessed using
the "new_name" field name.

MatCal's :class:`~matcal.core.data.DataCollection`
class also has a few data manipulation methods and tools 
available for users. These features simplify 
some of the manipulations above specifically for
data collections. Since data collections are larger, 
nested structures for data, manipulation can be more difficult
due to the layers that must be navigated to access data. 
The following tools are available:

#. The :meth:`~matcal.core.data.DataCollection.remove_field`
   method for data collections which behaves similarly to the 
   method for :class:`~matcal.core.data.Data`, but acts 
   on all data set held in the data collection.
#. The :meth:`~matcal.core.data.DataCollection.plot`
   method which will plot data from a data collection
   when an independent and dependent field is passed 
   to the method. It plots the data on a 2D plot with 
   each state on a separate figure.
#. The :func:`~matcal.core.data.scale_data_collection` function
   that can be used to apply scales and offsets to all 
   data sets in a data collection that have the field name
   requested for scaling. It returns a new scaled data collection
   and does not modify the data collection passed to it.

Examples of these data manipulation
features being used for both 
data preprocessing and 
objective function residual weighting can be found here 
:ref:`Successful Calibration`
and here
:ref:`304L stainless steel viscoplastic calibration`

Full-field Data Specific Features
---------------------------------

.. note::
   Full-field data is currently intended to be only be used on nearly planar two-dimensional
   surfaces. MatCal's :ref:`Virtual Fields Method` is strictly limited to two-dimensional data.
   Technically, MatCal's :ref:`Full-field Interpolation and Extrapolation` and 
   :ref:`Hierarchical Wavelet Decomposition` methods can be used with non-planar surfaces as 
   long as the points are uniquely identified by a given two-dimensional 
   set of spatial cooridnates, e.g. "X" and "Y" pairs.

   Extensions to three-dimensions are planned for future releases.

We simplifiy full-field data manipulation and importing by providing 
an importer that works with common file formats.
The :func:`~matcal.full_field.data_importer.FieldSeriesData` imports field data from CSV 
and exodus files into MatCal. This function stores full-field data and 
its associated global data in 
a single :class:`~matcal.full_field.data.FieldData` object that 
supports all functions of the :class:`~matcal.core.data.Data`
objects but has a few specific full-field data specific attributes.
Since it is derived from the :class:`~matcal.core.data.Data` class, 
it can be used in most MatCal functions and classes that 
operate on :class:`~matcal.core.data.Data` objects.  
:class:`~matcal.full_field.data.FieldData` objects can also be manipulated 
as described in :ref:`Data manipulation` for any of their data 
fields. For field data an important added attribute is the 
:attr:`~matcal.full_field.data.FieldData.spatial_coords`. These hold the 
reference configuration coordinates for the data held in 
the :class:`~matcal.full_field.data.FieldData` objects. 
Except for the :attr:`~matcal.full_field.data.FieldData.spatial_coords`, 
all other field data is stored in two-dimensional arrays that 
can be accessed from the :class:`~matcal.full_field.data.FieldData` object 
using the field name. The rows of the accessed two-dimensional
array correspond to each time step and the data in each 
column corresponds to each point. The indices for each column 
should match accross field in the stored field data by time step.
When accessing the reference configuration spatial coordinates 
from :attr:`~matcal.full_field.data.FieldData.spatial_coords`, a 
single two dimensional array is returned where the columns 
correspond to the "position_names" arguments passed to 
:func:`~matcal.full_field.data_importer.FieldSeriesData` and 
the rows correspond to the number of points. The row 
indices for the spatial coordinates  
correspond to the column indices in the field data 
fields that vary with time for a point. For example, 
the following block of code gets the reference configuration
position and displacement for the same point from a 
:class:`~matcal.full_field.data.FieldData` object.::

   my_field_data = FieldSeriesData("field_data_global.csv")
   point_position_of_interest = my_field_data.spatial_coords[101, :]

   #point_position_of_interest now stores the X/Y postion of point 101
   point_x_displacement_of_interest = my_field_data["X"][:, 101]
   point_y_displacement_of_interest = my_field_data["Y"][:, 101]

   point_x_deformed_position = point_position_of_interest+point_x_displacement_of_interest
   point_y_deformed_position = point_position_of_interest+point_x_displacement_of_interest


.. warning::
   Performing data manipulation, such as slicing on, the data fields in a
   :class:`~matcal.full_field.data.FieldData` object 
   will not also be applied to the spatial coordinates. You will 
   need to perform similar manipulations on the spatial coordinates
   to get the desired results. This due to the fact that
   field data for a given field, e.g. "u" for x displacement, is 
   expected to be a two-dimensional array where the rows correspond to each time 
   step and the columns correspond to each point where the field 
   data is recorded. Since spatial coordinates are constant through 
   time, they are stored as one-dimensional vectors and the slices 
   used on the fields may not apply. Perform the appropriate
   equivalent slice on the spatial coordinate data and then update it 
   in the :class:`~matcal.full_field.data.FieldData` object using
   :meth:`~matcal.full_field.data.FieldData.set_spatial_coords`.

Full-field Data Importing from CSV
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using :func:`~matcal.full_field.data_importer.FieldSeriesData` has some 
notable differences when importing CSV data when compared to the 
:func:`~matcal.core.data_importer.FileData` function.
For CSV data, you must point to a global file that contains
any global data of interest such as "time", "load" or "displacement". 
The global file is required to have a field named "file" 
that can be in any column. The "file" field contains the filename 
of other CSV files that contain all of the relevant field data for 
each time step. For example, the global data CSV file 
for a series of CSV files of field data would look 
similar to::

    #
    # Optional comments...
    #
    global_field_name_1, ..., global_field_name_n, ..., file
     value[1, 1],  ..., value[1, ], ..., field_filename_time_step_1
    ...
    #
    # Optional comments dispersed through data
    #
    ...
     value[m, 1],  ..., value[m, n], ...,  field_filename_time_step_m
    #
    # More optional comments...
    #

.. note::
   Currently, :func:`~matcal.full_field.data_importer.FieldSeriesData` will 
   not read state data from the files. The state information has to 
   be passed as the "state" keyword argument to 
   :func:`~matcal.full_field.data_importer.FieldSeriesData` or can be set 
   using :meth:`~matcal.full_field.data.FieldData.set_state`

The files that contain field data listed in the "file" field of the 
global data file should look like a typical CSV data file
that is read in using the :func:`~matcal.core.data_importer.FileData`
function as documented in :ref:`CSV file data importing details`. Their 
path that is specified in the "file" field can be absolute or 
relative with respect to the path specified in the "series_directory"
keyword argument to :func:`~matcal.full_field.data_importer.FieldSeriesData`.
As with the global file, any state information at the top of the time step
data files are 
ignored. The :attr:`~matcal.full_field.data.FieldData.spatial_coords` are set
from the data in the first file provided in the global filename list 
under the "file" field. The coordinates are imported from the data 
fields named according to the "position_names" keyword argument in 
:func:`~matcal.full_field.data_importer.FieldSeriesData`. They default 
to "X" and "Y", but the "position_names" keyword argument will 
accept a list or tuple of two strings that are valid keys 
in the data file containing the first time step field data.

A useful argument for importing large data sets from CSV is the "n_cores" argument.
It allows a parallel read of file data using subprocesses. The full data object is built 
on the parent thread in serial, but reading the files on separate processes
has shown to reduce the data importing time for CSV files to 
about half of the time loading them in serial. Building the full 
:class:`~matcal.full_field.data.FieldData` object in parallel
is currently not supported because it cannot benefit 
from parallelization due to Python's Global Interpreter Lock.

Full-field Data Importing from Exodus
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :func:`~matcal.full_field.data_importer.FieldSeriesData`
function uses ExodusPy to read in data from an 
exodus mesh results file. It currently ignores 
the "series_directory" and "n_cores" keyword 
arguments when importing data. It will load 
decomposed mesh files as long as the base filename 
for the decomposed mesh files is the same 
as that passed as the "global_filename" argument
to the function. When opening a decomposed set 
of files it uses the Seacas tool EPU to 
reconstruct the decomposed mesh into a single 
mesh. True parallel reads are not currently supported.
Every time step, node variable and element variable are
loaded into the :attr:`~matcal.full_field.data.FieldData` object. 
These can then be accessed through the field names 
and array slicing if necessary.

.. warning::
   If using full-field data tools with your own model, make 
   sure to only output variables and time steps that are needed
   for the study being performed.
   When importing the exodus data, the :func:`~matcal.full_field.data_importer.FieldSeriesData`
   stores all available data in memory and large results files can 
   quickly result in out-of-memory errors.

Generally, the user is not expected to import data from exodus files
unless they are perfoming studies on synthetic data. However, for studies 
using Sierra models with full-field objectives, all of MatCal 
expects results to be in exodus format.