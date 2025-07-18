Options for Parallel Computing
==============================
In MatCal, the use of parallel processing is supported in several ways. 
The primary methods involve how many processes can be used by a model and 
how many processes a study is limited to use as a maximum through out the study.
The former is set using :meth:`~matcal.core.UserExecutableModel.set_number_of_cores`
and the latter is set using :math:`~matcal.core.parameter_studies.ParameterStudy.set_core_limit`.

Model `set_number_of_cores` method
----------------------------------
First, we will describe specifics of the `set_number_of_cores` method. 
This method is valid for all MatCal models. The value passed to `set_number_of_cores` 
is how many cores the job will use to 
execute one evaluation of the model. 

.. note::
    The :class:`~matcal.core.models.PythonModel` and the 
    :class:`~matcal.core.models.UserExecutableModel` do not actually enforce the use 
    of the number of cores. MatCal just uses this for calculating how many cores are currently 
    in use when these models are run. The user is expected to control how the models are run in parallel
    through their use provided functions or files. 

For :ref:`MatCal Generated SIERRA Standard Models`, 
MatCal passes the number of cores to SIERRA which runs the model on that many processes. This
number should be strictly enforced.

Study `set_core_limit` method
-----------------------------
The `set_core_limit` method for :ref:`MatCal Studies` sets a hard limit to how many 
cores can be used by models run by the study. Since many :ref:`MatCal Studies` 
can evaluate objectives for different parameter sets concurrently and a single parameter
evaluation may require the execution of several models, this is used to ensure
that the available processes or cores on a machine is not oversubscribed. 
For example, if a study needs to evaluate 10 models for a single objective and 10 parameter set values
for a given parameter batch, it would run 100 models concurrently for maximum parallelism. However, 
if each model uses 5 cores and the available cores on the machine where the study is run is 32, 
then it is not possible to run all 100 models concurrently. If 15 was passed to `set_core_limit`, 
then three models would be run concurrently. A new one would be started every time one finished 
until all 100 models for the current parameter set batch completed for the study and a new 
parameter set batch was determined by the study algorithm so that it could continue.

The previous paragraph describes how this works when all models are run on 
the same computing platform or node as the study. However, 
on high performance computing platforms, jobs may be run on different nodes or
even on remote platforms through a queueing system. When a model is set to run in this 
way using :meth:`matcal.core.UserExecutableModel.run_in_queue`, then the model 
only counts as the study using one processes when the the study enforces the 
study core limit even if the model uses more than one core. This is because the 
cores for the model are not loading the machine where the study is being run. The study 
just needs to account for a single process for job monitoring and results processing. 

.. note::
    Once again, the :class:`~matcal.core.models.PythonModel` and the 
    :class:`~matcal.core.models.UserExecutableModel` do not actually submit the 
    jobs to the queueing system. The user is expected to do so within 
    files and functions supplied for these models.

