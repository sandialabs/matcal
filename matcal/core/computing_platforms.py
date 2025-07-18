from abc import ABC, abstractmethod
import multiprocessing
import numbers
import socket

from matcal.core.object_factory import IdentifierByTestFunction


class JobSubmitCommandCreatorInterface:
    def __init__(self, commands, environment_setup_commands):
        self._commands = commands
        self._environment_setup_commands = environment_setup_commands

    def get_commands(self):
        return self._commands

    def get_environment_setup_commands(self):
        return self._environment_setup_commands


class Queue:
    def __init__(self, name, node_limit, time_limit_seconds):
        self._name = name
        self._node_limit = node_limit
        self._time_limit_seconds = time_limit_seconds

    @property
    def name(self):
        return self._name

    @property
    def node_limit(self):
        return self._node_limit

    @property
    def time_limit_seconds(self):
        return self._time_limit_seconds


class ComputingPlatform(ABC):
    def __init__(self, name, total_processors, processors_per_node, queues):
        self._name = name
        self._processors_per_node = processors_per_node
        self._queues = queues
        self._total_processors = total_processors

    @property
    def name(self):
        return self._name

    @property
    def processors_per_node(self):
        return self._processors_per_node

    @property
    def queues(self):
        return self._queues

    @property
    def total_processors(self):
        return self._total_processors

    @abstractmethod
    def get_usable_queue_names(self, time_limit, number_of_cores):
        """Required method to return allowable queue names for the platform."""


class LocalComputingPlatform(ComputingPlatform):

    def __init__(self, name=None, total_processors=None, 
                 processors_per_node=None, queues=None):
        name = socket.gethostname()
        total_processors = multiprocessing.cpu_count()
        processor_per_node = total_processors

        super().__init__(name, total_processors, processor_per_node, queues)

    def get_usable_queue_names(self, time_limit, number_of_cores):
        return None


local_computer = LocalComputingPlatform()


class RemoteComputingPlatform(ComputingPlatform):
    def get_usable_queue_names(self, time_limit, number_of_cores):
        """Required method to return allowable queue names for the platform."""


class ImproperTimeFormatError(RuntimeError):
    def __init__(self, *args):
        super().__init__(*args)
        

def _convert_wall_time_string_to_seconds(wall_time_str):
    try:
        wall_time_entries = wall_time_str.split(":")
        time_seconds = 0
        if len(wall_time_entries) == 4:
            time_seconds = float(wall_time_entries[0])*24*60*60+ \
                           float(wall_time_entries[1])*60*60+ \
                           float(wall_time_entries[2])*60+float(wall_time_entries[3])
        elif len(wall_time_entries) == 3:
            time_seconds = float(wall_time_entries[0])*60*60+ \
                            float(wall_time_entries[1])*60+float(wall_time_entries[2])                   
        elif len(wall_time_entries) == 2:
            time_seconds = float(wall_time_entries[0])*60+float(wall_time_entries[1])    
        elif len(wall_time_entries) == 1:
            time_seconds = float(wall_time_entries[0]) 
        else:
            raise ImproperTimeFormatError()
    except (ValueError,AttributeError):
        raise ImproperTimeFormatError()

    return time_seconds


class HPCComputingPlatform(RemoteComputingPlatform):
    """A class to hold hardware information for a high performance computing (HPC)
    machine. This includes queue information, total processors and processors per node. 
    It can be used to inform job setup when a job is run on a particular HPC platform."""
    def __init__(self, name, total_processors, processors_per_node, queues):

        super().__init__(name, total_processors, processors_per_node, queues)

    def get_usable_queue_names(self, job_wall_time, job_processor_count):
        """Return queues that the job can be submitted to for a given run time 
        and processor count."""
        if not isinstance(job_wall_time, numbers.Real):
            time_limit_seconds = _convert_wall_time_string_to_seconds(job_wall_time)
        else:
            time_limit_seconds = job_wall_time
        usable_queues = []
        for queue in self._queues:
            if (queue.time_limit_seconds >= time_limit_seconds and 
                queue.node_limit * self._processors_per_node >= job_processor_count):
                usable_queues.append(queue.name)

        return usable_queues
    
    def get_processors_per_node(self):
        return self._processors_per_node


def return_local_computing_default():
    return LocalComputingPlatform()
MatCalComputingPlatformFunctionIdentifier = \
    IdentifierByTestFunction(return_local_computing_default)


def return_zero():
    return 0
MatCalJobDispatchDelayFunctionIdentifier = IdentifierByTestFunction(return_zero)


def no_check_checker(queue_id, computer, name):
    return True
MatCalPermissionsCheckerFunctionIdentifier = \
    IdentifierByTestFunction(no_check_checker)

