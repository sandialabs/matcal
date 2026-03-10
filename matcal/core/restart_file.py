from abc import ABC, abstractmethod
from io import IOBase

from matcal.core.logger import initialize_matcal_logger


logger = initialize_matcal_logger(__name__)


class BatchRestartBase(ABC):

    @abstractmethod
    def record(self, job_keys:list, results_filename:str)->None:
        """"""
    
    @abstractmethod
    def _retrieve_results_file_impl(self, job_keys:list)->str:
        """"""

    @abstractmethod
    def file_extension(self)->str:
        """"""

    @abstractmethod
    def open_restart_file(self)->IOBase:
        """"""

    def __init__(self, restart_file_handle:str, restart:bool):
        self._restart = restart
        self._restart_file_handle = restart_file_handle
        self._finished_jobs = {}

    def _create_h5_group(self, job_keys:list)->str:
        group_name = ""
        for i, key_element in enumerate(job_keys):
            if i > 0:
                group_name += "/"
            group_name += f"{key_element}"
        return group_name
    
    @property
    def default_lookup_return(self):
        return None
        
    def retrieve_results_file(self, job_keys:list)->str:
        if not self._restart:
            return self.default_lookup_return
        return self._retrieve_results_file_impl(job_keys)


class BatchRestartNone(BatchRestartBase):
    # Used to turn off file saving which may interfere with 3rd party libraries. 

    def record(self, job_keys, results_filename):
        print("Batch Restart Recording")

    def _retrieve_results_file_impl(self, job_keys):
        print("Batch Restart Retrieving")
        return self.default_lookup_return

    def file_extension():
        return None


class BatchRestartCSV(BatchRestartBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._restart:
            self._finished_jobs = self._get_finished_jobs_info()

    def _get_finished_jobs_info(self):
        finished_jobs = {}
        for line in self._restart_file_handle.readlines():
            job_key, results_filename = line.split(",")
            finished_jobs[job_key] = results_filename.strip()
        return finished_jobs
    
    def record(self, job_keys:list, results_filename:str)->None:
        if not isinstance(results_filename, str):
            return None
        group_name = self._create_h5_group(job_keys)
        self._finished_jobs[group_name] = results_filename
        self._restart_file_handle.write(f'{group_name},{results_filename}\n') 
        self._restart_file_handle.flush()

    def _retrieve_results_file_impl(self, job_keys:list)->str:
        group_name = self._create_h5_group(job_keys)
        if group_name in self._finished_jobs:
            res_filename = self._finished_jobs[group_name]
        else:
            res_filename = self.default_lookup_return
        return res_filename

    @staticmethod
    def file_extension():
        return ".csv"

    @staticmethod
    def open_restart_file():
        return open


class BatchRestartHDF5(BatchRestartBase):

    def record(self, job_keys:list, results_filename:str)->None:
        if not isinstance(results_filename, str):
            return None
        group_name = self._create_h5_group(job_keys)
        g = self._restart_file_handle.create_group(group_name)
        g.create_dataset('results', data=[results_filename])

    def _retrieve_results_file_impl(self, job_keys:list)->str:
        group_name = self._create_h5_group(job_keys)
        if group_name in self._restart_file_handle:
            res_filename = self._restart_file_handle[group_name]['results'][0].decode('ascii')
        else:
            res_filename = self.default_lookup_return
        return res_filename

    @staticmethod
    def file_extension():
        return ".h5"
    
    @staticmethod
    def open_restart_file():
        import h5py
        return h5py.File
    
    
SelectedBatchRestartClass = BatchRestartHDF5
