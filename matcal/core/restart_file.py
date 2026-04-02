from abc import ABC, abstractmethod
from io import IOBase
import os

from matcal.core.logger import initialize_matcal_logger


logger = initialize_matcal_logger(__name__)


class BatchRestartBase(ABC):

    @abstractmethod
    def _write_record(self, fh, job_keys: list, results_filename: str) -> None:
        """"""

    @abstractmethod
    def _read_record(self, fh, job_keys: list) -> str:
        """"""

    @property
    @abstractmethod
    def file_extension(self)->str:
        """"""

    @staticmethod
    @abstractmethod
    def get_open_command(self)->IOBase:
        """"""

    def __init__(self, restart_filename: str, restart: bool):
        self._restart = restart
        self._restart_filename = restart_filename
        self._finished_jobs = {}

        # Ensure the restart file exists and is clean/valid for this run
        self._initialize_restart_file()

        # Load existing entries into memory if this is a restart run
        if self._restart:
            self._load_finished_jobs()

    def _initialize_restart_file(self) -> None:
        """
        Create/truncate or open the restart file then close immediately.

        - restart=True and file exists: open with r+ (do not truncate)
        - otherwise: open with w (truncate/clean)
        """
        open_cmd = self.get_open_command()

        # BatchRestartNone or misconfigured
        if open_cmd is None or self._restart_filename is None:
            return

        if self._restart and os.path.exists(self._restart_filename):
            mode = "r+"
        else:
            mode = "w"

        # Note: for h5py.File, "w" truncates/creates; "r+" opens existing read/write.
        with open_cmd(self._restart_filename, mode):
            pass

    @classmethod
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

    @property
    def restart(self):
        return self._restart

    def _open_for_read(self):
        open_cmd = self.get_open_command()
        if open_cmd is None:
            return None
        return open_cmd(self._restart_filename, "r")

    def _open_for_update(self):
        open_cmd = self.get_open_command()
        if open_cmd is None:
            return None
        return open_cmd(self._restart_filename, "r+")

    def _load_finished_jobs(self):
        try:
            with self._open_for_read() as fh:
                if fh is None:
                    return
                self._finished_jobs = self._read_all_finished_jobs(fh)
        except FileNotFoundError:
            # restart requested but file missing: treat as no finished jobs
            self._finished_jobs = {}

    def _read_all_finished_jobs(self, fh) -> dict:
        """
        Subclasses should override for efficient whole-file loading.
        """
        return {}

    def retrieve_results_file(self, job_keys: list) -> str:
        if not self._restart:
            return self.default_lookup_return

        group_name = self._create_h5_group(job_keys)
        # Fast path: in-memory map
        if group_name in self._finished_jobs:
            return self._validated_lookup_return(self._finished_jobs[group_name])

        # Fallback: check file in case something updated it since init
        fh = self._open_for_read()
        if fh is None:
            return self.default_lookup_return
        with fh:
            res = self._read_record(fh, job_keys)

        if res is not None:
            self._finished_jobs[group_name] = res

        return self._validated_lookup_return(res)

    def _validated_lookup_return(self, results_filename: str) -> str:
        """
        Apply the robustness policy for restart entries (e.g., file must exist).
        Returns the filename if valid, otherwise returns default_lookup_return (None).
        """
        if results_filename is None:
            return self.default_lookup_return
        if not isinstance(results_filename, str):
            return self.default_lookup_return
        if not os.path.exists(results_filename):
            return self.default_lookup_return
        return results_filename

    def record(self, job_keys: list, results_filename: str) -> None:
        if not isinstance(results_filename, str):
            return None
        if not os.path.exists(results_filename):
            return None

        fh = self._open_for_update()
        if fh is None:
            return None

        group_name = self._create_h5_group(job_keys)
        with fh:
            self._write_record(fh, job_keys, results_filename)

        self._finished_jobs[group_name] = results_filename


class BatchRestartCSV(BatchRestartBase):

    file_extension = ".csv"

    def _read_all_finished_jobs(self, fh) -> dict:
        finished_jobs = {}
        fh.seek(0)

        for line_num, line in enumerate(fh.readlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                job_key, results_filename = line.split(",", 1)
            except ValueError:
                # Likely a truncated/corrupt line from an unclean exit during write.
                # Ignore it and keep everything else.
                logger.warning(
                    f"Skipping malformed restart line {line_num} in {self._restart_filename!r}: {line!r}"
                )
                continue

            finished_jobs[job_key] = results_filename.strip()

        return finished_jobs
    
    def _write_record(self, fh, job_keys: list, results_filename: str) -> None:
        group_name = self._create_h5_group(job_keys)
        fh.seek(0, os.SEEK_END)
        fh.write(f"{group_name},{results_filename}\n")
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except Exception:
            # Some environments/filesystems may not support fsync
            pass

    def _read_record(self, fh, job_keys: list) -> str:
        group_name = self._create_h5_group(job_keys)
        fh.seek(0)
        found = self.default_lookup_return

        for line_num, line in enumerate(fh.readlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                job_key, results_filename = line.split(",", 1)
            except ValueError:
                # Ignore malformed/truncated lines
                continue

            if job_key == group_name:
                # keep scanning so "last write wins" even if duplicates exist
                found = results_filename.strip()

        return found

    @staticmethod
    def get_open_command():
        return open


class BatchRestartHDF5(BatchRestartBase):

    file_extension = ".h5"

    def _read_all_finished_jobs(self, fh) -> dict:
        finished = {}

        def visitor(name, obj):
            try:
                if hasattr(obj, "keys") and "results" in obj:
                    val = obj["results"][0]
                    if isinstance(val, (bytes, bytearray)):
                        val = val.decode("ascii")
                    finished[name] = str(val)
            except Exception:
                return

        fh.visititems(visitor)
        return finished

    def _write_record(self, fh, job_keys: list, results_filename: str) -> None:
        group_name = self._create_h5_group(job_keys)

        # Replace existing group if present to avoid create_group error
        if group_name in fh:
            del fh[group_name]

        g = fh.create_group(group_name)
        g.create_dataset("results", data=[results_filename])

        try:
            fh.flush()
        except Exception:
            pass


    def _read_record(self, fh, job_keys: list) -> str:
        group_name = self._create_h5_group(job_keys)
        if group_name in fh and "results" in fh[group_name]:
            val = fh[group_name]["results"][0]
            if isinstance(val, (bytes, bytearray)):
                return val.decode("ascii")
            return str(val)
        return self.default_lookup_return
    
    @staticmethod
    def get_open_command():
        import h5py
        return h5py.File


class BatchRestartNone(BatchRestartBase):
    # Used to turn off file saving for testing
    file_extension = None

    def _write_record(self, fh, job_keys: list, results_filename: str) -> None:
        return None

    def _read_record(self, fh, job_keys: list) -> str:
        return None


    @staticmethod
    def get_open_command():
        return None


SelectedBatchRestartClass = BatchRestartCSV
