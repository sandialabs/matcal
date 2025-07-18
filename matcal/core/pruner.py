import glob
import os
from pathlib import Path
from abc import ABC, abstractmethod
from time import sleep

from matcal.core.constants import MATCAL_WORKDIR_STR
from matcal.core.utilities import _sort_numerically

class DirectoryPrunerBase(ABC):
  """
  Base class for all directory pruners, which each have criteria for which directories to prune and which to keep
  Note: all subclasses keep the first work directory, and return a list of all directories which are not kept
  """

  @staticmethod
  def _get_all_work_dirs() -> [str]:
    """
    returns all the MatCal work directories created by the study
    """
    workdirs = _sort_numerically(glob.glob(MATCAL_WORKDIR_STR+".*"))
    return [os.path.join(os.getcwd(), x) for x in workdirs]


  @abstractmethod
  def assess(self) -> [str]:
    """
    Returns a list of directories to be deleted
    """
    

class DirectoryPrunerKeepLast(DirectoryPrunerBase):
  """
  Keeps only the first and last directory
  """
  def assess(self) -> [str]:
    all_work_dirs = self._get_all_work_dirs()
    return all_work_dirs[1:-1]


class DirectoryPrunerKeepLastXPercent(DirectoryPrunerBase):
  """
  Keeps first directory and last (percent)% of directories
  """
  def __init__(self, percent):
    self.percent = percent

  def assess(self) -> [str]:
    all_work_dirs = self._get_all_work_dirs()
    return all_work_dirs[1:-int(len(all_work_dirs) * self.percent / 100)]


class DirectoryPrunerKeepAll(DirectoryPrunerKeepLastXPercent):
  """
  Keeps all directories, meaning that assess() will always return the empty list
  This is the default behavior of MatCal
  """
  def __init__(self):
    super().__init__(0)


class DirectoryPrunerKeepLastTenPercent(DirectoryPrunerKeepLastXPercent):
  """
  Keeps first directory and last 10% of directories
  """
  def __init__(self):
    super().__init__(10)


class DirectoryPrunerKeepLastTwentyPercent(DirectoryPrunerKeepLastXPercent):
  """
  Keeps first directory and last 20% of directories
  """
  def __init__(self):
    super().__init__(20)


class DirectoryPrunerKeepBestXPercent(DirectoryPrunerBase):
  """
  Keeps best (percent)% of directories
  Best directories are those with the lowest objective
  """

  def __init__(self, percent):
    self.percent = percent

  @staticmethod
  def _work_dir_objective(workdir):
    with open(os.path.join(workdir, 'objective.out')) as objective:
      content = objective.readlines()
    return DirectoryPrunerKeepBestXPercent._norm([float(line.strip()) for line in content])
    
  @staticmethod
  def _norm(objectives):
    return sum(objectives)

  def assess(self) -> [str]:
    sorted_work_dirs = sorted(self._get_all_work_dirs(), key=self._work_dir_objective)
    return sorted_work_dirs[int(len(sorted_work_dirs) * self.percent / 100) : ]


class DirectoryPrunerKeepBestTenPercent(DirectoryPrunerKeepBestXPercent):
  """
  Keeps best 10% of directories
  """
  def __init__(self):
    super().__init__(10)


class DirectoryPrunerKeepBestTwentyPercent(DirectoryPrunerKeepBestXPercent):
  """
  Keeps best 20% of directories
  """
  def __init__(self):
    super().__init__(20)


class Eliminator:

  def eliminate(self, paths: [str]) -> None:
    """
    removes all files present in paths
    if path is a file, they are simply removed
    if path is a directory, it is recursively removed (DANGEROUS)
    if paths is an empty iterable, None, empty string, etc., return without doing anything
    """
    if not paths:
      return

    for path in paths:
      if os.path.isfile(path):
        os.remove(path)
      elif os.path.isdir(path):
        self.remove_directory_recursive(path)

  def remove_directory_recursive(self, dir: str) -> None:
    directory = Path(dir)
    for item in directory.iterdir():
      if item.is_dir():
        self.remove_directory_recursive(item)
      else:
        item.unlink()
        self.pause_to_ensure_ample_file_access_time()
    directory.rmdir()

  def pause_to_ensure_ample_file_access_time(self):
      sleep(1e-2)





