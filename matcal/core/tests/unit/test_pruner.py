from matcal.core.tests.MatcalUnitTest import MatcalUnitTest
from matcal.core.pruner import *
from matcal.core.constants import MATCAL_WORKDIR_STR


def _make_matcal_workdirs_with_objectives(dir: str, n: int):
  """
  writes objective files with strictly increasing objectives
  not at all a real world situation but good for testing,
  otherwise the behavior of KeepBest is the same as KeepLast
  """
  dirs = []
  for i in range(n):
    new_dir = _make_test_dir(dir, f'{MATCAL_WORKDIR_STR}.{i + 1}')
    dirs.append(new_dir)
    with open(os.path.join(new_dir, 'objective.out'), 'w') as objective_file:
      objective_file.write(str(float(f'{i}e-1')))
  _assert_all_exist(dirs)
  return dirs


def _make_matcal_workdirs_with_multiple_objectives(dir: str, n: int):
  """
  writes objective files with strictly increasing pairs of objectives
  not at all a real world situation but good for testing,
  otherwise the behavior of KeepBest is the same as KeepLast
  """
  dirs = []
  for i in range(n):
    new_dir = _make_test_dir(dir, f'{MATCAL_WORKDIR_STR}.{i + 1}')
    dirs.append(new_dir)
    with open(os.path.join(new_dir, 'objective.out'), 'w') as objective_file:
      objective_file.write(str(float(f'{i}e-1')) + '\n')
      objective_file.write(str(float(f'{i}e-1')))
  _assert_all_exist(dirs)
  return dirs


def _make_test_dir(dir: str, subdir='test_dir') -> str:
  new_dir = os.path.join(dir, subdir)
  os.mkdir(new_dir)
  assert os.path.isdir(new_dir)
  return new_dir


def _make_test_file_paths(dir: str) -> [str]:
  files = ['test1', 'test2', 'test3']
  file_paths = [os.path.join(dir, f) for f in files]
  return file_paths

def _touch(file_path: str) -> None:
  """
  opens and closes a file to create a new empty file
  """
  open(file_path, 'a').close()


def _assert_all_exist(file_paths: [str]):
  for file_path in file_paths:
    assert os.path.exists(file_path)


def _assert_none_exist(file_paths: [str]):
  for file_path in file_paths:
    assert not os.path.exists(file_path)


def _make_matcal_workdirs(dir: str, n: int):
  dirs = [_make_test_dir(dir, f'{MATCAL_WORKDIR_STR}.{i + 1}') for i in range(n)]
  _assert_all_exist(dirs)
  return dirs


def _make_test_files(dir: str) -> [str]:
  file_paths = _make_test_file_paths(dir)
  for file_path in file_paths:
    _touch(file_path)
  _assert_all_exist(file_paths)
  return file_paths


class AssessorTesting(MatcalUnitTest):

  def setUp(self):
    super().setUp(__file__)

  def test_no_directories(self):
    assessor = DirectoryPrunerKeepAll()
    result = assessor.assess()
    self.assertEqual(result, [])

  def test_1_directory(self):
    dirs = _make_matcal_workdirs(self.build_dir, 1)
    assessor = DirectoryPrunerKeepAll()
    result = assessor.assess()
    self.assertEqual(result, [])

  def test_2_directory(self):
    dirs = _make_matcal_workdirs(self.build_dir, 2)
    assessor = DirectoryPrunerKeepAll()
    result = assessor.assess()
    self.assertEqual(result, [])

  def test_keep_all(self):
    dirs = _make_matcal_workdirs(self.build_dir, 10)
    assessor = DirectoryPrunerKeepAll()
    result = assessor.assess()
    self.assertEqual(result, [])

  def test_keep_last_1(self):
    dirs = _make_matcal_workdirs(self.build_dir, 30)
    assessor = DirectoryPrunerKeepLast()
    result = assessor.assess()
    self.assertEqual(sorted(result), sorted(dirs[1:-1]))

  def test_keep_last_10_percent(self):
    dirs = _make_matcal_workdirs(self.build_dir, 30)
    assessor = DirectoryPrunerKeepLastTenPercent()
    result = assessor.assess()
    self.assertEqual(sorted(result), sorted(dirs[1:-3]))

  def test_keep_last_20_percent(self):
    dirs = _make_matcal_workdirs(self.build_dir, 30)
    assessor = DirectoryPrunerKeepLastTwentyPercent()
    result = assessor.assess()
    self.assertEqual(sorted(result), sorted(dirs[1:-6]))

  def test_keep_last_custom_percent(self):
    dirs = _make_matcal_workdirs(self.build_dir, 100)
    assessor = DirectoryPrunerKeepLastXPercent(37)
    result = assessor.assess()
    goal = dirs[1:-37]
    
    for goal_item in goal:
      self.assertTrue(goal_item in result)
    
  def test_work_dir_objective(self):
    dirs = _make_matcal_workdirs(self.build_dir, 1)
    test_val = 3.14159265e4
    with open(os.path.join(dirs[0], 'objective.out'), 'w') as objective_file:
      objective_file.write(str(test_val))
    assessor = DirectoryPrunerKeepBestXPercent(0)
    self.assertEqual(assessor._work_dir_objective(dirs[0]), test_val)

  def test_work_dir_objective_multiple(self):
    dirs = _make_matcal_workdirs(self.build_dir, 1)
    test_val1 = 1.5
    test_val2 = 2.718
    with open(os.path.join(dirs[0], 'objective.out'), 'w') as objective_file:
      objective_file.write(str(test_val1) + '\n')
      objective_file.write(str(test_val2))
    assessor = DirectoryPrunerKeepBestXPercent(0)
    self.assertEqual(assessor._work_dir_objective(dirs[0]), test_val1 + test_val2)

  def test_keep_best_custom_percent(self):
    dirs = _make_matcal_workdirs_with_objectives(self.build_dir, 30)
    assessor = DirectoryPrunerKeepBestXPercent(30)
    result = assessor.assess()
    self.assertEqual(result, dirs[9:])

  def test_keep_best_10_percent(self):
    dirs = _make_matcal_workdirs_with_objectives(self.build_dir, 30)
    assessor = DirectoryPrunerKeepBestTenPercent()
    result = assessor.assess()
    self.assertEqual(result, dirs[3:])

  def test_keep_best_20_percent(self):
    dirs = _make_matcal_workdirs_with_multiple_objectives(self.build_dir, 30)
    assessor = DirectoryPrunerKeepBestTwentyPercent()
    result = assessor.assess()
    self.assertEqual(result, dirs[6:])

  def test_keep_best_custom_percent_multiple(self):
    dirs = _make_matcal_workdirs_with_multiple_objectives(self.build_dir, 30)
    assessor = DirectoryPrunerKeepBestXPercent(30)
    result = assessor.assess()
    self.assertEqual(result, dirs[9:])

  def test_keep_best_10_percent_multiple(self):
    dirs = _make_matcal_workdirs_with_multiple_objectives(self.build_dir, 30)
    assessor = DirectoryPrunerKeepBestTenPercent()
    result = assessor.assess()
    self.assertEqual(result, dirs[3:])

  def test_keep_best_20_percent_multiple(self):
    dirs = _make_matcal_workdirs_with_multiple_objectives(self.build_dir, 30)
    assessor = DirectoryPrunerKeepBestTwentyPercent()
    result = assessor.assess()
    self.assertEqual(result, dirs[6:])


class EliminatorTesting(MatcalUnitTest):

  def setUp(self):
    super().setUp(__file__)

  def test_empty_list(self):
    eliminator = Eliminator()
    eliminator.eliminate([])
    eliminator.eliminate(None)

  def test_nonexistent_files(self):
    file_paths = ['does not exist', 'garbage', 'nothing']
    eliminator = Eliminator()
    eliminator.eliminate(file_paths)

  def test_mixed_files(self):
    """
    some files which exist, and some which don't
    """
    file_paths = _make_test_files(self.build_dir)
    eliminator = Eliminator()
    os.remove(file_paths[0])
    eliminator.eliminate(file_paths + ['nonexistent', 'garbage'])
    _assert_none_exist(file_paths)

  def test_existing_files(self):
    file_paths = _make_test_files(self.build_dir)
    eliminator = Eliminator()
    eliminator.eliminate(file_paths)
    _assert_none_exist(file_paths)

  def test_empty_directory(self):
    new_dir = _make_test_dir(self.build_dir)
    eliminator = Eliminator()
    eliminator.eliminate([new_dir])
    assert not os.path.isdir(new_dir)

  def test_full_directory(self):
    new_dir = _make_test_dir(self.build_dir)
    file_paths = _make_test_files(new_dir)
    eliminator = Eliminator()
    eliminator.eliminate([new_dir])
    _assert_none_exist(file_paths + [new_dir])

  def test_recursive(self):
    # create top-level directory
    dir1 = _make_test_dir(self.build_dir)
    # populate top-level directory
    dir1_files = _make_test_files(dir1)
    # create directory inside top-level
    dir2 = _make_test_dir(dir1)
    # populate nested directory
    dir2_files = _make_test_files(dir2)
    all_paths = dir1_files + dir2_files + [dir1, dir2]
    _assert_all_exist(all_paths)
    eliminator = Eliminator()
    eliminator.eliminate([dir1])
    _assert_none_exist(all_paths)
