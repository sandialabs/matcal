import gc
import os

from matcal.core.constants import BATCH_RESTART_FILENAME
from matcal.core.restart_file import (BatchRestartCSV, BatchRestartHDF5)
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest


class BatchRestartTests(MatcalUnitTest):

    class CommonSetUp(MatcalUnitTest):
        @property
        def _batch_restart_class(self):
            """"""

        def setUp(self):
            super().setUp(__file__)

    class CommonTests(CommonSetUp):

        def test_init(self):
            save_only = True
            br = self._batch_restart_class(save_only)
            br.close()

        def test_record_and_retrieve_if_exists_else_None_and_not_save_only(self):
            empty_br = self._batch_restart_class(True)
            empty_br.close()
            gc.collect()

            save_only = False
            br = self._batch_restart_class(save_only)
            eval_name = 'eval.1'
            model_name = 'model'
            state_name = 'matcal_default_state'
            results_filename = 'results.csv'
            job_key = [eval_name, model_name, state_name]
            none_job_key = ['eval.2', model_name, state_name]
            goal_file= results_filename
            br.record(job_key, results_filename)
            self.assertEqual(br.retrieve_results_file(job_key), goal_file)
            self.assertIsNone(br.retrieve_results_file(none_job_key))
            br.close()

        def test_if_save_only_retrieve_returns_None_always(self):
            save_only = True
            br = self._batch_restart_class(save_only)
            eval_name = 'eval.1'
            model_name = 'model'
            state_name = 'matcal_default_state'
            results_filename = 'results.csv'
            job_key = [eval_name, model_name, state_name]
            none_job_key = ['eval.2', model_name, state_name]
            goal_file= results_filename
            br.record(job_key, results_filename)
            self.assertIsNone(br.retrieve_results_file(job_key), goal_file)
            self.assertIsNone(br.retrieve_results_file(none_job_key))
            br.close()

        def test_None_filename_does_not_get_written(self):
            save_only = True
            br = self._batch_restart_class(save_only)

            restart_file = f"{BATCH_RESTART_FILENAME}"+br.file_extension()
            old_file_size = os.path.getsize(restart_file)

            br.record(['a', 'b', 'c'], None)
            self.assertIsNone(br.retrieve_results_file(['a', 'b', 'c']))
            os.path.getsize(restart_file)
            new_file_szie = os.path.getsize(restart_file)
            self.assertEqual(new_file_szie, old_file_size)

        def test_write_to_file_durring_a_record(self):
            save_only = True
            br = self._batch_restart_class(save_only)
            restart_file = f"{BATCH_RESTART_FILENAME}"+br.file_extension()

            old_file_size = os.path.getsize(restart_file)
            br.record(['a', 'b', 'c'], 'a.txt')
            new_file_size = os.path.getsize(restart_file)
            self.assertGreater(new_file_size, old_file_size)
            old_file_size = new_file_size
            br.record(['a', 'b', 'd'], 'b.txt')
            new_file_size = os.path.getsize(restart_file)
            self.assertGreater(new_file_size, old_file_size)
            old_file_size = new_file_size
            br.record(['a', 'b', '3'], 'c.txt')
            new_file_size = os.path.getsize(restart_file)
            self.assertGreater(new_file_size, old_file_size)
            old_file_size = new_file_size


class TestBatchRestartHDF5(BatchRestartTests.CommonTests):

    _batch_restart_class = BatchRestartHDF5


class TestBatchRestartCSV(BatchRestartTests.CommonTests):

    _batch_restart_class = BatchRestartCSV