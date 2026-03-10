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
            self._restart_filename = BATCH_RESTART_FILENAME
            file_extension = self._batch_restart_class.file_extension
            if file_extension is not None:
                self._restart_filename += file_extension
            self._open_method = self._batch_restart_class.get_open_command()

    class CommonTests(CommonSetUp):

        def test_instantiation_and_no_file_created(self):
            # The file should not exist before we open it
            self.assertFalse(os.path.exists(self._restart_filename))

            with self._open_method(self._restart_filename, "w") as fh:
                br = self._batch_restart_class(fh, restart=False)
                self.assertIsInstance(br, self._batch_restart_class)

            # Even after closing the handle the file should exist
            self.assertTrue(os.path.isfile(self._restart_filename))

        def test_record_and_retrieve_if_exists_else_None_when_restart_run(self):
            with self._open_method(self._restart_filename, "w") as restart_file_handle:
                br = self._batch_restart_class(restart_file_handle, restart=False)
            gc.collect()

            with self._open_method(self._restart_filename, "r+") as restart_file_handle:
                br = self._batch_restart_class(restart_file_handle, restart=True)
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

        def test_if_not_restart_retrieve_returns_None_always(self):
            with self._open_method(self._restart_filename, "w") as restart_file_handle:
                br = self._batch_restart_class(restart_file_handle, restart=False)
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

        def test_None_filename_does_not_get_written(self):
            with self._open_method(self._restart_filename, "w") as restart_file_handle:
                br = self._batch_restart_class(restart_file_handle, restart=False)
                restart_file = f"{BATCH_RESTART_FILENAME}"+br.file_extension
                old_file_size = os.path.getsize(restart_file)

                br.record(['a', 'b', 'c'], None)
                self.assertIsNone(br.retrieve_results_file(['a', 'b', 'c']))
                os.path.getsize(restart_file)
                new_file_szie = os.path.getsize(restart_file)
                self.assertEqual(new_file_szie, old_file_size)

        def test_write_to_file_during_a_record(self):
            with self._open_method(self._restart_filename, "w") as restart_file_handle:
                br = self._batch_restart_class(restart_file_handle, restart=False)
                restart_file = f"{BATCH_RESTART_FILENAME}"+br.file_extension

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

        def test_persistence_across_close_open(self):
            # Record while file is open, close it, then reopen and read
            with self._open_method(self._restart_filename, "w") as fh:
                br = self._batch_restart_class(fh, restart=False)
                job_key = ['j', 'k', 'l']
                br.record(job_key, 'out.h5')
            # Re‑open for reading
            with self._open_method(self._restart_filename, "r") as fh:
                br = self._batch_restart_class(fh, restart=True)
                self.assertEqual(br.retrieve_results_file(['j', 'k', 'l']), 'out.h5')

        def test_empty_file_restart(self):
            # Create an empty file, then open with restart=True – should not crash
            open(self._restart_filename, "w").close()
            with self._open_method(self._restart_filename, "r+") as fh:
                br = self._batch_restart_class(fh, restart=True)
                self.assertIsNone(br.retrieve_results_file(['x', 'y', 'z']))

        def test_create_h5_group_path(self):
            # Verify that the helper builds the expected path string
            path = self._batch_restart_class._create_h5_group(['a', 'b', 'c'])
            self.assertEqual(path, "a/b/c")


class TestBatchRestartHDF5(BatchRestartTests.CommonTests):

    _batch_restart_class = BatchRestartHDF5

    def test_get_open_command_returns_correct_callable(self):
        import h5py
        self.assertIs(self._batch_restart_class.get_open_command(), h5py.File)

class TestBatchRestartCSV(BatchRestartTests.CommonTests):

    _batch_restart_class = BatchRestartCSV

    def test_get_open_command_returns_correct_callable(self):
        self.assertIs(self._batch_restart_class.get_open_command(), open)

