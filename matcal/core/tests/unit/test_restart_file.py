import os

from matcal.core.constants import BATCH_RESTART_FILENAME
from matcal.core.restart_file import (BatchRestartCSV, BatchRestartHDF5,
                                      BatchRestartNone)
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest


def _touch(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w") as f:
        f.write("x")


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

        def test_instantiation_and_file_created(self):
            # Should not exist before instantiation (test harness should clean workdir)
            self.assertFalse(os.path.exists(self._restart_filename))

            br = self._batch_restart_class(self._restart_filename, restart=False)
            self.assertIsInstance(br, self._batch_restart_class)

            # Instantiation should create/truncate file
            if self._batch_restart_class.file_extension is not None:
                self.assertTrue(os.path.isfile(self._restart_filename))

        def test_restart_property(self):
            br = self._batch_restart_class(self._restart_filename, restart=False)
            self.assertFalse(br.restart)

            br = self._batch_restart_class(self._restart_filename, restart=True)
            self.assertTrue(br.restart)

        def test_record_and_retrieve_when_restart_run(self):
            # First, create data in a "non-restart" run (writes file)
            br = self._batch_restart_class(self._restart_filename, restart=False)
            job_key = ['eval.1', 'model', 'matcal_default_state']
            _touch("results.csv")
            br.record(job_key, 'results.csv')

            # Now restart=True should load prior entries and allow retrieval
            br2 = self._batch_restart_class(self._restart_filename, restart=True)
            self.assertEqual(br2.retrieve_results_file(job_key), 'results.csv')
            self.assertIsNone(br2.retrieve_results_file(['eval.2', 'model', 
                                                         'matcal_default_state']))

        def test_if_not_restart_retrieve_returns_None_always(self):
            br = self._batch_restart_class(self._restart_filename, restart=False)
            job_key = ['eval.1', 'model', 'matcal_default_state']
            br.record(job_key, 'results.csv')

            # retrieve_results_file must return None when restart=False
            self.assertIsNone(br.retrieve_results_file(job_key))
            self.assertIsNone(br.retrieve_results_file(['eval.2', 'model', 
                                                        'matcal_default_state']))

        def test_None_filename_does_not_get_written(self):
            br = self._batch_restart_class(self._restart_filename, restart=False)

            # File exists (except BatchRestartNone, but these tests don't run for None)
            old_size = os.path.getsize(self._restart_filename)

            br.record(['a', 'b', 'c'], None)
            self.assertIsNone(br.retrieve_results_file(['a', 'b', 'c']))

            new_size = os.path.getsize(self._restart_filename)
            self.assertEqual(new_size, old_size)

        def test_write_to_file_during_a_record(self):
            br = self._batch_restart_class(self._restart_filename, restart=False)

            _touch("a.txt")
            _touch("b.txt")
            _touch("c.txt")

            br2 = self._batch_restart_class(self._restart_filename, restart=True)
            br.record(['a', 'b', 'c'], 'a.txt')
            self.assertEqual(br2.retrieve_results_file(['a', 'b', 'c']), 'a.txt')
            br.record(['a', 'b', 'd'], 'b.txt')
            self.assertEqual(br2.retrieve_results_file(['a', 'b', 'd']), 'b.txt')

            br.record(['a', 'b', '3'], 'c.txt')
            self.assertEqual(br2.retrieve_results_file(['a', 'b', '3']), 'c.txt')

        def test_persistence_across_instances(self):
            # Record with one instance, then create a new instance restart=True and read it
            br = self._batch_restart_class(self._restart_filename, restart=False)
            job_key = ['j', 'k', 'l']
            _touch('out.h5')
            br.record(job_key, 'out.h5')

            br2 = self._batch_restart_class(self._restart_filename, restart=True)
            self.assertEqual(br2.retrieve_results_file(['j', 'k', 'l']), 'out.h5')

        def test_empty_file_restart(self):
            # Create an empty file and then restart=True should not crash
            if self._open_method is None:
                self.skipTest("No file backing for BatchRestartNone")
            # Create empty file in correct format:
            with self._open_method(self._restart_filename, "w"):
                pass
            br = self._batch_restart_class(self._restart_filename, restart=True)
            self.assertIsNone(br.retrieve_results_file(['x', 'y', 'z']))

        def test_create_h5_group_path(self):
            path = self._batch_restart_class._create_h5_group(['a', 'b', 'c'])
            self.assertEqual(path, "a/b/c")

        def test_restart_true_does_not_truncate_existing_file(self):
            # Write something, then instantiate with restart=True and verify it is still there
            br = self._batch_restart_class(self._restart_filename, restart=False)
            _touch('f1')
            br.record(['e1', 'm', 's'], 'f1')

            size_before = os.path.getsize(self._restart_filename)

            br2 = self._batch_restart_class(self._restart_filename, restart=True)
            self.assertEqual(br2.retrieve_results_file(['e1', 'm', 's']), 'f1')

            size_after = os.path.getsize(self._restart_filename)
            self.assertEqual(size_after, size_before)


class TestBatchRestartHDF5(BatchRestartTests.CommonTests):

    _batch_restart_class = BatchRestartHDF5

    def test_get_open_command_returns_correct_callable(self):
        import h5py
        self.assertIs(self._batch_restart_class.get_open_command(), h5py.File)

    def test_record_overwrite_same_key_is_allowed(self):
        br = self._batch_restart_class(self._restart_filename, restart=False)
        key = ['eval.1', 'model', 'state']
        _touch('file1')
        _touch('file2')

        br.record(key, 'file1')
        br.record(key, 'file2')

        br2 = self._batch_restart_class(self._restart_filename, restart=True)
        self.assertEqual(br2.retrieve_results_file(key), 'file2')


class TestBatchRestartCSV(BatchRestartTests.CommonTests):

    _batch_restart_class = BatchRestartCSV

    def test_get_open_command_returns_correct_callable(self):
        self.assertIs(self._batch_restart_class.get_open_command(), open)

    def test_duplicate_key_last_write_wins_in_memory(self):
        br = self._batch_restart_class(self._restart_filename, restart=False)
        key = ['eval.1', 'model', 'state']
        _touch('file1')
        br.record(key, 'file1')
        _touch('file2')
        br.record(key, 'file2')

        # restart=False means retrieve_results_file returns None; verify via restart=True
        br2 = self._batch_restart_class(self._restart_filename, restart=True)
        # Because CSV appends, retrieval may find first match if naive scan is used.
        # Our implementation scans from top; thus for correctness we rely on in-memory
        # map built during load which will end with the last occurrence.
        self.assertEqual(br2.retrieve_results_file(key), 'file2')


class TestBatchRestartNone(MatcalUnitTest):

    _batch_restart_class = BatchRestartNone

    def setUp(self):
        super().setUp(__file__)
        self._restart_filename = BATCH_RESTART_FILENAME

    def test_get_open_command_returns_None(self):
        self.assertIsNone(self._batch_restart_class.get_open_command())

    def test_file_extension_returns_None(self):
        self.assertIsNone(self._batch_restart_class.file_extension)

    def test_record_returns_None(self):
        br = self._batch_restart_class(None, None)
        self.assertIsNone(br.record([], "test.csv"))

    def test_retrieve_results_file_returns_None(self):
        br = self._batch_restart_class(None, None)
        self.assertIsNone(br.retrieve_results_file([]))


class TestBatchRestartCSVRobustness(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)
        self._restart_filename = BATCH_RESTART_FILENAME + BatchRestartCSV.file_extension

    def test_restart_load_ignores_truncated_last_line(self):
        # create real files for valid entries
        open("results_1.joblib", "w").close()
        open("results_2.joblib", "w").close()
        open("results_2.joblib", "w").close()
        # do NOT create results_3.job

        lines = [
            "eval.1/model/state,results_1.joblib\n",
            "eval.2/model/state,results_2.joblib\n",
            "eval.3/model/state,results_3.job"  # syntactically valid but file doesn't exist
        ]
        with open(self._restart_filename, "w") as f:
            f.writelines(lines)

        br = BatchRestartCSV(self._restart_filename, restart=True)
        self.assertEqual(br.retrieve_results_file(["eval.1", "model", "state"]), "results_1.joblib")
        self.assertEqual(br.retrieve_results_file(["eval.2", "model", "state"]), "results_2.joblib")
        self.assertIsNone(br.retrieve_results_file(["eval.3", "model", "state"]))

    def test_restart_load_ignores_malformed_line_missing_comma(self):
        lines = [
            "eval.1/model/state,results_1.joblib\n",
            "THIS_LINE_HAS_NO_COMMA\n",
            "eval.2/model/state,results_2.joblib\n",
        ]
        with open(self._restart_filename, "w") as f:
            f.writelines(lines)
        _touch("results_1.joblib")
        _touch("results_2.joblib")
        br = BatchRestartCSV(self._restart_filename, restart=True)

        self.assertEqual(br.retrieve_results_file(["eval.1", "model", "state"]), "results_1.joblib")
        self.assertEqual(br.retrieve_results_file(["eval.2", "model", "state"]), "results_2.joblib")

    def test_read_record_ignores_malformed_lines_and_last_write_wins(self):
        # Include malformed lines and duplicated key. We want:
        # - no crash
        # - last write wins
        # - malformed lines ignored
        lines = [
            "eval.1/model/state,old.joblib\n",
            "BADLINE\n",
            "eval.1/model/state,new.joblib\n",
            "eval.2/model/state,other.joblib\n",
            "eval.1/model/state,final.joblib\n",
            "TRUNCATED_NO_COMMA"
        ]
        for f in ["old.joblib", "new.joblib", "final.joblib", "other.joblib"]:
            _touch(f)
        with open(self._restart_filename, "w") as f:
            f.writelines(lines)

        br = BatchRestartCSV(self._restart_filename, restart=True)
        self.assertEqual(br.retrieve_results_file(["eval.1", "model", "state"]), "final.joblib")
        self.assertEqual(br.retrieve_results_file(["eval.2", "model", "state"]), "other.joblib")

    def test_record_then_manual_truncate_last_line_still_loads_previous(self):
        # Write valid entries via API
        br = BatchRestartCSV(self._restart_filename, restart=False)
        _touch("r1.joblib")
        _touch("r2.joblib")
        br.record(["eval.1", "model", "state"], "r1.joblib")
        br.record(["eval.2", "model", "state"], "r2.joblib")

        # Now simulate corruption by truncating the file mid-last-line
        # (remove last few bytes)
        with open(self._restart_filename, "rb+") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            # truncate a small amount, but not below zero
            f.truncate(max(0, size - 5))

        # Restart should still load what remains valid; at least eval.1 should survive
        br2 = BatchRestartCSV(self._restart_filename, restart=True)
        self.assertEqual(br2.retrieve_results_file(["eval.1", "model", "state"]), "r1.joblib")
        # eval.2 may or may not be present depending on where truncation landed; ensure no crash:
        _ = br2.retrieve_results_file(["eval.2", "model", "state"])

    def test_restart_returns_None_if_results_file_missing(self):
        with open(self._restart_filename, "w") as f:
            f.write("eval.1/model/state,does_not_exist.joblib\n")

        br = BatchRestartCSV(self._restart_filename, restart=True)
        self.assertIsNone(br.retrieve_results_file(["eval.1", "model", "state"]))