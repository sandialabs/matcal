from matcal.core.tests.MatcalUnitTest import MatcalUnitTest

from matcal.cubit.cubit_runner import CubitExternalExecutable


class TestCubitExternalExecutable(MatcalUnitTest):

    def setUp(self): 
      super().setUp(__file__)

    def _confirm_clean_return(self, cbr_proc):
        self.assertEqual(cbr_proc[-1], 0)

    def _confirm_error_return(self, cbr_proc):
        self.assertNotEqual(cbr_proc[-1], 0)

    def test_run_cubit_with_empty_journal(self):
        j_file = _write_test_journal_file([""])
        cbr = CubitExternalExecutable([j_file])
        cbr_proc = cbr.run()
        self._confirm_clean_return(cbr_proc)

    def test_run_cubit_with_exit_command(self):
        j_file = _write_test_journal_file(["exit"])
        cbr = CubitExternalExecutable([j_file])
        cbr_proc = cbr.run()
        self._confirm_clean_return(cbr_proc)

    def test_run_cubit_with_bad_command(self):
        j_file = _write_test_journal_file(["create brick", "exit"])
        cbr = CubitExternalExecutable([j_file])
        cbr_proc = cbr.run()
        self._confirm_error_return(cbr_proc)

def _write_test_journal_file(cmds):
    j_file = "test.jou"
    with open(j_file, "w") as f:
        for cmd in cmds:
            f.write(f"{cmd}\n")
    return j_file