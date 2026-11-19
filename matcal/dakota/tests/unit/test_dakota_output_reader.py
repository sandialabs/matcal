import os
import numpy as np
from matcal.dakota.dakota_interfaces import (
    DakotaOutputReader,
    DakotaTermination,
    parse_termination_text,
    read_dakota_mcmc_chain,
)
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest


class TestDakotaOutputReader(MatcalUnitTest):

    @classmethod
    def setUpClass(cls) -> None:
        dir_name = os.path.join(os.path.dirname(__file__), "..", 'test_support_files')
        
        cls.dakota_calibration_example = os.path.join(dir_name, "dakota.out")
        cls.dakota_sobol_example = os.path.join(dir_name, "dakota_sens.out")
        cls.dakota_pearson_example = os.path.join(dir_name, "dakota_pearson.out")

    def setUp(self):
        super().setUp(__file__)

    def test_initWithNonExistentFile_WillThrow(self):
        with self.assertRaises(FileNotFoundError):
            DakotaOutputReader("nonexistent.file")

    def test_parseCalibrationFile(self):
        dor = DakotaOutputReader(self.dakota_calibration_example)
        param = dor.parse_calibration()
        self.assertAlmostEqual(param["best:Y"], 500.49999998)

    def test_parseSobolFile(self):
        dor = DakotaOutputReader(self.dakota_sobol_example)
        param = dor.parse_sobol()
        self.assert_close_arrays(param["sobol:Y"], np.array([[2.1111921515, 1.9781906299]], dtype=float))
        self.assert_close_arrays(param["sobol:nu"], np.array([[0.,0.]]))

    def test_parsePearsonFile(self):
        dor = DakotaOutputReader(self.dakota_pearson_example)
        param = dor.parse_pearson()
        goal  = {'a':np.array([np.nan, 1., 1, 1, 1], dtype=float), 'b':np.array([np.nan, -1., -1, -1, -1], dtype=float)}
        self.assert_close_arrays(param['pearson:a'][1:], goal['a'][1:])
        self.assert_close_arrays(param["pearson:b"][1:], goal["b"][1:])
        

class TestDakotaMCMCChainReader(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)
        
    def _write_mock_mcmc(self, filename):
        n_terms = 20
        headers = ["%mcmc_id", "interface", "spring_const", "drag"]
        lst = "least_sq_term_"
        n_lst = 10
        for i in range(n_lst):
            headers.append(lst+str(i+1))
        inter_val = "NO_ID"
        sc = np.random.uniform(0, 10, n_terms)
        dr = np.random.uniform(-10, 0, n_terms)
        lst_val = np.random.uniform(0, 100, n_lst)
        delim = "     "

        def make_line(l_idx, inter_val, sc, dr, lst_val, delim):
            line = str(l_idx) + delim
            line += inter_val + delim
            line += str(sc[l_idx]) + delim
            line += str(dr[l_idx]) + delim
            for lst in lst_val:
                line += str(lst)+ delim
            line += "\n"
            return line

        lines = ""
        for h in headers:
            lines += h+delim
        lines += "\n"

        for l_idx in range(n_terms):
            lines += make_line(l_idx, inter_val, sc, dr, lst_val, delim)
        with open(filename, 'w') as f:
            f.write(lines)
        return sc, dr
            
    def test_mcmc_import(self):
        fn = 'mock_mcmc.csv'
        spring, drag = self._write_mock_mcmc(fn)
        n_param = 2
        chain  = read_dakota_mcmc_chain(fn, n_param)
        self.assert_close_arrays(spring, chain['spring_const'], show_on_fail=True)
        self.assert_close_arrays(drag, chain['drag'], show_on_fail=True)
        

NL2SOL_OUTPUT = "  Begin\n\n ***** ABSOLUTE FUNCTION CONVERGENCE *****\n\n FUNCTION 1.0E-24\n"
NL2SOL_FALSE_OUTPUT = " ***** FALSE CONVERGENCE *****\n"
NL2SOL_LIMIT_OUTPUT = " ***** ITERATION LIMIT *****\n"
NPSOL_OUTPUT = (
    "NPSOL exits with INFORM code = 4 "
    '(see "Interpretation of output" section in NPSOL manual)\n'
)
NLSSOL_OUTPUT = (
    "NLSSOL exits with INFORM code = 0 "
    '(see "Interpretation of output" section of NPSOL manual)\n'
)
OPTPP_DAKOTA_OUT = (
    "********************************************************\n"
    "             OPT++ TERMINATION CRITERION                \n"
    "             Return Code                3\n"
    "\t  SUCCESS - optpp_q_newton converged to a solution\n"
    "Algorithm converged - Norm of gradient is less than gradient tolerance\n"
)
OPTPP_DEFAULT_OUT = (
    "Return code               = -4 (Algorithm terminated - Number of "
    "iterations or fevals exceeds the specified limit)\n"
)
ROL_OUTPUT = "ROL: Optimization Terminated with Status: Converged\n"
ROL_LIMIT_OUTPUT = (
    "ROL: Optimization Terminated with Status: Iteration Limit Exceeded\n"
)
DOT_OUTPUT = (
    "  TERMINATION CRITERIA\n\n"
    "  ABSOLUTE CONVERGENCE CRITERION WAS MET FOR  2 CONSECUTIVE ITERATIONS\n"
)
DOT_LIMIT_OUTPUT = "  TERMINATION CRITERIA\n\n  NUMBER OF ITERATIONS = ITMAX\n"
CONMIN_OUTPUT = (
    "     TERMINATION CRITERION\n"
    "          ABS(OBJ(I)-OBJ(I-1))   LESS THAN DABFUN FOR  3 ITERATIONS\n\n"
    "     NUMBER OF ITERATIONS =    7\n"
)
CONMIN_LIMIT_OUTPUT = (
    "     TERMINATION CRITERION\n          ITER EQUALS ITMAX\n\n"
    "     NUMBER OF ITERATIONS =    2\n"
)


class TestParseTerminationText(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)

    def _assert_termination(self, text, method, success, status, message_part):
        termination = parse_termination_text(text, method)
        self.assertIsInstance(termination, DakotaTermination)
        self.assertEqual(termination.success, success)
        self.assertEqual(termination.status, status)
        self.assertIn(message_part, termination.message)

    def test_nl2sol_convergence(self):
        self._assert_termination(
            NL2SOL_OUTPUT, "nl2sol", True, None, "ABSOLUTE FUNCTION CONVERGENCE"
        )

    def test_nl2sol_false_convergence_is_not_success(self):
        self._assert_termination(
            NL2SOL_FALSE_OUTPUT, "nl2sol", False, None, "FALSE CONVERGENCE"
        )

    def test_nl2sol_iteration_limit_is_not_success(self):
        self._assert_termination(
            NL2SOL_LIMIT_OUTPUT, "nl2sol", False, None, "ITERATION LIMIT"
        )

    def test_npsol_iteration_limit(self):
        self._assert_termination(
            NPSOL_OUTPUT, "npsol_sqp", False, 4, "iteration limit"
        )

    def test_nlssol_optimal(self):
        self._assert_termination(NLSSOL_OUTPUT, "nlssol_sqp", True, 0, "INFORM 0")

    def test_optpp_dakota_out_success(self):
        self._assert_termination(
            OPTPP_DAKOTA_OUT, "optpp_q_newton", True, 3, "Norm of gradient"
        )

    def test_optpp_default_out_failure(self):
        self._assert_termination(
            OPTPP_DEFAULT_OUT, "optpp_g_newton", False, -4, "exceeds the specified"
        )

    def test_rol_converged(self):
        self._assert_termination(ROL_OUTPUT, "rol", True, "Converged", "Converged")

    def test_rol_iteration_limit(self):
        self._assert_termination(
            ROL_LIMIT_OUTPUT, "rol", False, "Iteration Limit Exceeded", "Limit"
        )

    def test_dot_convergence(self):
        self._assert_termination(
            DOT_OUTPUT, "dot_sqp", True, None, "ABSOLUTE CONVERGENCE CRITERION"
        )

    def test_dot_kuhn_tucker_is_success(self):
        text = "  TERMINATION CRITERIA\n\n  KUHN-TUCKER CONDITIONS ARE SATISFIED\n"
        self._assert_termination(text, "dot_sqp", True, None, "KUHN-TUCKER")

    def test_npsol_inform_one_is_success(self):
        text = "NLSSOL exits with INFORM code = 1 (see manual)\n"
        self._assert_termination(text, "nlssol_sqp", True, 1, "requested accuracy")

    def test_dot_iteration_limit(self):
        self._assert_termination(DOT_LIMIT_OUTPUT, "dot_mmfd", False, None, "ITMAX")

    def test_conmin_convergence(self):
        self._assert_termination(
            CONMIN_OUTPUT, "conmin_mfd", True, None, "LESS THAN DABFUN"
        )

    def test_conmin_iteration_limit(self):
        self._assert_termination(
            CONMIN_LIMIT_OUTPUT, "conmin_mfd", False, None, "ITER EQUALS ITMAX"
        )

    def test_conmin_relative_convergence(self):
        text = (
            "     TERMINATION CRITERION\n"
            "          ABS(1-OBJ(I-1)/OBJ(I)) LESS THAN DELFUN FOR  3 ITERATIONS\n"
        )
        self._assert_termination(text, "conmin_mfd", True, None, "DELFUN")

    def test_conmin_unrecognised_reason_is_not_success(self):
        text = "     TERMINATION CRITERION\n          CONSTRAINT VIOLATION\n"
        self._assert_termination(text, "conmin_mfd", False, None, "CONSTRAINT")

    def test_conmin_parser_ignores_optpp_header(self):
        self._assert_termination(
            OPTPP_DAKOTA_OUT, "conmin_mfd", None, None, "not found"
        )

    def test_last_message_wins(self):
        text = NL2SOL_LIMIT_OUTPUT + NL2SOL_OUTPUT
        self._assert_termination(text, "nl2sol", True, None, "ABSOLUTE")

    def test_unknown_method_is_not_found(self):
        self._assert_termination(NL2SOL_OUTPUT, "coliny_cobyla", None, None, "not found")

    def test_solver_families_without_message_are_not_found(self):
        for method in ["npsol_sqp", "optpp_q_newton", "rol", "dot_sqp", "conmin_mfd"]:
            self._assert_termination("nothing here", method, None, None, "not found")

    def test_no_message_is_not_found(self):
        self._assert_termination("nothing here", "nl2sol", None, None, "not found")


class TestDakotaOutputReaderTermination(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)

    def _write(self, filename, text):
        with open(filename, "w") as file_handle:
            file_handle.write(text)

    def test_parse_termination_reads_dakota_out(self):
        self._write("dakota.out", NPSOL_OUTPUT)
        termination = DakotaOutputReader("dakota.out").parse_termination("npsol_sqp")
        self.assertEqual(termination.status, 4)

    def test_parse_termination_appends_additional_text(self):
        self._write("dakota.out", "no solver text")
        reader = DakotaOutputReader("dakota.out")
        termination = reader.parse_termination("nl2sol", [NL2SOL_OUTPUT])
        self.assertTrue(termination.success)
