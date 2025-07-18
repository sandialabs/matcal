import os
import numpy as np
from matcal.dakota.dakota_interfaces import DakotaOutputReader, read_dakota_mcmc_chain
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
        