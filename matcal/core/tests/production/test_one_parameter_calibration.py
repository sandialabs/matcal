

from matcal.core import *
from matcal.core.objective import SumSquaresMetricFunction
from matcal.core.calibration_studies import (ScipyLeastSquaresStudy)
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest
from matcal.core.tests.unit.test_calibration_studies import (run_serial_study_for_method, 
                                                             run_study_for_method)

class ScipyMinimizeOneParameterPythonTest(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)
        self.error_tol = 1e-5

    def test_default_minimize_calibration(self):
        run_study_for_method(self)

    def test_cg_minimize_calibration(self):

        run_study_for_method(self, method="cg",
                             tol=1e-6)

    def test_bfgs_minimize_calibration(self):
        run_study_for_method(self, method="bfgs")

    def test_newton_cg_minimize_calibration(self):
        run_study_for_method(self, method="newton-cg", 
                             initial_guess=90)

    def test_trust_ncg_minimize_calibration(self):
        run_study_for_method(self, method="trust-ncg")

    def test_dogleg_minimize_calibration(self):
        self.error_tol=1e-4
        run_study_for_method(self, method="dogleg")

    def test_trust_exact_minimize_calibration(self):
        run_study_for_method(self, method="trust-exact")
   
    def test_nelder_mead_minimize_calibration(self):
        run_study_for_method(self, method="Nelder-Mead", 
                             initial_guess=90, options={'xatol':1e-5})

    def test_L_BFGS_B_minimize_calibration(self):
        run_study_for_method(self, method="L-BFGS-B")

    def test_powell_minimize_calibration(self):
        run_study_for_method(self, method="Powell")

    def test_TNC_minimize_calibration(self):
        run_study_for_method(self, method="TNC")
    
    def test_COBYLA_minimize_calibration(self):
        run_study_for_method(self, method="cobyla", tol=1e-7)

    def test_SLSQP_minimize_calibration(self):
        run_study_for_method(self, method="SLSQP")

    def test_trust_constr_minimize_calibration(self):
        run_study_for_method(self, method="trust-constr")


class ScipyLeastSquaresOneParameterPythonTest(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)
        self.error_tol = 1e-6

    def test_default_minimize_calibration(self):
        run_study_for_method(self, study=ScipyLeastSquaresStudy)

    def test_dogbox_minimize_calibration(self):
        run_study_for_method(self, method='dogbox', study=ScipyLeastSquaresStudy)

    def test_lm_minimize_calibration(self):
        run_study_for_method(self, method='lm', study=ScipyLeastSquaresStudy)
        
class RunStudiesInSerialTest(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)
        self.error_tol = 1e-6
    
    def test_default_minimize_calibration(self):
        run_serial_study_for_method(self, study=ScipyLeastSquaresStudy)

    def test_default_minimize_calibration(self):
        run_serial_study_for_method(self)

    def test_cg_minimize_calibration(self):
        run_serial_study_for_method(self, method="cg",
                             tol=1e-7)

    def test_bfgs_minimize_calibration(self):
        run_serial_study_for_method(self, method="bfgs")

    def test_newton_cg_minimize_calibration(self):
        run_serial_study_for_method(self, method="newton-cg", 
                             initial_guess=90)