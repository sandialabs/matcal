from matcal.core.tests.MatcalUnitTest import MatcalUnitTest

from matcal.sierra.input_file_writer.solvers import (
    FetiSolver,
    GdswSolver,
    TpetraSolver,
    SolidMechanicsControlContact,
    SolidMechanicsLoadstepPredictor,
    SolidMechanicsFullTangentPreconditioner,
    SolidMechanicsConjugateGradient,
)


class TestSolverBlocks(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)

    def test_feti_solver_(self):
        input_ = FetiSolver()
        test_str = input_.get_string()
        self.assertTrue("Begin feti equation solver feti" in test_str)

    def test_gdsw_solver_(self):
        input_ = GdswSolver()
        test_str = input_.get_string()
        self.assertTrue("Begin gdsw equation solver gdsw" in test_str)

    def test_tpetra_solver_(self):
        input_ = TpetraSolver()
        test_str = input_.get_string()
        self.assertTrue("Begin tpetra equation solver tpetra" in test_str)


class TestSolidMechanicsSolverSubblocks(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)

    def test_solid_mechanics_control_contact(self):
        input_block = SolidMechanicsControlContact()

        min_iters = input_block.get_line_value("minimum iterations")
        self.assertEqual(min_iters, 5)
        accept_rel_resid = input_block.get_line_value("acceptable relative residual")
        self.assertEqual(accept_rel_resid, 1e-2)
        target_rel_resid = input_block.get_line_value("target relative residual")
        self.assertEqual(target_rel_resid, 1e-3)

    def test_solid_mechanics_loadstep_predictor(self):
        input_block = SolidMechanicsLoadstepPredictor()
        scale_factor = input_block.get_line_value("scale factor")
        self.assertEqual(scale_factor, 0.0)
        input_block.set_scale_factor(1.0)
        self.assertEqual(scale_factor, 0.0)

    def test_solid_mechanics_full_tangent_preconditioner(self):
        input_block = SolidMechanicsFullTangentPreconditioner()

        small_num_iters = input_block.get_line_value("small number of iterations")
        self.assertEqual(small_num_iters, 20)

        min_smooth_iters = input_block.get_line_value("minimum smoothing iterations")
        self.assertEqual(min_smooth_iters, 15)

        iters_update = input_block.get_line_value("iteration update")
        self.assertEqual(iters_update, 25)
        self.assertEqual(len(input_block.lines), 3)

        solver = GdswSolver()
        input_block = SolidMechanicsFullTangentPreconditioner(solver)
        self.assertEqual(len(input_block.lines), 4)
        linear_solver = input_block.get_line_value("linear solver")
        self.assertEqual(linear_solver, "gdsw")

    def test_solid_mechanics_conjugate_gradient(self):
        input_block = SolidMechanicsConjugateGradient()
        self.assertEqual(len(input_block.lines), 6)
        reference = input_block.get_line_value("reference")
        self.assertEqual(reference, "Belytschko")
        self.assertFalse(input_block._print_name)

        full_tan_precond = SolidMechanicsFullTangentPreconditioner()
        input_block = SolidMechanicsConjugateGradient(full_tangent_preconditioner=full_tan_precond)
        self.assertIn(SolidMechanicsFullTangentPreconditioner.type, input_block.subblocks)
        input_block.set_full_tangent_preconditioner(None)
        self.assertNotIn(SolidMechanicsFullTangentPreconditioner, input_block.subblocks)

    def test_solid_mechanics_conjugate_gradient_set_tolerances(self):
        input_block = SolidMechanicsConjugateGradient()
        self.assertEqual(input_block.get_target_relative_residual(), 1e-9)
        input_block.set_target_relative_residual(1e-7)
        self.assertEqual(input_block.get_target_relative_residual(), 1e-7)

        self.assertAlmostEqual(input_block.get_target_residual(), 1e-7)
        input_block.set_target_residual(1e-9)
        self.assertEqual(input_block.get_target_residual(), 1e-9)

        self.assertEqual(input_block.get_acceptable_relative_residual(), 1e-8)
        input_block.set_acceptable_relative_residual(1e-6)
        self.assertEqual(input_block.get_acceptable_relative_residual(), 1e-6)

        self.assertIsNone(input_block.get_acceptable_residual())
        input_block.set_acceptable_residual(1e-4)
        self.assertEqual(input_block.get_acceptable_residual(), 1e-4)