"""
Solver-related SIERRA input-deck blocks for MatCal-generated decks.

This module includes:
- Linear solver blocks (FETI/GDSW/TPETRA)
- Solid mechanics nonlinear solver components (CG, full tangent preconditioner, contact control, etc.)
"""

from matcal.core.input_file_writer import InputFileBlock, InputFileLine

from .blocks_base import _BaseSierraInputFileBlock


class FetiSolver(_BaseSierraInputFileBlock):
    type = "feti equation solver"
    required_keys = []
    default_values = {}

    def __init__(self, name="feti"):
        super().__init__(name)


class GdswSolver(_BaseSierraInputFileBlock):
    type = "gdsw equation solver"
    required_keys = []
    default_values = {}

    def __init__(self, name="gdsw"):
        super().__init__(name)


class TpetraSolver(_BaseSierraInputFileBlock):
    type = "tpetra equation solver"
    required_keys = []
    default_values = {}

    def __init__(self, name="tpetra"):
        super().__init__(name=name)

        preset_solver_block = InputFileBlock("preset solver", begin_end=True)
        preset_solver_options = {
            "convergence tolerance": 1e-10,
            "maximum iterations": 10000,
            "residual scaling": "r0",
            "solver type": "thermal_symmetric",
        }
        preset_solver_block.add_lines_from_dictionary(preset_solver_options)
        self.add_subblock(preset_solver_block)


class SolidMechanicsNonlinearSolverBase(_BaseSierraInputFileBlock):
    def __init__(
        self,
        name,
        target_relative_residual,
        acceptable_relative_residual,
        minimum_iterations,
        maximum_iterations,
        *args,
        **kwargs,
    ):
        super().__init__(name=name)
        self.set_minimum_iterations(minimum_iterations)
        self.set_acceptable_relative_residual(acceptable_relative_residual)
        self.set_target_relative_residual(target_relative_residual)
        self.set_maximum_iterations(maximum_iterations)

    def set_minimum_iterations(self, minimum_iterations=5):
        self.add_line(InputFileLine("minimum iterations", minimum_iterations), replace=True)

    def set_acceptable_relative_residual(self, acceptable_relative_residual=1e-2):
        self.add_line(
            InputFileLine("acceptable relative residual", acceptable_relative_residual),
            replace=True,
        )

    def set_target_relative_residual(self, target_relative_residual=1e-2):
        self.add_line(
            InputFileLine("target relative residual", target_relative_residual),
            replace=True,
        )

    def set_target_residual(self, target_residual=1e-2):
        self.add_line(InputFileLine("target residual", target_residual), replace=True)

    def set_acceptable_residual(self, acceptable_residual=1e-2):
        self.add_line(InputFileLine("acceptable residual", acceptable_residual), replace=True)

    def set_maximum_iterations(self, maximum_iterations=100):
        self.add_line(InputFileLine("maximum iterations", maximum_iterations), replace=True)

    def get_target_relative_residual(self):
        return self.get_line_value("target relative residual")

    def get_target_residual(self):
        return self.get_line_value("target residual")

    def get_acceptable_relative_residual(self):
        return self.get_line_value("acceptable relative residual")

    def get_acceptable_residual(self):
        if "acceptable residual" in self.lines:
            return self.get_line_value("acceptable residual")
        return None


class SolidMechanicsControlContact(SolidMechanicsNonlinearSolverBase):
    type = "control contact"
    required_keys = []
    default_values = {
        "lagrange adaptive penalty": "off",
        "lagrange initialize": "none",
        "lagrange maximum updates": 0,
    }

    def __init__(
        self,
        target_relative_residual=1e-3,
        acceptable_relative_residual=1e-2,
        minimum_iterations=5,
        maximum_iterations=100,
        name="contact_control",
    ):
        super().__init__(
            name=name,
            target_relative_residual=target_relative_residual,
            acceptable_relative_residual=acceptable_relative_residual,
            minimum_iterations=minimum_iterations,
            maximum_iterations=maximum_iterations,
        )


class SolidMechanicsLoadstepPredictor(_BaseSierraInputFileBlock):
    type = "loadstep predictor"
    required_keys = ["scale factor"]
    default_values = {}

    def __init__(self, scale_factor=0.0):
        super().__init__()
        self.set_print_name(False)
        self.set_scale_factor(scale_factor)

    def set_scale_factor(self, scale_factor=0.0):
        self.add_line(InputFileLine("scale factor", scale_factor), replace=True)


class SolidMechanicsFullTangentPreconditioner(_BaseSierraInputFileBlock):
    type = "full tangent preconditioner"
    required_keys = []
    default_values = {
        "small number of iterations": 20,
        "minimum smoothing iterations": 15,
        "iteration update": 25,
    }

    def __init__(self, linear_solver=None):
        super().__init__()
        self.set_print_name(False)
        self.set_linear_solver(linear_solver)

    def set_linear_solver(self, linear_solver=None):
        if linear_solver is not None:
            self.add_line(InputFileLine("linear solver", linear_solver.name), replace=True)


class SolidMechanicsConjugateGradient(SolidMechanicsNonlinearSolverBase):
    type = "cg"
    required_keys = []
    default_values = {"reference": "Belytschko"}

    def __init__(
        self,
        target_relative_residual=1e-9,
        acceptable_relative_residual=1e-8,
        minimum_iterations=15,
        maximum_iterations=100,
        full_tangent_preconditioner=None,
        name=None,
    ):
        super().__init__(
            name=name,
            target_relative_residual=target_relative_residual,
            acceptable_relative_residual=acceptable_relative_residual,
            minimum_iterations=minimum_iterations,
            maximum_iterations=maximum_iterations,
        )
        self.set_print_name(False)
        self.set_full_tangent_preconditioner(full_tangent_preconditioner)

        # default: target residual two orders higher than target relative residual
        self.set_target_residual(target_relative_residual * 100)

    def set_full_tangent_preconditioner(self, full_tangent_preconditioner=None):
        if full_tangent_preconditioner is not None:
            self.add_subblock(full_tangent_preconditioner)
        elif SolidMechanicsFullTangentPreconditioner.type in self.subblocks:
            self.remove_subblock(SolidMechanicsFullTangentPreconditioner.type)


class SolidMechanicsNonlinearSolverContainer(_BaseSierraInputFileBlock):
    type = "solver"
    required_keys = []
    default_values = {}