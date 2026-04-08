"""
Finite element models/regions and related blocks for MatCal-generated SIERRA decks.

Includes:
- finite element model blocks and parameters
- solid mechanics and thermal regions
- implicit dynamics
- element death blocks (SM + thermal)
"""

from matcal.core.input_file_writer import InputFileLine

from .blocks_base import _BaseSierraInputFileBlock
from .sections import _SectionNames
from .outputs import SolidMechanicsUserVariable


class FiniteElementParameters(_BaseSierraInputFileBlock):
    type = "parameters for block"
    required_keys = ["material"]
    default_values = {}

    def __init__(self, material_name, *blocks):
        name = " ".join(blocks).strip()
        super().__init__(name)
        material_line = InputFileLine(self.required_keys[0], material_name)
        self.add_line(material_line)


class SolidMechanicsFiniteElementParameters(FiniteElementParameters):
    type = "parameters for block"
    required_keys = FiniteElementParameters.required_keys + ["model", "section"]
    default_values = {}

    def __init__(self, material_name, material_model, *blocks):
        super().__init__(material_name, *blocks)
        model_line = InputFileLine(self.required_keys[1], material_model)
        self.add_line(model_line)
        self.set_section(_SectionNames.total_lagrange)

    def set_section(self, section_name):
        section_line = InputFileLine(self.required_keys[2], section_name)
        self.add_line(section_line, replace=True)

    def get_section(self):
        return self.get_line_value(self.required_keys[2])

    def get_blocks(self):
        return list(self.name.split(" "))


class _FiniteElementModelNames:
    solid_mechanics = "matcal_solid_mechanics_model"
    thermal = "matcal_thermal_model"


class FiniteElementModel(_BaseSierraInputFileBlock):
    type = "finite element model"
    required_keys = ["database name", "database type"]
    default_values = {required_keys[1]: "exodusII"}

    def __init__(
        self, *finite_element_model_parameters, name=_FiniteElementModelNames.solid_mechanics
    ):
        super().__init__(name)
        for finite_element_model_parameter in finite_element_model_parameters:
            self.add_subblock(finite_element_model_parameter)

    @property
    def mesh_filename(self):
        return self.lines[self.required_keys[0]].get_values()[-1]

    def set_mesh_filename(self, mesh_file):
        if self.required_keys[0] not in self.lines:
            self.add_line(InputFileLine(self.required_keys[0], mesh_file))
        else:
            self.lines[self.required_keys[0]].set(mesh_file, 1)

    def set_element_section(self, section_name):
        for block in self.subblocks.values():
            block.set_section(section_name)

    def get_element_section(self):
        sections = [block.get_section() for block in self.subblocks.values()]
        if len(set(sections)) == 1:
            return list(set(sections))[0]
        if len(set(sections)) == 0:
            raise ValueError(
                "No element sections found. Add "
                "FiniteElementModelParameters block to this FiniteElementModel."
            )
        return set(sections)

    def get_blocks(self):
        blocks = []
        for subblock in self.get_subblocks_by_type(FiniteElementParameters.type):
            blocks += subblock.get_blocks()
        return blocks


class SolidMechanicsImplicitDynamics(_BaseSierraInputFileBlock):
    type = "implicit dynamics"
    required_keys = []
    default_values = {"contact timestep": "off"}

    def __init__(self):
        super().__init__()
        self.set_print_name(False)
        self.set_print_title()


class _BaseDeath(_BaseSierraInputFileBlock):
    type = "element death"
    required_keys = ["criterion"]

    def __init__(self, death_variable, critical_value, criterion_eval_operator=">=", name="hades"):
        super().__init__(name=name)
        criterion_line = InputFileLine(
            self.required_keys[0],
            "element value of",
            death_variable,
            criterion_eval_operator,
            critical_value,
        )
        criterion_line.set_symbol("is")
        self.add_line(criterion_line)

    def get_critical_value(self):
        return self.get_line_value(self.required_keys[0], -1)


class SolidMechanicsDeath(_BaseDeath):
    type = "element death"
    required_keys = _BaseDeath.required_keys + ["block"]
    default_values = {"skip criteria evaluation at start of load step": "on"}

    def __init__(
        self, death_variable, critical_value, *death_blocks, 
        criterion_eval_operator=">=", name="hades"
    ):
        super().__init__(death_variable, critical_value, criterion_eval_operator, name=name)
        block_line = InputFileLine("block", *death_blocks)
        self.add_line(block_line)


class ThermalDeath(_BaseDeath):
    type = "element death"
    required_keys = _BaseDeath.required_keys + ["Add volume"]
    default_values = {}

    def __init__(
        self, death_variable, critical_value, *death_blocks, 
        criterion_eval_operator=">=", name="hades"
    ):
        super().__init__(death_variable, critical_value, criterion_eval_operator, name=name)
        volume_line = InputFileLine("Add volume", *death_blocks)
        volume_line.suppress_symbol()
        self.add_line(volume_line)


class SolidMechanicsRegion(_BaseSierraInputFileBlock):
    type = "adagio region"
    required_keys = ["use finite element model"]
    default_values = {}

    def __init__(self, name, finite_element_model_name):
        super().__init__(name=name)
        fem_line = InputFileLine(self.required_keys[0], finite_element_model_name)
        fem_line.suppress_symbol()
        self.add_line(fem_line)


class ThermalRegion(_BaseSierraInputFileBlock):
    type = "aria region"
    required_keys = ["use finite element model", "use linear solver"]
    default_values = {}

    def __init__(self, name, finite_element_model_name, solver):
        super().__init__(name=name)

        fem_line = InputFileLine(self.required_keys[0], finite_element_model_name)
        fem_line.suppress_symbol()
        self.add_line(fem_line, replace=True)

        self.add_solver(solver)
        self.add_nonlinear_solve_options()
        self.add_equations_to_solve(aria_quadrature_rule="Q1")

    def add_solver(self, solver):
        solver_line = InputFileLine(self.required_keys[1], solver.name)
        solver_line.suppress_symbol()
        self.add_line(solver_line, replace=True)

    def add_nonlinear_solve_options(self):
        self.add_line(InputFileLine("nonlinear solution strategy", "NEWTON"), replace=True)
        self.add_line(InputFileLine("maximum nonlinear iterations", 250), replace=True)
        self.add_line(InputFileLine("nonlinear residual tolerance", 1e-8), replace=True)
        self.add_line(InputFileLine("nonlinear relaxation factor", 1.0), replace=True)

    def add_equations_to_solve(self, aria_quadrature_rule):
        energy_eq = InputFileLine(
            "EQ energy for TEMPERATURE on all_blocks using",
            aria_quadrature_rule,
            "with DIFF SRC MASS",
        )
        energy_eq.suppress_symbol()
        self.add_line(energy_eq, replace=True)

        mesh_eq = InputFileLine(
            "EQ mesh for MESH_DISPLACEMENTS on all_blocks using",
            aria_quadrature_rule,
            "with XFER",
        )
        mesh_eq.suppress_symbol()
        self.add_line(mesh_eq, replace=True)

    def add_heating_source(self, plastic_work_variable, num_integration_pts):
        """
        Add plastic work as volumetric heat source.

        Creates an element user variable (vector length = num_integration_pts) and
        adds an Aria source definition referencing it.
        """
        self.remove_subblocks_by_type(SolidMechanicsUserVariable.type)

        plastic_work_var = SolidMechanicsUserVariable(
            plastic_work_variable,
            "element",
            "real",
            *([0] * num_integration_pts),
        )
        blocks_line = InputFileLine("Add part", "all_blocks")
        blocks_line.suppress_symbol()
        plastic_work_var.add_line(blocks_line)
        self.add_subblock(plastic_work_var, replace=True)

        source_line = InputFileLine(
            "source for energy on all_blocks",
            "user_field_volume_heating name =",
            plastic_work_variable,
            "scaling = 1",
        )
        self.add_line(source_line, replace=True)

    def add_element_death(self, *death_blocks):
        death_status_aria_line = InputFileLine(
            "User field real element scalar death_status_aria on all_blocks value",
            1,
        )
        self.add_line(death_status_aria_line, replace=True)

        transfer_line = InputFileLine("transfer", "element death")
        transfer_line.suppress_symbol()
        self.add_line(transfer_line)

        thermal_death = ThermalDeath(
            "death_status_aria", 0.99, *death_blocks, criterion_eval_operator="<="
        )
        self.add_subblock(thermal_death, replace=True)

    def add_initial_condition(self, initial_temperature):
        ic_line = InputFileLine("IC const on all_blocks TEMPERATURE", initial_temperature)
        self.add_line(ic_line, replace=True)

    def add_dirichlet_temperature_boundary_condition(self, nodeset_name, temperature):
        bc_line = InputFileLine(f"BC Const Dirichlet at {nodeset_name} Temperature", temperature)
        self.add_line(bc_line, replace=True)