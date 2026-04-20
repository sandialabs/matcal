from matcal.core.tests.MatcalUnitTest import MatcalUnitTest

from matcal.sierra.input_file_writer.regions_models import (
    SolidMechanicsRegion,
    SolidMechanicsFiniteElementParameters,
    FiniteElementModel,
    SolidMechanicsImplicitDynamics,
    SolidMechanicsDeath,
    ThermalRegion,
    _FiniteElementModelNames,
    ThermalDeath,
)
from matcal.sierra.input_file_writer.sections import _SectionNames
from matcal.sierra.input_file_writer.solvers import TpetraSolver


class TestFiniteElementParametersAndModel(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)

    def _get_finite_element_params_block(self, blocks=("block_1", "block_2")):
        return SolidMechanicsFiniteElementParameters(
            "test_material", "elastic", *blocks
        )

    def test_finite_element_block_parameters(self):
        input_block = self._get_finite_element_params_block()
        test_str = input_block.get_string()
        self.assertTrue("Begin parameters for block block_1 block_2" in test_str)
        self.assertTrue("section = total_lagrange" in test_str)

    def test_finite_element_block_parameters_UG(self):
        input_block = SolidMechanicsFiniteElementParameters(
            "test_material", "elastic", "block_1", "block_2"
        )
        input_block.set_section(_SectionNames.uniform_gradient)
        test_str = input_block.get_string()
        self.assertTrue("Begin parameters for block block_1 block_2" in test_str)
        self.assertTrue("section = uniform_gradient" in test_str)

    def test_finite_element_block_parameters_get_section(self):
        input_block = SolidMechanicsFiniteElementParameters(
            "test_material", "elastic", "block_1", "block_2"
        )
        input_block.set_section(_SectionNames.uniform_gradient)
        section = input_block.get_section()
        self.assertEqual(section, _SectionNames.uniform_gradient)

    def test_finite_element_model_block(self):
        FE_params_block = self._get_finite_element_params_block()
        input_block = FiniteElementModel(FE_params_block)
        input_block.set_mesh_filename("test_mesh.g")
        test_str = input_block.get_string()
        self.assertTrue("Begin finite element model matcal_solid_mechanics" in test_str)
        database_values = input_block.lines[FiniteElementModel.required_keys[1]].get_values()
        self.assertEqual("exodusII", database_values[-1])
        mesh_name_values = input_block.lines[FiniteElementModel.required_keys[0]].get_values()
        self.assertEqual("test_mesh.g", mesh_name_values[-1])
        self.assertTrue("Begin parameters for block block_1 block_2" in test_str)

    def test_finite_element_model_block_raise_val_error_no_params(self):
        input_block = FiniteElementModel()
        input_block.set_mesh_filename("test_mesh.g")
        with self.assertRaises(ValueError):
            input_block.get_element_section()

    def test_finite_element_model_block_set_element_section(self):
        FE_params_block = self._get_finite_element_params_block()
        input_block = FiniteElementModel(FE_params_block)
        input_block.set_element_section(_SectionNames.uniform_gradient)
        input_block.set_mesh_filename("test_mesh.g")
        section = input_block.get_element_section()
        self.assertEqual(section, _SectionNames.uniform_gradient)

    def test_finite_element_model_block_get_element_section_two_sections(self):
        FE_params_block1 = self._get_finite_element_params_block()
        FE_params_block2 = self._get_finite_element_params_block(blocks=("block3", "block_4"))
        FE_params_block2.set_section(_SectionNames.uniform_gradient)
        input_block = FiniteElementModel(FE_params_block1, FE_params_block2)
        input_block.set_mesh_filename("test_mesh.g")
        section = input_block.get_element_section()
        self.assertEqual(
            section,
            set([_SectionNames.total_lagrange, _SectionNames.uniform_gradient]),
        )


class TestRegionsAndDeathBlocks(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)

    def test_SM_implicit_dynamics_block(self):
        input_block = SolidMechanicsImplicitDynamics()
        test_str = input_block.get_string()

        self.assertEqual("Begin implicit dynamics", test_str.split("\n")[0])
        self.assertEqual("contact timestep = off", test_str.split("\n")[1].strip())

    def test_solid_mechanics_death_block(self):
        input_block = SolidMechanicsDeath("eqps", 0.15, "necking_section", "gauge_section")
        test_str = input_block.get_string()
        self.assertEqual("Begin element death hades", test_str.split("\n")[0])
        block_values = input_block.lines["block"].get_values()
        self.assertEqual("necking_section", block_values[1])
        self.assertEqual("gauge_section", block_values[2])
        criterion_values = input_block.lines["criterion"].get_values()
        self.assertEqual("eqps", criterion_values[-3])
        self.assertEqual(">=", criterion_values[-2])
        self.assertEqual(0.15, criterion_values[-1])
        self.assertEqual(0.15, input_block.get_critical_value())

    def tests_olid_mechanics_region(self):
        region_block = SolidMechanicsRegion(
            "adagio_region", _FiniteElementModelNames.solid_mechanics
        )
        self.assertEqual(len(region_block.lines), 1)
        finite_ele_model_name = region_block.get_line_value(
            SolidMechanicsRegion.required_keys[0]
        )
        self.assertEqual(finite_ele_model_name, _FiniteElementModelNames.solid_mechanics)

    def test_get_subblock_by_type(self):
        region_block = SolidMechanicsRegion(
            "adagio_region", _FiniteElementModelNames.solid_mechanics
        )
        self.assertEqual(len(region_block.lines), 1)
        finite_ele_model_name = region_block.get_line_value(
            SolidMechanicsRegion.required_keys[0]
        )
        self.assertEqual(finite_ele_model_name, _FiniteElementModelNames.solid_mechanics)

        from matcal.sierra.input_file_writer.boundary_conditions import (
            SolidMechanicsPrescribedDisplacement,
        )

        bc1 = SolidMechanicsPrescribedDisplacement("test", "test_ns", "X")
        bc2 = SolidMechanicsPrescribedDisplacement("test", "test_ns2", "X")
        bc3 = SolidMechanicsPrescribedDisplacement("test", "test_ns3", "X")
        bc4 = SolidMechanicsPrescribedDisplacement("test", "test_ns4", "X")

        subblock = region_block.get_subblock_by_type(bc1.type)
        self.assertEqual(subblock, None)

        region_block.add_subblock(bc1)
        region_block.add_subblock(bc2)
        region_block.add_subblock(bc3)
        region_block.add_subblock(bc4)

        subblock = region_block.get_subblock_by_type(bc1.type)
        self.assertEqual(subblock, bc1)
        region_block.remove_subblock(bc1)
        region_block.remove_subblock(bc2)

        subblock = region_block.get_subblock_by_type(bc3.type)
        self.assertEqual(subblock, bc3)

    def test_remove_subblocks_by_type(self):
        region_block = SolidMechanicsRegion(
            "adagio_region", _FiniteElementModelNames.solid_mechanics
        )
        self.assertEqual(len(region_block.lines), 1)

        from matcal.sierra.input_file_writer.boundary_conditions import (
            SolidMechanicsPrescribedDisplacement,
        )

        bc1 = SolidMechanicsPrescribedDisplacement("test", "test_ns", "X")
        bc2 = SolidMechanicsPrescribedDisplacement("test", "test_ns2", "X")
        bc3 = SolidMechanicsPrescribedDisplacement("test", "test_ns3", "X")
        bc4 = SolidMechanicsPrescribedDisplacement("test", "test_ns4", "X")

        region_block.add_subblock(bc1)
        region_block.add_subblock(bc2)
        region_block.add_subblock(bc3)
        region_block.add_subblock(bc4)

        subblock = region_block.remove_subblocks_by_type(bc1.type)
        self.assertEqual(subblock, None)

    def test_thermal_death(self):
        input_block = ThermalDeath(
            "death_status",
            0.99,
            "block1",
            "block2",
            criterion_eval_operator="<=",
        )
        self.assertEqual(input_block.get_line_value("Add volume", 1), "block1")
        self.assertEqual(input_block.get_line_value("Add volume", 2), "block2")
        self.assertEqual(input_block.get_line_value("criterion", 2), "death_status")
        self.assertEqual(input_block.get_line_value("criterion", 3), "<=")
        self.assertEqual(input_block.get_line_value("criterion", 4), 0.99)


from matcal.sierra.input_file_writer.regions_models import (
    ThermalRegion,
    _FiniteElementModelNames,
)
from matcal.sierra.input_file_writer.solvers import TpetraSolver


class TestThermalRegion(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)
        self._region_name = "thermal_region"
        self._fe_model = _FiniteElementModelNames.thermal
        self._solver = TpetraSolver()

    def _make_thermal_region(self):
        return ThermalRegion(self._region_name, self._fe_model, self._solver)

    def test_init(self):
        thermal_region = self._make_thermal_region()
        self.assertEqual(
            thermal_region.get_line_value("nonlinear solution strategy"), "NEWTON"
        )
        self.assertEqual(
            thermal_region.get_line_value("use finite element model"), self._fe_model
        )
        self.assertEqual(
            thermal_region.get_line_value("use linear solver"), self._solver.name
        )
        test_str = thermal_region.get_input_string()
        self.assertIn("EQ energy", test_str)
        self.assertIn("EQ mesh", test_str)

    def test_add_heating_source(self):
        thermal_region = self._make_thermal_region()
        thermal_region.add_heating_source("plastic_work", 8)
        test_str = thermal_region.get_input_string()
        self.assertIn("source for energy", test_str)
        self.assertIn("plastic_work", thermal_region.subblocks)

    def test_add_element_death(self):
        thermal_region = self._make_thermal_region()
        thermal_region.add_element_death("death_block1", "death_block2")
        test_str = thermal_region.get_input_string()
        self.assertIn("User field real element scalar death_status_aria", test_str)
        self.assertIn("transfer element death", test_str)
        self.assertIn("hades", thermal_region.subblocks)

    def test_add_initial_condition(self):
        thermal_region = self._make_thermal_region()
        thermal_region.add_initial_condition(298)
        test_str = thermal_region.get_input_string()
        self.assertIn("IC const on all_blocks TEMPERATURE = 298", test_str)

    def test_add_dirichlet_temperature_boundary_condition(self):
        thermal_region = self._make_thermal_region()
        thermal_region.add_dirichlet_temperature_boundary_condition("grip", 298)
        test_str = thermal_region.get_input_string()
        self.assertIn("BC Const Dirichlet at grip Temperature = 298", test_str)