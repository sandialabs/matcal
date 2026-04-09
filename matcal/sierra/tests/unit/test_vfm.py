# test_vfm.py
from collections import OrderedDict
import numpy as np
import os

from matcal.core.boundary_condition_calculators import BoundaryConditionDeterminationError
from matcal.core.state import SolitaryState
from matcal.core.utilities import matcal_name_format

from matcal.full_field.data import convert_dictionary_to_field_data
from matcal.full_field.data_exporter import export_full_field_data_to_json

from matcal.sierra.models.vfm import _vfm_field_series_data
from matcal.sierra.input_file_writer.boundary_conditions import (
    SolidMechanicsFixedDisplacement,
    SolidMechanicsPrescribedDisplacement,
)

from model_tests_base import MatcalStandardModelUnitTestNewBase


class TestVFMUniaxialTensionModelCommon:
    def __init__():
        pass

    class VFMCommonTests(MatcalStandardModelUnitTestNewBase.CommonTests):

        def setUp(self):
            super().setUp()
            self._has_temperature_function_in_input = False

        def test_boundary_condition_function_calculations(self):
            """"Not needed, reads from mesh"""

        def test_add_boundary_condition_add_data_collections(self):
            """skipping due to shared infrastructure"""

        def test_add_boundary_condition_add_data(self):
            """skipping due to shared infrastructure"""

        def test_add_nodal_output_variable(self):
            super().test_add_nodal_output_variable(initial_false=False)

        def test_add_element_output_variable(self):
            super().test_add_element_output_variable(initial_false=False)

        def test_activate_exodus_output_interval_adjust(self):
            with self.assertRaises(TypeError):
                super().test_activate_exodus_output_interval_adjust(initial_false=False)

        def test_activate_exodus_output(self):
            model = self.init_model()
            model.activate_exodus_output()
            self.assertTrue(model.exodus_output)
            with self.assertRaises(TypeError):
                model.activate_exodus_output("yay")
            with self.assertRaises(TypeError):
                model.activate_exodus_output(-1)

        def test_boundary_condition_scale_factor_makes_it_to_input(self):
            with self.assertRaises(AttributeError):
                super().test_boundary_condition_scale_factor_makes_it_to_input()

        def test_set_boundary_condition_scale_factor(self):
            with self.assertRaises(AttributeError):
                super().test_set_boundary_condition_scale_factor()

        def _get_temp_data(self):
            return self._field_data

        def test_set_field_names(self):
            model = self.init_model(x_disp_name=None, y_disp_name=None)
            self.assertEqual(model._x_displacement_field_name, "U")
            self.assertEqual(model._y_displacement_field_name, "V")
            model.set_displacement_field_names("X", "Y")
            self.assertEqual(model._x_displacement_field_name, "X")
            self.assertEqual(model._y_displacement_field_name, "Y")

        def test_repeat_boundary_data(self):
            model = self.init_model()
            from copy import deepcopy
            from matcal import DataCollection

            field_data2 = deepcopy(self._field_data)
            field_data2.set_name("new_data")
            my_dc = DataCollection("too many field datas", self._field_data, field_data2)

            with self.assertRaises(BoundaryConditionDeterminationError):
                model.add_boundary_condition_data(my_dc)

        def test_not_field_boundary_data(self):
            from matcal import convert_dictionary_to_data

            model = self.init_model()
            data_dict = {"x": [0, 1], "y": [0, 1]}
            data = convert_dictionary_to_data(data_dict)
            with self.assertRaises(BoundaryConditionDeterminationError):
                model.add_boundary_condition_data(data)

        def test_activate_element_death(self):
            model = self.init_model()
            model.activate_element_death()

        def test_mesh_not_found(self):
            with self.assertRaises(FileNotFoundError):
                self._model_class(self._example_material, "no_mesh.g", thickness=0.1)

        def test_thickness(self):
            model = self.init_model()
            thickness = model.get_thickness()
            self.assertAlmostEqual(0.1, thickness)

        def test_bad_inits(self):
            self.init_model()
            with self.assertRaises(TypeError):
                self._model_class(self._example_material, self._mesh_grid, thickness="a")
            with self.assertRaises(FileNotFoundError):
                self._model_class(self._example_material, "not a mesh", thickness=0.1)
            with self.assertRaises(TypeError):
                self._model_class("not a material class", self._mesh_grid, thickness=0.1)

        def test_use_under_integrated_element(self):
            model = self.init_model()
            model.add_boundary_condition_data(self._field_data)
            model.set_displacement_field_names("Ux", "Uy")
            exo_output = model._input_file.exodus_output
            self.assertTrue(
                exo_output.has_element_output("first_pk_stress_vol_avg", "first_pk_stress")
            )
            model.use_under_integrated_element()
            self.assertTrue(
                exo_output.has_element_output("first_pk_stress_vol_avg", "first_pk_stress")
            )

        def test_change_map(self):
            model = self.init_model()
            self.assertAlmostEqual(model._polynomial_order, 1)
            self.assertAlmostEqual(model._search_radius_multiplier, 2.75)
            model.set_mapping_parameters(3, 5)
            self.assertAlmostEqual(model._polynomial_order, 3)
            self.assertAlmostEqual(model._search_radius_multiplier, 5)

        def test_generate_input_deck_with_temperature(self):
            model = self.init_model()
            model.add_boundary_condition_data(self._field_data)
            model.set_displacement_field_names("Ux", "Uy")
            model.read_temperature_from_boundary_condition_data("temperature")
            state = SolitaryState()
            model._setup_state(state, build_mesh=False)
            self.assertTrue(os.path.exists(matcal_name_format(model.name) + ".i"))

        def test_generate_input_deck(self):
            model = self.init_model()
            model.add_boundary_condition_data(self._field_data)
            model.set_displacement_field_names("Ux", "Uy")
            state = SolitaryState()
            model._setup_state(state, build_mesh=False)
            self.assertTrue(os.path.exists(matcal_name_format(model.name) + ".i"))

        def test_generate_fail_wrong_bc_fields(self):
            model = self.init_model(x_disp_name="bad_x", y_disp_name="bad_y")
            model.add_boundary_condition_data(self._field_data)
            state = SolitaryState()
            with self.assertRaises(BoundaryConditionDeterminationError):
                model._setup_state(state, build_mesh=False)

        def test_generate_input_bad_field_errors_before_mesh_build(self):
            model = self.init_model(x_disp_name="bad_x", y_disp_name="bad_y")
            model.add_boundary_condition_data(self._field_data)
            state = SolitaryState()
            with self.assertRaises(BoundaryConditionDeterminationError):
                model._setup_state(state, build_mesh=False)

        def test_activate_coupling(self):
            model = self.init_model()
            model.add_boundary_condition_data(self._field_data)
            model.set_displacement_field_names("Ux", "Uy")
            model.activate_thermal_coupling()

        def test_get_boundary_condition_function(self):
            model = self.init_model(x_disp_name="Ux", y_disp_name="Uy")
            for bc_dc in self.boundary_condition_data_sets:
                model.add_boundary_condition_data(bc_dc)
                for state in bc_dc.states.values():
                    model._setup_state(state, build_mesh=False)
                    disp_func = model._get_loading_boundary_condition_displacement_function(
                        state, {}
                    )
                    self.assertTrue("time" in disp_func.field_names)
                model.reset_boundary_condition_data()

        def test_check_input_bcs(self):
            model = self.init_model(x_disp_name="Ux", y_disp_name="Uy")
            for bc_dc in self.boundary_condition_data_sets:
                model.add_boundary_condition_data(bc_dc)
                for state in bc_dc.states.values():
                    model._setup_state(state, build_mesh=False)
                    sm_region = model.input_file.solid_mechanics_region

                    prescribed_disp_type = SolidMechanicsPrescribedDisplacement.type
                    self.assertTrue(len(sm_region.get_subblocks_by_type(prescribed_disp_type)), 4)

                    fixed_disp_type = SolidMechanicsFixedDisplacement.type
                    self.assertTrue(len(sm_region.get_subblocks_by_type(fixed_disp_type)), 1)

                    fixed_disp = sm_region.get_subblock_by_type(fixed_disp_type)
                    self.assertEqual(fixed_disp.get_line_value("component"), "z")
                    self.assertEqual(fixed_disp.get_line_value("node set"), "back_node_set")

                    bc_inputs = zip(
                        model._loading_bc_node_sets,
                        model._loading_bc_directions,
                        model._loading_bc_direction_keys,
                        model._loading_bc_read_variables,
                    )
                    for node_set, direction, direction_key, read_var in bc_inputs:
                        bc_name = node_set + " " + direction
                        bc = sm_region.subblocks[bc_name]
                        self.assertIn(direction_key, bc.lines)
                        self.assertEqual(bc.get_line_value("read variable"), read_var)

                model.reset_boundary_condition_data()


from matcal.sierra.tests.sierra_sm_models_for_tests import VFMUniaxialTensionHexModelForTests


class TestVFMUniaxialTensionHexModel(
    VFMUniaxialTensionHexModelForTests,
    TestVFMUniaxialTensionModelCommon.VFMCommonTests,
):

    def test_staggered_coupling_input(self):
        model = self.init_model()
        with self.assertRaises(AttributeError):
            model.activate_thermal_coupling(1)

    def test_use_iterative_coupling(self):
        model = self.init_model()
        with self.assertRaises(AttributeError):
            model.use_iterative_coupling()

    def test_input_has_output(self):
        model = self.init_model(x_disp_name="Ux", y_disp_name="Uy")
        for bc_dc in self.boundary_condition_data_sets:
            model.add_boundary_condition_data(bc_dc)
            for state in bc_dc.states.values():
                model._setup_state(state, build_mesh=False)
                exo_output = model.input_file.exodus_output
                self.assertTrue(
                    exo_output.has_element_output("first_pk_stress_vol_avg", "first_pk_stress")
                )
                self.assertTrue(exo_output.has_element_output("centroid"))
                self.assertTrue(exo_output.has_element_output("volume"))
            model.reset_boundary_condition_data()

    def _make_simple_mesh_with_info(self):
        n_time = 3
        n_loc = 4
        time = np.linspace(0, 1, n_time)
        T = np.random.uniform(0, 1, [n_time, n_loc])

        ref_ff_data = OrderedDict(
            {
                "T": T,
                "first_pk_stress": T,
                "centroid": T,
                "volume": T,
                "first_pk_stress_xx": T,
                "time": time,
                "x": np.array([0, 1, 1, 0]),
                "y": np.array([0, 0, 1, 1]),
                "con": [[0, 1, 2, 3]],
            }
        )
        return convert_dictionary_to_field_data(ref_ff_data, ["x", "y"], "con")

    def test_vfm_field_series_data(self):
        data = self._make_simple_mesh_with_info()
        export_full_field_data_to_json("data.json", data)
        res = _vfm_field_series_data("data.json")
        sorted_fields = sorted(res.field_names)
        goal = sorted(["first_pk_stress", "centroid", "volume", "first_pk_stress_xx", "time"])
        self.assertEqual(goal, sorted_fields)


from matcal.sierra.tests.sierra_sm_models_for_tests import (
    VFMUniaxialTensionConnectedHexModelForTests,
)


class TestVFMUniaxialTensionConnectedHexModel(
    VFMUniaxialTensionConnectedHexModelForTests,
    TestVFMUniaxialTensionModelCommon.VFMCommonTests,
):
    def test_input_has_output(self):
        model = self.init_model(x_disp_name="Ux", y_disp_name="Uy")
        for bc_dc in self.boundary_condition_data_sets:
            model.add_boundary_condition_data(bc_dc)
            for state in bc_dc.states.values():
                model._setup_state(state, build_mesh=False)
                exo_output = model.input_file.exodus_output
                self.assertTrue(
                    exo_output.has_element_output("first_pk_stress_vol_avg", "first_pk_stress")
                )
                self.assertTrue(exo_output.has_element_output("centroid"))
                self.assertTrue(exo_output.has_element_output("double_volume", "volume"))
            model.reset_boundary_condition_data()