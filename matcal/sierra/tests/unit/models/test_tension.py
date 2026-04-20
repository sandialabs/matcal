# test_tension.py
import numpy as np

from matcal.core.boundary_condition_calculators import BoundaryConditionDeterminationError
from matcal.core.constants import (
    TIME_KEY,
    ENG_STRAIN_KEY,
    ENG_STRESS_KEY,
    DISPLACEMENT_KEY,
    STRAIN_RATE_KEY,
    DISPLACEMENT_RATE_KEY,
)
from matcal.core.data import convert_dictionary_to_data, DataCollection
from matcal.core.state import SolitaryState, State

from matcal.cubit.geometry import GeometryParameters

from matcal.sierra.tests.unit.models.model_tests_base import (
    MatcalThreeDimensionalStandardModelUnitTestNewBase,
    _parse_denominator_value_from_expression_string,
)

from matcal.sierra.tests.sierra_sm_models_for_tests import (
    RoundUniaxialTensionModelForTests,
    RectangularUniaxialTensionModelForTests,
    RoundNotchedTensionModelForTests,
)


class EigthSymmetryModelTests:
    def __init__():
        pass

    class CommonTests(MatcalThreeDimensionalStandardModelUnitTestNewBase.CommonTests):

        def test_add_boundary_condition_mixed_boundary_data_fields(self):
            model = self.init_model()
            eng_data_dict = {
                "engineering_stress": np.linspace(0, 100, 10),
                "engineering_strain": np.linspace(0, 1, 10),
            }
            data_stress_strain = convert_dictionary_to_data(eng_data_dict)

            data_dict = {
                "load": np.linspace(0, 100, 10),
                "displacement": np.linspace(0, 1, 10),
            }
            data_load_disp = convert_dictionary_to_data(data_dict)

            data_col = DataCollection("test", data_load_disp, data_stress_strain)
            model.add_boundary_condition_data(data_col)

            with self.assertRaises(BoundaryConditionDeterminationError):
                model._prepare_loading_boundary_condition_displacement_function(
                    SolitaryState(), {}
                )

        def test_state_geo_param_override(self):
            model = self.init_model()
            data_dict = {"displacement": np.linspace(0, 100, 10)}
            data = convert_dictionary_to_data(data_dict)
            state = State("test", extensometer_length=0.5 * 0.0254)
            data.set_state(state)

            data2 = convert_dictionary_to_data(data_dict)
            default_state = data2.state
            dc = DataCollection("test", data, data2)

            model.add_boundary_condition_data(dc)

            model._setup_state(state, ".", build_mesh=False)
            self.assertEqual(
                model._current_state_geo_params["extensometer_length"], 0.5 * 0.0254
            )
            self.assertEqual(model._base_geo_params["extensometer_length"], 1.0 * 0.0254)

            model._setup_state(default_state, ".", build_mesh=False)
            self.assertEqual(
                model._current_state_geo_params["extensometer_length"], 1.0 * 0.0254
            )

        def test_state_geo_param_override_model_constants_take_precedent(self):
            model = self.init_model()
            data_dict = {"displacement": np.linspace(0, 100, 10)}
            data = convert_dictionary_to_data(data_dict)
            state = State(
                "test", extensometer_length=0.5 * 0.0254, element_size=0.01 * 0.0254
            )
            data.set_state(state)

            data2 = convert_dictionary_to_data(data_dict)
            default_state = data2.state
            dc = DataCollection("test", data, data2)

            model.add_boundary_condition_data(dc)

            model.add_state_constants(
                state,
                extensometer_length=0.25 * 0.0254,
                element_size=0.02 * 0.0254,
            )
            model._setup_state(state, ".", build_mesh=False)

            self.assertEqual(
                model._current_state_geo_params["extensometer_length"], 0.25 * 0.0254
            )
            self.assertEqual(model._base_geo_params["extensometer_length"], 1.0 * 0.0254)

            model._setup_state(default_state, ".", build_mesh=False)
            self.assertEqual(
                model._current_state_geo_params["extensometer_length"], 1.0 * 0.0254
            )


class UniaxialTensionStandardModelUnitTestBase:
    def __init__():
        pass

    class CommonTests(EigthSymmetryModelTests.CommonTests):

        def test_fail_if_mixed_bc_fields_in_a_state(self):
            model = self.init_model()
            disp_data = convert_dictionary_to_data({"displacement": [0, 1]})
            strain_data = convert_dictionary_to_data({"engineering_strain": [0, 1]})
            bc_dc = DataCollection("test", disp_data, strain_data)

            model.add_boundary_condition_data(bc_dc)
            with self.assertRaises(BoundaryConditionDeterminationError):
                model._setup_state(disp_data.state, build_mesh=False)

        def test_specific_outputs_added(self):
            model = self.init_model()
            data = convert_dictionary_to_data({"displacement": [0, 1]})
            model.add_boundary_condition_data(data)
            model._setup_state(SolitaryState(), build_mesh=False)

            sm_region = model._input_file.solid_mechanics_region

            self.assertIn("global_strain", sm_region.subblocks)
            lines = sm_region.subblocks["global_strain"].lines
            self.assertIn("global engineering_strain", lines)

            self.assertIn("global_stress", sm_region.subblocks)
            lines = sm_region.subblocks["global_stress"].lines
            self.assertIn("global engineering_stress", lines)

            self.assertIn("x_contraction", sm_region.subblocks)
            lines = sm_region.subblocks["x_contraction"].lines
            self.assertIn("global x_contraction", lines)

            self.assertIn("z_contraction", sm_region.subblocks)
            lines = sm_region.subblocks["z_contraction"].lines
            self.assertIn("global z_contraction", lines)

            hb_output = model._input_file.heartbeat_output
            self.assertTrue(hb_output.has_global_output("engineering_stress"))
            self.assertTrue(hb_output.has_global_output("engineering_strain"))
            self.assertTrue(hb_output.has_global_output("x_contraction"))
            self.assertTrue(hb_output.has_global_output("z_contraction"))

        def test_derived_outputs_added_full_field(self):
            model = self.init_model()
            data = convert_dictionary_to_data({self._displacement_var: [0, 1]})
            model.add_boundary_condition_data(data)
            model.activate_full_field_data_output(0.1, 0.1)
            model._setup_state(SolitaryState(), build_mesh=False)

            ff_output = model._input_file._full_field_output
            self.assertTrue(ff_output.has_global_output("engineering_strain"))
            self.assertTrue(ff_output.has_global_output("engineering_stress"))
            self.assertTrue(ff_output.has_global_output("x_contraction"))
            self.assertTrue(ff_output.has_global_output("z_contraction"))


class RoundUniaxialTensionModelUnitTests(
    UniaxialTensionStandardModelUnitTestBase.CommonTests,
    RoundUniaxialTensionModelForTests,
):
    def test_bad_geo_caught_on_setup_state_and_init(self):
        mat = self._get_material(plasticity=True)
        geo_params_bad = {
            "extensometer_length": 1.5,
            "gauge_length": 1.25,
            "gauge_radius": 0.125,
            "grip_radius": 0.25,
            "total_length": 4,
            "fillet_radius": 0.188,
            "taper": 0.0015,
            "necking_region": 0.375,
            "element_size": 0.0125,
            "mesh_method": 3,
            "grip_contact_length": 1,
        }
        with self.assertRaises(GeometryParameters.ValueError):
            self._model_class(mat, **geo_params_bad)

        geo_params_good = dict(geo_params_bad)
        geo_params_good["extensometer_length"] = 1.0
        model = self._model_class(mat, **geo_params_good)

        # geo params add in order that could result in failure. Need to
        # check params only after all parameters have been processed.
        model.add_constants(extensometer_length=1.5, gauge_length=1.55)
        params_by_precedence, param_source = model._get_parameters_by_precedence(SolitaryState())
        model._update_geometry_parameters(params_by_precedence, param_source)


class RectangularUniaxialTensionModelUnitTests(
    UniaxialTensionStandardModelUnitTestBase.CommonTests,
    RectangularUniaxialTensionModelForTests,
):
    """"""


class RoundNotchedTensionModelUnitTests(
    EigthSymmetryModelTests.CommonTests,
    RoundNotchedTensionModelForTests,
):
    def test_boundary_condition_accepts_engineering_strain(self):
        model = self.init_model()
        data = convert_dictionary_to_data({ENG_STRAIN_KEY: [0.0, 0.1]})
        model.add_boundary_condition_data(data)
        model._setup_state(SolitaryState(), build_mesh=False)

    def test_engineering_stress_strain_output_blocks_added(self):
        model = self.init_model()

        # Minimal BC data needed to allow setup_state to complete (displacement-based BC)
        data = convert_dictionary_to_data({"displacement": [0.0, 1.0]})
        model.add_boundary_condition_data(data)

        # Build the input deck (no mesh required for this test)
        model._setup_state(SolitaryState(), build_mesh=False)

        sm_region = model._input_file.solid_mechanics_region

        # Verify the named user output blocks exist in the input deck
        self.assertIn("global_strain", sm_region.subblocks)
        self.assertIn("global_stress", sm_region.subblocks)

        # Verify the global variable lines are present in those blocks
        strain_block_lines = sm_region.subblocks["global_strain"].lines
        self.assertIn(f"global {ENG_STRAIN_KEY}", strain_block_lines)

        stress_block_lines = sm_region.subblocks["global_stress"].lines
        self.assertIn(f"global {ENG_STRESS_KEY}", stress_block_lines)

    def test_bc_function_from_engineering_strain_with_time_uses_extensometer_length_and_symmetry(self):
        model = self.init_model()

        eng_strain = np.array([0.0, 0.10, 0.20])
        time = np.array([0.0, 0.5, 1.0])

        data = convert_dictionary_to_data({TIME_KEY: time, ENG_STRAIN_KEY: eng_strain})
        model.add_boundary_condition_data(data)

        ext_len = 0.01
        model.add_constants(extensometer_length=ext_len)

        state = data.state
        model._setup_state(state, build_mesh=False)

        disp_func = model._get_loading_boundary_condition_displacement_function(state, {})

        expected_disp = 0.5 * eng_strain * float(ext_len)

        self.assertIn(TIME_KEY, disp_func.field_names)
        self.assertIn(DISPLACEMENT_KEY, disp_func.field_names)
        self.assertTrue(np.allclose(disp_func[TIME_KEY], time))
        self.assertTrue(np.allclose(disp_func[DISPLACEMENT_KEY], expected_disp))

    def test_bc_function_from_engineering_strain_no_time_uses_max_strain_and_symmetry(self):
        model = self.init_model()

        eng_strain = np.array([0.0, 0.10, 0.20])
        data = convert_dictionary_to_data({ENG_STRAIN_KEY: eng_strain})
        model.add_boundary_condition_data(data)

        ext_len = 0.01
        model.add_constants(extensometer_length=ext_len)

        state = data.state
        model._setup_state(state, build_mesh=False)

        disp_func = model._get_loading_boundary_condition_displacement_function(state, {})

        # Without TIME_KEY (and without STRAIN_RATE_KEY), the BC calculator creates a
        # 2-point linear ramp to the maximum strain value.
        self.assertIn(TIME_KEY, disp_func.field_names)
        self.assertIn(DISPLACEMENT_KEY, disp_func.field_names)
        self.assertEqual(len(disp_func[TIME_KEY]), 2)
        self.assertEqual(len(disp_func[DISPLACEMENT_KEY]), 2)

        expected_end_disp = 0.5 * float(ext_len) * float(np.max(np.abs(eng_strain)))
        self.assertAlmostEqual(disp_func[DISPLACEMENT_KEY][-1], expected_end_disp, places=14)

    def test_bc_function_from_engineering_strain_no_time_with_strain_rate_sets_end_time_correctly(self):
        model = self.init_model()

        eng_strain = np.array([0.0, 0.10, 0.20])
        data = convert_dictionary_to_data({ENG_STRAIN_KEY: eng_strain})

        # Put engineering_strain_rate on the state so the BC calculator uses it
        state = State("strain_rate_state", **{STRAIN_RATE_KEY: 2.0})  # 1/s
        data.set_state(state)

        model.add_boundary_condition_data(data)

        ext_len = 0.01
        model.add_constants(extensometer_length=ext_len)

        model._setup_state(state, build_mesh=False)

        disp_func = model._get_loading_boundary_condition_displacement_function(state, {})

        eps_max = float(np.max(np.abs(eng_strain)))
        strain_rate = float(state.params[STRAIN_RATE_KEY])
        expected_end_time = eps_max / strain_rate

        expected_end_disp = 0.5 * eps_max * float(ext_len)

        self.assertAlmostEqual(disp_func[TIME_KEY][-1], expected_end_time, places=14)
        self.assertAlmostEqual(disp_func[DISPLACEMENT_KEY][-1], expected_end_disp, places=14)

    def test_bc_function_from_displacement_no_time_with_displacement_rate_sets_end_time_correctly(self):
        model = self.init_model()

        disp = np.array([0.0, 0.5, 1.0])
        data = convert_dictionary_to_data({DISPLACEMENT_KEY: disp})

        state = State("disp_rate_state", **{DISPLACEMENT_RATE_KEY: 4.0})  # length / s
        data.set_state(state)

        model.add_boundary_condition_data(data)
        model._setup_state(state, build_mesh=False)

        disp_func = model._get_loading_boundary_condition_displacement_function(state, {})

        max_disp = float(np.max(np.abs(disp)))
        disp_rate = float(state.params[DISPLACEMENT_RATE_KEY])
        expected_end_time = max_disp / disp_rate

        # RoundNotchedTensionModel applies 0.5 symmetry scaling to the displacement function
        expected_end_disp = 0.5 * max_disp

        self.assertAlmostEqual(disp_func[TIME_KEY][-1], expected_end_time, places=14)
        self.assertAlmostEqual(disp_func[DISPLACEMENT_KEY][-1], expected_end_disp, places=14)

    def test_engineering_strain_denominator_uses_extensometer_length(self):
        model = self.init_model()

        data = convert_dictionary_to_data({"displacement": [0.0, 1.0]})
        model.add_boundary_condition_data(data)

        ext_len = 0.0123
        model.add_constants(extensometer_length=ext_len)

        model._setup_state(SolitaryState(), build_mesh=False)

        sm_region = model._input_file.solid_mechanics_region
        self.assertIn("global_strain", sm_region.subblocks)

        strain_block = sm_region.subblocks["global_strain"]
        line_obj = strain_block.lines[f"global {ENG_STRAIN_KEY}"]
        line_str = line_obj.get_string()

        denom = _parse_denominator_value_from_expression_string(line_str)
        self.assertAlmostEqual(denom, float(ext_len), places=14)

    def test_engineering_stress_denominator_uses_notch_gauge_radius_area(self):
        model = self.init_model()

        data = convert_dictionary_to_data({"displacement": [0.0, 1.0]})
        model.add_boundary_condition_data(data)

        notch_r = 0.0025
        model.add_constants(
            notch_gauge_radius=notch_r,
            element_size=notch_r / 10.0,  # safely below notch_r/2
        )

        model._setup_state(SolitaryState(), build_mesh=False)

        expected_area = np.pi * float(notch_r) ** 2

        sm_region = model._input_file.solid_mechanics_region
        self.assertIn("global_stress", sm_region.subblocks)

        stress_block = sm_region.subblocks["global_stress"]
        line_obj = stress_block.lines[f"global {ENG_STRESS_KEY}"]
        line_str = line_obj.get_string()

        denom = _parse_denominator_value_from_expression_string(line_str)
        self.assertAlmostEqual(denom, expected_area, places=14)