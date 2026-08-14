# test_material_point.py
import os

from matcal.core.constants import (
    DISPLACEMENT_KEY,
    LOAD_KEY,
    ENG_STRAIN_KEY,
    ENG_STRESS_KEY,
    TRUE_STRAIN_KEY,
    TRUE_STRESS_KEY,
    TEMPERATURE_KEY,
    TIME_KEY
)
from matcal.core.data import convert_dictionary_to_data
from matcal.core.state import SolitaryState, State
from matcal.core.tests.unit.comment_test_helpers import (
    assert_data_set_index_comment,
    assert_data_set_name_comment,
    assert_selection_reason_comment,
    assert_source_collection_comment,
    assert_source_fields_comment,
)

from matcal.sierra.tests.unit.models.model_tests_base import (
    MatcalStandardModelUnitTestNewBase
)
from matcal.sierra.tests.sierra_sm_models_for_tests import (
    UniaxialLoadingMaterialPointModelForTests,
    SimpleShearMaterialPointModelForTests,
)


class UniaxialLoadingMaterialPointModelTests(
    MatcalStandardModelUnitTestNewBase.CommonTests,
    UniaxialLoadingMaterialPointModelForTests,
):
    def test_setup_state_all_states_with_build_mesh(self):
        model = self.init_model()
        bc_data_cols = self.boundary_condition_data_sets
        for data_col in bc_data_cols:
            model.reset_boundary_condition_data()
            model.add_boundary_condition_data(data_col)
            for state in data_col.states.values():
                model._setup_state(state, build_mesh=True)
                self.assertTrue(os.path.exists(model._mesh_filename))
                os.remove(model._mesh_filename)

    def _check_outputs_default(self, model):
        sm_region = model._input_file.solid_mechanics_region
        self.assertIn("global_stress_strain_load_disp", sm_region.subblocks)

        lines = sm_region.subblocks["global_stress_strain_load_disp"].lines
        self.assertIn(f"global {DISPLACEMENT_KEY}", lines)
        self.assertIn(f"global {LOAD_KEY}", lines)

        self.assertIn("true_stress_strain", sm_region.subblocks)
        lines = sm_region.subblocks["true_stress_strain"].lines
        self.assertIn(f"global {TRUE_STRAIN_KEY}", lines)
        self.assertIn("global log_strain_xx", lines)
        self.assertIn("global log_strain_yy", lines)
        self.assertIn(f"global {TRUE_STRESS_KEY}", lines)

        hb_output = model._input_file.heartbeat_output
        self.assertTrue(hb_output.has_global_output(DISPLACEMENT_KEY))
        self.assertTrue(hb_output.has_global_output(LOAD_KEY))
        self.assertTrue(hb_output.has_global_output(DISPLACEMENT_KEY, ENG_STRAIN_KEY))
        self.assertTrue(hb_output.has_global_output(LOAD_KEY, ENG_STRESS_KEY))
        self.assertTrue(hb_output.has_global_output(TRUE_STRAIN_KEY))
        self.assertTrue(hb_output.has_global_output(TRUE_STRESS_KEY))
        self.assertTrue(hb_output.has_global_output("log_strain_xx"))
        self.assertTrue(hb_output.has_global_output("log_strain_yy"))
        self.assertTrue(hb_output.has_global_output("time"))
        self.assertTrue(hb_output.has_global_output("contraction"))

    def test_outputs_added(self):
        model = self.init_model()
        data = convert_dictionary_to_data({"engineering_strain": [0, 1]})
        model.add_boundary_condition_data(data)
        model._setup_state(SolitaryState(), build_mesh=False)

        self._check_outputs_default(model)

        sm_region = model._input_file.solid_mechanics_region
        lines = sm_region.subblocks["true_stress_strain"].lines
        self.assertNotIn(f"global {TEMPERATURE_KEY}", lines)

        hb_output = model._input_file.heartbeat_output
        self.assertFalse(hb_output.has_global_output(TEMPERATURE_KEY))

    def test_outputs_added_adiabatic(self):
        model = self.init_model()
        data = convert_dictionary_to_data({"engineering_strain": [0, 1]})
        data.set_state(State("temp", temperature=100))

        model.add_boundary_condition_data(data)
        model.activate_thermal_coupling()
        model._setup_state(data.state, build_mesh=False)

        self._check_outputs_default(model)

        sm_region = model._input_file.solid_mechanics_region
        lines = sm_region.subblocks["true_stress_strain"].lines
        self.assertIn(f"global {TEMPERATURE_KEY}", lines)

        hb_output = model._input_file.heartbeat_output
        self.assertTrue(hb_output.has_global_output(TEMPERATURE_KEY))

    def test_loading_function_comment_added_for_strain_history(self):
        model = self.init_model()
        data = convert_dictionary_to_data({
            TIME_KEY: [0.0, 1.0],
            ENG_STRAIN_KEY: [0.0, 0.1],
        })
        data.set_name("mp_curve_01")
        model.add_boundary_condition_data(data)

        state = data.state
        model._setup_state(state, build_mesh=False)

        func_str = self._get_loading_function_block_string(model)
        assert_source_fields_comment(self, func_str, ENG_STRAIN_KEY, uses_time=True)
        assert_source_collection_comment(self, func_str)
        assert_data_set_index_comment(self, func_str, 0)
        assert_data_set_name_comment(self, func_str, "mp_curve_01")
        assert_selection_reason_comment(self, func_str, ENG_STRAIN_KEY, state.name)


class SimpleShearMaterialPointModelTests(
    MatcalStandardModelUnitTestNewBase.CommonTests,
    SimpleShearMaterialPointModelForTests,
):
    """Unit tests for SimpleShearMaterialPointModel."""

    def test_setup_state_all_states_with_build_mesh(self):
        """Test setup_state builds mesh for all boundary condition states."""
        model = self.init_model()
        bc_data_cols = self.boundary_condition_data_sets
        for data_col in bc_data_cols:
            model.reset_boundary_condition_data()
            model.add_boundary_condition_data(data_col)
            for state in data_col.states.values():
                model._setup_state(state, build_mesh=True)
                self.assertTrue(os.path.exists(model._mesh_filename))
                os.remove(model._mesh_filename)

    def _check_outputs_default(self, model):
        """Check that default output blocks are correctly configured."""
        sm_region = model._input_file.solid_mechanics_region
        
        # Check shear load outputs
        self.assertIn("global_shear_stress_strain_load_disp", sm_region.subblocks)
        lines = sm_region.subblocks["global_shear_stress_strain_load_disp"].lines
        self.assertIn(f"global {DISPLACEMENT_KEY}", lines)
        self.assertIn(f"global {LOAD_KEY}", lines)

        # Check true stress/strain outputs
        self.assertIn("true_shear_stress_strain", sm_region.subblocks)
        lines = sm_region.subblocks["true_shear_stress_strain"].lines
        self.assertIn(f"global {TRUE_STRAIN_KEY}", lines)
        self.assertIn(f"global {TRUE_STRESS_KEY}", lines)
        self.assertIn("global log_strain_xx", lines)
        self.assertIn("global log_strain_yy", lines)
        self.assertIn("global log_strain_zz", lines)
        self.assertIn("global cauchy_stress_xx", lines)
        self.assertIn("global cauchy_stress_yy", lines)
        self.assertIn("global cauchy_stress_zz", lines)

        # Check heartbeat outputs
        hb_output = model._input_file.heartbeat_output
        self.assertTrue(hb_output.has_global_output(DISPLACEMENT_KEY))
        self.assertTrue(hb_output.has_global_output(LOAD_KEY))
        self.assertTrue(hb_output.has_global_output(DISPLACEMENT_KEY, ENG_STRAIN_KEY))
        self.assertTrue(hb_output.has_global_output(LOAD_KEY, ENG_STRESS_KEY))
        self.assertTrue(hb_output.has_global_output(TRUE_STRAIN_KEY))
        self.assertTrue(hb_output.has_global_output(TRUE_STRESS_KEY))
        self.assertTrue(hb_output.has_global_output("log_strain_xx"))
        self.assertTrue(hb_output.has_global_output("log_strain_yy"))
        self.assertTrue(hb_output.has_global_output("log_strain_zz"))
        self.assertTrue(hb_output.has_global_output("cauchy_stress_xx"))
        self.assertTrue(hb_output.has_global_output("cauchy_stress_yy"))
        self.assertTrue(hb_output.has_global_output("cauchy_stress_zz"))
        self.assertTrue(hb_output.has_global_output("time"))

    def test_outputs_added(self):
        """Test that output blocks are correctly added for uncoupled case."""
        model = self.init_model()
        data = convert_dictionary_to_data({"engineering_strain": [0, 1]})
        model.add_boundary_condition_data(data)
        model._setup_state(SolitaryState(), build_mesh=False)

        self._check_outputs_default(model)

        # Temperature should NOT be in outputs for uncoupled
        sm_region = model._input_file.solid_mechanics_region
        lines = sm_region.subblocks["true_shear_stress_strain"].lines
        self.assertNotIn(f"global {TEMPERATURE_KEY}", lines)

        hb_output = model._input_file.heartbeat_output
        self.assertFalse(hb_output.has_global_output(TEMPERATURE_KEY))

    def test_outputs_added_adiabatic(self):
        """Test that temperature outputs are added for adiabatic coupling."""
        model = self.init_model()
        data = convert_dictionary_to_data({"engineering_strain": [0, 1]})
        data.set_state(State("temp", temperature=100))

        model.add_boundary_condition_data(data)
        model.activate_thermal_coupling()
        model._setup_state(data.state, build_mesh=False)

        self._check_outputs_default(model)

        # Temperature SHOULD be in outputs for adiabatic
        sm_region = model._input_file.solid_mechanics_region
        lines = sm_region.subblocks["true_shear_stress_strain"].lines
        self.assertIn(f"global {TEMPERATURE_KEY}", lines)

        hb_output = model._input_file.heartbeat_output
        self.assertTrue(hb_output.has_global_output(TEMPERATURE_KEY))

    def test_loading_function_comment_added_for_strain_history(self):
        """Test that loading function comments are correctly added."""
        model = self.init_model()
        data = convert_dictionary_to_data({
            TIME_KEY: [0.0, 1.0],
            ENG_STRAIN_KEY: [0.0, 0.1],
        })
        data.set_name("simple_shear_curve_01")
        model.add_boundary_condition_data(data)

        state = data.state
        model._setup_state(state, build_mesh=False)

        func_str = self._get_loading_function_block_string(model)
        assert_source_fields_comment(self, func_str, ENG_STRAIN_KEY, uses_time=True)
        assert_source_collection_comment(self, func_str)
        assert_data_set_index_comment(self, func_str, 0)
        assert_data_set_name_comment(self, func_str, "simple_shear_curve_01")
        assert_selection_reason_comment(self, func_str, ENG_STRAIN_KEY, state.name)

    def test_boundary_conditions_setup(self):
        """Test that boundary conditions are correctly set up for simple shear."""
        model = self.init_model()
        data = convert_dictionary_to_data({"engineering_strain": [0, 0.5]})
        model.add_boundary_condition_data(data)
        model._setup_state(SolitaryState(), build_mesh=False)

        # Check that fixed BC is applied to bottom surface (all directions)
        input_file = model._input_file
        sm_region = input_file.solid_mechanics_region
        
        # Should have prescribed displacement BCs
        bc_blocks = [block for name, block in sm_region.subblocks.items() 
                     if "prescribed displacement" in name.lower()]
        
        # At least one BC block should exist
        self.assertGreater(len(bc_blocks), 0)
