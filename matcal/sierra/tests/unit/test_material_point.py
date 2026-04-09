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
)
from matcal.core.data import convert_dictionary_to_data
from matcal.core.state import SolitaryState, State

from model_tests_base import MatcalStandardModelUnitTestNewBase
from matcal.sierra.tests.sierra_sm_models_for_tests import (
    UniaxialLoadingMaterialPointModelForTests,
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