from abc import ABC, abstractmethod, abstractproperty
from matcal.core.parameter_studies import ParameterStudy
import numpy as np
import os

from matcal.core.data import DataCollection, scale_data_collection
from matcal.core.data_importer import FileData
from matcal.core.objective import CurveBasedInterpolatedObjective, ObjectiveCollection, ObjectiveSet
from matcal.core.parameters import Parameter, ParameterCollection
from matcal.core.state import SolitaryState
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest

from matcal.full_field.data_importer import FieldSeriesData
from matcal.full_field.qoi_extractor import InternalVirtualPowerExtractor

from matcal.full_field.objective import MechanicalVFMObjective

from matcal.sierra.models import UserDefinedSierraModel 
from matcal.sierra.tests.platform_options import MatCalTestPlatformOptionsFunctionIdentifier
from matcal.sierra.tests.utilities import (TEST_SUPPORT_FILES_FOLDER, 
     replace_string_in_file)


SET_PLATFORM_OPTIONS = MatCalTestPlatformOptionsFunctionIdentifier.identify()


class MatcalGeneratedModelProductionTestsBase(ABC):

    def __init__():
        pass

    class CommonTests(MatcalUnitTest):
        @abstractproperty
        def _default_output_field_names(self):
            """"""

        @abstractproperty
        def _coupled_output_field_names(self):
            """"""

        def _check_results_data(self, results_data, model, check_objective):
            for field_name in self._default_output_field_names:
                self.assertTrue(field_name in results_data.field_names)

            if model.coupling != "uncoupled" and model.coupling is not None:
                for field_name in self._coupled_output_field_names:
                    self.assertTrue(field_name in results_data.field_names)

            self.assertTrue(len(results_data["time"]) > 0)
            if model.failure == "local" or model.failure == "nonlocal":
                load_field = "load"
                if "torque" in results_data.field_names:
                    load_field = "torque"

                self.assertAlmostEqual(results_data[load_field][-1], 0, places=2)
            else:
                try:
                    #remove after all models convert to the new model setup
                    self.assertAlmostEqual(results_data["time"][-1], model._time_inputs["end_time"])
                except AttributeError:
                    self.assertAlmostEqual(results_data["time"][-1], 
                        model.input_file.solid_mechanics_procedure._termination_time)
                
            if(check_objective):
                self._check_objectives(results_data)
        
        @abstractmethod
        def setup_model(self, coupled=False):
            """"""

        def setUp(self):
            super().setUp(__file__)

        def _run_model_check_input(self, model, state):
            results = model.run_check_input(state, self.get_material_parameter_collection())
            self.assertEqual(results.return_code, 0)
            self.assertEqual(results.results_data, None)

        def _run_model_and_check_results(self, model, state, check_objective=False):
            results = model.run(state, self.get_material_parameter_collection())
            self._check_results_data(results.results_data, model, check_objective)

        def _check_objectives(self, results_data):
            gold_data = self._get_gold_data()
            gold_data.set_state(results_data.state)
            gold_data_collection = DataCollection("gold data", gold_data)
            data_collection = DataCollection("test data", results_data)
            field_names = results_data.field_names
            field_names.remove("time")
            obj = CurveBasedInterpolatedObjective("time", *field_names)
            obj.set_name("test_obj")
            obj_set = ObjectiveSet(ObjectiveCollection('obj', obj), gold_data_collection,
                                    gold_data_collection.states)

            results = obj_set.calculate_objective_set_results(data_collection)
            for obj_results in results[0]["test_obj"].objectives.values():
                for obj_result in obj_results:
                    for field in obj_result.field_names:
                        self.assertAlmostEqual(0, obj_result[field][0], places=2)

        def test_basic_under_integrated_element(self, only_exodus=False):
                model = self.setup_model(coupled=False)  
                model.use_under_integrated_element()
                for bc_dc in self.boundary_condition_data_sets:
                    model.reset_boundary_condition_data()
                    model.add_boundary_condition_data(bc_dc)
                    for state in bc_dc.states.values():
                        self._run_model_and_check_results(model, state)
                        run_dir = model.get_target_dir_name(state)
                        exodus_results_dir = os.path.join(run_dir, "results")
                        if only_exodus:
                            self.assertTrue(os.path.exists(exodus_results_dir)) 
                        else:
                            self.assertTrue(not os.path.exists(exodus_results_dir)) 
                            self.assertTrue(os.path.exists(exodus_results_dir+".csv"))

        def test_basic(self):
            model = self.setup_model(coupled=False)
            bc_dc = self.boundary_condition_data_sets[0]
            model.add_boundary_condition_data(bc_dc)
            for state in bc_dc.states.values():
                self._run_model_check_input(model, state)

        def test_basic_composite_tet(self):
            model = self.setup_model(coupled=False)
            bc_dc = self.boundary_condition_data_sets[0]
            model.add_boundary_condition_data(bc_dc)
            model.use_composite_tet_element()
            for state in bc_dc.states.values():
                self._run_model_check_input(model, state)

        def test_basic_death(self):
            model = self.setup_model(coupled=False)
            bc_dc = self.boundary_condition_data_sets[0]
            model.add_boundary_condition_data(bc_dc)
            model.activate_exodus_output()
            model.add_element_output_variable("eqps")
            model.add_element_output_variable("eqdot")
            model.activate_element_death("eqps", critical_value=0.01)
            try:
                model.activate_implicit_dynamics()
            except AttributeError:
                pass

            for state in bc_dc.states.values():
                self._run_model_check_input(model, state)
                break

        def test_coupled_under_integrated_element(self):
            model = self.setup_model(coupled=True)
            bc_dc = self.boundary_condition_data_sets[0]
            model.add_boundary_condition_data(bc_dc)
            model.activate_exodus_output()
            model.add_element_output_variable("eqps")
            model.add_element_output_variable("eqdot")
            model.use_under_integrated_element()
        
            for state in bc_dc.states.values():
                if "temperature" in state.params or "temperature" in model.get_model_constants():
                    self._run_model_and_check_results(model, state, check_objective=True)
                else:
                    with self.assertRaises(RuntimeError):
                        self._run_model_and_check_results(model, state)
                break

        def test_adiabatic(self):
            model = self.setup_model()
            bc_dc = self.boundary_condition_data_sets[0]
            model.add_boundary_condition_data(bc_dc)
            model.activate_exodus_output()
            model.add_element_output_variable("eqdot")
            model.add_constants(displacement_rate=10, coupling='adiabatic')
            model.activate_thermal_coupling()
            model.activate_element_death("eqps", 0.01)
            model.use_under_integrated_element()
            for state in bc_dc.states.values():
                if "temperature" in state.params:
                    self._run_model_and_check_results(model, state)
                else:
                    with self.assertRaises(RuntimeError):
                        self._run_model_and_check_results(model, state)
                break

        def test_minimum_time_step(self):
            model = self.setup_model()
            model.add_boundary_condition_data(self.boundary_condition_data_sets[0])
            try:
                model.set_minimum_timestep(1e-2)
                model.set_number_of_time_steps(10000)
                for state in self.boundary_condition_data_sets[0].states.values():
                    results = model.run(state, self.get_material_parameter_collection())
                    break
                model_log_file = os.path.join(model.get_target_dir_name(state), 
                    f"{model.name}.log")
                with open(model_log_file, "r") as f:
                    file_contents = f.read()
                    self.assertIn("terminate global timestep < 0.01", file_contents)
                    self.assertIn("SIERRA termination reason: User solution termination criteria", 
                        file_contents)
            except AttributeError:
                pass


from matcal.sierra.tests.sierra_sm_models_for_tests import (
    UniaxialLoadingMaterialPointModelForTests)
class UniaxialLoadingMaterialPointModelProductionTests(
    MatcalGeneratedModelProductionTestsBase.CommonTests, 
    UniaxialLoadingMaterialPointModelForTests):

    _default_output_field_names = ["time", "displacement", "load", "engineering_strain", 
                                   "engineering_stress", "true_stress", "true_strain", 
                                   "contraction", ]

    _coupled_output_field_names = ["temperature"]

    def setup_model(self, coupled=False):
        model = self.init_model(plasticity=True, coupled=coupled)
        SET_PLATFORM_OPTIONS(model)
        return model 

    def test_coupled_under_integrated_element(self):
        """"""""

    def test_basic_composite_tet(self):
        """"""

    def test_compression(self):
        model = self.setup_model()
        bc_dc = self.boundary_condition_data_sets[0]
        neg_bc_dc = scale_data_collection(bc_dc, "engineering_stress", -1)
        neg_bc_dc = scale_data_collection(neg_bc_dc, "engineering_strain", -1)

        model.add_boundary_condition_data(neg_bc_dc)

        for state in bc_dc.states.values():
            results = model.run(state, self.get_material_parameter_collection())
            self._check_compression(results.results_data)
            self._check_results_data(results.results_data, model, check_objective=True)
            break

    def _check_compression(self, results_data):
            self.assertLess(results_data["engineering_stress"][-1], 0)
            self.assertLess(results_data["engineering_strain"][-1], 0)
            self.assertLess(results_data["true_stress"][-1], 0)
            self.assertLess(results_data["true_strain"][-1], 0)
      
class UniaxialTensionModelProductionTestsBase(ABC):
    def __init__():
        pass
    class CommonTests(MatcalGeneratedModelProductionTestsBase.CommonTests):

        _default_output_field_names = ["time", "displacement", "load", "engineering_strain", 
                                   "engineering_stress", "z_contraction", "x_contraction"]

        _coupled_output_field_names = ["low_temperature", "med_temperature", "high_temperature" ]

        def setup_model(self, coupled=False):
            model = self.init_model(plasticity=True, coupled=coupled)
            if not coupled:
                model.add_constants(coupling="uncoupled")
            model.set_number_of_cores(16)
            model.set_number_of_time_steps(100)
            SET_PLATFORM_OPTIONS(model)
            return model 

        @abstractmethod
        def _additional_geo_params(self):
            """"""

        def test_basic_under_integrated_full_field_output(self):
                model = self.setup_model(coupled=False)  
                model.use_under_integrated_element()
                model.activate_full_field_data_output(0.1*0.0254, 0.25*0.0254) 
                for bc_dc in self.boundary_condition_data_sets:
                    model.reset_boundary_condition_data()
                    model.add_boundary_condition_data(bc_dc)
                    for state in bc_dc.states.values():
                        self._run_model_and_check_results(model, state)
                        ff_data = FieldSeriesData(f"{model.name}/{state.name}/results/full_field_results.e")
                        for field_name in self._default_output_field_names:
                            self.assertTrue(field_name in ff_data.field_names)
                        self.assertTrue("displacement_x" in ff_data.field_names)
                        self.assertTrue("displacement_y" in ff_data.field_names)
                        self.assertTrue("displacement_z" in ff_data.field_names)
                        len_coords = len(ff_data.spatial_coords[:,0])
                        self.assertTrue(len_coords > 0)
                        self.assertTrue(np.all(ff_data.spatial_coords[:,0] <= 0.1*0.0254))
                        self.assertTrue(np.all(ff_data.spatial_coords[:,1] <= 0.25*0.0254))
                        break
                    break

from matcal.sierra.tests.sierra_sm_models_for_tests import RectangularUniaxialTensionModelForTests
class RectangularUniaxialTensionModelProductionTests(
    UniaxialTensionModelProductionTestsBase.CommonTests, 
    RectangularUniaxialTensionModelForTests):
    """"""

    def test_coupled_death(self):
        model = self.setup_model(coupled=True)
        bc_dc = self.boundary_condition_data_sets[0]
        model.add_boundary_condition_data(bc_dc)
        model.activate_exodus_output()
        model.add_element_output_variable("eqps")
        model.add_element_output_variable("eqdot")
        model.activate_element_death("eqps", 0.01)
        model.activate_implicit_dynamics()
        for state in bc_dc.states.values():
            if "temperature" in state.params:
                self._run_model_and_check_results(model, state)
            else:
                with self.assertRaises(model.InputError):
                    self._run_model_and_check_results(model, state)
            break

    def test_coupled_death_composite_tet(self):
        model = self.setup_model(coupled=True)
        bc_dc = self.boundary_condition_data_sets[0]
        model.add_boundary_condition_data(bc_dc)
        model.activate_exodus_output()
        model.add_element_output_variable("eqps")
        model.add_element_output_variable("eqdot")
        model.activate_element_death("eqps", 0.01)
        model.use_composite_tet_element()
        model.activate_implicit_dynamics()
        for state in bc_dc.states.values():
            if "temperature" in state.params:
                self._run_model_and_check_results(model, state)
            else:
                with self.assertRaises(model.InputError):
                    self._run_model_and_check_results(model, state)
            break

    def test_coupled(self):
        model = self.setup_model(coupled=True)
        bc_dc = self.boundary_condition_data_sets[0]
        model.add_boundary_condition_data(bc_dc)
        model.activate_exodus_output()
        model.add_element_output_variable("eqps")
        model.add_element_output_variable("eqdot")

        for state in bc_dc.states.values():
            if "temperature" in state.params:
                self._run_model_check_input(model, state)
            else:
                with self.assertRaises(model.InputError):
                    self._run_model_check_input(model, state)

    def test_basic_nonlocal_death(self):
        model = self.setup_model(coupled=False)
        bc_dc = self.boundary_condition_data_sets[0]
        model.add_boundary_condition_data(bc_dc)
        model.activate_exodus_output()
        model.add_element_output_variable("eqps")
        model.add_element_output_variable("eqdot")
        model.activate_element_death("eqps", critical_value=0.01, 
                nonlocal_radius=self.geo_params["element_size"])

        model.activate_implicit_dynamics()
        for state in bc_dc.states.values():
            self._run_model_and_check_results(model, state)
            break


from matcal.sierra.tests.sierra_sm_models_for_tests import RoundUniaxialTensionModelForTests
class RoundUniaxialTensionModelProductionTests(UniaxialTensionModelProductionTestsBase.CommonTests, 
                                               RoundUniaxialTensionModelForTests):
    """"""

from matcal.sierra.tests.sierra_sm_models_for_tests import RoundNotchedTensionModelForTests
class RoundNotchedTensionModelProductionTests(RoundNotchedTensionModelForTests, 
                                              MatcalGeneratedModelProductionTestsBase.CommonTests):

    _default_output_field_names = ["time", "displacement", "load" ]

    _coupled_output_field_names = ["low_temperature", "med_temperature", "high_temperature" ]

    def setup_model(self, coupled=False):
        model = self.init_model(plasticity=True, coupled=coupled)                
        model.set_number_of_cores(16)
        model.set_number_of_time_steps(100)
        SET_PLATFORM_OPTIONS(model)
        return model 

    def test_basic_under_integrated_full_field_output(self):
            model = self.setup_model(coupled=False)  
            model.use_under_integrated_element()
            model.activate_full_field_data_output(0.1*0.0254, 0.25*0.0254) 
            for bc_dc in self.boundary_condition_data_sets:
                model.reset_boundary_condition_data()
                model.add_boundary_condition_data(bc_dc)
                for state in bc_dc.states.values():
                    self._run_model_and_check_results(model, state)
                    ff_data = FieldSeriesData(f"{model.name}/{state.name}/results/full_field_results.e")
                    for field_name in self._default_output_field_names:
                        self.assertTrue(field_name in ff_data.field_names)
                    self.assertTrue("displacement_x" in ff_data.field_names)
                    self.assertTrue("displacement_y" in ff_data.field_names)
                    self.assertTrue("displacement_z" in ff_data.field_names)
                    len_coords = len(ff_data.spatial_coords[:,0])
                    self.assertTrue(len_coords > 0)
                    self.assertTrue(np.all(ff_data.spatial_coords[:,0] <= 0.1*0.0254))
                    self.assertTrue(np.all(ff_data.spatial_coords[:,1] <= 0.25*0.0254))
                    break
                break

    def test_basic_single_core(self):
        model = self.setup_model(coupled=False)
        bc_dc = self.boundary_condition_data_sets[0]
        model.add_boundary_condition_data(bc_dc)
        model.activate_exodus_output()
        model.add_element_output_variable("eqdot")
        model.set_number_of_cores(1)

        for state in bc_dc.states.values():
            self._run_model_and_check_results(model, state)
            run_dir = model.get_target_dir_name(state)
            exodus_results_dir = os.path.join(run_dir, "results")
            self.assertTrue(os.path.exists(exodus_results_dir))
            break


from matcal.sierra.tests.sierra_sm_models_for_tests import TopHatShearModelForTests
class TopHatShearModelProductionTests(TopHatShearModelForTests, 
                                      MatcalGeneratedModelProductionTestsBase.CommonTests):
    
    _default_output_field_names = ["time", "displacement", "load" ]

    _coupled_output_field_names = ["low_temperature", "med_temperature", "high_temperature" ]

    def setup_model(self, coupled=False):
        model = self.init_model(plasticity=True, coupled=coupled)                
        model.set_number_of_cores(16)
        model.set_number_of_time_steps(100)
        SET_PLATFORM_OPTIONS(model)
        return model 
    
    def test_self_contact(self):
        model = self.setup_model(coupled=False)  
        model.activate_exodus_output()
        model.use_under_integrated_element()
        model.activate_self_contact()
        for bc_dc in self.boundary_condition_data_sets:
            model.reset_boundary_condition_data()
            model.add_boundary_condition_data(bc_dc)
            for state in bc_dc:
                self._run_model_check_input(model, state)

from matcal.sierra.tests.sierra_sm_models_for_tests import SolidBarTorsionModelForTests
class SolidBarTorsionModelProductionTests(SolidBarTorsionModelForTests, 
                                          MatcalGeneratedModelProductionTestsBase.CommonTests):
    
    _default_output_field_names = ["time", "applied_rotation", "grip_rotation", "torque" ]

    _coupled_output_field_names = ["low_temperature", "med_temperature", "high_temperature" ]

    def setup_model(self, coupled=False):
        model = self.init_model(plasticity=True, coupled=coupled)                
        model.set_number_of_time_steps(100)
        model.set_number_of_cores(16)
        SET_PLATFORM_OPTIONS(model)

        return model 

class VFMUniaxialTensionModelBaseProductionTests():
    def __init__():
        pass
    class Tests( MatcalGeneratedModelProductionTestsBase.CommonTests):

        _default_output_field_names = ["time", "centroid_x", "centroid_y", "centroid_z", "volume",
            "first_pk_stress_xx", "first_pk_stress_xy" , "first_pk_stress_xz", 
            "first_pk_stress_yx", "first_pk_stress_yy" , "first_pk_stress_yz",
            "first_pk_stress_zx", "first_pk_stress_zy" , "first_pk_stress_zz"]
        _coupled_output_field_names = []
        def setup_model(self, coupled=False):
            model = self.init_model(plasticity=True, coupled=coupled)                
            model.set_number_of_time_steps(100)
            SET_PLATFORM_OPTIONS(model)

            return model 

        def test_basic_under_integrated_element(self, only_exodus=True):
            super().test_basic_under_integrated_element(only_exodus)

        def test_VFM_model_multicore(self):
            vfm_model = self.prepare_VFM_model()
            vfm_model.set_number_of_cores(2)
            self._model_name = vfm_model.name
            state = SolitaryState()
            results = vfm_model.run(state, 
                ParameterCollection("test", Parameter("RC", 0, 1, 1e-1))).results_data
            
            self.assertTrue(os.path.exists(os.path.join(vfm_model.get_target_dir_name(state), 
                                                        "results", "results.e.2.0")))
            self.assertTrue(os.path.exists(os.path.join(vfm_model.get_target_dir_name(state),
                                                         "results", "results.e.2.1")))
            
            self.assertTrue(os.path.exists(os.path.join(vfm_model.get_target_dir_name(state),
                                                         "results", "results.e")))
            self.assertTrue("first_pk_stress_xx" in results.field_names)
            self.assertTrue("first_pk_stress_xy" in results.field_names)
            self.assertTrue("first_pk_stress_yy" in results.field_names)
            self.assertTrue("first_pk_stress_yx" in results.field_names)

            self._check_simulation_stderr_is_empty(vfm_model.get_target_dir_name(state))

        def test_prescribed_temp_field_VFM_model_single_core(self):
            vfm_model = self.prepare_VFM_model(adiabatic=False)
            vfm_model.set_number_of_cores(1)
            vfm_model.read_temperature_from_boundary_condition_data("temp")

            self._model_name = vfm_model.name
            state = SolitaryState()
            results = vfm_model.run(state, 
                        ParameterCollection("test", Parameter("RC", 0, 1e12, 2e10))).results_data

            self.assertTrue("first_pk_stress_xx" in results.field_names)
            self.assertTrue("first_pk_stress_xy" in results.field_names)
            self.assertTrue("first_pk_stress_yy" in results.field_names)
            self.assertTrue("first_pk_stress_yx" in results.field_names)
            self._check_simulation_stderr_is_empty(vfm_model.get_target_dir_name(state))

        def _check_simulation_stderr_is_empty(self, dir, goal_lines=[]):
            with open(os.path.join(dir, "simulation.err"), 'r') as f:
                lines = f.readlines()
            self.assertListEqual(goal_lines, lines)

        def test_basic_composite_tet(self):
            """"""

        def _check_objectives(self, results_data):
            gold_data = self._get_gold_data()
            gold_data.set_state(results_data.state)
            gold_data_collection = DataCollection("gold data", gold_data)
            data_collection = DataCollection("test data", results_data)
                        
            obj = MechanicalVFMObjective("time", "load")
            obj.set_experiment_qoi_extractor(InternalVirtualPowerExtractor("time"))
            obj.set_name("test_obj")
            obj_set = ObjectiveSet(ObjectiveCollection('obj', obj), gold_data_collection,
                                    gold_data_collection.states)

            results = obj_set.calculate_objective_set_results(data_collection)
            for obj_results in results[0]["test_obj"].objectives.values():
                for obj_result in obj_results:
                    for field in obj_result.field_names:
                        self.assertAlmostEqual(0, obj_result[field][0], places=2)


from matcal.sierra.tests.sierra_sm_models_for_tests import VFMUniaxialTensionHexModelForTests
class VFMUniaxialTensionHexModelProductionTests(VFMUniaxialTensionHexModelForTests,
                                               VFMUniaxialTensionModelBaseProductionTests.Tests):
        """"""""


from matcal.sierra.tests.sierra_sm_models_for_tests import (
    VFMUniaxialTensionConnectedHexModelForTests)
class VFMUniaxialTensionConnectedHexModelProductionTests(
    VFMUniaxialTensionConnectedHexModelForTests, VFMUniaxialTensionModelBaseProductionTests.Tests):
    """"""

    
from matcal.sierra.tests.sierra_sm_models_for_tests import UserDefinedSierraModelForTests
class TestUserDefinedSierraModel(MatcalUnitTest, UserDefinedSierraModelForTests):

    def setUp(self):
        super().setUp(__file__)
    
    def test_aria_user_model(self):
        test_support_dir= os.path.join(TEST_SUPPORT_FILES_FOLDER,
                                     "user_defined_model_tests")
        goal = FileData(os.path.join(test_support_dir, "user_defined_model_goal.csv"))
        input_file =os.path.join(test_support_dir, "conductivity.i")
        mesh_file = "square.g"
        mesh_file = os.path.join(test_support_dir, mesh_file)
        model = UserDefinedSierraModel('aria', input_file, mesh_file)
        model.set_number_of_cores(2)
        model.continue_when_simulation_fails()
        SET_PLATFORM_OPTIONS(model)
        state = SolitaryState()
        results = model.run(state, ParameterCollection("test", Parameter("K", 0, 5, 1)))
        self.assert_close_arrays(goal, results.results_data)

    def test_simulation_fails_half_the_time_complete_the_study(self):
        test_support_dir= os.path.join(TEST_SUPPORT_FILES_FOLDER,
                                     "user_defined_model_tests")
        goal_pass = FileData(os.path.join(test_support_dir, "user_defined_model_goal.csv"))
        input_file =os.path.join(test_support_dir, "conductivity.i")
        mesh_file = "square.g"
        mesh_file = os.path.join(test_support_dir, mesh_file)
        model = UserDefinedSierraModel('aria', input_file, mesh_file)
        model.continue_when_simulation_fails()
        obj = CurveBasedInterpolatedObjective('time', 'temperature')
        p = Parameter('K', -1, 1)
        study = ParameterStudy(p)
        study.add_evaluation_set(model,obj, goal_pass)
        study.add_parameter_evaluation(K=1)
        study.add_parameter_evaluation(K=-.1)
        study.add_parameter_evaluation(K=1)
        study.add_parameter_evaluation(K=-.1)
        study.set_core_limit(4)
        results = study.launch()
        sim_hist = results.simulation_history[model.name][goal_pass.state]
        failure_default = {'time':np.linspace(-1, 1, 20), 'temperature':np.linspace(-1, 1, 20)}
        for i, model_results in enumerate(sim_hist):
            if i%2 == 0:
                current_goal = goal_pass
            else:
                current_goal = failure_default
            for name in ['time', 'temperature']:
                self.assert_close_arrays(current_goal[name], model_results[name])

    def test_create_user_supplied_simulation_user_specified_filename_and_file_type(self):
        model_generator = UniaxialLoadingMaterialPointModelForTests()
        model = model_generator.init_model(plasticity=True)
        bc_data = model_generator.boundary_condition_data_sets[-1]
        model.add_boundary_condition_data(bc_data)
        state_name = bc_data.state_names[-1]
        state = bc_data.states[state_name]
        model.set_name("test_model")
        model.set_number_of_time_steps(50)
        results_gold = model.run(state, model_generator.get_material_parameter_collection())

        target_dir = model.get_target_dir_name(state)
        input_name = os.path.join(target_dir, "test_model.aprepro.i")
        mesh_name = os.path.join(target_dir, "test_model.g")
        user_out_filename = "sim_out.dat"
        replace_string_in_file(input_name, "results.csv", user_out_filename)
        model = UserDefinedSierraModel('adagio', input_name, mesh_name)
        model.set_results_filename(user_out_filename, file_type="csv")
        SET_PLATFORM_OPTIONS(model)
        def_state = SolitaryState()
        results = model.run(def_state, model_generator.get_material_parameter_collection())
        self.assert_close_arrays(results_gold.results_data, results.results_data)
