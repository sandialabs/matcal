import numpy as np
import os

from matcal.core.constants import STATE_PARAMETER_FILE
from matcal.core.data import convert_dictionary_to_data
from matcal.core.parameters import Parameter, ParameterCollection
from matcal.core.state import State, SolitaryState
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest

from matcal.sierra.models import UserDefinedSierraModel
from matcal.sierra.tests.utilities import write_empty_file


class MatCalStandardModelIntegrationTestBase:
    def __init__():
        pass
    class CommonTests(MatcalUnitTest):

        def setUp(self):
            return super().setUp(__file__)

        def test_run_without_boundary_condition_data(self):
            model = self.init_model()
            param = Parameter("elastic_modulus", 0, 1000, 100)
            pc = ParameterCollection("my PC", param)
            model.reset_boundary_condition_data()
            with self.assertRaises(RuntimeError):
                model.run(SolitaryState(), pc)

        def test_preprocess_model_and_build_files(self):
            model = self.init_model()
            new_state_constants= [{"c":4, "d":"D", "a":3}, 
                                  {"num_time_steps":1e3, "c":5, "d":"d"}]
            new_constants = {"b":2, "element":"TL", "num_time_steps":1e2}

            for bc_dc in self.boundary_condition_data_sets:
                for idx, state in enumerate(bc_dc.states.values()):
                    model.reset_constants()
                    expected_states = {}
                    expected_states.update(state.params)
                    expected_states.update(new_constants)
                    
                    model.add_constants(**new_constants)
                    if idx < len(new_state_constants):
                        model.add_state_constants(state, **new_state_constants[idx])
                        expected_states.update(new_state_constants[idx])
                        
                    model.reset_boundary_condition_data()
                    model.add_boundary_condition_data(bc_dc)
                    target_dir = model.get_target_dir_name(state)

                    model.preprocess(state)
                    mesh_filename = f"{model.name}.g"
                    input_filename = f"{model.name}.i"
                    expected_mesh_file = os.path.join(target_dir, mesh_filename)

                    expected_input_file = os.path.join(target_dir, input_filename)
                    expected_state_file = os.path.join(target_dir, STATE_PARAMETER_FILE)
                    self.assert_file_exists(expected_mesh_file)
                    self.assert_file_exists(expected_input_file)
                    self.assert_file_exists(expected_state_file)

                    from matcal.sierra.aprepro_format import parse_aprepro_variable_line
                    with open(expected_state_file, "r") as f:
                        state_vars = {}
                        for line in f.readlines():
                            key, var = parse_aprepro_variable_line(line)
                            if key:
                                state_vars[key] = var
                    for key in state_vars.keys():
                        expected_value = expected_states[key]
                        if isinstance(expected_value, str):
                            expected_value = "'"+expected_value+"'"
                        self.assertAlmostEqual(state_vars[key], expected_value)              
        
       
from matcal.sierra.tests.sierra_sm_models_for_tests import UniaxialLoadingMaterialPointModelForTests
class UniaxialLoadingMaterialPointModelIntegrationTest(MatCalStandardModelIntegrationTestBase.CommonTests, 
    UniaxialLoadingMaterialPointModelForTests):
    """"""

      
class UniaxialTensionModelIntegrationTestBase:
    def __init__():
        pass

    class CommonTests(MatCalStandardModelIntegrationTestBase.CommonTests):

        def test_update_geometry_params_from_state(self):
            state = State("new extensometer", extensometer_length=0.019)
            state2 = State("new extensometer2", extensometer_length=0.02, element_size=0.0005)
            data_dict = {"displacement":np.linspace(0,1,10)*0.0254}
            data = convert_dictionary_to_data(data_dict)
            data.set_state(state)
            data2 = convert_dictionary_to_data(data_dict)
            data2.set_state(state2)

            model = self.init_model()

            model.add_boundary_condition_data(data)
            model.add_boundary_condition_data(data2)

            model.preprocess(state)
            model.preprocess(state2)
            state_dir = model.get_target_dir_name(state)
            journal_file = os.path.join(state_dir, model._geometry_creator_class._journal_filename)

            with open(journal_file, "r") as journal_file:
                file_data = journal_file.read()
                self.assertTrue("extensometer_length = 0.019" in file_data)
            
            with open(os.path.join(state_dir, model._input_filename)) as input_file:
                file_data = input_file.read()
                self.assertTrue("displacement/0.019" in file_data)
            
            state2_dir = model.get_target_dir_name(state2)
            journal_file = os.path.join(state2_dir, model._geometry_creator_class._journal_filename)
            with open(journal_file, "r") as journal_file:
                file_data = journal_file.read()
                self.assertTrue("extensometer_length = 0.02" in file_data)
                self.assertTrue("element_size = 0.0005" in file_data)

            with open(os.path.join(state2_dir, model._input_filename), "r") as input_file:
                file_data = input_file.read()
                self.assertTrue("displacement/0.02" in file_data)


        def test_update_geometry_params_from_model_constants_take_precendent(self):
            state = State("new extensometer", extensometer_length=0.019)
            state2 = State("new extensometer2", extensometer_length=0.02, element_size=0.0005)
            data_dict = {"displacement":np.linspace(0,1,10)*0.0254}
            data = convert_dictionary_to_data(data_dict)
            data.set_state(state)
            data2 = convert_dictionary_to_data(data_dict)
            data2.set_state(state2)

            model = self.init_model()
            model.add_constants(extensometer_length=0.0195)
            model.add_state_constants(data2.state, extensometer_length=0.01975)
            model.add_boundary_condition_data(data)
            model.add_boundary_condition_data(data2)

            model.preprocess(state)
            model.preprocess(state2)
            state_dir = model.get_target_dir_name(state)
            journal_file = os.path.join(state_dir, model._geometry_creator_class._journal_filename)
            with open(journal_file, "r") as journal_file:
                file_data = journal_file.read()
                self.assertTrue("extensometer_length = 0.0195" in file_data)
            
            with open(os.path.join(state_dir, model._input_filename)) as input_file:
                file_data = input_file.read()
                self.assertTrue("displacement/0.0195" in file_data)
            
            state2_dir = model.get_target_dir_name(state2)
            journal_file = os.path.join(state2_dir, model._geometry_creator_class._journal_filename)
            with open(journal_file, "r") as journal_file:
                file_data = journal_file.read()
                self.assertTrue("extensometer_length = 0.01975" in file_data)
                self.assertTrue("element_size = 0.0005" in file_data)

            with open(os.path.join(state2_dir, model._input_filename), "r") as input_file:
                file_data = input_file.read()
                self.assertTrue("displacement/0.01975" in file_data)

from matcal.sierra.tests.sierra_sm_models_for_tests import RoundUniaxialTensionModelForTests
class RoundUniaxialTensionModelIntegrationTest(UniaxialTensionModelIntegrationTestBase.CommonTests, 
    RoundUniaxialTensionModelForTests):
    """"""


from matcal.sierra.tests.sierra_sm_models_for_tests import RectangularUniaxialTensionModelForTests
class RectangularUniaxialTensionModelIntegrationTest(UniaxialTensionModelIntegrationTestBase.CommonTests, 
    RectangularUniaxialTensionModelForTests):
    """"""


from matcal.sierra.tests.sierra_sm_models_for_tests import SolidBarTorsionModelForTests
class SolidBarTorsionIntegrationTest(MatCalStandardModelIntegrationTestBase.CommonTests, 
    SolidBarTorsionModelForTests):

  def test_update_geometry_params_from_state(self):
            state = State("new extensometer", extensometer_length=0.005)
            state2 = State("new extensometer2", extensometer_length=0.004, element_size=0.0005)
            data_dict = {"grip_rotation":np.linspace(0,250,100)}
            data = convert_dictionary_to_data(data_dict)
            data.set_state(state)
            data2 = convert_dictionary_to_data(data_dict)
            data2.set_state(state2)

            model = self.init_model()

            model.add_boundary_condition_data(data)
            model.add_boundary_condition_data(data2)

            model.preprocess(state)
            model.preprocess(state2)
            state_dir = model.get_target_dir_name(state)
            journal_file = os.path.join(state_dir, model._geometry_creator_class._journal_filename)
            with open(journal_file, "r") as journal_file:
                file_data = journal_file.read()
                self.assertTrue("extensometer_length = 0.005" in file_data)
            
            state2_dir = model.get_target_dir_name(state2)
            journal_file = os.path.join(state2_dir, model._geometry_creator_class._journal_filename)
            with open(journal_file, "r") as journal_file:
                file_data = journal_file.read()
                self.assertTrue("extensometer_length = 0.004" in file_data)
                self.assertTrue("element_size = 0.0005" in file_data)
            
            
from matcal.sierra.tests.sierra_sm_models_for_tests import RoundNotchedTensionModelForTests
class RoundNotchedTensionModelIntegrationTest(MatCalStandardModelIntegrationTestBase.CommonTests, 
    RoundNotchedTensionModelForTests):

    def test_update_geometry_params_from_state(self):
        state = State("new extensometer", extensometer_length=0.019)
        state2 = State("new extensometer2", extensometer_length=0.02, element_size=0.0005)
        data_dict = {"displacement":np.linspace(0,1,10)*0.0254}
        data = convert_dictionary_to_data(data_dict)
        data.set_state(state)
        data2 = convert_dictionary_to_data(data_dict)
        data2.set_state(state2)

        model = self.init_model()

        model.add_boundary_condition_data(data)
        model.add_boundary_condition_data(data2)

        model.preprocess(state)
        model.preprocess(state2)
        state_dir = model.get_target_dir_name(state)

        with open(os.path.join(state_dir, model._geometry_creator_class._journal_filename), "r") as journal_file:
            file_data = journal_file.read()
            self.assertTrue("extensometer_length = 0.019" in file_data)
       
        state2_dir = model.get_target_dir_name(state2)
        with open(os.path.join(state2_dir, model._geometry_creator_class._journal_filename), "r") as journal_file:
            file_data = journal_file.read()
            self.assertTrue("extensometer_length = 0.02" in file_data)
            self.assertTrue("element_size = 0.0005" in file_data)


from matcal.sierra.tests.sierra_sm_models_for_tests import TopHatShearModelForTests
class TopHatShearModelIntegrationTest(MatCalStandardModelIntegrationTestBase.CommonTests, 
    TopHatShearModelForTests):

    def test_update_geometry_params_from_state(self):
        state = State("new_base_width", base_width=1.625*0.0254*1.1)
        state2 = State("new base width 2", base_width=1.625*0.0254*0.9,
                       element_size=0.004*0.0254, numsplits=2)
        data_dict = {"displacement":np.linspace(0,1,10)*0.0254}
        data = convert_dictionary_to_data(data_dict)
        data.set_state(state)
        data2 = convert_dictionary_to_data(data_dict)
        data2.set_state(state2)

        model = self.init_model()

        model.add_boundary_condition_data(data)
        model.add_boundary_condition_data(data2)

        model.preprocess(state)
        model.preprocess(state2)
        state_dir = model.get_target_dir_name(state)

        with open(os.path.join(state_dir, model._geometry_creator_class._journal_filename), "r") as journal_file:
            file_data = journal_file.read()
            self.assertTrue(f"base_width = {1.625*0.0254*1.1}" in file_data)
       
        state2_dir = model.get_target_dir_name(state2)
        with open(os.path.join(state2_dir, model._geometry_creator_class._journal_filename), "r") as journal_file:
            file_data = journal_file.read()
            self.assertTrue(f"base_width = {1.625*0.0254*0.9}" in file_data)
            self.assertTrue(f"element_size = {0.004*0.0254}" in file_data)         


from matcal.sierra.tests.sierra_sm_models_for_tests import (
    VFMUniaxialTensionHexModelForTests)
class VFMUniaxialTensionHexModelIntegrationTests(MatCalStandardModelIntegrationTestBase.CommonTests, 
    VFMUniaxialTensionHexModelForTests):
    """"""
    def test_generate_geometry(self):
        model = self.init_model()
        model.add_boundary_condition_data(self._field_data)
        model.set_displacement_field_names("Ux", "Uy")
        state = SolitaryState()
        model.preprocess(state)
        self.assertTrue(os.path.exists(os.path.join(model.get_target_dir_name(state), 
                                                    model._mesh_filename)))

    def test_generate_geometry_with_temperature(self):
        model = self.init_model()
        model.add_boundary_condition_data(self._field_data)
        model.set_displacement_field_names("Ux", "Uy")
        model.read_temperature_from_boundary_condition_data('temperature')
        state = SolitaryState()
        model._setup_state(state, build_mesh=True)
        self.assertTrue(os.path.exists(model._input_filename))


from matcal.sierra.tests.sierra_sm_models_for_tests import (
    VFMUniaxialTensionConnectedHexModelForTests)
class VFMUniaxialTensionConnectedHexModelIntegrationTests(MatCalStandardModelIntegrationTestBase.CommonTests, 
    VFMUniaxialTensionConnectedHexModelForTests):
    """"""


from matcal.sierra.tests.sierra_sm_models_for_tests import UserDefinedSierraModelForTests
class UserDefinedSierraModelIntegrationTests(MatcalUnitTest, UserDefinedSierraModelForTests):
    def setUp(self):
        return super().setUp(__file__)

    def tearDown(self):
        super().tearDown()

    def _get_additional_aprepro_files(self):  
        apr_files = ["fake_apr.inc", "fake_apr2.inc"]
        for f in apr_files:
            write_empty_file(f)
        os.mkdir("test_dir")
        write_empty_file("test_dir/empty.inc")
        apr_files.append("test_dir")
        return apr_files

    def test_decompose_mesh_with_2_cores(self):
        from matcal.sierra.tests.utilities import (make_complex_mesh_for_tests, 
                                                   COARSE_COMPLEX_MESH_NAME)
        make_complex_mesh_for_tests()
        state=SolitaryState()
        input_file = 'fake_input.i'
        write_empty_file(input_file)

        model = UserDefinedSierraModel('aria', input_file, COARSE_COMPLEX_MESH_NAME, 
                                       *self._get_additional_aprepro_files())
        model.set_number_of_cores(2)
        model.preprocess(SolitaryState())
        target_dir = model.get_target_dir_name(state)
        self.assertTrue(os.path.exists(os.path.join(target_dir, input_file)))
        self.assertTrue(os.path.exists(os.path.join(target_dir, COARSE_COMPLEX_MESH_NAME+".2.0")))
        self.assertTrue(os.path.exists(os.path.join(target_dir, COARSE_COMPLEX_MESH_NAME+".2.1")))
        self.assertTrue(os.path.exists(os.path.join(target_dir,"fake_apr.inc")))
        self.assertTrue(os.path.exists(os.path.join(target_dir,"fake_apr2.inc")))
        self.assertTrue(os.path.exists(os.path.join(target_dir,"test_dir/empty.inc")))
 