from abc import abstractmethod
import copy
from matcal.core.logger import matcal_print_message
import numpy as np
import os

from matcal.core.boundary_condition_calculators import (DISPLACEMENT_KEY, 
    DISPLACEMENT_RATE_KEY, ENG_STRAIN_KEY, STRAIN_RATE_KEY, TIME_KEY, TRUE_STRAIN_KEY)
from matcal.core.data import DataCollection,  convert_dictionary_to_data
from matcal.core.data_importer import FileData
from matcal.core.parameters import Parameter, ParameterCollection
from matcal.core.state import State, SolitaryState
from matcal.core.tests.unit.test_models import ModelForTestsBase

from matcal.full_field.data import convert_dictionary_to_field_data
from matcal.full_field.data_importer import FieldSeriesData
from matcal.full_field.TwoDimensionalFieldGrid import auto_generate_two_dimensional_field_grid

from matcal.sierra.material import Material
from matcal.sierra.models import (RectangularUniaxialTensionModel, 
    RoundNotchedTensionModel, RoundUniaxialTensionModel,
    SolidBarTorsionModel, TopHatShearModel, UniaxialLoadingMaterialPointModel, 
    UserDefinedSierraModel, VFMUniaxialTensionConnectedHexModel, VFMUniaxialTensionHexModel)
from matcal.sierra.simulators import SierraSimulator
from matcal.sierra.tests.utilities import (write_j2_plasticity_material_file, write_empty_file, 
    write_linear_elastic_material_file, read_file_lines, create_goal_user_model_simulation_results,
    TEST_SUPPORT_FILES_FOLDER, GENERATED_TEST_DATA_FOLDER)


class SierraModelForTestsBase(ModelForTestsBase):
    _simulator_class = SierraSimulator


def get_linear_displacement_vs_time_data(end_time, end_displacement):
    time = np.linspace(1,end_time, 100)
    disp_rate = end_displacement/(time[-1]-time[0])
    displacement = disp_rate*(time-time[0])
    state = State(f"rate_{disp_rate}", temperature=298)
    state.update_state_variable(DISPLACEMENT_RATE_KEY, disp_rate)
    data_dict = {TIME_KEY:time, DISPLACEMENT_KEY:displacement}
    data = convert_dictionary_to_data(data_dict)
    data.set_state(state)
    return data


def convert_disp_time_to_engineering_strain_time(displacement_data, initial_length):
    displacement = copy.deepcopy(displacement_data[DISPLACEMENT_KEY])
    time = copy.deepcopy(displacement_data[TIME_KEY])
    
    eng_strain = displacement/initial_length
    eng_strain_dict = {ENG_STRAIN_KEY:eng_strain, TIME_KEY: time}
    eng_strain_data = convert_dictionary_to_data(eng_strain_dict)
    new_state =copy.deepcopy(displacement_data.state)
    new_state._name = new_state.name+"_eng_strain_rate"
    eng_strain_data.set_state(new_state)
    eng_strain_data.state.update_state_variable(STRAIN_RATE_KEY, eng_strain_data[ENG_STRAIN_KEY][-1]/(eng_strain_data[TIME_KEY][-1]-eng_strain_data[TIME_KEY][0]))
    return eng_strain_data

def convert_eng_strain_time_to_true_strain_time(eng_strain_data):
    eng_strain = copy.deepcopy(eng_strain_data[ENG_STRAIN_KEY])
    time = copy.deepcopy(eng_strain_data[TIME_KEY])
    true_strain = np.log(eng_strain+1)
    true_strain_dict = {TRUE_STRAIN_KEY:true_strain, TIME_KEY: time}
    true_strain_data = convert_dictionary_to_data(true_strain_dict)
    true_strain_data.set_state(eng_strain_data.state)

    return true_strain_data

def get_all_uniaxial_tension_data(end_time, end_displacement, initial_length):
    displace_time_data = get_linear_displacement_vs_time_data(end_time, end_displacement)
    eng_strain_data = convert_disp_time_to_engineering_strain_time(displace_time_data, initial_length)
    true_strain_data = convert_eng_strain_time_to_true_strain_time(eng_strain_data)

    return displace_time_data, eng_strain_data, true_strain_data

class MatcalGeneratedModelForTestsBase(SierraModelForTestsBase):

    results_parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "generated_sm_model_data_for_tests"))

    @property
    @abstractmethod
    def geo_params(self):
        """"""

    @property
    @abstractmethod
    def boundary_condition_data_sets(self):
        """"""

    @abstractmethod
    def _get_gold_data():
        """"""

    @property
    def get_model_results(self):
        results_folder = self._model_class.model_type
        files_dir = os.path.abspath(os.path.join(self.results_parent_folder, results_folder, results_folder+"_0"))
        states = self.boundary_condition_data_sets[0].states.values()
        all_results_dc = DataCollection(f"gold {results_folder} results")
        for state in states:
            sim_state_results_filename = os.path.join(files_dir, state.name, "results.csv")
            data = FileData(sim_state_results_filename) 
            data.set_state(state)
            all_results_dc.add(data)

        return all_results_dc

    def _get_material(self, plasticity=False):            
        if plasticity:
            material_file =  write_j2_plasticity_material_file()
            material = Material("matcal_test", 
            material_file, 
            "j2_plasticity")    
        else:
            material_file =  write_linear_elastic_material_file()
            material = Material("matcal_test", 
            material_file, 
            "linear_elastic")
        return material

    def init_model(self, plasticity=False, coupled=False):
        material = self._get_material(plasticity)
        model = self._model_class(material, **self.geo_params)
        mat_props=self.get_material_properties()
        model.add_constants(elastic_modulus=mat_props["elastic_modulus"], nu=mat_props["nu"],
                             specific_heat=mat_props["specific_heat"], beta_tq=mat_props["beta_tq"],
                               density=mat_props["density"])
        if coupled:
            model.activate_thermal_coupling(thermal_conductivity=mat_props["thermal_conductivity"], 
                                specific_heat=mat_props["specific_heat"],
                                density=mat_props["density"],
                                plastic_work_variable=mat_props["plastic_work_variable"])
            model.add_constants(coupling="coupled")
        else:
            model.add_constants(coupling="uncoupled")    
        return model

    @staticmethod
    def get_material_properties():
        return {"density": 7800, "elastic_modulus": 200e9, "nu": 0.27, 
                "thermal_conductivity":15, "specific_heat":500, "beta_tq":0.9,
                  "plastic_work_variable":"plastic_work_heat_rate"}

    @staticmethod
    def get_material_parameter_collection():
        y = Parameter("yield_stress", 10e6, 500e6, 250e6)
        A = Parameter("A", 10e6, 5000e6, 2500e6)
        b = Parameter("b", 0.1, 5, 2)
        pc=ParameterCollection("params", y, A, b)

        return pc

    @staticmethod
    def get_elastic_material_parameter_collection():
        elastic_modulus = Parameter("elastic_modulus", 10e6, 500e10, 300e9)
        nu = Parameter("nu", 0.01, 0.45, 0.4)
        pc=ParameterCollection("params", elastic_modulus, nu)

        return pc
    
class UniaxialLoadingMaterialPointModelForTests(MatcalGeneratedModelForTestsBase):
    _model_class = UniaxialLoadingMaterialPointModel
    geo_params = {}

    def init_model(self, plasticity=False, coupled=False):
        material = self._get_material(plasticity)
        model = self._model_class(material, **self.geo_params)
        mat_props=self.get_material_properties()
        model.add_constants(elastic_modulus=mat_props["elastic_modulus"], nu=mat_props["nu"],
                             specific_heat=mat_props["specific_heat"], beta_tq=mat_props["beta_tq"],
                               density=mat_props["density"])
        if coupled:
            model.activate_thermal_coupling()
            model.add_constants(coupling="adiabatic")    

        else:
            model.add_constants(coupling="uncoupled")    
        return model

    @property
    def boundary_condition_data_sets(self):

        displace_time_data_slow_1, eng_strain_data_slow_1, true_strain_data_slow_1 = get_all_uniaxial_tension_data(60*60*4, 0.5, 1.0)
        displace_time_data_fast_1, eng_strain_data_fast_1, true_strain_data_fast_1 = get_all_uniaxial_tension_data(4, 0.5, 1.0)

        eng_strain_dc = DataCollection("engineering strain data", eng_strain_data_slow_1, eng_strain_data_fast_1)
        true_strain_dc = DataCollection("true strain data", true_strain_data_slow_1, true_strain_data_fast_1)
        true_strain_dc_no_time = copy.deepcopy(true_strain_dc)
        true_strain_dc_no_time.remove_field(TIME_KEY)
        
        return [eng_strain_dc, true_strain_dc, true_strain_dc_no_time]

    def _get_gold_data(self):
        gold_data_file = os.path.join(self.results_parent_folder, 
                                        "uniaxial_loading_material_point", 
                                        "uniaxial_loading_material_point_0", 
                                        "rate_3.472463365511494e-05_eng_strain_rate", 
                                        "results.csv")
        return FileData(gold_data_file)

class UniaxialTensionModelForTestsBase(MatcalGeneratedModelForTestsBase):
    geo_params_base = {"extensometer_length": 1.*0.0254,
            "total_length":5.*0.0254,
            "fillet_radius": 2.125*0.0254,
            "taper": .001*0.0254,
            "element_size": 0.04*0.0254,
            "necking_region": .375, 
            "mesh_method":1, 
            "grip_contact_length":1*0.0254, 
            "gauge_length": 1.25*0.0254, 
            "element_type":"total_lagrange"}

    @property
    def boundary_condition_data_sets(self):
        slow_data = get_all_uniaxial_tension_data(60*60*4, 0.8*0.0254, 1.0*0.0254)
        displace_time_data_slow_1, eng_strain_data_slow_1, true_strain_data_slow_1 = slow_data
        fast_data = get_all_uniaxial_tension_data(4, 0.8*0.0254, 1.0*0.0254)
        displace_time_data_fast_1, eng_strain_data_fast_1, true_strain_data_fast_1 = fast_data


        eng_strain_disp_rate_state = copy.deepcopy(eng_strain_data_slow_1)
        state = eng_strain_disp_rate_state.state
        strain_rate = state._state_variables.pop(STRAIN_RATE_KEY)
        state.update_state_variable(DISPLACEMENT_RATE_KEY, strain_rate*0.0254)
        eng_strain_disp_rate_state.set_state(state)

        displace_strain_rate_state = copy.deepcopy(displace_time_data_slow_1)
        state = displace_strain_rate_state.state
        disp_rate = state._state_variables.pop(DISPLACEMENT_RATE_KEY)
        state.update_state_variable(STRAIN_RATE_KEY, disp_rate/0.0254)
        displace_strain_rate_state.set_state(state)
       
        mix_dc_without_time = DataCollection("mixed", eng_strain_disp_rate_state, 
        displace_strain_rate_state)
        mix_dc_without_time.remove_field(TIME_KEY)

        eng_strain_dc_w_time = DataCollection("engineering strain data", eng_strain_data_slow_1, 
            eng_strain_data_fast_1, )
        disp_dc_w_time = DataCollection("displacement data", displace_time_data_slow_1, 
            displace_time_data_fast_1)

        disp_dc = copy.deepcopy(disp_dc_w_time)
        disp_dc.remove_field(TIME_KEY)

        eng_strain_dc = copy.deepcopy(eng_strain_dc_w_time)
        eng_strain_dc.remove_field(TIME_KEY)
        
        return [eng_strain_dc, eng_strain_dc_w_time, disp_dc_w_time, disp_dc, mix_dc_without_time]


class RoundUniaxialTensionModelForTests(UniaxialTensionModelForTestsBase):

    _model_class = RoundUniaxialTensionModel

    @property
    def geo_params(self):
        geo_params= {"gauge_radius":0.125*0.0254, "grip_radius":0.25*0.0254}
        geo_params.update(self.geo_params_base)
        return geo_params

    def _get_gold_data(self):
        gold_data_file = os.path.join(self.results_parent_folder, 
                                        "round_uniaxial_tension_model", 
                                        "round_uniaxial_tension_model_0", 
                                        "rate_1.4112091117438712e-06_eng_strain_rate", 
                                        "results.csv")
        return FileData(gold_data_file)


class RectangularUniaxialTensionModelForTests(UniaxialTensionModelForTestsBase):
    _model_class = RectangularUniaxialTensionModel

    @property
    def geo_params(self):
        geo_params= {"gauge_width":0.25*0.0254, "grip_width":0.5*0.0254, "thickness":0.0625*0.0254}
        geo_params.update(self.geo_params_base)
        return geo_params
    
    def _get_gold_data(self):
        gold_data_file = os.path.join(self.results_parent_folder, 
                                        "rectangular_uniaxial_tension_model", 
                                        "rectangular_uniaxial_tension_model_0", 
                                        "rate_1.4112091117438712e-06_eng_strain_rate", 
                                        "results.csv")
        return FileData(gold_data_file)


class RoundNotchedTensionModelForTests(MatcalGeneratedModelForTestsBase):
    _model_class = RoundNotchedTensionModel

    geo_params = {"extensometer_length": 1.0*0.0254,
            "total_length":3.0*0.0254,
            "fillet_radius": 1.0*0.0254,
            "element_size": 0.05*0.0254,
            "necking_region": .375, 
            "mesh_method":1, 
            "grip_contact_length":0.2*0.0254, 
            "gauge_length": 1.5*0.0254,
            "grip_radius": 0.3*0.0254,
            "gauge_radius":0.25*0.0254,
            "notch_gauge_radius":0.125*0.0254,
            "notch_radius":0.078*0.0254, 
            "element_type":"total_lagrange"}

    @property
    def boundary_condition_data_sets(self):
        displace_time_data_slow_1 = get_linear_displacement_vs_time_data(60*60*4, 0.2*0.0254)
        displace_time_data_fast_1 = get_linear_displacement_vs_time_data(4, 0.2*0.0254)

        disp_dc_w_time = DataCollection("displacement data", displace_time_data_slow_1, displace_time_data_fast_1)
        disp_dc = copy.deepcopy(disp_dc_w_time)
        disp_dc.remove_field(TIME_KEY)

        return [disp_dc_w_time, disp_dc]
    
    def _get_gold_data(self):
        gold_data_file = os.path.join(self.results_parent_folder, 
                                        "round_notched_tension_model", 
                                        "round_notched_tension_model_0", 
                                        "rate_3.528022779359678e-07", 
                                        "results.csv")
        return FileData(gold_data_file)

class TopHatShearModelForTests(MatcalGeneratedModelForTestsBase):
    _model_class = TopHatShearModel

    geo_params = {"total_height": 1.25*0.0254,
            "base_height":0.75*0.0254,
            "trapezoid_angle": 10.0,
            "top_width": 0.417*2*0.0254,
            "base_width": 1.625*0.0254, 
            "base_bottom_height": (0.75-0.425)*0.0254,
            "thickness":0.375*0.0254, 
            "external_radius": 0.05*0.0254,
            "internal_radius": 0.05*0.0254,
            "hole_height": 0.3*0.0254,
            "lower_radius_center_width":0.390*2*0.0254,
            "localization_region_scale":0.0,
            "element_size":0.013*0.0254, 
            "numsplits":1, 
            "element_type":"total_lagrange"}

    @property
    def boundary_condition_data_sets(self):
        displace_time_data_slow_1 = get_linear_displacement_vs_time_data(60*60*4, 0.4*0.0254)
        displace_time_data_fast_1 = get_linear_displacement_vs_time_data(4, 0.4*0.0254)
        
        disp_dc_w_time = DataCollection("displacement data", displace_time_data_slow_1, displace_time_data_fast_1)
        disp_dc = copy.deepcopy(disp_dc_w_time)
        disp_dc.remove_field(TIME_KEY)

        return [disp_dc_w_time, disp_dc]
    
    def _get_gold_data(self):
        gold_data_file = os.path.join(self.results_parent_folder, 
                                        "top_hat_shear_model", 
                                        "top_hat_shear_model_0", 
                                        "rate_7.056045558719356e-07", 
                                        "results.csv")
        return FileData(gold_data_file)

class SolidBarTorsionModelForTests(MatcalGeneratedModelForTestsBase):
    _model_class = SolidBarTorsionModel
    geo_params = {"extensometer_length": ((0.277-0.277*0.5)/2+0.277*0.5)*0.0254,
                  "gauge_length": 0.227*0.0254,
                  "gauge_radius": 0.125*0.0254,
                  "grip_radius": 0.25*0.0254,
                  "total_length": 1.25*0.0254,
                  "fillet_radius": 0.350*0.0254,
                  "taper": 0.0001*0.0254,
                  "necking_region":0.5,
                  "element_size": 0.125*0.0254/3,
                  "mesh_method": 1,
                  "grip_contact_length":0.125*0.0254, 
                  "element_type":"total_lagrange"}

    @property
    def boundary_condition_data_sets(self):
        displace_time_data_slow_1 = get_linear_displacement_vs_time_data(60*60*4, 725)
        displace_time_data_fast_1 = get_linear_displacement_vs_time_data(4, 725)

        rotation_time_data_slow_1 = copy.deepcopy(displace_time_data_slow_1)
        rotation_time_data_slow_1.rename_field("displacement", "grip_rotation")
        rotation_time_data_fast_1 = copy.deepcopy(displace_time_data_fast_1)
        rotation_time_data_fast_1.rename_field("displacement", "grip_rotation")

        rotation_dc_w_time = DataCollection("rotation data", rotation_time_data_slow_1, rotation_time_data_fast_1)
        rotation_dc = copy.deepcopy(rotation_dc_w_time)
        rotation_dc.remove_field(TIME_KEY)

        return [rotation_dc_w_time, rotation_dc]
    
    def _get_gold_data(self):
        gold_data_file = os.path.join(self.results_parent_folder, 
                                        "solid_bar_torsion_model", 
                                        "solid_bar_torsion_model_0", 
                                        "rate_0.05035071879991666", 
                                        "results.csv")
        return FileData(gold_data_file)


class VFMBaseModelForTests(MatcalGeneratedModelForTestsBase):

    geo_params = {}
       
    def init_model(self, plasticity=False, coupled=False, x_disp_name="Ux", y_disp_name="Uy"):
        self._material = self._get_material(plasticity)
        bc_data = self.boundary_condition_data_sets[0][SolitaryState()][0]
        self._mesh_grid = auto_generate_two_dimensional_field_grid(3,3, bc_data)
        model = self._model_class(self._material, self._mesh_grid, thickness=0.1)
        if x_disp_name is not None and y_disp_name is not None:
            model.set_displacement_field_names(x_disp_name, y_disp_name)
        mat_props=self.get_material_properties()
        model.add_constants(elastic_modulus=mat_props["elastic_modulus"], nu=mat_props["nu"],
                             specific_heat=mat_props["specific_heat"], beta_tq=mat_props["beta_tq"],
                               density=mat_props["density"])
        if coupled:
            model.activate_thermal_coupling()
            model.add_constants(coupling="adiabatic", temperature=298)    

        else:
            model.add_constants(coupling="uncoupled")    
        return model

    def set_rectangle_filenames_and_paths(self):
        self.test_files_subdir =  "rectangle_vfm_test_files"
        self.template_files_dir = os.path.join(TEST_SUPPORT_FILES_FOLDER, self.test_files_subdir)
        self.input_filename = os.path.join(self.template_files_dir, "rectangle_input.i")
        self.solid_aprepro = os.path.join(self.template_files_dir, "make_fine_thinner_solid.inc")

        self.shell_aprepro = os.path.join(self.template_files_dir, "make_coarse_wider_surface.inc")
        self.journal_filename = os.path.join(self.template_files_dir, "make_rect_mesh_needs_N_and_solid_mesh.jou")
        self.constants["aprepro_file"] = self.shell_aprepro
        self.constants["mat_model"] = "j2_plasticity"
        self.goal_files_dir = os.path.join(GENERATED_TEST_DATA_FOLDER, self.test_files_subdir)
        self.plastic_results_filename = "plastic_results_shell.e"
        self.gold_results_filename = os.path.join(self.goal_files_dir, self.plastic_results_filename)
        
        self.shell_mesh_filename = os.path.join(self.goal_files_dir, "thin_rect_surface.g")
        self.solid_mesh_filename = os.path.join(self.goal_files_dir, "thin_rect.g")
    
    def set_complex_filenames_and_paths(self):
        self.test_files_subdir =  "complex_vfm_test_files"
        self.template_files_dir = os.path.join(TEST_SUPPORT_FILES_FOLDER, self.test_files_subdir)
        self.input_filename = os.path.join(self.template_files_dir, "complex_vfm_gold.i")
        self.solid_aprepro = os.path.join(self.template_files_dir, "solid_mesh.inc")

        self.shell_aprepro = os.path.join(self.template_files_dir, "shell_mesh.inc")
        self.journal_filename = os.path.join(self.template_files_dir, "complex_vfm_mesh.jou")
        self.constants["aprepro_file"] = self.shell_aprepro

        self.goal_files_dir = os.path.join(GENERATED_TEST_DATA_FOLDER, self.test_files_subdir)
        self.results_filename = "results_with_temp.e"
        self.gold_results_filename = os.path.join(self.goal_files_dir, self.results_filename)
        
        self.shell_mesh_filename = os.path.join(self.goal_files_dir, "complex_vfm_mesh_shell.g")
        self.solid_mesh_filename = os.path.join(self.goal_files_dir, "complex_vfm_mesh.g")

    def prepare_gold_results(self):
        self.set_rectangle_filenames_and_paths()
        gold_mesh_str = read_file_lines(self.shell_aprepro)+ \
            read_file_lines(self.journal_filename)
        create_goal_user_model_simulation_results(self.input_filename, self.shell_mesh_filename, 
                                    self.gold_results_filename, self.shell_aprepro,
                                    self.material_file, 
                                    mesh_str=gold_mesh_str,
                                    run_dir=self.goal_files_dir, 
                                    constants=self.constants, cores=1,
                                    **self.goal_param_vals)
        
        field_data = FieldSeriesData(self.gold_results_filename)
        field_data.rename_field("temperature", "temp")
        field_data.rename_field("displacement_x", "U")
        field_data.rename_field("displacement_y", "V")

        return field_data

    def prepare_VFM_model(self, adiabatic=True):
        self.constants = MatcalGeneratedModelForTestsBase.get_material_properties()
        if adiabatic:
            self.constants["coupling"] = "adiabatic"
        else:
            self.constants["coupling"] = "uncoupled"

        self.constants["mat_model"] = "j2_plasticity"
        self.material_file = write_j2_plasticity_material_file()
        self.goal_param_vals = MatcalGeneratedModelForTestsBase.get_material_parameter_collection().get_current_value_dict()
        self.goal_param_vals.update(MatcalGeneratedModelForTestsBase.get_material_properties())

        field_data = self.prepare_gold_results()
        
        mat = Material("matcal_test", self.material_file, "j2_plasticity")

        vfm_model = self._model_class(mat, self.shell_mesh_filename, thickness=0.0625*0.0254)
        vfm_model.add_constants(**self.goal_param_vals, coupling=self.constants["coupling"])
        vfm_model.add_boundary_condition_data(field_data)
        vfm_model.set_displacement_field_names("U", "V")
        vfm_model.set_mapping_parameters(2,2)
        vfm_model.set_number_of_time_steps(100)
        return vfm_model
    
    @property
    def boundary_condition_data_sets(self):
        field_data_dict = {}
        field_data_dict["time"] = np.array([0.0, 1.0, 2.0])
        field_data_dict["load"] = np.array([0, 100, 200])
        field_data_dict["displacement"] = np.array([0, 0.02, 0.04])
        field_data_dict["Ux"] = np.array([[0]*20, 
                                            [0, 1.0/30, 2./30, 1/10]*5, 
                                            [0, 2.0/30, 4./30, 2/10]*5])
        field_data_dict["Uy"] = np.array([[0]*20, 
                                            [0.025]*4+[0.0125]*4+[0]*4+[-0.0125]*4+[-0.025]*4,
                                            [0.025*2]*4+[0.0125*2]*4+[0]*4+[-0.0125*2]*4+[-0.025*2]*4])
        field_data_dict["Uz"] = np.array([[0]*20, [0]*20, [0]*20])
        field_data_dict["temperature"] = np.array([[300]*20, [300]*20, [300]*20])

        field_data = convert_dictionary_to_field_data(field_data_dict)
        spatial_coords = np.array([[0, 1.0/3, 2./3, 1]*5, [1.0]*4+[1.5]*4+[2.0]*4+[2.5]*4+[3.0]*4]).T
        field_data.set_spatial_coords(spatial_coords)
        field_data_dc = DataCollection("test field data", field_data)
        
        return [field_data_dc]
    
    def _get_gold_data(self):
        gold_data_file = os.path.join(self.results_parent_folder, 
                                        "solid_bar_torsion_model", 
                                        "solid_bar_torsion_model_0", 
                                        "rate_0.05035071879991666", 
                                        "results.csv")
        return FileData(gold_data_file)

    @property
    def _field_data(self):
        return self.boundary_condition_data_sets[0][SolitaryState()][0]


class VFMUniaxialTensionHexModelForTests(VFMBaseModelForTests):
    _model_class = VFMUniaxialTensionHexModel

    def _get_gold_data(self):
        gold_data_file = os.path.join(self.results_parent_folder, 
                                        "vfm_hex_model", 
                                        "VFM_0", 
                                        "matcal_default_state", 
                                        "results",
                                        "results.e")
        return FieldSeriesData(gold_data_file)


class VFMUniaxialTensionConnectedHexModelForTests(VFMBaseModelForTests):
    _model_class = VFMUniaxialTensionConnectedHexModel

    def _get_gold_data(self):
        gold_data_file = os.path.join(self.results_parent_folder, 
                                        "vfm_connected_hex_model", 
                                        "VFM_0", 
                                        "matcal_default_state", 
                                        "results",
                                        "results.e")
        return FieldSeriesData(gold_data_file)


class UserDefinedSierraModelForTests(SierraModelForTestsBase):
    _model_class = UserDefinedSierraModel
    _mesh_file = "fake_geo.g"
    _input_file = "user_supplied_input.i"
    
    def init_model(self):
        write_empty_file(self._input_file)
        write_empty_file(self._mesh_file)
        model = self._model_class("adagio", self._input_file, self._mesh_file)
        return model

def generate_sierra_reference_data(self, simulation_dir, execution_dir, matcal_script, *jou_files):
    init_dir = os.getcwd()
    try:
        os.chdir(simulation_dir)
        if os.path.exists(execution_dir):
            import shutil
            shutil.rmtree(execution_dir)
        generate_mesh_command = ""
        for current_jou in jou_files:
            if len(generate_mesh_command) > 0:
                generate_mesh_command += ";"
            generate_mesh_command += f"/projects/cubit/cubit -nojournal -nobanner -nogui -nographics {current_jou}"
        run_model_command = f"module load matcal/sprint;python {matcal_script}"
        convert_data_command = f"module load matcal/sprint; python ../convert_for_dic.py {self._model_dir}/results {self._data_dir} {self._data_name}"
        matcal_print_message("making meshes")
        os.system(generate_mesh_command)
        matcal_print_message('running model')
        os.system(run_model_command)
        matcal_print_message('convert data')
        os.system(convert_data_command)
        os.chdir(init_dir)
    except Exception:
        raise RuntimeError()
