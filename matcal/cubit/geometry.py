import os
import re
from enum import Enum

from abc import ABC, abstractmethod, abstractproperty
from collections import OrderedDict
import numbers
import numpy as np

from matcal.core.logger import initialize_matcal_logger

from matcal.cubit.cubit_runner import CubitExternalExecutable

logger = initialize_matcal_logger(__name__)


BASE_WRITE_VOLUME_TO_MESH_COMMANDS = """
#!python
all_elements = cubit.parse_cubit_list('*entity_name*', 'all')
element_volumes = cubit.get_quality_values( '*entity_name*', all_elements, 'element volume')
global_ids = []
for ele in all_elements:
    id = cubit.get_global_element_id('*entity_name*', ele)
    global_ids.append(id)
cubit.set_element_variable( global_ids, 'field_variable', element_volumes )
#!cubit
    """


class Elements(Enum):
    SIDESET = "sideset"
    NODESET = "nodeset"
    BLOCK = "block"


class GeometryParameters(ABC):

    class ValueError(RuntimeError):
        """"""

    @property
    @abstractmethod
    def _required_geometry_parameters(self):
        """"""

    @abstractmethod
    def _check_parameters():
        """"""

    @abstractmethod
    def _set_derived_parameters():
        """"""

    def _verify_parameter_numeric(self, parameter_name, numeric_type=numbers.Real):
        if not isinstance(self._parameters[parameter_name], numeric_type):
            raise self.ValueError(f"The geometric parameter with name \"{parameter_name}\"must " \
                             f"be of type \"{numeric_type}\", but received a value of " \
                             f"type \"{type(self._parameters[parameter_name])}\"")

    def _verify_parameter_less_than_value(self, parameter_name, value, message, 
                                          numeric_type=numbers.Real):
        self._verify_parameter_numeric(parameter_name, numeric_type)
        self._verify_value_less_than_value(self._parameters[parameter_name], value, 
                                           message)

    def _verify_parameter_greater_than_value(self, parameter_name, value, message, 
                                             numeric_type=numbers.Real):
        self._verify_parameter_numeric(parameter_name, numeric_type)
        self._verify_value_less_than_value(value, self._parameters[parameter_name], 
                                           message)

    def _verify_parameter_less_than_equal_to_value(self, parameter_name, value, message, 
                                                   numeric_type=numbers.Real):
        self._verify_parameter_numeric(parameter_name, numeric_type)
        self._verify_value_less_than_equal_to_value(self._parameters[parameter_name], value, 
                                           message)

    def _verify_parameter_greater_than_equal_to_value(self, parameter_name, value, message, 
                                                      numeric_type=numbers.Real):
        self._verify_parameter_numeric(parameter_name, numeric_type)
        self._verify_value_less_than_equal_to_value(value, 
                                                    self._parameters[parameter_name], 
                                                    message)

    def _verify_parameter_less_than_parameter(self, parameter_name1, 
                                              parameter_name2, message):
        self._verify_parameter_numeric(parameter_name1)
        self._verify_parameter_numeric(parameter_name2)
        self._verify_value_less_than_value(self._parameters[parameter_name1], 
                                           self._parameters[parameter_name2], 
                                            message)

    def _verify_parameter_between_values(self, parameter_name, low_value, 
                                         high_value, message, 
                                         numeric_type=numbers.Real):
        self._verify_parameter_numeric(parameter_name, numeric_type)
        param_value = self._parameters[parameter_name]
        if not (param_value > low_value and param_value < high_value):   
            raise self.ValueError(message)

    def _verify_value_less_than_value(self, low_value, high_value, message):
        if not low_value < high_value:
            raise self.ValueError(message)

    def _verify_value_less_than_equal_to_value(self, low_value, high_value, message):
        if not low_value <= high_value:
            raise self.ValueError(message)

    def __init__(self, **passed_parameters):      
        self._parameters = OrderedDict(passed_parameters)
        for req_param in self.required_parameters:
            if req_param not in self._parameters.keys():
                raise ValueError(
                    f"Cannot initialize GeometryParameters, missing parameter {req_param}." 
                    f"Provided parameters are:\n {self._parameters.keys()}")
        self._set_derived_parameters()
        self._check_parameters()

    def __getitem__(self, key):
        return self._parameters[key]

    def __setitem__(self, key, value):
        self._parameters[key] = value
        self._set_derived_parameters()
        self._check_parameters()

    def keys(self):
        return self._parameters.keys()

    def update(self, other):
        self._parameters.update(other)
        self._set_derived_parameters()
        self._check_parameters()

    @property 
    def parameters(self):
        return self._parameters

    @property
    def required_parameters(self):
        return self._required_geometry_parameters


class GeometryBase(ABC):
    # this is a nested class that needs to be created in order to track what information the geometry needs
    class Parameters(object):
        def __init__(self, **kwargs):
            raise NotImplementedError("Geometry Class requires an overwritten Parameter Nested class.")

    class MeshBuildError(RuntimeError):
        def __init__(self, *args):
            super().__init__(*args)

    def __init__(self, mesh_filename, geometry_parameters={}, **kwargs):     
        full_journal_filename = os.path.join(self._journal_file_path, self._journal_filename)
        try:
            self._raw_cmds = self._read_journal(full_journal_filename)
        except FileNotFoundError:
            raise ValueError("Journal file {} does not exist.".format(full_journal_filename))
        self._mesh_filename = mesh_filename
        self._params = geometry_parameters

    def create_mesh(self, template_dir=None):
        if template_dir is None:
            template_dir = "."
        journal_file = os.path.join(template_dir, self._journal_filename)
        journal_cmds = self._update_cubit_commands()
        self._write_journal_file(journal_file, journal_cmds)
        cmds = [journal_file]
        mesh_filename_for_logging = os.path.split(self._mesh_filename)[-1]
        logger.info(f"\t\tBuilding mesh \"{mesh_filename_for_logging}\" using Cubit")
        cubit = CubitExternalExecutable(cmds)
        stdout, stderr, return_code = cubit.run()
        if return_code > 0 and not "libGL error" in stderr:
            abs_mesh_fn = os.path.abspath(self._mesh_filename)
            raise self.MeshBuildError(f"Cubit failed with errors."
                                      f" Check the mesh file: \n {abs_mesh_fn}."
                                      f" Cubit failed with error code {return_code}"
                                      f" the following errors: {stderr}, {stdout}")
        logger.debug(stderr)
        logger.debug(stdout)
        logger.info("\t\tMesh generation complete.")

        return stdout, stderr, return_code

    @property
    def name(self):
        return self._mesh_filename

    @property
    def elements(self):
        reg_exp = r"(sideset|nodeset|block) [{}a-zA-Z0-9~+_\-()* ]+ name \"([a-zA-Z0-9_~]+)\""
        pattern = re.compile(reg_exp)
        elements = []
        for line in self._raw_cmds:
            m = pattern.match(line)
            if m:
                elements.append((m.group(1), m.group(2)))
        return elements

    def _update_cubit_commands(self):
        cmds = self._create_param_commands(self._raw_cmds)
        cmds = self._add_write_volumes_as_field_variable_to_commands(cmds)
        cmds.append('export mesh "' + self._mesh_filename + '" overwrite')
        cmds.append('exit')
        return cmds

    def _add_write_volumes_as_field_variable_to_commands(self, commands):
        if "element_type" in self._params:
            element_type = self._params["element_type"]
        else:
            element_type = "hex8"

        if element_type == "hex8":
            new_commands = BASE_WRITE_VOLUME_TO_MESH_COMMANDS.replace("*entity_name*", "hex")
        else:
            new_commands = BASE_WRITE_VOLUME_TO_MESH_COMMANDS.replace("*entity_name*", "tet")
        new_commands = new_commands.split("\n")
        return commands+new_commands
    
    def _create_param_commands(self, raw_cmds):
        cmds = ["graphics window create"]
        for line in raw_cmds:
            new_line = line
            for key, value in self._params.items():
                try:
                    new_line = new_line.replace("~" + key + "~", str(value))
                except TypeError as te:
                    raise TypeError("Error in geometry scripting. with terms:"
                                    f" {key} and {value}")
            cmds.append(new_line)
        return cmds

    def _read_journal(self, journal_file):
        cmds = []
        with open(journal_file) as fid:
            for l in fid:
                cmds.append(l.strip('\n'))
        return cmds

    @staticmethod
    def _write_log(logfile, stdout, stderr):
        with open(logfile, "wb") as flog:
            flog.write(b"stdout:\n")
            flog.write(stdout)

            flog.write(b"stderr:\n")
            flog.write(stderr)

    def _write_journal_file(self, input_command_file, cmds):
        with open(input_command_file, "w") as fid:
            for c in cmds:
                fid.write("{}\n".format(c))

    
class MaterialPointGeometry(GeometryBase):
    _journal_file_path = os.path.join(os.path.dirname(__file__),
                                       "matcal_generated_model_journal_files", 
                                       "material_point_model_files")
    _journal_filename = "cubit_materialpoint_single_state.jou"

    class Parameters(GeometryParameters):
        _offset = 0
        _node_set_offset = 10
        _side_set_offset = 10
        _block_offset = 10
        _required_geometry_parameters = []

        @property
        def reference_area(self):
            return 1.

        def _check_parameters(self):
            pass

        def _set_derived_parameters(self):
            pass

    def __init__(self, mesh_filename, geometry_parameters=Parameters(), **kwargs):
        if not isinstance(geometry_parameters, self.Parameters):
            raise TypeError("params is not an instance of MaterialPointGeometry.Parameters")

        self._offset_parameters = OrderedDict()
        self._offset_parameters["offset"] = str(self.Parameters._offset)
        self._offset_parameters["node_set_offset"] =  str(self.Parameters._node_set_offset)
        self._offset_parameters["side_set_offset"] =  str(self.Parameters._side_set_offset)
        self._offset_parameters["block_offset"] =  str(self.Parameters._block_offset)
        geometry_parameters.update(self._offset_parameters)

        super().__init__(mesh_filename, 
                         geometry_parameters=geometry_parameters.parameters, 
                         **kwargs)


class TopHatShearGeometry(GeometryBase):
    _journal_file_path = os.path.join(os.path.dirname(__file__), 
                                      "matcal_generated_model_journal_files", 
                                      "top_hat_model_files")
    _journal_filename = "cubit_top_hat.jou"
    _block_names = ["localization_section", "platten_interface_section"]

    class Parameters(GeometryParameters):
        _required_geometry_parameters = ["total_height", "base_height", 
                                         "base_bottom_height",
                                        "trapezoid_angle", "top_width",
                                          "base_width",
                                         "thickness", "external_radius", 
                                         "internal_radius", "hole_height", 
                                         "lower_radius_center_width", 
                                         "localization_region_scale", 
                                         "element_size", 
                                         "numsplits", "element_type"]

        @property
        def reference_area(self):
            return None

        def _set_derived_parameters(self):
            theta, external_rad, internal_rad = self._get_radii_and_angle_parameters_to_calculate_dervied_params()

            base_height, total_height, \
            hole_height, base_bottom_height = self._get_height_parameters_to_calculate_derived_params()
            top_width, lower_radius_center_width = self._get_width_parameters_to_calculate_dervied_params()
            hole_top_radius_center_width = self._calculate_hole_top_center_width(hole_height, internal_rad, 
                                                                                theta, lower_radius_center_width)
            top_width_rad_transition_x, top_width_rad_transition_y = \
            self._calculate_top_hat_upper_platten_interface_geometry(theta, external_rad, total_height, base_height)
            self._calculate_bottom_hole_geometry_values(lower_radius_center_width, internal_rad, theta, base_bottom_height)
            middle_radius_transition_x, middle_radius_transition_y = \
            self._calculate_top_hole_geometry_values(internal_rad, theta, hole_top_radius_center_width, 
                                                     hole_height, base_bottom_height)
            self._calculate_localization_region_cylinder_center(top_width_rad_transition_x, top_width_rad_transition_y,
                                                                middle_radius_transition_x, middle_radius_transition_y)
            self._calculate_large_mesh_size()
            cylinder_radius = \
            self._calculate_localization_region_cylinder_radius(top_width_rad_transition_x, middle_radius_transition_x, 
                                                                top_width_rad_transition_y, middle_radius_transition_y,
                                                                top_width, hole_top_radius_center_width, base_height, 
                                                                hole_height, base_bottom_height)
            self._calculate_scaled_localization_region_parameters(cylinder_radius)

        def _get_height_parameters_to_calculate_derived_params(self):

            base_height = self._parameters["base_height"]
            total_height = self._parameters["total_height"]
            hole_height = self._parameters["hole_height"]
            base_bottom_height = self._parameters["base_bottom_height"]

            return base_height, total_height, hole_height, base_bottom_height

        def _get_radii_and_angle_parameters_to_calculate_dervied_params(self):
            theta = self._parameters["trapezoid_angle"]*np.pi/180
            external_rad = self._parameters["external_radius"]
            internal_rad = self._parameters["internal_radius"]
            return theta, external_rad, internal_rad

        def _get_width_parameters_to_calculate_dervied_params(self):
            top_width = self._parameters["top_width"]
            lower_radius_center_width = self._parameters["lower_radius_center_width"]

            return  top_width, lower_radius_center_width

        def _calculate_hole_top_center_width(self, hole_height, internal_rad, 
                                             theta, lower_radius_center_width):
            hole_top_radius_center_width = lower_radius_center_width - \
                2*(hole_height-2*internal_rad)*np.tan(theta)
            self._parameters["hole_top_radius_center_width"] = hole_top_radius_center_width
            return hole_top_radius_center_width

        def _calculate_top_hat_upper_platten_interface_geometry(self, theta, external_rad, 
                                                                total_height, base_height):
            top_width_rad_transition_x = self._parameters["top_width"]/2-external_rad*np.cos(theta)
            self._parameters["top_width_radius_transition_x"] = top_width_rad_transition_x
            top_width_rad_transition_y =base_height+external_rad-external_rad*np.sin(theta)
            self._parameters["top_width_radius_transition_y"] = top_width_rad_transition_y
            self._parameters["top_outer_vertex_x"] = top_width_rad_transition_x - \
                (total_height-base_height)*np.tan(theta)

            return top_width_rad_transition_x, top_width_rad_transition_y

        def _calculate_bottom_hole_geometry_values(self, lower_radius_center_width, 
                                                   internal_rad, theta, base_bottom_height):
            self._parameters["bottom_radius_transition_x"] = lower_radius_center_width/2+ \
                internal_rad*np.cos(theta)
            self._parameters["bottom_radius_transition_y"] = base_bottom_height+ \
                internal_rad*(1+np.sin(theta))

        def _calculate_top_hole_geometry_values(self, internal_rad, theta, 
                                                hole_top_radius_center_width, 
                                                hole_height, base_bottom_height):
            middle_radius_transition_x = hole_top_radius_center_width/2+internal_rad*np.cos(theta)
            self._parameters["middle_radius_transition_x"] = middle_radius_transition_x
            middle_radius_transition_y = base_bottom_height+hole_height-\
                internal_rad*(1-np.sin(theta))
            self._parameters["middle_radius_transition_y"] = middle_radius_transition_y

            return middle_radius_transition_x, middle_radius_transition_y

        def _calculate_localization_region_cylinder_center(self, top_width_rad_transition_x, 
                                                           top_width_rad_transition_y, 
                                                           middle_radius_transition_x, 
                                                           middle_radius_transition_y):
            cylinder_x_center = (top_width_rad_transition_x+middle_radius_transition_x)/2
            self._parameters["cylinder_x_center"] = cylinder_x_center
            cylinder_y_center =  (top_width_rad_transition_y+middle_radius_transition_y)/2
            self._parameters["cylinder_y_center"] = cylinder_y_center

        def _calculate_large_mesh_size(self):
            large_mesh_size = self._parameters["element_size"]*3**(self._parameters["numsplits"])
            self._parameters["large_mesh_size"] = large_mesh_size

        def _calculate_localization_region_cylinder_radius(self, top_width_rad_transition_x, 
                                                           middle_radius_transition_x, 
                                                           top_width_rad_transition_y,
                                                           middle_radius_transition_y, 
                                                           top_width, hole_top_radius_center_width, 
                                                           base_height, 
                                                           hole_height, base_bottom_height):
            cylinder_radius = 0.5*np.sqrt((top_width_rad_transition_x-middle_radius_transition_x)**2 + 
                                          (top_width_rad_transition_y-middle_radius_transition_y)**2)
            horizontal_cylinder_radius = 0.5*np.sqrt((top_width/2-hole_top_radius_center_width/2)**2 + 
                                                     (base_height-hole_height+base_bottom_height)**2)
            self._parameters["cylinder_radius"] = cylinder_radius
            if (abs(cylinder_radius-horizontal_cylinder_radius) < 
                self._parameters["large_mesh_size"] and 
                self._parameters["localization_region_scale"] < 1):
                self._parameters["localization_region_scale"] = 1
            return cylinder_radius

        def _calculate_scaled_localization_region_parameters(self, cylinder_radius):
            scaled_cylinder_radius = (cylinder_radius + 
                self._parameters["large_mesh_size"]*self._parameters["localization_region_scale"])
            self._parameters["scaled_cylinder_radius"] = scaled_cylinder_radius

        def _check_parameters(self):
            self._check_base_height_parameters_are_valid()
            self._check_base_width_parametes_are_valid()           
            self._check_top_trapezoid_section_parameters_are_valid()
            self._check_localization_region_parameters_are_valid()
            self._check_hole_parameters_are_valid()
            self._check_mesh_size_parameters_are_valid()

        def _check_base_height_parameters_are_valid(self):
            base_height_upper_bound_err_message = "The base height cannot be greater than or " \
                                 "equal to the total height minus 2 times the external radius."
            base_height_upper_bound = self._parameters["total_height"] - 2*self._parameters["external_radius"]
            self._verify_parameter_less_than_value("base_height", base_height_upper_bound, 
                                                                base_height_upper_bound_err_message)

            base_height_lower_bound_err_message = "The base height must be greater than " \
                                 "the hole height plus the base bottom height."
            base_height_lower_bound = self._parameters["hole_height"] + self._parameters["base_bottom_height"]
            self._verify_parameter_greater_than_value("base_height", base_height_lower_bound, 
                                                      base_height_lower_bound_err_message)         

        def _check_base_width_parametes_are_valid(self):
            base_width_err_message = "The base width must be greater than " \
                                    "the top width plus 2*(element size)*3^(numsplits)"
            base_width_lower_bound = self._parameters["top_width"] + \
                                     2*self._parameters["element_size"]*3**self._parameters["numsplits"]
            self._verify_parameter_greater_than_value("base_width", base_width_lower_bound, 
                                                                    base_width_err_message)

        def _check_top_trapezoid_section_parameters_are_valid(self):
            trapezoid_angle_err_message = "The 'trapezoid_angle' parameter must be between 0 and 50. "
            self._verify_parameter_between_values("trapezoid_angle", 0, 50, trapezoid_angle_err_message)

            top_vertex_err_message = "The provided geometry is invalid and the top surface would not be generated." \
                                    "This is related to the 'base_width', 'top_width', 'trapezoid_angle and 'external_radius' geometry parameters " \
                                    "causing the top section of the element to have not top surface."
            self._verify_parameter_greater_than_value("top_outer_vertex_x", 0, top_vertex_err_message)


        def _check_localization_region_parameters_are_valid(self):
            localization_region_lower_bound_err_message = "The localization region is too small. It must be at least " \
                                                          "'base_height' - ('hole_height' + 'base_bottom_height'). Increase the " \
                                                          "'localization_region_scale' geometry parameter."
            scaled_cylinder_radius_lower_bound = (self._parameters["base_height"] - (self._parameters["hole_height"] + 
                                                  self._parameters["base_bottom_height"]))/2
            self._verify_parameter_greater_than_value("scaled_cylinder_radius", scaled_cylinder_radius_lower_bound, localization_region_lower_bound_err_message)        

        def _check_hole_parameters_are_valid(self):
            hole_width_upper_bound_err_message = "The lower radius center width must be less than base width " \
                                                 " minus 2*(element size)*3^(numsplits)."
            hole_width_upper_bound = self._parameters["base_width"] - 2*self._parameters["element_size"]*3**self._parameters["numsplits"]
            self._verify_parameter_less_than_value("lower_radius_center_width", hole_width_upper_bound, hole_width_upper_bound_err_message)

            top_hole_width_err_message = "The provided geometry is invalid and the top surface of the hole would not be generated correctly." \
                                    "This is usually related to the 'base_width', `lower_radius_center_width`, `trapezoid_angle` and `internal radius`" \
                                    " geometry parameters causing the top surface and internal radius not being generated."
            self._verify_parameter_greater_than_value("hole_top_radius_center_width", 0, top_hole_width_err_message)

            hole_height_too_small_err_message = "The 'hole_height' parameter is too small for the provided 'internal_radius' and 'trapezoid_angle'."
            self._verify_parameter_less_than_parameter("bottom_radius_transition_y", "middle_radius_transition_y", hole_height_too_small_err_message)

        def _check_mesh_size_parameters_are_valid(self):
            numsplit_err_message = "The 'numsplits' value must be an integer between 0 and 2."
            self._verify_parameter_less_than_equal_to_value("numsplits", 2, numsplit_err_message, numbers.Integral)
            self._verify_parameter_greater_than_equal_to_value("numsplits", 0, numsplit_err_message, numbers.Integral)
            
            element_size_err_message = "The geometry parameters for 'numsplits' and 'element_size' must be provided such that " \
                                    "(element_size)*3^(numsplits+1) <= (base_height-(hole_height+base_bottom_height)))"
            
            max_ele_size_thru_localization_region = self._parameters["base_height"] - (self._parameters["hole_height"]+self._parameters["base_bottom_height"])
            ele_size_thru_localization_region = self._parameters["element_size"]*3**(self._parameters["numsplits"]+1)
            self._verify_value_less_than_equal_to_value(ele_size_thru_localization_region, max_ele_size_thru_localization_region, element_size_err_message)

    def __init__(self, mesh_filename, geometry_parameters, **kwargs):
        if not isinstance(geometry_parameters, self.Parameters):
            raise TypeError("geometry_parameters is not an instance of TopHatGeometry.Parameters")
        super().__init__(mesh_filename, geometry_parameters=geometry_parameters.parameters, **kwargs)

class UniaxialLoadingGeometry(GeometryBase):
    _block_names = ["grip_section", "gauge_section", "necking_section"]

    class Parameters(GeometryParameters):
        _required_geometry_parameters = ["extensometer_length", "gauge_length", 
                                        "total_length", "fillet_radius", "element_size",
                                         "mesh_method", "necking_region", "grip_contact_length", 
                                         "element_type"]

        @abstractproperty
        def _model_type_name(self):
            """"""

        def _check_parameters(self):
            extensometer_err_message = "The extensometer length cannot be greater " \
                                       f"than the gauge length for the {self._model_type_name}."
            self._verify_parameter_less_than_parameter("extensometer_length", "gauge_length", 
                                                          extensometer_err_message)
            gauge_err_message = "The gauge length length cannot be greater than the " \
                                f"total length length for the {self._model_type_name}."
            self._verify_parameter_less_than_parameter("gauge_length", "total_length", 
                                                        gauge_err_message)
            
            self._check_necking_region_parameters()

            blend_height_err_message = "The resulting blend height is too large and results in no grip "\
                                        "length. Check the specimen dimensions"
            max_blend_height = (self._parameters["total_length"]-self._parameters["gauge_length"])/2
            self._verify_parameter_less_than_value("blend_height", max_blend_height, blend_height_err_message)

            grip_contact_length_err_message = "The grip contact length is too large for the available grip length. "\
                                              " Check the specimen dimensions."
            max_grip_contact_length = (self._parameters["total_length"]-self._parameters["gauge_length"])/2-self._parameters["blend_height"]
            self._verify_parameter_less_than_value("grip_contact_length", max_grip_contact_length, grip_contact_length_err_message)
            if "full_field_window_height" not in self._parameters and "full_field_window_width" not in self._parameters:
                self._parameters["full_field_window_height"] = 0
                self._parameters["full_field_window_width"] = 0
                
        def _check_necking_region_parameters(self):
            necking_region_upper_limit = (self._parameters["extensometer_length"]-2*self._parameters["element_size"])/self._parameters["extensometer_length"]
            necking_region_lower_limit = 2*self._parameters["element_size"]/self._parameters["extensometer_length"]

            if necking_region_upper_limit > necking_region_lower_limit:
                necking_region_err_message = f"The necking region must be between {necking_region_lower_limit} "\
                                             f"and {necking_region_upper_limit} for "\
                                              f"the {self._model_type_name}."
                self._verify_parameter_between_values("necking_region", necking_region_lower_limit, 
                                                   necking_region_upper_limit, necking_region_err_message)
            else:
                err_message = "The element_size it too large for the specified extensometer_length and necking_region. "\
                    "Refine mesh or update the specimen dimensions."
                raise self.ValueError(err_message)


        def _set_derived_parameters(self):
            self._parameters["blend_height"] = self._calculate_blend_height()

        @property
        def gauge_length(self):
            return np.double(self._parameters["gauge_length"])

        @property
        def extensometer_length(self):           
            return np.double(self._parameters["extensometer_length"])

        def _calculate_height_of_radius_transition(self, inner_length, outter_length, tranisition_radius):
            radial_length = outter_length-inner_length
            if radial_length < tranisition_radius and radial_length > 0:
                blend_arc_angle = np.arccos((tranisition_radius-radial_length)/tranisition_radius)
                height = tranisition_radius*np.sin(blend_arc_angle)
            else:
                height = tranisition_radius
            return height


class UniaxialStressTensionGeometry(UniaxialLoadingGeometry):
    _journal_file_path = os.path.join(os.path.dirname(__file__), 
        "matcal_generated_model_journal_files", "uniaxial_tension_model_files")
    _journal_filename = "cubit_tension.jou"

    class Parameters(UniaxialLoadingGeometry.Parameters):
        _required_geometry_parameters = (["taper"] + 
            UniaxialLoadingGeometry.Parameters._required_geometry_parameters)
        def _calculate_blend_height(self):
            fillet_radius = self._parameters["fillet_radius"]
            if "grip_radius" in self._parameters and "gauge_radius" in self._parameters:
                outer_length = self._parameters["grip_radius"]
                inner_length = self._parameters["gauge_radius"]+self._parameters["taper"]/2
            else:
                outer_length = self._parameters["grip_width"]/2
                inner_length = self._parameters["gauge_width"]/2+self._parameters["taper"]/2
            blend_height = self._calculate_height_of_radius_transition(inner_length, 
                outer_length, fillet_radius)
            return blend_height


class RoundUniaxialTensionGeometry(UniaxialStressTensionGeometry):
    
    class Parameters(UniaxialStressTensionGeometry.Parameters):
        _required_geometry_parameters = ["gauge_radius", "grip_radius"] + \
                                         UniaxialStressTensionGeometry.Parameters._required_geometry_parameters

        _model_type_name = "round uniaxial tension model"
        def _check_parameters(self):
            super()._check_parameters()
            guage_radius_err_message = "The gauge radius cannot be greater than the grip "\
                                       f"radius for the {self._model_type_name}."
            self._verify_parameter_less_than_parameter("gauge_radius", "grip_radius", guage_radius_err_message)

            element_max_size = self._parameters["gauge_radius"]/2
            element_size_err_message = "The element size cannot be greater than 1/2 "\
                                       f"the gauge_radius for the {self._model_type_name}."
            self._verify_parameter_less_than_equal_to_value("element_size", element_max_size, element_size_err_message)

            taper_max_limit = 2*(self._parameters["grip_radius"]-self._parameters["gauge_radius"])
            taper_err_message = "The taper cannot be greater than 2*(grip_radius - gauge_radius) " \
                                f"for the {self._model_type_name}."
            self._verify_parameter_less_than_value("taper", taper_max_limit, taper_err_message)

            mesh_method = self._parameters["mesh_method"]
            if mesh_method == 5:
                ele_size_max_limit = self._parameters["gauge_radius"]/24
                mesh_method_message = "If the element size is greater than 1/24 of "\
                                      "the gauge radius,  you cannot use \"mesh_method\"=5."
                self._verify_parameter_less_than_equal_to_value("element_size", ele_size_max_limit, mesh_method_message)

            elif mesh_method == 4 :
                ele_size_max_limit = self._parameters["gauge_radius"]/9
                mesh_method_message = "If the element size is greater than 1/9 of "\
                                      "the gauge radius,  you cannot use \"mesh_method\"=4."
                self._verify_parameter_less_than_equal_to_value("element_size", ele_size_max_limit, mesh_method_message)
            elif mesh_method > 1:
                ele_size_max_limit = self._parameters["gauge_radius"]/4
                mesh_method_message = "If the element size is greater than 1/4 of "\
                                      "the gauge radius,  you must use \"mesh_method\"=1."
                self._verify_parameter_less_than_equal_to_value("element_size", ele_size_max_limit, mesh_method_message)
                
        def _set_derived_parameters(self):
            super()._set_derived_parameters()
            self._parameters["gauge_width"] = self._parameters["gauge_radius"]*2
            self._parameters["grip_width"] = self._parameters["grip_radius"]*2
            self._parameters["model_type"] = "round"

    def __init__(self, mesh_filename, geometry_parameters, **kwargs):
        if not isinstance(geometry_parameters, self.Parameters):
            raise TypeError("geometry_parameters is not an instance of RoundUniaxialLoadingGeometry.Parameters")
        super().__init__(mesh_filename, geometry_parameters=geometry_parameters.parameters, **kwargs)


class RectangularUniaxialTensionGeometry(UniaxialStressTensionGeometry):
    
    class Parameters(UniaxialStressTensionGeometry.Parameters):
        _required_geometry_parameters = ["gauge_width", "grip_width", "thickness"] + \
                                         UniaxialStressTensionGeometry.Parameters._required_geometry_parameters
        _model_type_name = "rectangular uniaxial tension model"

        def _check_parameters(self):
            super()._check_parameters()
            gauge_width_err_message = "The gauge width cannot be greater than the gauge "\
                                      f"width for the {self._model_type_name}."
            self._verify_parameter_less_than_parameter("gauge_width", "grip_width", gauge_width_err_message)

            thickness_err_message = "The thickness cannot be greater than the gauge width "\
                                    f"for the {self._model_type_name}."
            self._verify_parameter_less_than_parameter("thickness", "gauge_width", thickness_err_message)

            element_max_size = self._parameters["gauge_width"]/4
            element_size_err_message = "The element size cannot be greater than 1/4 "\
                                       f"the gauge width for the {self._model_type_name}."
            self._verify_parameter_less_than_equal_to_value("element_size", element_max_size, element_size_err_message)

            taper_max_limit = self._parameters["grip_width"]-self._parameters["gauge_width"]
            taper_err_message = "The taper cannot be greater than (grip_width - gauge_width) " \
                                "for the uniaxial tension models."
            self._verify_parameter_less_than_value("taper", taper_max_limit, taper_err_message)

            mesh_method = self._parameters["mesh_method"]
            if mesh_method == 5:
                ele_size_limit = self._parameters["thickness"]/9
                mesh_method_message = "If the element size is greater than 1/9 of the thickness, "\
                                      "you cannot use \"mesh_method\"=5."
                self._verify_parameter_less_than_equal_to_value("element_size", ele_size_limit, mesh_method_message)
            elif mesh_method > 1 :
                ele_size_limit = self._parameters["thickness"]/3
                mesh_method_message = "If the element size is greater than 1/3 of the thickness, "\
                                      "you must use \"mesh_method\"=1."
                self._verify_parameter_less_than_equal_to_value("element_size", ele_size_limit, mesh_method_message)

        def _set_derived_parameters(self):
            super()._set_derived_parameters()
            self._parameters["model_type"] = "rectangular"

        @property
        def reference_area(self):
            return self._parameters["gauge_width"]*self._parameters["thickness"]

    def __init__(self, mesh_filename, geometry_parameters, **kwargs):
        if not isinstance(geometry_parameters, self.Parameters):
            raise TypeError("geometry_parameters is not an instance of RectangularUniaxialLoadingGeometry.Parameters")
        super().__init__(mesh_filename, geometry_parameters=geometry_parameters.parameters, **kwargs)


class SolidBarTorsionGeometry(RoundUniaxialTensionGeometry):
    class Parameters(RoundUniaxialTensionGeometry.Parameters):
        _model_type_name = "solid bar round torsion model"

    def _read_journal(self, journal_file):
        cmds = super()._read_journal(journal_file)
        cmds.append("set copy_block_on_geometry_copy use_original")
        cmds.append("set copy_nodeset_on_geometry_copy use_original")
        cmds.append("set copy_sideset_on_geometry_copy use_original")
        cmds.append("Volume all copy reflect x")
        cmds.append("Volume all copy reflect z")
        cmds.append("volume all scale {1/total_length}")
        cmds.append("merge vol all")
        cmds.append("volume all scale {total_length}")
        cmds.append("#{if(element_type == \"composite_tet\")}")
        cmds.append("del mesh vol all prop")
        cmds.append("mesh vol all")
        cmds.append("#{endif}")
        cmds.append("group \"side_grip\" add surface in volume in block 1000 with area >" 
            " {side_grip_surface_area*0.99999} and area < {1.000001*side_grip_surface_area})")
        cmds.append("nodeset 5000 node in {face_element} in surface in side_grip with y_coord >" 
                    "{total_length/2-grip_contact_length}")
        
        return cmds

class RoundNotchedTensionGeometry(UniaxialLoadingGeometry):
    _journal_file_path = os.path.join(os.path.dirname(__file__), 
                                      "matcal_generated_model_journal_files", 
                                      "round_notched_tension_model_files")
    _journal_filename = "cubit_round_notched_tension.jou"

    class Parameters(UniaxialLoadingGeometry.Parameters):
        _required_geometry_parameters = (["gauge_radius", "grip_radius", "notch_gauge_radius",
        "notch_radius"] + UniaxialLoadingGeometry.Parameters._required_geometry_parameters)
        _model_type_name = "round notched tension model"

        def _check_parameters(self):
            super()._check_parameters()

            element_max_size = self._parameters["notch_radius"]
            element_size_err_message = ("The element size cannot be greater than "+
                                    "the notch radius for the round notched tension models.")
            self._verify_parameter_less_than_equal_to_value("element_size", element_max_size, 
                                                            element_size_err_message)

            element_max_size = self._parameters["notch_gauge_radius"]/2
            element_size_err_message = ("The element size cannot be greater than "+
                        "half of the notch gauge radius for the round notched tension models.")
            self._verify_parameter_less_than_equal_to_value("element_size", element_max_size,
                                                             element_size_err_message)

            guage_radius_err_message = ("The gauge radius cannot be greater than the grip "+
                                       "radius for the round notched tension model.")
            self._verify_parameter_less_than_parameter("gauge_radius", "grip_radius",
                                                        guage_radius_err_message)

            notch_gauge_radius_err_message = ("The notch gauge radius cannot be greater "+
                "than the gauge radius for the round notched tension model." )           
            self._verify_parameter_less_than_parameter("notch_gauge_radius", "gauge_radius",
                                                        notch_gauge_radius_err_message)

            notch_height_err_message = ("The notch height is too large for the specimen with "+
                "the provided dimensions. The notch height must be less than the "+
                "\"extensometer_length\". Check the specimen dimensions.")
            self._verify_parameter_less_than_parameter("notch_height", "extensometer_length", 
                                                       notch_height_err_message)
   
            mesh_method = self._parameters["mesh_method"]
            if mesh_method == 5:
                ele_size_max_limit = self._parameters["notch_gauge_radius"]/24
                mesh_method_message = "If the element size is greater than 1/24 of "\
                                      "the notch gauge radius,  you cannot use \"mesh_method\"=5."
                self._verify_parameter_less_than_equal_to_value("element_size", ele_size_max_limit, 
                                                                mesh_method_message)
            elif mesh_method == 4 :
                ele_size_max_limit = self._parameters["notch_gauge_radius"]/9
                mesh_method_message = "If the element size is greater than 1/9 of "\
                                      "the notch gauge radius,  you cannot use \"mesh_method\"=4."
                self._verify_parameter_less_than_equal_to_value("element_size", ele_size_max_limit,
                                                                 mesh_method_message)
            elif mesh_method > 1:
                ele_size_max_limit = self._parameters["notch_gauge_radius"]/4
                mesh_method_message = "If the element size is greater than 1/4 of "\
                                      "the notch gauge radius,  you must use \"mesh_method\"=1."
                self._verify_parameter_less_than_equal_to_value("element_size", ele_size_max_limit, 
                                                                mesh_method_message)

        def _calculate_notch_height(self):
            notch_gauge_radius = self._parameters["notch_gauge_radius"]
            gauge_radius = self._parameters["gauge_radius"]
            notch_radius = self._parameters["notch_radius"]
            notch_height = self._calculate_height_of_radius_transition(notch_gauge_radius,
                                                                        gauge_radius, 
                                                                        notch_radius)*2
            return notch_height

        def _calculate_blend_height(self):
            fillet_radius = self._parameters["fillet_radius"]
            outer_length = self._parameters["grip_radius"]
            inner_length = self._parameters["gauge_radius"]
            blend_height = self._calculate_height_of_radius_transition(inner_length, 
                                                                       outer_length, 
                                                                       fillet_radius)
            return blend_height

        def _set_derived_parameters(self):
            super()._set_derived_parameters()
            self._parameters["notch_height"] = self._calculate_notch_height()

        @property
        def reference_area(self):
            r = self._parameters["notch_gauge_radius"]
            return np.pi * np.double(r) ** 2

    def __init__(self, mesh_filename, geometry_parameters, **kwargs):
        if not isinstance(geometry_parameters, self.Parameters):
            raise TypeError("geometry_parameters is not an instance "+
            "of RoundNotchedTensionLoadingGeometry.Parameters")
        super().__init__(mesh_filename, geometry_parameters=geometry_parameters.parameters, 
                         **kwargs)

