"""
Contact-related SIERRA/SM input-deck blocks for MatCal-generated decks.

Includes:
- friction model block
- interaction defaults
- remove initial overlap
- contact definition container
"""

from matcal.core.input_file_writer import InputFileLine

from .blocks_base import _BaseSierraInputFileBlock


class SolidMechanicsInteractionDefaults(_BaseSierraInputFileBlock):
    type = "interaction defaults"
    required_keys = ["friction model", "general contact", "self contact"]
    default_values = {"general contact": "on"}

    def __init__(self, friction_model_name, self_contact=True):
        super().__init__()
        self.set_print_name(False)
        self.set_self_contact(self_contact)

        friction_model_line = InputFileLine(self.required_keys[0], friction_model_name)
        self.add_line(friction_model_line)

    def set_self_contact(self, self_contact=True):
        contact_val = "on" if self_contact else "off"
        self_contact_line = InputFileLine(self.required_keys[2], contact_val)
        self.add_line(self_contact_line, replace=True)


class SolidMechanicsConstantFrictionModel(_BaseSierraInputFileBlock):
    type = "constant friction model"
    required_keys = ["friction coefficient"]
    default_values = {}

    def __init__(self, friction_model_name, friction_coefficient=0.3):
        super().__init__(name=friction_model_name)
        self.set_friction_coefficient(friction_coefficient)

    def set_friction_coefficient(self, friction_coefficient):
        friction_coeff_line = InputFileLine(self.required_keys[0], friction_coefficient)
        self.add_line(friction_coeff_line, replace=True)

    def get_friction_coefficient(self):
        return self.get_line_value(self.required_keys[0])


class SolidMechanicsRemoveInitialOverlap(_BaseSierraInputFileBlock):
    type = "remove initial overlap"
    required_keys = []
    default_values = {}

    def __init__(self):
        super().__init__()
        self.set_print_name(False)


class SolidMechanicsContactDefinitions(_BaseSierraInputFileBlock):
    type = "contact definition"
    required_keys = []
    default_values = {"skin all blocks": "on"}

    def __init__(self, friction_model_block, name="contact_defs"):
        super().__init__(name=name)

        self.add_subblock(friction_model_block)
        interactions_default_block = SolidMechanicsInteractionDefaults(friction_model_block.name)
        self.add_subblock(interactions_default_block)
        self.add_subblock(SolidMechanicsRemoveInitialOverlap())

    def get_interaction_defaults_block(self):
        return self.subblocks[SolidMechanicsInteractionDefaults.type]

    def get_constant_friction_model_block(self):
        return self.get_subblock_by_type(SolidMechanicsConstantFrictionModel.type)

    def get_remove_initial_overlap_block(self):
        return self.subblocks[SolidMechanicsRemoveInitialOverlap.type]