"""
Boundary condition input blocks for MatCal-generated SIERRA/SM decks.

Includes:
- fixed displacement
- prescribed displacement (optionally read from mesh)
- prescribed temperature (function, transfer, or read from mesh)
- initial temperature
"""

from matcal.core.input_file_writer import InputFileLine

from .blocks_base import _BaseSierraInputFileBlock


class _SolidMechanicsWithMeshEntity(_BaseSierraInputFileBlock):
    required_keys = []
    default_values = {}

    def __init__(self, mesh_entity_name=None, mesh_entity="node set", **kwargs):
        super().__init__(**kwargs)
        if mesh_entity_name is not None:
            self._add_mesh_entity_line(mesh_entity_name=mesh_entity_name, mesh_entity=mesh_entity)

    def _add_mesh_entity_line(self, mesh_entity_name, mesh_entity):
        if mesh_entity_name.lower().strip() == "include all blocks":
            mesh_entity_line = InputFileLine(mesh_entity_name)
        else:
            mesh_entity_line = InputFileLine(mesh_entity, mesh_entity_name)

        self.add_line(mesh_entity_line)
        self.set_print_name(False)
        self.set_print_title()

    def read_from_mesh(self, read_variable):
        read_variable_line = InputFileLine("read variable", read_variable)
        self.add_line(read_variable_line, replace=True)


class _SolidMechanicsBaseConditionWithFunction(_SolidMechanicsWithMeshEntity):
    def __init__(self, function_name=None, scale_factor=1.0, **kwargs):
        super().__init__(**kwargs)

        if function_name is not None:
            function_line = InputFileLine("function", function_name)
            self.add_line(function_line)

        if scale_factor != 1:
            scale_factor_line = InputFileLine("scale factor", scale_factor)
            self.add_line(scale_factor_line)


class _SolidMechanicsBaseConditionWithDirection(_SolidMechanicsWithMeshEntity):
    def __init__(self, direction_name=None, direction_key="component", **kwargs):
        super().__init__(**kwargs)

        if direction_name is not None:
            direction_line = InputFileLine(direction_key, direction_name)
            self.add_line(direction_line)


class SolidMechanicsFixedDisplacement(_SolidMechanicsBaseConditionWithDirection):
    type = "fixed displacement"

    def __init__(
        self, mesh_entity_name, direction_name, mesh_entity="node set", direction_key="component"
    ):
        name = mesh_entity_name + " " + direction_name
        super().__init__(
            mesh_entity_name=mesh_entity_name,
            name=name,
            mesh_entity=mesh_entity,
            direction_name=direction_name,
            direction_key=direction_key,
        )


class SolidMechanicsPrescribedDisplacement(
    _SolidMechanicsBaseConditionWithFunction,
    _SolidMechanicsBaseConditionWithDirection,
):
    type = "prescribed displacement"

    def __init__(
        self,
        function_name,
        mesh_entity_name,
        direction_name,
        mesh_entity="node set",
        direction_key="component",
        scale_factor=1.0,
    ):
        name = mesh_entity_name + " " + direction_name
        if function_name is not None:
            name += " " + function_name

        super().__init__(
            function_name=function_name,
            mesh_entity_name=mesh_entity_name,
            direction_name=direction_name,
            mesh_entity=mesh_entity,
            direction_key=direction_key,
            scale_factor=scale_factor,
            name=name,
        )


class SolidMechanicsPrescribedTemperature(_SolidMechanicsBaseConditionWithFunction):
    type = "prescribed temperature"

    def __init__(
        self,
        mesh_entity_name,
        scale_factor=1.0,
        mesh_entity="node set",
        function_name=None,
        transfer=None,
    ):
        if function_name is not None:
            name = mesh_entity_name + " " + function_name
        elif transfer is not None:
            name = mesh_entity_name + " " + "temperature transfer"
        else:
            name = mesh_entity_name + " " + "read temperature from mesh"

        super().__init__(
            function_name=function_name,
            mesh_entity_name=mesh_entity_name,
            mesh_entity=mesh_entity,
            name=name,
            scale_factor=scale_factor,
        )

        if transfer:
            transfer_line = InputFileLine("receive", "from", "transfer", name="transfer_temp")
            transfer_line.suppress_symbol()
            self.add_line(transfer_line)


class SolidMechanicsInitialTemperature(_SolidMechanicsWithMeshEntity):
    type = "initial temperature"

    def __init__(self, mesh_entity_name, magnitude, mesh_entity="block"):
        super().__init__(mesh_entity_name=mesh_entity_name, mesh_entity=mesh_entity)
        temp_mag_line = InputFileLine("magnitude", magnitude)
        self.add_line(temp_mag_line)