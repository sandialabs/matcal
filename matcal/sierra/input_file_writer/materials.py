"""
Thermal material blocks for MatCal-generated SIERRA/Aria input decks.
"""

from matcal.core.input_file_writer import InputFileLine

from .blocks_base import THERMAL_MATERIAL_NAME, _BaseSierraInputFileBlock


class ThermalMaterial(_BaseSierraInputFileBlock):
    """
    Basic Aria material model with conduction.
    """

    type = "aria material"
    required_keys = ["density", "thermal conductivity", "specific heat"]
    default_values = {}

    def __init__(
        self,
        density,
        thermal_conductivity,
        specific_heat,
        name=THERMAL_MATERIAL_NAME,
    ):
        super().__init__(name)

        density_line = InputFileLine("density", "constant", "rho", "=", density)
        self.add_line(density_line)

        conductivity_line = InputFileLine(
            "thermal conductivity",
            "constant",
            "K",
            "=",
            thermal_conductivity,
        )
        self.add_line(conductivity_line)

        spec_heat_line = InputFileLine("specific heat", "constant", "cp", "=", specific_heat)
        self.add_line(spec_heat_line)

        self.add_line(InputFileLine("heat conduction", "basic"))