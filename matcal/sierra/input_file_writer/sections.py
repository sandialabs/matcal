"""
Element section blocks for MatCal-generated SIERRA input decks.
"""

from matcal.core.input_file_writer import InputFileLine

from .blocks_base import _BaseSierraInputFileBlock


class _SectionNames:
    total_lagrange = "total_lagrange"
    uniform_gradient = "uniform_gradient"
    composite_tet = "composite_tet"


class TotalLagrangeSection(_BaseSierraInputFileBlock):
    """
    SIERRA total lagrange section block.

    Users can add additional options using the matcal.core.input_file_writer tools.
    """

    type = "total lagrange section"
    required_keys = []
    default_values = {}

    def __init__(self, name=_SectionNames.total_lagrange):
        super().__init__(name)
        self.add_line(InputFileLine("volume average J", "on"))

    def use_composite_tet(self, use_composite_tet=True):
        """
        Toggle composite tet formulation.

        :param use_composite_tet: If True, switch to composite_tet formulation.
        :type use_composite_tet: bool
        """
        if use_composite_tet:
            if "formulation" not in self._lines:
                self.add_line(InputFileLine("formulation", "composite_tet"), replace=True)
            self.set_name(_SectionNames.composite_tet)
        elif not use_composite_tet and "formulation" in self._lines:
            self._lines.pop("formulation")
            self.set_name(_SectionNames.total_lagrange)


class SolidSectionDefault(_BaseSierraInputFileBlock):
    """
    Default solid section block.

    Sets strain incrementation to 'strongly_objective'.
    """

    type = "solid section"
    required_keys = []
    default_values = {"strain incrementation": "strongly_objective"}

    def __init__(self, name=_SectionNames.uniform_gradient):
        super().__init__(name)