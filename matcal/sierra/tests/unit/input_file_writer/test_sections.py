from matcal.core.tests.MatcalUnitTest import MatcalUnitTest

from matcal.sierra.input_file_writer.sections import (
    SolidSectionDefault,
    TotalLagrangeSection,
)


class TestSections(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)

    def test_total_lagrange_(self):
        input_ = TotalLagrangeSection()
        test_str = input_.get_string()
        self.assertTrue("Begin total lagrange section total_lagrange" in test_str)
        self.assertTrue("volume average J = on" in test_str)
        self.assertTrue("composite_tet" not in test_str)

        input_.use_composite_tet()
        test_str = input_.get_string()
        self.assertTrue("composite_tet" in test_str)
        self.assertTrue("total_lagrange" not in test_str)

        input_.use_composite_tet(False)
        test_str = input_.get_string()
        self.assertTrue("composite_tet" not in test_str)
        self.assertTrue("total_lagrange" in test_str)

    def test_default_section_(self):
        input_ = SolidSectionDefault()
        test_str = input_.get_string()
        self.assertTrue("Begin solid section uniform_gradient" in test_str)
        self.assertTrue("strain incrementation = strongly_objective" in test_str)