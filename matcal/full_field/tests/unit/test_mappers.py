from abc import ABC, abstractmethod
import numpy as np


from matcal.core.data import Data
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest

from matcal.full_field.data import convert_dictionary_to_field_data
from matcal.full_field.data_importer import (
    FieldSeriesData,
    ImportedTwoDimensionalMesh,
    _import_full_field_data_from_json,
)
from matcal.full_field.data_exporter import export_full_field_data_to_json
from matcal.full_field.field_mappers import (
    BadPolynomialOrderError,
    BadSearchTypeError,
    BadTimeDataShape,
    FullFieldCalculator,
    InsuffichentFieldDataSetsError,
    MeshlessMapperGMLS,
    NoMeasurementError,
    NonIntegerPolynomialOrderError,
    SameDataNameError,
    SameMeasurementNameError,
    SmallSearchRadiusError,
    _TwoDimensionalFieldInterpolator,
    _TwoDimensionalFieldProjector,
    _LabToParametricSpaceMapper,
    _check_gmls_parameters,
    meshless_remapping,
)
from matcal.full_field.field_mappers import _check_pycompadre_available
import unittest
from unittest.mock import patch
from matcal.full_field.shapefunctions import TwoDim4NodeBilinearShapeFunction
from matcal.full_field.TwoDimensionalFieldGrid import (
    MeshSkeleton,
    GridAxis,
    TwoDimensionalFieldGrid,
    MeshSkeletonTwoDimensionalMesh,
)


class TestTwoDimensionalFieldProjector(MatcalUnitTest):
    def setUp(self) -> None:  # type: ignore[override]
        super().setUp(__file__)
        self._global_field_data_csv = (
            self.get_current_files_path(__file__)
            + "/input_files/simple_2D_global_data.csv"
        )
        self._series_directory = self.get_current_files_path(__file__) + "/input_files/"
        self._series_data = FieldSeriesData(
            self._global_field_data_csv, self._series_directory, ["X", "Y"]
        )

        self.x = GridAxis(-0.01, 6.01, 4)
        self.y = GridAxis(-0.01, 4.01, 3)
        self._target_grid = TwoDimensionalFieldGrid(self.x, self.y)
        self.tdfp = _TwoDimensionalFieldProjector(self._series_data, self._target_grid)

    def test_raise_error_when_using_bad_field(self):
        with self.assertRaises(_TwoDimensionalFieldProjector.InvalidFieldnameError):
            self.tdfp.project_field("bad_name", 0)

    def test_translated_parametric_location(self):
        test_point = np.array([[1, 1]])
        cell_points = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
        lpm = _LabToParametricSpaceMapper(test_point, cell_points)
        location = lpm.calculate_parametric_location()
        goal_location = np.array([0, 0])
        self.assertTrue(np.allclose(location, goal_location))

    def _project_test_base(self, fieldname, goal):
        results = self.tdfp.project_field(fieldname, 0)
        self.assertTrue(np.allclose(results[fieldname], goal))

    def test_project_a_zero_field_get_zero(self):
        zero_field = "U_y"
        goal = np.zeros(self._target_grid.node_count)
        self._project_test_base(zero_field, goal)

    def test_project_field_returns_a_data_type_class(self):
        results = self.tdfp.project_field("U_y", 0)
        self.assertIsInstance(results, Data)

    def test_project_a_constant_value_get_100(self):
        field = "E"
        goal = np.ones(self._target_grid.node_count) * 100
        self._project_test_base(field, goal)

    def test_project_linear_field(self):
        field = "T"
        goal = (
            self._target_grid.node_positions[:, 0] * 10
            + self._target_grid.node_positions[:, 1] * 10
            + 275
        )
        self._project_test_base(field, goal)

    def test_project_multiple_fields(self):
        fields = ["E", "U_y", "T"]
        T_goal = (
            self._target_grid.node_positions[:, 0] * 10
            + self._target_grid.node_positions[:, 1] * 10
            + 275
        )
        goals = {
            "E": np.ones(self._target_grid.node_count) * 100,
            "U_y": np.zeros(self._target_grid.node_count),
            "T": T_goal,
        }
        results_datas = {}
        for field in fields:
            results_datas[field] = self.tdfp.project_field(field, 0)

        for field in fields:
            self.assertTrue(np.allclose(results_datas[field][field], goals[field]))


class TestProjectionFromCopy(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)
        self._global_field_data_csv = (
            self.get_current_files_path(__file__)
            + "/input_files/simple_2D_global_data.csv"
        )
        self._series_directory = self.get_current_files_path(__file__) + "/input_files/"
        self._series_data = FieldSeriesData(
            self._global_field_data_csv, self._series_directory, ["X", "Y"]
        )

        self.x = GridAxis(-0.01, 6.01, 4)
        self.y = GridAxis(-0.01, 4.01, 3)
        thickness = 0.1
        self._target_grid = TwoDimensionalFieldGrid(self.x, self.y)
        self.tdfp = _TwoDimensionalFieldProjector(self._series_data, self._target_grid)

    def test_copy_projection_gets_correct_answers(self):
        fieldname = "E"
        self.tdfp.project_field(fieldname, 0)
        new_data_file = (
            self.get_current_files_path(__file__)
            + "/input_files/simple_2D_global_data2.csv"
        )
        new_data_file_dir = self.get_current_files_path(__file__) + "/input_files/"
        source_data = FieldSeriesData(new_data_file, new_data_file_dir)

        E_goal = np.ones(self._target_grid.node_count) * 200
        U_y_goal = np.zeros(self._target_grid.node_count)
        T_goal = (
            self._target_grid.node_positions[:, 0] * 20
            + self._target_grid.node_positions[:, 1] * 10
            + 275
        )

        test_goals = {"E": E_goal, "U_y": U_y_goal, "T": T_goal}
        self.tdfp.reset(source_data)

        results_datas = {}
        for field in test_goals.keys():
            results_datas[field] = self.tdfp.project_field(field, 0)

        for key, goal in test_goals.items():
            self.assert_close_arrays(goal, results_datas[key])


class TestProjectionWithHoles(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def test_multi_hole_plate_with_passed_mesh(self):
        data_file = (
            self.get_current_files_path(__file__)
            + "/input_files/cut_plate/adv_plastic_linear_global.csv"
        )
        data_dir = self.get_current_files_path(__file__) + "/input_files/cut_plate/"
        field_data = FieldSeriesData(data_file, data_dir)
        reference_meshfile = "/".join(
            [self.get_current_files_path(__file__), "input_files", "flat_surface.json"]
        )

        thickness = 0.1 / 50
        self._target_grid = ImportedTwoDimensionalMesh(reference_meshfile)
        self.tdfp = _TwoDimensionalFieldProjector(field_data, self._target_grid)
        self.tdfp.project_field("Ux", 0)
        results = self.tdfp.get_results_data()
        x_unsort = self._target_grid.node_positions[:, 0]
        goal = 0.1 * x_unsort
        test = results["Ux"]

        self.assert_close_arrays(test, goal)

    def test_projection_with_exterior_cloud_points(self):
        reference_meshfile = "/".join(
            [
                self.get_current_files_path(__file__),
                "input_files",
                "flat_rectangle.json",
            ]
        )
        x_range = [-0.55, 0.55]
        y_range = [-0.55, 0.55]
        n_axis = 7
        x = np.linspace(x_range[0], x_range[1], n_axis)
        y = np.linspace(y_range[0], y_range[1], n_axis)
        x, y = np.meshgrid(x, y)
        x = x.flatten()
        y = y.flatten()
        T = (x + 2 * y) * 10
        T = T.reshape(-1, n_axis * n_axis)
        n_time = 1
        data = {"time": np.linspace(0, 10, n_time), "x": x, "y": y, "T": T}
        data = convert_dictionary_to_field_data(data, coordinate_names=["x", "y"])
        target_grid = ImportedTwoDimensionalMesh(reference_meshfile)
        proj = _TwoDimensionalFieldProjector(data, target_grid)
        proj.project_field("T", 0)
        results = proj.get_results_data()
        T_goal = (
            target_grid.node_positions[:, 0] + 2 * target_grid.node_positions[:, 1]
        ) * 10
        self.assert_close_arrays(results["T"], T_goal)


class TestLabToParametericSpaceMapper(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def test_translated_parametric_location_center(self):
        test_point = np.array([[1, 1]])
        cell_points = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
        lpm = _LabToParametricSpaceMapper(test_point, cell_points)
        location = lpm.calculate_parametric_location()
        goal_location = np.array([0, 0])
        self.assertTrue(np.allclose(location, goal_location))

    def test_translated_parametric_location_corner(self):
        test_point = np.array([[2, 2]])
        cell_points = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
        lpm = _LabToParametricSpaceMapper(test_point, cell_points)
        location = lpm.calculate_parametric_location()
        goal_location = np.array([1, 1])
        self.assertTrue(np.allclose(location, goal_location))

    def test_translated_parametric_location_general(self):
        goal_location = np.array([[0.123, -0.52]])
        sf_values = TwoDim4NodeBilinearShapeFunction().values(goal_location)
        cell_points = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
        test_point = np.dot(sf_values, cell_points)
        lpm = _LabToParametricSpaceMapper(test_point, cell_points)
        location = lpm.calculate_parametric_location()
        self.assertTrue(np.allclose(location, goal_location))

    def test_scaled_parameter_location_center(self):
        goal_location = np.array([[0.0, -0.0]])
        sf_values = TwoDim4NodeBilinearShapeFunction().values(goal_location)
        cell_points = np.array([[0, 0], [3, 0], [3, 10], [0, 10]])
        test_point = np.array([[1.5, 5]])
        lpm = _LabToParametricSpaceMapper(test_point, cell_points)
        location = lpm.calculate_parametric_location()
        self.assertTrue(np.allclose(location, goal_location))

    def test_scaled_parameter_location_general(self):
        goal_location = np.array([[0.4, -0.3]])
        sf_values = TwoDim4NodeBilinearShapeFunction().values(goal_location)
        cell_points = np.array([[0, 0], [3, 0], [3, 10], [0, 10]])
        test_point = np.dot(sf_values, cell_points)
        lpm = _LabToParametricSpaceMapper(test_point, cell_points)
        location = lpm.calculate_parametric_location()
        self.assertTrue(np.allclose(location, goal_location))

    def test_skewed_parameter_location_general(self):
        goal_location = np.array([[-0.499, 0.333]])
        sf_values = TwoDim4NodeBilinearShapeFunction().values(goal_location)
        cell_points = np.array([[-1, 2], [2, 1.5], [3, 4], [0, 3]])
        test_point = np.dot(sf_values, cell_points)
        lpm = _LabToParametricSpaceMapper(test_point, cell_points)
        location = lpm.calculate_parametric_location()
        self.assertTrue(np.allclose(location, goal_location))


class TestFieldInterpolator(MatcalUnitTest):

    class FieldInterpolatorSpy(_TwoDimensionalFieldInterpolator):

        def __init__(self, grid_geometry, cloud_points):
            self.matrix_generation_count = 0
            super().__init__(grid_geometry, cloud_points)

        def get_interpolation_matrix(self):
            return self._matrix.todense()

        def _make_interpolation_matrix_from_transposed_projection_matrix(self):
            self.matrix_generation_count += 1
            return (
                super()._make_interpolation_matrix_from_transposed_projection_matrix()
            )

    def setUp(self):
        super().setUp(__file__)

    def test_generates_correct_one_element_interp_mesh(self):
        grid_geometry = self.make_one_element_grid()
        cloud_points = np.array([[0], [0]]).T

        fi_spy = self.FieldInterpolatorSpy(grid_geometry, cloud_points)
        goal_matrix = np.array([[1, 1, 1, 1]]) * 0.25
        self.assert_close_arrays(fi_spy.get_interpolation_matrix(), goal_matrix)

    def test_generates_correct_constant_interp_for_one_points(self):
        grid_geometry = self.make_one_element_grid()
        cloud_points = np.array([[0], [0]]).T

        fi = _TwoDimensionalFieldInterpolator(grid_geometry, cloud_points)
        result_val = fi.interpolate(np.ones((4, 1)))
        goal = np.array([[1]]).T
        self.assert_close_arrays(result_val, goal)

    def test_generates_correct_constant_interp_for_two_of_the_same_point(self):
        grid_geometry = self.make_one_element_grid()
        cloud_points = np.array([[0, 0], [0, 0]]).T

        fi = _TwoDimensionalFieldInterpolator(grid_geometry, cloud_points)
        result_val = fi.interpolate(np.ones((4, 1)))
        goal = np.array([[1, 1]]).T
        self.assert_close_arrays(result_val, goal)

    def test_generates_correct_linear_interp_for_one_point(self):
        grid_geometry = self.make_one_element_grid()
        cloud_points = np.array(
            [
                [
                    0,
                ],
                [0],
            ]
        ).T

        fi = _TwoDimensionalFieldInterpolator(grid_geometry, cloud_points)
        grid_data = np.array([[-1.0, -1, 1, 1]]).T
        result_val = fi.interpolate(grid_data)
        goal = np.array([[0.0]]).T
        self.assert_close_arrays(result_val, goal)

    def test_generates_correct_linear_interp_for_one_point_two_fields(self):
        grid_geometry = self.make_one_element_grid()
        cloud_points = np.array(
            [
                [
                    0,
                ],
                [0],
            ]
        ).T

        fi = _TwoDimensionalFieldInterpolator(grid_geometry, cloud_points)
        grid_data = np.array([[-1.0, -1, 1, 1], [10, 20, 20, 10]]).T
        result_val = fi.interpolate(grid_data)
        goal = np.array([[0.0], [15]]).T
        self.assert_close_arrays(result_val, goal)

    def test_generates_correct_linear_interp_for_one_point_two_fields_off_center(self):
        grid_geometry = self.make_one_element_grid()
        cloud_points = np.array([[0.25], [-1]]).T

        fi = _TwoDimensionalFieldInterpolator(grid_geometry, cloud_points)
        grid_data = np.array([[-1.0, -1, 1, 1], [10, 20, 20, 10]]).T
        result_val = fi.interpolate(grid_data)
        goal = np.array([[-1], [10 + 10 * (5 / 8.0)]]).T
        self.assert_close_arrays(result_val, goal)

    def test_generates_correct_linear_interp_for_two_points(self):
        grid_geometry = self.make_one_element_grid()
        cloud_points = np.array([[0, 0], [0, 1]]).T

        fi = _TwoDimensionalFieldInterpolator(grid_geometry, cloud_points)
        grid_data = np.array([[-1.0, -1, 1, 1]]).T
        result_val = fi.interpolate(grid_data)
        goal = np.array([[0.0, 1.0]]).T
        self.assert_close_arrays(result_val, goal)

    def test_confirm_matrix_only_generated_once_for_multiple_interpolations(self):
        grid_geometry = self.make_one_element_grid()
        cloud_points = np.array([[0, 0], [0, 1]]).T

        fi = self.FieldInterpolatorSpy(grid_geometry, cloud_points)
        grid_data = np.array([[-1.0, -1, 1, 1]]).T
        result_val = fi.interpolate(grid_data)
        result_val = fi.interpolate(grid_data)
        result_val = fi.interpolate(grid_data)
        self.assertEqual(fi.matrix_generation_count, 1)

    def test_generates_correct_linear_interp_for_one_point_three_fields(self):
        grid_geometry = self.make_one_element_grid()
        cloud_points = np.array([[0, 0, 1], [0, 1, 0]]).T

        fi = _TwoDimensionalFieldInterpolator(grid_geometry, cloud_points)
        grid_data = np.array([[-1.0, -1, 1, 1], [10, 20, 20, 10]]).T
        result_val = fi.interpolate(grid_data)
        goal = np.array([[0.0, 1, 0], [15, 15, 20]]).T
        self.assert_close_arrays(result_val, goal)

    def make_one_element_grid(self):
        grid = MeshSkeleton()
        grid.spatial_coords = np.array([[-1, 1, 1, -1], [-1, -1, 1, 1]]).T
        grid.connectivity = np.array([[0, 1, 2, 3]])
        grid_geometry = MeshSkeletonTwoDimensionalMesh(grid)
        return grid_geometry

    def test_collocation_reconstruction(self):
        grid = MeshSkeleton()
        grid.spatial_coords = np.array(
            [
                [-0.025, -0.05, 0.0],
                [-0.025, -0.01666667, 0.0],
                [0.0, -0.05, 0.0],
                [0.0, -0.01666667, 0.0],
                [-0.025, 0.01666667, 0.0],
                [0.0, 0.01666667, 0.0],
                [-0.025, 0.05, 0.0],
                [0.0, 0.05, 0.0],
                [0.025, -0.05, 0.0],
                [0.025, -0.01666667, 0.0],
                [0.025, 0.01666667, 0.0],
                [0.025, 0.05, 0.0],
            ]
        )
        grid.connectivity = np.array(
            [
                [2, 8, 9, 3],
                [3, 9, 10, 5],
                [5, 10, 11, 7],
                [0, 2, 3, 1],
                [1, 3, 5, 4],
                [4, 5, 7, 6],
            ]
        )
        cloud_locations = np.array(
            [
                [0.025, -0.05],
                [0.025, 0.05],
                [0.025, -0.01666667],
                [0.025, 0.01666667],
                [-0.025, 0.05],
                [0.0, 0.05],
                [-0.025, -0.05],
                [-0.025, 0.01666667],
                [-0.025, -0.01666667],
                [0.0, -0.05],
                [0.0, -0.01666667],
                [0.0, 0.01666667],
            ]
        )
        mesh = MeshSkeletonTwoDimensionalMesh(grid)
        interp_tool = _TwoDimensionalFieldInterpolator(mesh, cloud_locations)
        interp_mat = interp_tool._matrix.todense()

        goal = np.zeros([12, 12])
        # cloud  = map * grid
        colocation_map = [
            [0, 8],
            [1, 11],
            [2, 9],
            [3, 10],
            [4, 6],
            [5, 7],
            [6, 0],
            [7, 4],
            [8, 1],
            [9, 2],
            [10, 3],
            [11, 5],
        ]
        for pair in colocation_map:
            goal[pair[0], pair[1]] = 1.0

        self.assert_close_arrays(goal, interp_mat)


class _MeshlessMappingTestBase(MatcalUnitTest):
    """Base class for meshless mapping tests.

    Not collected by pytest directly (leading underscore).
    Derived classes select a specific GMLS backend.
    """

    __test__ = False

    # Subclasses may set this to explicitly pass ``backend=`` to the
    # mapper.  When *None* (default), the old patching behaviour is
    # used so that the pycompadre backend tests still work correctly
    # even when invoked via the mock.
    _backend: str | None = None

    # Legacy patching knob -- only used when ``_backend`` is *None*.
    _force_pycompadre_available: bool = False

    def setUp(self):
        super().setUp(__file__)
        # When an explicit backend is requested we skip the mock
        # entirely so that the ``backend=`` code-path in the
        # constructor is exercised end-to-end.
        if self._backend is None:
            patcher = patch(
                "matcal.full_field.field_mappers._check_pycompadre_available",
                return_value=self._force_pycompadre_available,
            )
            self._mock_pycompadre = patcher.start()
            self.addCleanup(patcher.stop)

    def cubic_function_1d(self, coords):
        consts = [1, 2, 3, 4]
        vals = np.zeros_like(coords)
        for p, c in enumerate(consts):
            vals += c * np.power(coords, p)
        return vals

    def cubic_function_2d(self, coords):
        consts = [1, 2, 3, 4]
        vals = np.zeros_like(coords[:, 0])
        for p, c in enumerate(consts):
            vals += c * np.power(coords[:, 0], p) * np.power(coords[:, 1], 3 - p)
        return vals

    def test_1d_polynomial(self):
        test_function = self.cubic_function_1d
        n_points = 20
        n_dim = 1
        poly_order = 3
        eps = 2
        self._confirm_interp(test_function, n_points, n_dim, poly_order, eps)

    def test_2d_polynomial(self):
        test_function = self.cubic_function_2d
        n_points = 10
        n_dim = 2
        poly_order = 4
        eps = 1.5
        self._confirm_interp(test_function, n_points, n_dim, poly_order, eps)

    def test_2d_polynomial_meshless_remapping_func(self):
        test_function = self.cubic_function_2d
        n_points = 10
        n_dim = 2
        poly_order = 4
        eps = 1.5
        self._confirm_interp_meshless_remapping_function(
            test_function, n_points, n_dim, poly_order, eps
        )

    def cubic_3d(self, coords):
        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2]
        val = (
            2 * x
            + np.multiply(x, np.multiply(y, z))
            + np.power(z, 3) * 4
            + np.multiply(np.power(y, 2), x) * -1
        )
        return val

    def test_3d_polynomial(self):
        test_function = self.cubic_3d
        n_points = 10
        n_dim = 3
        poly_order = 4
        eps = 2.0
        self._confirm_interp(test_function, n_points, n_dim, poly_order, eps)

    def linear_sin(self, coords):
        x = coords[:, 0]
        y = coords[:, 1]
        val = np.zeros_like(x)
        val += -1 + 2 * x + y / 2
        val += np.sin(np.pi * (x + y))
        return val

    def test_2d_trig(self):
        test_function = self.linear_sin
        n_points = 120
        n_dim = 2
        poly_order = 7
        eps = 1.5
        self._confirm_interp(test_function, n_points, n_dim, poly_order, eps)

    def test_2d_trig_incorrect_length_error(self):
        test_function = self.linear_sin
        n_points = 120
        n_dim = 2
        poly_order = 7
        eps = 1.5
        with self.assertRaises(MeshlessMapperGMLS.IncorrectLengthError):
            self._confirm_interp_bad_length(
                test_function, n_points, n_dim, poly_order, eps
            )

    def test_errors(self):
        n_points = 12
        n_dim = 2
        coords = np.random.uniform(0, 1, [n_points, n_dim])
        with self.assertRaises(NonIntegerPolynomialOrderError):
            _check_gmls_parameters(1.2, 1)
        with self.assertRaises(SmallSearchRadiusError):
            _check_gmls_parameters(1, 1.0)
        with self.assertRaises(BadPolynomialOrderError):
            _check_gmls_parameters(0, 1.0)
        with self.assertRaises(BadPolynomialOrderError):
            _check_gmls_parameters(11, 1.0)
        with self.assertRaises(BadSearchTypeError):
            _check_gmls_parameters(10, "0")

    def test_perform_time_interp(self):
        n_space = 20
        n_dim = 2
        n_time = 10

        target_coords = np.random.uniform(0, 1, [n_space, n_dim])
        target_time = np.linspace(0, 1, n_time)
        source_coords = np.random.uniform(0, 1, [10 * n_space, n_dim])
        source_time = np.linspace(0, 1, n_time * 5)

        def test_function(x, t):
            y = np.outer(t, x[:, 0] + x[:, 1] + np.power(x[:, 0], 2))
            return y

        source_vals = test_function(source_coords, source_time)
        target_goal_vals = test_function(target_coords, target_time)

        source_data_dict = {
            "X": source_coords[:, 0],
            "Y": source_coords[:, 1],
            "Z": source_vals,
            "time": source_time,
        }
        source_data = convert_dictionary_to_field_data(
            source_data_dict, coordinate_names=["X", "Y"]
        )

        target_map_vals = meshless_remapping(
            source_data,
            ["Z"],
            target_coords,
            polynomial_order=2,
            target_time=target_time,
            time_field="time",
            backend=self._backend,
        )

        self.assert_close_arrays(target_goal_vals, target_map_vals["Z"])

    def _confirm_interp(self, test_function, n_points, n_dim, poly_order, eps):
        target_coords = np.random.uniform(0, 1, [n_points, n_dim])
        source_coords = np.random.uniform(0, 1, [10 * n_points, n_dim])

        source_vals = test_function(source_coords)
        target_goal_vals = test_function(target_coords)

        kwargs = {}
        if self._backend is not None:
            kwargs["backend"] = self._backend
        mapper = MeshlessMapperGMLS(
            target_coords, source_coords, poly_order, eps, **kwargs
        )
        target_map_vals = mapper.map(source_vals)
        mapper.finish()
        self.assert_close_arrays(target_goal_vals, target_map_vals)

    def _confirm_interp_bad_length(
        self, test_function, n_points, n_dim, poly_order, eps
    ):
        target_coords = np.random.uniform(0, 1, [n_points, n_dim])
        source_coords = np.random.uniform(0, 1, [10 * n_points, n_dim])

        source_vals = test_function(source_coords)
        target_goal_vals = test_function(target_coords)

        kwargs = {}
        if self._backend is not None:
            kwargs["backend"] = self._backend
        mapper = MeshlessMapperGMLS(
            target_coords, source_coords, poly_order, eps, **kwargs
        )
        target_map_vals = mapper.map(target_goal_vals)
        mapper.finish()

    def _confirm_interp_bad_dim_neighbor_detection_error(
        self, test_function, n_points, n_dim, poly_order, eps
    ):
        target_coords = np.random.uniform(0, 1, [n_points, n_dim])
        source_coords = np.random.uniform(0, 1, [10 * n_points, n_dim])

        source_vals = test_function(source_coords)
        source_data_dict = {
            "X": source_coords[:, 0].reshape(-1, 1),
            "Y": source_coords[:, 1].reshape(-1, 1),
            "Z": source_vals.reshape(1, -1),
            "time": [0],
        }
        source_data = convert_dictionary_to_field_data(
            source_data_dict, coordinate_names=["X", "Y"]
        )

        target_goal_vals = test_function(target_coords)
        target_data_dict = {
            "X": target_coords[:, 0].reshape(-1, 1),
            "Y": target_coords[:, 1].reshape(-1, 1),
            "Z": target_goal_vals.reshape(1, -1),
            "time": [0],
        }
        target_data = convert_dictionary_to_field_data(
            target_data_dict, coordinate_names=["X", "Y"]
        )

        target_map_vals = meshless_remapping(
            source_data,
            ["Z"],
            target_data.spatial_coords,
            poly_order,
            eps,
            backend=self._backend,
        )
        self.assert_close_arrays(target_goal_vals, target_map_vals["Z"])

    def _confirm_interp_meshless_remapping_function(
        self, test_function, n_points, n_dim, poly_order, eps
    ):
        target_coords = np.random.uniform(0, 1, [n_points, n_dim])
        source_coords = np.random.uniform(0, 1, [10 * n_points, n_dim])

        source_vals = test_function(source_coords)
        source_data_dict = {
            "X": source_coords[:, 0].flatten(),
            "Y": source_coords[:, 1].flatten(),
            "Z": source_vals.reshape(1, -1),
            "time": [0],
        }
        source_data = convert_dictionary_to_field_data(
            source_data_dict, coordinate_names=["X", "Y"]
        )

        target_goal_vals = test_function(target_coords)
        target_data_dict = {
            "X": target_coords[:, 0].flatten(),
            "Y": target_coords[:, 1].flatten(),
            "Z": target_goal_vals.reshape(1, -1),
            "time": [0],
        }
        target_data = convert_dictionary_to_field_data(
            target_data_dict, coordinate_names=["X", "Y"]
        )

        target_map_vals = meshless_remapping(
            source_data,
            ["Z"],
            target_data.spatial_coords,
            poly_order,
            eps,
            backend=self._backend,
        )
        self.assert_close_arrays(target_goal_vals, target_map_vals["Z"])


class TestMeshlessMappingScipyBackend(_MeshlessMappingTestBase):
    """Run all meshless mapping tests using the built-in scipy backend.

    Uses the explicit ``backend="scipy"`` parameter so that the scipy
    code-path is exercised even when pycompadre is installed.
    """

    __test__ = True
    _backend = "scipy"

    def test_2d_trig_high_order_scipy(self):
        """Verify scipy backend handles high-order mapping without error."""
        test_function = self.linear_sin
        n_points = 120
        n_dim = 2
        poly_order = 7
        eps = 1.5
        self._confirm_interp_bad_dim_neighbor_detection_error(
            test_function, n_points, n_dim, poly_order, eps
        )


@unittest.skipUnless(
    _check_pycompadre_available(),
    "pycompadre not installed; skipping pycompadre backend tests",
)
class TestMeshlessMappingPycompadreBackend(_MeshlessMappingTestBase):
    """Run all meshless mapping tests using the pycompadre backend."""

    __test__ = True
    _force_pycompadre_available = True

    def test_2d_trig_neighbor_detection_error_pycompadre(self):
        """Verify pycompadre wraps neighbor errors as NeighborDetectionError.

        A 7th-order polynomial in 2D requires C(9,7)=36 basis terms,
        so at least 36 source neighbors per target.  With only 30
        source points total and a tight search radius (eps=1.5),
        pycompadre cannot find sufficient support and raises during
        neighbor list construction.  The error must be wrapped as
        ``NeighborDetectionError``.
        """
        np.random.seed(123456)
        n_dim = 2
        poly_order = 7
        eps = 1.5
        source_coords = np.random.uniform(0, 1, [30, n_dim])
        target_coords = np.random.uniform(0, 1, [10, n_dim])
        with self.assertRaises(MeshlessMapperGMLS.NeighborDetectionError):
            MeshlessMapperGMLS(target_coords, source_coords, poly_order, eps)


class TestMeshlessMappingBackendParameter(MatcalUnitTest):
    """Tests for the ``backend`` parameter on MeshlessMapperGMLS and meshless_remapping."""

    def setUp(self):
        super().setUp(__file__)

    def _make_coords(self):
        n_dim = 2
        source_coords = np.random.uniform(0, 1, [50, n_dim])
        target_coords = np.random.uniform(0, 1, [10, n_dim])
        return source_coords, target_coords

    def test_invalid_backend_raises_value_error(self):
        source, target = self._make_coords()
        with self.assertRaises(ValueError):
            MeshlessMapperGMLS(target, source, backend="invalid")

    def test_backend_none_selects_automatically(self):
        """backend=None should select a backend without error."""
        source, target = self._make_coords()
        mapper = MeshlessMapperGMLS(target, source, backend=None)
        self.assertIn(mapper._backend, ("pycompadre", "scipy"))
        mapper.finish()

    def test_backend_scipy_forces_scipy(self):
        """backend='scipy' must use the scipy path even when pycompadre is available."""
        source, target = self._make_coords()
        mapper = MeshlessMapperGMLS(target, source, backend="scipy")
        self.assertEqual(mapper._backend, "scipy")
        self.assertIsNotNone(mapper._weight_matrix)
        mapper.finish()

    def test_backend_scipy_produces_correct_results(self):
        """Verify that backend='scipy' produces accurate interpolation."""
        n_dim = 2
        source_coords = np.random.uniform(0, 1, [200, n_dim])
        target_coords = np.random.uniform(0, 1, [20, n_dim])
        source_vals = source_coords[:, 0] + 2 * source_coords[:, 1]
        target_goal = target_coords[:, 0] + 2 * target_coords[:, 1]

        mapper = MeshlessMapperGMLS(
            target_coords, source_coords, 2, 2.0, backend="scipy"
        )
        mapped = mapper.map(source_vals)
        mapper.finish()
        self.assert_close_arrays(target_goal, mapped)

    @unittest.skipUnless(
        _check_pycompadre_available(),
        "pycompadre not installed",
    )
    def test_backend_pycompadre_forces_pycompadre(self):
        """backend='pycompadre' must use the pycompadre path."""
        source, target = self._make_coords()
        mapper = MeshlessMapperGMLS(target, source, backend="pycompadre")
        self.assertEqual(mapper._backend, "pycompadre")
        mapper.finish()

    def test_backend_pycompadre_missing_raises_import_error(self):
        """Requesting pycompadre when it's not importable should raise ImportError."""
        source, target = self._make_coords()
        with patch(
            "matcal.full_field.field_mappers._check_pycompadre_available",
            return_value=False,
        ):
            with self.assertRaises(ImportError):
                MeshlessMapperGMLS(target, source, backend="pycompadre")

    def test_meshless_remapping_backend_scipy(self):
        """meshless_remapping should forward backend='scipy' correctly."""
        n_dim = 2
        source_coords = np.random.uniform(0, 1, [200, n_dim])
        target_coords = np.random.uniform(0, 1, [20, n_dim])
        source_vals = source_coords[:, 0] + 2 * source_coords[:, 1]
        target_goal = target_coords[:, 0] + 2 * target_coords[:, 1]

        source_data_dict = {
            "X": source_coords[:, 0],
            "Y": source_coords[:, 1],
            "Z": source_vals.reshape(1, -1),
        }
        source_data = convert_dictionary_to_field_data(
            source_data_dict, coordinate_names=["X", "Y"]
        )

        mapped = meshless_remapping(
            source_data,
            ["Z"],
            target_coords,
            polynomial_order=2,
            search_radius_multiplier=2.0,
            backend="scipy",
        )
        self.assert_close_arrays(target_goal, mapped["Z"])

    def test_meshless_remapping_backend_invalid_raises(self):
        """meshless_remapping should raise ValueError for invalid backend."""
        n_dim = 2
        source_coords = np.random.uniform(0, 1, [50, n_dim])
        target_coords = np.random.uniform(0, 1, [10, n_dim])
        source_vals = source_coords[:, 0]

        source_data_dict = {
            "X": source_coords[:, 0],
            "Y": source_coords[:, 1],
            "Z": source_vals.reshape(1, -1),
        }
        source_data = convert_dictionary_to_field_data(
            source_data_dict, coordinate_names=["X", "Y"]
        )

        with self.assertRaises(ValueError):
            meshless_remapping(
                source_data,
                ["Z"],
                target_coords,
                backend="bogus",
            )

    def test_finish_clears_state(self):
        """After finish(), internal resources should be released."""
        source, target = self._make_coords()
        mapper = MeshlessMapperGMLS(target, source, backend="scipy")
        self.assertIsNotNone(mapper._weight_matrix)
        mapper.finish()
        self.assertIsNone(mapper._weight_matrix)
        self.assertIsNone(mapper._pycompadre_gmls)
        self.assertIsNone(mapper._pycompadre_helper)

    def test_3d_target_coords_reshaped(self):
        """3-D target_coords should be reshaped to 2-D automatically."""
        n_dim = 2
        source_coords = np.random.uniform(0, 1, [50, n_dim])
        # Provide target_coords with 3 dimensions: (2, 5, 2) -> should flatten to (10, 2)
        target_coords = np.random.uniform(0, 1, [2, 5, n_dim])
        source_vals = source_coords[:, 0] + source_coords[:, 1]

        mapper = MeshlessMapperGMLS(
            target_coords, source_coords, 1, 2.0, backend="scipy"
        )
        mapped = mapper.map(source_vals)
        mapper.finish()
        self.assertEqual(mapped.shape[0], 10)

    def test_backend_none_fallback_logs_info(self):
        """When backend=None and pycompadre unavailable, an info log should be emitted."""
        source, target = self._make_coords()
        with patch(
            "matcal.full_field.field_mappers._check_pycompadre_available",
            return_value=False,
        ):
            with self.assertLogs("matcal.full_field.field_mappers", level="INFO") as cm:
                mapper = MeshlessMapperGMLS(target, source, backend=None)
                mapper.finish()
        self.assertTrue(any("numpy/scipy GMLS" in msg for msg in cm.output))


def subtract_function(reference_field, specific_field, spatial_corrds, time):
    return specific_field - reference_field


def add_function(reference_field, specific_field, spatial_coords, time):
    return reference_field + specific_field


class FieldStatsExportTests(ABC):

    def __init__(self):
        pass

    @property
    @abstractmethod
    def export_filename(self):
        """"""

    class CommonExportTests(MatcalUnitTest):

        def test_two_function_export_and_import(self):
            n_time = 10
            n_pts = 20
            n_ele = 11
            ele_size = 4
            field_vars = ["A", "T"]
            global_vars = ["total_heat"]
            ref_ff_data = _make_same_polynomial_field_data(
                n_time, n_pts, n_ele, ele_size, field_vars, global_vars, 3, 1
            )
            test_ff_data1 = _make_same_polynomial_field_data(
                n_time * 2, n_pts * 2, n_ele, ele_size, field_vars, global_vars, 3, 1
            )
            test_ff_data2 = _make_same_polynomial_field_data(
                n_time * 3, n_pts * 2, n_ele, ele_size, field_vars, global_vars, 3, 1
            )

            source_filename1 = "source1.json"
            source_filename2 = "source2.json"
            export_full_field_data_to_json(source_filename1, test_ff_data1)
            export_full_field_data_to_json(source_filename2, test_ff_data2)

            field_stats = FullFieldCalculator(
                ref_ff_data, ref_ff_data["time"], position_names=["X", "Y", "Z"]
            )
            field_stats.add_spatial_calculation("diff", subtract_function, "T")
            field_stats.add_spatial_calculation("sum", add_function, "T")
            field_stats.add_data("d1", source_filename1)
            field_stats.add_data("d2", source_filename2)
            field_stats.set_mapping_parameters(3, 1.5)

            field_stats.calculate_and_export(self.export_filename)

            if self.export_filename[-5] == "json":
                calc_data = _import_full_field_data_from_json(self.export_filename)
            else:
                calc_data = FieldSeriesData(self.export_filename)

            goal_val = np.zeros_like(calc_data["diff_d1_T"])
            self._assert_in_and_val(calc_data, "diff_d1_T", goal_val)
            self._assert_in_and_val(calc_data, "diff_d2_T", goal_val)

            goal_val = ref_ff_data["T"] * 2
            self._assert_in_and_val(calc_data, "sum_d1_T", goal_val)
            self._assert_in_and_val(calc_data, "sum_d2_T", goal_val)

            goal_val = ref_ff_data["T"]
            self._assert_in_and_val(calc_data, "T_interp_ref", goal_val)
            self._assert_in_and_val(calc_data, "T_interp_d1", goal_val)
            self._assert_in_and_val(calc_data, "T_interp_d2", goal_val)

        def _assert_in_and_val(self, calc_data, target_field, goal_val):
            self.assertIn(target_field, list(calc_data.keys()))
            self.assert_close_arrays(
                calc_data[target_field], goal_val, show_on_fail=True
            )


class TestFieldStats(FieldStatsExportTests.CommonExportTests):

    def setUp(self):
        super().setUp(__file__)

    @property
    def export_filename(self):
        return "target.json"

    class FFCalcSpy(FullFieldCalculator):

        def test_time_field_name(self, goal_name):
            return self._independent_field_name == goal_name

        def test_reference_times(self, goal_times):
            tol = 1e-8
            delta = self._independent_field_vals - goal_times
            scale_factor = np.max(np.abs(goal_times))
            rel_delta = delta / scale_factor
            return np.max(rel_delta) < tol

    def _make_simple_mesh_with_info(self):
        n_time = 3
        n_loc = 4
        time = np.linspace(0, 1, n_time)
        T = np.random.uniform(0, 1, [n_time, n_loc])
        bc_temp = np.power(time, 2)
        ref_ff_data = {
            "T": T,
            "time": time,
            "x": np.array([0, 1, 1, 0]),
            "y": np.array([0, 0, 1, 1]),
            "con": [[0, 1, 2, 3]],
            "bc_temp": bc_temp,
        }
        ref_ff_data = convert_dictionary_to_field_data(ref_ff_data, ["x", "y"], "con")
        return ref_ff_data

    def test_initialize_reference_geometry_file(self):
        ref_ff_data = self._make_simple_mesh_with_info()
        ref_filename = "my_mesh.json"
        export_full_field_data_to_json(ref_filename, ref_ff_data)
        field_stats = FullFieldCalculator(ref_filename, ref_ff_data["time"])

    def test_bad_init_independent_field_vals(self):
        ref_ff_data = self._make_simple_mesh_with_info()
        ref_filename = "my_mesh.json"
        export_full_field_data_to_json(ref_filename, ref_ff_data)
        field_stats = FullFieldCalculator(ref_filename, [0.1, 1])
        field_stats = FullFieldCalculator(ref_filename, np.array([0.1, 1]))
        with self.assertRaises(TypeError):
            field_stats = FullFieldCalculator(
                ref_filename, "bad independent field vals type"
            )

    def test_bad_init_independent_field_name(self):
        ref_ff_data = self._make_simple_mesh_with_info()
        ref_filename = "my_mesh.json"
        export_full_field_data_to_json(ref_filename, ref_ff_data)
        field_stats = FullFieldCalculator(ref_filename, [0.1, 1], "time")
        with self.assertRaises(TypeError):
            field_stats = FullFieldCalculator(ref_filename, [0.1, 1], ["bad type "])

    def test_bad_init_pos_names(self):
        ref_ff_data = self._make_simple_mesh_with_info()
        ref_filename = "my_mesh.json"
        export_full_field_data_to_json(ref_filename, ref_ff_data)
        field_stats = FullFieldCalculator(
            ref_filename, [0.1, 1], position_names=["X", "Y"]
        )
        with self.assertRaises(TypeError):
            field_stats = FullFieldCalculator(
                ref_filename, [0.1, 1], position_names="bad type"
            )
        with self.assertRaises(TypeError):
            field_stats = FullFieldCalculator(
                ref_filename, [0.1, 1], position_names=[1, 2]
            )

    def test_bad_init_global_vars(self):
        ref_ff_data = self._make_simple_mesh_with_info()
        ref_filename = "my_mesh.json"
        export_full_field_data_to_json(ref_filename, ref_ff_data)
        field_stats = FullFieldCalculator(
            ref_filename, [0.1, 1], add_global_variables=False
        )
        with self.assertRaises(TypeError):
            field_stats = FullFieldCalculator(
                ref_filename, [0.1, 1], add_global_variables="bad type"
            )

    def test_bad_init_reference_mesh(self):
        ref_ff_data = self._make_simple_mesh_with_info()
        ref_filename = "my_mesh.json"
        export_full_field_data_to_json(ref_filename, ref_ff_data)
        field_stats = FullFieldCalculator(ref_filename, [0.1, 1])
        with self.assertRaises(TypeError):
            field_stats = FullFieldCalculator(1, [0.1, 1])

    def test_bad_add_spatial_calc(self):
        ref_ff_data = self._make_simple_mesh_with_info()
        ref_filename = "my_mesh.json"
        export_full_field_data_to_json(ref_filename, ref_ff_data)
        field_stats = FullFieldCalculator(ref_filename, [0.1, 1])
        with self.assertRaises(TypeError):
            field_stats.add_spatial_calculation(1, subtract_function, "T")
        with self.assertRaises(TypeError):
            field_stats.add_spatial_calculation("calc", "not a func", "T")
        with self.assertRaises(TypeError):
            field_stats.add_spatial_calculation("calc", "not a func", 1, 1)
        with self.assertRaises(TypeError):
            field_stats.add_spatial_calculation("calc", "not a func", "T", 1)

    def test_initialize_reference_geometry_data(self):
        ref_ff_data = self._make_simple_mesh_with_info()
        field_stats = FullFieldCalculator(ref_ff_data, ref_ff_data["time"])

    def test_default_initialize_has_time_for_time_field(self):
        field_stats = self._make_simple_field_stats()
        goal = "time"
        self.assertTrue(field_stats.test_time_field_name(goal))

    def test_init_new_time_field_name(self):
        ref_ff_data = self._make_simple_mesh_with_info()
        new_time_name = "bc_temp"
        field_stats = self.FFCalcSpy(
            ref_ff_data, ref_ff_data["bc_temp"], independent_field_name=new_time_name
        )
        self.assertTrue(field_stats.test_time_field_name(new_time_name))

    def test_init_raise_error_if_time_field_is_not_global(self):
        ref_ff_data = self._make_simple_mesh_with_info()
        new_time_name = "T"
        with self.assertRaises(BadTimeDataShape):
            field_stats = self.FFCalcSpy(
                ref_ff_data, ref_ff_data["T"], independent_field_name=new_time_name
            )
            field_stats.add_data("d1", source_filename1)
            field_stats.add_spatial_calculation("diff", subtract_function, "T")
            field_stats.calculate_and_export("test.joblib")

    def test_propagation_of_reference_time(self):
        ref_ff_data = self._make_simple_mesh_with_info()
        field_stats = self.FFCalcSpy(ref_ff_data, ref_ff_data["time"])
        self.assertTrue(field_stats.test_reference_times(ref_ff_data["time"]))

    def test_set_mapping_parameters(self):
        ref_ff_data = self._make_simple_mesh_with_info()
        field_stats = self.FFCalcSpy(ref_ff_data, ref_ff_data["time"])
        self.assertEqual(
            field_stats._mapping_polynomial_order,
            MeshlessMapperGMLS.default_polynomial_order,
        )
        self.assertEqual(
            field_stats._mapping_search_radius_multiplier,
            MeshlessMapperGMLS.default_epsilon_multiplier,
        )
        field_stats.set_mapping_parameters(2, 2)
        self.assertEqual(field_stats._mapping_polynomial_order, 2)
        self.assertEqual(field_stats._mapping_search_radius_multiplier, 2)
        with self.assertRaises(TypeError):
            field_stats.set_mapping_parameters("a", 1)
        with self.assertRaises(TypeError):
            field_stats.set_mapping_parameters(1, "b")

    def test_add_function_to_calculate_and_show_increase_in_record(self):
        field_stats = self._make_simple_field_stats()

        self.assertEqual(len(field_stats.get_calculation_functions()["spatial"]), 0)
        field_stats.add_spatial_calculation("diff", subtract_function, "T")
        self.assertEqual(len(field_stats.get_calculation_functions()["spatial"]), 1)

    def test_add_function_to_calculate_raise_error_if_shared_name(self):
        field_stats = self._make_simple_field_stats()

        field_stats.add_spatial_calculation("diff", subtract_function, "T")
        with self.assertRaises(SameMeasurementNameError):
            field_stats.add_spatial_calculation("diff", add_function, "T")

    def _make_simple_field_stats(self):
        ref_ff_data = self._make_simple_mesh_with_info()
        ref_filename = "my_mesh.json"
        export_full_field_data_to_json(ref_filename, ref_ff_data)
        field_stats = self.FFCalcSpy(ref_filename, ref_ff_data["time"])
        return field_stats

    def test_add_examination_data_and_increase_record(self):
        field_stats = self._make_simple_field_stats()
        simple_data = self._make_simple_mesh_with_info()

        self._assert_data_size(field_stats, 0)
        field_stats.add_data("d1", simple_data)
        self._assert_data_size(field_stats, 1)
        field_stats.add_data("d2", simple_data)
        self._assert_data_size(field_stats, 2)

    def test_raise_error_if_data_has_same_name(self):
        field_stats = self._make_simple_field_stats()
        simple_data = self._make_simple_mesh_with_info()
        field_stats.add_data("d1", simple_data)
        with self.assertRaises(SameDataNameError):
            field_stats.add_data("d1", simple_data)

    def _assert_data_size(self, field_stats, expected_length):
        self.assertEqual(len(field_stats.get_data()), expected_length)

    def test_raise_error_if_export_called_without_adding_examination_files(self):
        field_stats = self._make_simple_field_stats()
        field_stats.add_spatial_calculation("diff", add_function, "T")
        export_filename = "results.json"
        with self.assertRaises(InsuffichentFieldDataSetsError):
            field_stats.calculate_and_export(export_filename)

    def test_raise_error_if_export_called_without_adding_measurements(self):
        field_stats = self._make_simple_field_stats()
        export_filename = "results.json"
        with self.assertRaises(NoMeasurementError):
            field_stats.calculate_and_export(export_filename)

    def test_bad_calc_and_export(self):
        n_time = 10
        n_pts = 20
        n_ele = 11
        ele_size = 5
        field_vars = ["A", "T"]
        global_vars = ["total_heat"]
        ref_ff_data = _make_same_polynomial_field_data(
            n_time, n_pts, n_ele, ele_size, field_vars, global_vars, 0, 0
        )
        source_filename = "source.json"
        target_filename = "target.json"
        export_full_field_data_to_json(source_filename, ref_ff_data)
        field_stats = FullFieldCalculator(
            ref_ff_data, ref_ff_data["time"], position_names=["X", "Y", "Z"]
        )

        field_stats.add_spatial_calculation("diff", subtract_function, "T")
        field_stats.add_data("d1", source_filename)
        field_stats.set_mapping_parameters(2, 2)
        with self.assertRaises(TypeError):
            field_stats.calculate_and_export(1)
        with self.assertRaises(TypeError):
            field_stats.calculate_and_export(target_filename, 1)

    def test_bad_add_data(self):
        n_time = 10
        n_pts = 20
        n_ele = 11
        ele_size = 5
        field_vars = ["A", "T"]
        global_vars = ["total_heat"]
        ref_ff_data = _make_same_polynomial_field_data(
            n_time, n_pts, n_ele, ele_size, field_vars, global_vars, 0, 0
        )
        source_filename = "source.json"
        export_full_field_data_to_json(source_filename, ref_ff_data)
        field_stats = FullFieldCalculator(
            ref_ff_data, ref_ff_data["time"], position_names=["X", "Y", "Z"]
        )

        field_stats.add_spatial_calculation("diff", subtract_function, "T")
        with self.assertRaises(TypeError):
            field_stats.add_data(1, source_filename)
        with self.assertRaises(TypeError):
            field_stats.add_data("d1", 1)

    def test_subtract_measure_produces_zero_measurement_for_constant_same_data(self):
        n_time = 10
        n_pts = 20
        n_ele = 11
        ele_size = 5
        field_vars = ["A", "T"]
        global_vars = ["total_heat"]
        ref_ff_data = _make_same_polynomial_field_data(
            n_time, n_pts, n_ele, ele_size, field_vars, global_vars, 0, 0
        )
        source_filename = "source.json"
        target_filename = "target.json"
        export_full_field_data_to_json(source_filename, ref_ff_data)
        field_stats = FullFieldCalculator(
            ref_ff_data, ref_ff_data["time"], position_names=["X", "Y", "Z"]
        )

        field_stats.add_spatial_calculation("diff", subtract_function, "T")
        field_stats.add_data("d1", source_filename)
        field_stats.set_mapping_parameters(2, 2)

        field_stats.calculate_and_export(target_filename)
        calc_data = _import_full_field_data_from_json(target_filename)

        target_field = "diff_d1_T"
        self.assertIn(target_field, list(calc_data.keys()))
        self.assert_close_arrays(
            calc_data[target_field], np.zeros_like(calc_data[target_field])
        )

    def test_global_data_in_fields(self):
        n_time = 10
        n_pts = 20
        n_ele = 11
        ele_size = 5
        field_vars = ["A", "T"]
        global_vars = ["total_heat"]
        ref_ff_data = _make_same_polynomial_field_data(
            n_time, n_pts, n_ele, ele_size, field_vars, global_vars, 0, 0
        )
        source_filename = "source.json"
        target_filename = "target.json"
        export_full_field_data_to_json(source_filename, ref_ff_data)
        field_stats = FullFieldCalculator(
            ref_ff_data, ref_ff_data["time"], position_names=["X", "Y", "Z"]
        )

        field_stats.add_spatial_calculation("diff", subtract_function, "T")
        field_stats.add_data("d1", source_filename)
        field_stats.set_mapping_parameters(2, 2)

        field_stats.calculate_and_export(target_filename)
        calc_data = _import_full_field_data_from_json(target_filename)

        target_field = "total_heat_interp_d1"
        self.assertIn(target_field, list(calc_data.keys()))

        target_field = "total_heat_interp_ref"
        self.assertIn(target_field, list(calc_data.keys()))

    def test_subtract_measure_produces_zero_measurement_for_linear_same_data(self):
        n_time = 10
        n_pts = 20
        n_ele = 11
        ele_size = 5
        field_vars = ["A", "T"]
        global_vars = ["total_heat"]
        ref_ff_data = _make_same_polynomial_field_data(
            n_time, n_pts, n_ele, ele_size, field_vars, global_vars, 1, 1
        )
        source_filename = "source.json"
        target_filename = "target.json"
        export_full_field_data_to_json(source_filename, ref_ff_data)
        field_stats = FullFieldCalculator(
            ref_ff_data, ref_ff_data["time"], position_names=["X", "Y", "Z"]
        )
        field_stats.add_spatial_calculation("diff", subtract_function, "T")
        field_stats.add_data("d1", source_filename)
        field_stats.set_mapping_parameters(2, 2)

        field_stats.calculate_and_export(target_filename)
        calc_data = _import_full_field_data_from_json(target_filename)

        target_field = "diff_d1_T"
        self.assertIn(target_field, list(calc_data.keys()))
        self.assert_close_arrays(
            calc_data[target_field], np.zeros_like(calc_data[target_field])
        )

    def test_subtract_measure_produces_zero_measurement_for_quadratic_same_data(self):
        n_time = 10
        n_pts = 20
        n_ele = 11
        ele_size = 5
        field_vars = ["A", "T"]
        global_vars = ["total_heat"]
        ref_ff_data = _make_same_polynomial_field_data(
            n_time, n_pts, n_ele, ele_size, field_vars, global_vars, 2, 2
        )
        source_filename = "source.json"
        target_filename = "target.json"
        export_full_field_data_to_json(source_filename, ref_ff_data)
        field_stats = FullFieldCalculator(
            ref_ff_data, ref_ff_data["time"], position_names=["X", "Y", "Z"]
        )
        field_stats.add_spatial_calculation("diff", subtract_function, "T")
        field_stats.add_data("d1", source_filename)
        field_stats.set_mapping_parameters(2, 2)

        field_stats.calculate_and_export(target_filename)
        calc_data = _import_full_field_data_from_json(target_filename)

        target_field = "diff_d1_T"
        self.assertIn(target_field, list(calc_data.keys()))
        self.assert_close_arrays(
            calc_data[target_field], np.zeros_like(calc_data[target_field])
        )

    def test_subtract_measure_produces_zero_measurement_for_random_same_data(self):
        n_time = 10
        n_pts = 20
        n_ele = 11
        ele_size = 5
        field_vars = ["A", "T"]
        global_vars = ["total_heat"]
        ref_ff_data = self._make_random_polynomial_field_data(
            n_time, n_pts, n_ele, ele_size, field_vars, global_vars, 3
        )
        source_filename = "source.json"
        target_filename = "target.json"
        export_full_field_data_to_json(source_filename, ref_ff_data)
        field_stats = FullFieldCalculator(
            ref_ff_data, ref_ff_data["time"], position_names=["X", "Y", "Z"]
        )
        field_stats.add_spatial_calculation("diff", subtract_function, "T", "A")
        field_stats.add_data("d1", source_filename)
        field_stats.set_mapping_parameters(2, 2)
        field_stats.calculate_and_export(target_filename)
        calc_data = _import_full_field_data_from_json(target_filename)

        target_field = "diff_d1_T"
        self.assertIn(target_field, list(calc_data.keys()))
        self.assert_close_arrays(
            calc_data[target_field], np.zeros_like(calc_data[target_field])
        )

        target_field = "diff_d1_A"
        self.assertIn(target_field, list(calc_data.keys()))
        self.assert_close_arrays(
            calc_data[target_field], np.zeros_like(calc_data[target_field])
        )

    def test_subtract_measure_produces_zero_measurement_for_same_quadratic_data_different_instance_space(
        self,
    ):
        n_time = 10
        n_pts = 20
        n_ele = 11
        ele_size = 5
        field_vars = ["A", "T"]
        global_vars = ["total_heat"]
        ref_ff_data = _make_same_polynomial_field_data(
            n_time, n_pts, n_ele, ele_size, field_vars, global_vars, 2, 2
        )
        test_ff_data = _make_same_polynomial_field_data(
            n_time, n_pts, n_ele, ele_size, field_vars, global_vars, 2, 2
        )

        source_filename = "source.json"
        target_filename = "target.json"
        export_full_field_data_to_json(source_filename, test_ff_data)
        field_stats = FullFieldCalculator(
            ref_ff_data, ref_ff_data["time"], position_names=["X", "Y", "Z"]
        )
        field_stats.add_spatial_calculation("diff", subtract_function, "T")
        field_stats.add_data("d1", source_filename)
        field_stats.set_mapping_parameters(2, 2)

        field_stats.calculate_and_export(target_filename)
        calc_data = _import_full_field_data_from_json(target_filename)

        target_field = "diff_d1_T"
        self.assertIn(target_field, list(calc_data.keys()))
        self.assert_close_arrays(
            calc_data[target_field],
            np.zeros_like(calc_data[target_field]),
            show_arrays=False,
        )

    def test_subtract_measure_produces_zero_measurement_for_same_quad_lin_data_different_instance_space_and_time(
        self,
    ):
        n_time = 10
        n_pts = 20
        n_ele = 11
        ele_size = 5
        field_vars = ["A", "T"]
        global_vars = ["total_heat"]
        ref_ff_data = _make_same_polynomial_field_data(
            n_time, n_pts, n_ele, ele_size, field_vars, global_vars, 2, 1
        )
        test_ff_data = _make_same_polynomial_field_data(
            n_time * 2, n_pts, n_ele, ele_size, field_vars, global_vars, 2, 1
        )

        source_filename = "source.json"
        target_filename = "target.json"
        export_full_field_data_to_json(source_filename, test_ff_data)
        field_stats = FullFieldCalculator(
            ref_ff_data, ref_ff_data["time"], position_names=["X", "Y", "Z"]
        )
        field_stats.add_spatial_calculation("diff", subtract_function, "T")
        field_stats.add_data("d1", source_filename)
        field_stats.set_mapping_parameters(2, 2)

        field_stats.calculate_and_export(target_filename)
        calc_data = _import_full_field_data_from_json(target_filename)

        target_field = "diff_d1_T"
        self.assertIn(target_field, list(calc_data.keys()))
        self.assert_close_arrays(
            calc_data[target_field],
            np.zeros_like(calc_data[target_field]),
            show_arrays=False,
        )

    def test_subtract_makes_0_for_higher_polys(self):
        n_time = 10
        n_pts = 20
        n_ele = 11
        ele_size = 5
        field_vars = ["A", "T"]
        global_vars = ["total_heat"]
        ref_ff_data = _make_same_polynomial_field_data(
            n_time, n_pts, n_ele, ele_size, field_vars, global_vars, 3, 1
        )
        test_ff_data = _make_same_polynomial_field_data(
            n_time * 2, n_pts * 2, n_ele, ele_size, field_vars, global_vars, 3, 1
        )

        source_filename = "source.json"
        target_filename = "target.json"
        export_full_field_data_to_json(source_filename, test_ff_data)

        field_stats = FullFieldCalculator(
            ref_ff_data, ref_ff_data["time"], position_names=["X", "Y", "Z"]
        )
        field_stats.add_spatial_calculation("diff", subtract_function, "T")
        field_stats.add_data("d1", source_filename)
        field_stats.set_mapping_parameters(3, 1.5)

        field_stats.calculate_and_export(target_filename)
        calc_data = _import_full_field_data_from_json(target_filename)

        target_field = "diff_d1_T"
        self.assertIn(target_field, list(calc_data.keys()))
        self.assert_close_arrays(
            calc_data[target_field],
            np.zeros_like(calc_data[target_field]),
            show_arrays=False,
        )

    def test_two_functions_and_two_data_sets(self):
        n_time = 10
        n_pts = 20
        n_ele = 11
        ele_size = 5
        field_vars = ["A", "T"]
        global_vars = ["total_heat"]
        ref_ff_data = _make_same_polynomial_field_data(
            n_time, n_pts, n_ele, ele_size, field_vars, global_vars, 3, 2
        )
        ref_ff_data = _make_same_polynomial_field_data(
            n_time, n_pts, n_ele, ele_size, field_vars, global_vars, 3, 1
        )
        test_ff_data1 = _make_same_polynomial_field_data(
            n_time * 2, n_pts * 2, n_ele, ele_size, field_vars, global_vars, 3, 1
        )
        test_ff_data2 = _make_same_polynomial_field_data(
            n_time * 3, n_pts * 2, n_ele, ele_size, field_vars, global_vars, 3, 1
        )

        source_filename1 = "source1.json"
        source_filename2 = "source2.json"
        target_filename = "target.json"
        export_full_field_data_to_json(source_filename1, test_ff_data1)
        export_full_field_data_to_json(source_filename2, test_ff_data2)

        field_stats = FullFieldCalculator(
            ref_ff_data, ref_ff_data["time"], position_names=["X", "Y", "Z"]
        )
        field_stats.add_spatial_calculation("diff", subtract_function, "T")
        field_stats.add_spatial_calculation("sum", add_function, "T")
        field_stats.add_data("d1", source_filename1)
        field_stats.add_data("d2", source_filename2)
        field_stats.set_mapping_parameters(3, 1.5)

        field_stats.calculate_and_export(target_filename)

        calc_data = _import_full_field_data_from_json(target_filename)

        goal_val = np.zeros_like(calc_data["diff_d1_T"])
        self._assert_in_and_val(calc_data, "diff_d1_T", goal_val)
        self._assert_in_and_val(calc_data, "diff_d2_T", goal_val)

        goal_val = ref_ff_data["T"] * 2
        self._assert_in_and_val(calc_data, "sum_d1_T", goal_val)
        self._assert_in_and_val(calc_data, "sum_d2_T", goal_val)

        goal_val = ref_ff_data["T"]
        self._assert_in_and_val(calc_data, "T_interp_ref", goal_val)
        self._assert_in_and_val(calc_data, "T_interp_d1", goal_val)
        self._assert_in_and_val(calc_data, "T_interp_d2", goal_val)

    def _make_random_polynomial_field_data(
        self,
        n_timesteps,
        n_points,
        n_ele,
        ele_size,
        field_var_names,
        global_var_names,
        max_poly,
    ):
        time, x, y, z, data_dict = _make_basics(n_timesteps, n_points, n_ele, ele_size)
        for var in field_var_names:
            data_dict[var] = poly(
                x,
                y,
                z,
                time,
                r_int(max_poly),
                r_int(max_poly),
                r_int(max_poly),
                r_int(max_poly),
                r_int(max_poly),
                r_int(max_poly),
            )
        for var in global_var_names:
            data_dict[var] = np.random.uniform(0, 100, [n_timesteps])
        return convert_dictionary_to_field_data(
            data_dict, ["x", "y", "z"], "connectivity"
        )


def _make_same_polynomial_field_data(
    n_timesteps,
    n_points,
    n_ele,
    ele_size,
    field_var_names,
    global_var_names,
    n_space,
    n_time,
):
    time, x, y, z, data_dict = _make_basics(n_timesteps, n_points, n_ele, ele_size)
    max_poly = 4
    for var in field_var_names:
        data_dict[var] = poly(
            x, y, z, time, n_space, n_space, n_space, n_time, n_time, n_time
        )
    for var in global_var_names:
        data_dict[var] = np.random.uniform(0, 100, [n_timesteps])
    return convert_dictionary_to_field_data(data_dict, ["x", "y", "z"], "connectivity")


def _make_basics(n_time, n_points, n_ele, ele_size):
    time = np.linspace(0, 5, n_time)
    x = np.sort(np.random.uniform(-1, 1, n_points))
    y = np.random.uniform(-1, 1, n_points)
    z = np.random.uniform(-1, 1, n_points)

    data_dict = {"time": time, "x": x, "y": y, "z": z}
    data_dict["connectivity"] = np.random.randint(0, n_points, [n_ele, ele_size])
    return time, x, y, z, data_dict


def poly(x, y, z, t, nx, ny, nz, nxt, nyt, nzt):
    val = np.power(np.outer(x, np.ones_like(t)), nx)
    val += np.power(np.outer(y, np.ones_like(t)), ny)
    val += np.power(np.outer(z, np.ones_like(t)), nz)
    val += np.power(np.outer(x, t), nxt)
    val += np.power(np.outer(y, t), nyt)
    val += np.power(np.outer(z, t), nzt)
    return val.T


def r_int(top):
    return np.random.randint(0, top)
