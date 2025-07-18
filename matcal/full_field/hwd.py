from abc import ABC, abstractmethod
from matcal.full_field import shapes, trees 
import numpy as np


class HWDGeneratorBase(ABC):

    @abstractmethod
    def _set_up(self, max_tree_depth, scale_inputs=True):
        pass

    def __init__(self, polynomial_order, max_tree_depth=5, scale_inputs=True, return_m=False):
        self._warn_for_tree_depth(max_tree_depth)
        self._return_m = return_m
        self._momentum_matrix_generator = None
        self._Q = None
        self._R = None
        self._M = None
        self._set_up(polynomial_order, max_tree_depth, scale_inputs)

    def create_basis(self, locations):
        M = self._build_M(locations)
        if M.shape[0] < M.shape[1]:
            raise self.ExcessiveWaveletsError(M.shape)
        self._M = M
        self._Q, self._R = np.linalg.qr(self._M)

    def _build_M(self, locations):
        if self._momentum_matrix_generator.is_built:
            M = self._momentum_matrix_generator.populate(locations)
        else:
            M = self._momentum_matrix_generator.build_and_populate(locations)
        return M

    def map_data(self, data, locations=None, R_ref=None):
        if locations is not None:
            self.create_basis(locations)
        c = self._current_basis_map(data)
        c = self._reference_basis_map(R_ref, c)
        return c

    def _reference_basis_map(self, R_ref, c):
        if R_ref is not None:
            c = np.dot(R_ref, np.linalg.solve(self._R, c))
        return c

    def _current_basis_map(self, data):
        c = np.dot(self._Q.T, data)
        return c
 
    def _warn_for_tree_depth(self, max_tree_depth):
        if max_tree_depth > 8:
            message = "\nWarning: Tree depths above 8 are expensive."
            message += f"\nCurrent Depth {max_tree_depth}, which as {2**max_tree_depth} number of subsections"
            print(message)

    def get_basis(self):
        if self._return_m:
            return self._Q, self._R, self._M
        else:
            return self._Q, self._R
        
    class ExcessiveWaveletsError(RuntimeError):

        def __init__(self, m_shape):
            message = f"\nNumber of wavelets({m_shape[1]}) exceeeds that of the number of points({m_shape[0]})."
            message += f"\nDecrease depth or polynomial order"
            super().__init__(message)


class RO_HWDGeneratorBase(ABC):

    @abstractmethod
    def _set_up(self, max_tree_depth, scale_inputs=True):
        pass

    def __init__(self, polynomial_order, max_tree_depth=5, reduction_ratio = 1e-2, scale_inputs=True):
        self._warn_for_tree_depth(max_tree_depth)
        self._momentum_matrix_generator = None
        self._reduction_ratio = reduction_ratio
        self._Q = None
        self._set_up(polynomial_order, max_tree_depth, scale_inputs)

    def create_basis(self, locations):
        M = self._build_M(locations)
        if M.shape[0] < M.shape[1]:
            raise self.ExcessiveWaveletsError(M.shape)
        self._Q, _ = np.linalg.qr(M)

    def _build_M(self, locations):
        if self._momentum_matrix_generator.is_built:
            M = self._momentum_matrix_generator.populate(locations)
        else:
            M = self._momentum_matrix_generator.build_and_populate(locations)
        return M

    def build_compressed_space(self, locations, data):
        self.create_basis(locations)
        c = self._current_basis_map(data)
        c = np.abs(c)
        max_c = np.max(c)
        thresh = max_c * self._reduction_ratio
        keep_modes = np.argwhere(c >= thresh).flatten()
        self._Q = self._Q[:, keep_modes]


    def map_data(self, data):
        c = self._current_basis_map(data)
        return c


    def _current_basis_map(self, data):
        c = np.dot(self._Q.T, data)
        return c
 
    def _warn_for_tree_depth(self, max_tree_depth):
        if max_tree_depth > 8:
            message = "\nWarning: Tree depths above 8 are expensive."
            message += f"\nCurrent Depth {max_tree_depth}, which as {2**max_tree_depth} number of subsections"
            print(message)
        
    class ExcessiveWaveletsError(RuntimeError):

        def __init__(self, m_shape):
            message = f"\nNumber of wavelets({m_shape[1]}) exceeeds that of the number of points({m_shape[0]})."
            message += f"\nDecrease depth or polynomial order"
            super().__init__(message)


class MomentMatrixGeneratorBase(ABC):
    
    @abstractmethod
    def _set_up(self):
        pass

    def __init__(self, max_tree_depth=5, scale_inputs=True):
        self._check_inputs(max_tree_depth, scale_inputs)
        self._max_tree_depth = max_tree_depth
        self._use_scaling = scale_inputs
        self._active_side_index = 0
        self._min_cluster_size = None
        self._wavelets_per_domain = None
        self._evaluate_shape = None
        self._tree = None
        self._scaling_tool = LinearConditioner()

    @property
    def is_built(self):
        return not self._tree is None

    def build_and_populate(self, passed_locations):
        self._overwrite_tree_warning()
        self._scaling_tool.fit(passed_locations)
        locations = self._process_locations(passed_locations)
        self._tree, cluster_results = trees.initialize_cluster_tree(locations, self._min_cluster_size, self._max_tree_depth)
        return self._populate_impl(locations, cluster_results)

    def populate(self, passed_locations):        
        locations = self._process_locations(passed_locations)
        cluster_results = trees.cluster_from_tree(locations, self._tree, self._min_cluster_size)
        return self._populate_impl(locations, cluster_results)

    def _process_locations(self, passed_locations):
        if self._use_scaling:
           locations = self._scaling_tool.condition(passed_locations)
        else:
           locations = passed_locations
        return locations



    def _populate_impl(self, locations, cluster_results):
        active_indices = self._parse_active_domains(cluster_results)
        n_active_domains = len(active_indices)
        total_wavelet_count = n_active_domains * self._wavelets_per_domain
        M = np.zeros([locations.shape[0], total_wavelet_count])
        for wave_group_index, domain_index in enumerate(active_indices):
            sub_population = cluster_results.clusters[domain_index]
            sub_locations = locations[sub_population, :]
            wave_index_end, wave_index_start = self._get_start_and_end(wave_group_index)
            M[sub_population, wave_index_start:wave_index_end] = self._evaluate_shape(sub_locations)
        return M

    def _get_start_and_end(self, wave_group_index):
        wave_index_end = (wave_group_index + 1) * self._wavelets_per_domain
        wave_index_start = (wave_group_index) * self._wavelets_per_domain
        return wave_index_end,wave_index_start
    
    def _parse_active_domains(self, cluster_results):
        return np.argwhere(cluster_results.side_index==self._active_side_index).flatten()

    def _overwrite_tree_warning(self):
        if self._tree is not None:
            print("WARNING:: Overwriting existing spatial decomposition")
            print("     If same decomposition was intended to be used use the method 'populate'.")
        
    def _check_inputs(self, max_tree_depth, scale_inputs):
        if max_tree_depth < 0  or not isinstance(max_tree_depth, int):
           raise self.NonCountingNumberTreeDepthError()
        if not isinstance(scale_inputs, bool):
           raise self.NonBoolianScaleInputsFlagError()

    class NonCountingNumberTreeDepthError(RuntimeError):
        pass

    class NonBoolianScaleInputsFlagError(RuntimeError):
        pass

class PolynomialMomentMatrixGeneratorBase(MomentMatrixGeneratorBase):

    @abstractmethod
    def _set_up(self, polynomial_order):
        pass

    def __init__(self, polynomial_order, max_tree_depth=5, scale_inputs=True):
        super().__init__(max_tree_depth, scale_inputs)
        if not isinstance(polynomial_order, int):
           raise self.NonIntegerPolynomialOrderError()
        self._set_up(polynomial_order)
    
    class NonIntegerPolynomialOrderError(RuntimeError):
        pass


class ConditionerBase(ABC):
    
    @abstractmethod
    def fit(self, x):
        pass

    @abstractmethod
    def condition(self, x):
        pass

    def fit_and_condition(self, x):
       self.fit(x)
       return self.condition(x)

class LinearConditioner(ConditionerBase):
    
    def __init__(self):
        self._offsets = None
        self._multipliers = None

    def fit(self, x):
        if self._offsets is not None:
            print("WARNING: Overwriting conditioning data, use 'condition' to avoid overwriting")
        x_min = np.min(x, axis=0)
        x_max = np.max(x, axis=0)
        delta = x_max - x_min
        self._offsets = x_min
        self._multipliers = 2 / delta

    def condition(self, x):
        return np.multiply(self._multipliers, (x - self._offsets)) - 1




##### Final Polynomial Classes


class PolynomialMomentMatrixGeneratorOneD(PolynomialMomentMatrixGeneratorBase):
   
    def _set_up(self, polynomial_order):
       self._min_cluster_size = polynomial_order + 1
       self._wavelets_per_domain = polynomial_order + 1
       self._evaluate_shape = shapes.GlobalNaivePolynomialCalculatorOneD(polynomial_order).calculate

class PolynomialMomentMatrixGeneratorTwoD(PolynomialMomentMatrixGeneratorBase):

    def _set_up(self, polynomial_order):
        self._min_cluster_size = shapes.entries_count_for_full_two_dim_polynomial(polynomial_order)
        self._wavelets_per_domain = self._min_cluster_size
        self._evaluate_shape = shapes.PolynomialCalculatorTwoD(polynomial_order).calculate

class NoCrossPolynomialMomentMatrixGeneratorTwoD(PolynomialMomentMatrixGeneratorBase):

    def _set_up(self, polynomial_order):
        n_dim = 2
        self._min_cluster_size = shapes.entries_count_for_no_cross_ndim_polynomial(polynomial_order, n_dim)
        self._wavelets_per_domain = self._min_cluster_size
        self._evaluate_shape = shapes.NoCrossPolynomialCalculatorTwoD(polynomial_order).calculate

class ReducedPolynomialMomentMatrixGeneratorTwoD(PolynomialMomentMatrixGeneratorBase):
    def _set_up(self, polynomial_order):
        n_dim = 2
        self._min_cluster_size = shapes.entries_count_for_mid_and_edge_polynomial(polynomial_order)
        self._wavelets_per_domain = self._min_cluster_size
        self._evaluate_shape = shapes.ReducedPolynomialCaclulatorTwoD(polynomial_order).calculate


class PolynomialMomentMatrixGeneratorThreeD(PolynomialMomentMatrixGeneratorBase):

    def _set_up(self, polynomial_order):
        self._min_cluster_size = shapes.entries_count_for_full_three_dim_polynomial(polynomial_order)
        self._wavelets_per_domain = self._min_cluster_size
        self._evaluate_shape = shapes.PolynomialCalculatorThreeD(polynomial_order).calculat


class OneDPolynomialHWD(HWDGeneratorBase):
    def _set_up(self, polynomial_order, max_tree_depth, scale_inputs=True):
        self._momentum_matrix_generator = PolynomialMomentMatrixGeneratorOneD(polynomial_order,max_tree_depth, scale_inputs)

class TwoDPolynomialHWD(HWDGeneratorBase):
    def _set_up(self, polynomial_order, max_tree_depth, scale_inputs=True):
        self._momentum_matrix_generator = PolynomialMomentMatrixGeneratorTwoD(polynomial_order,max_tree_depth, scale_inputs)

class NoCrossTwoDPolynomialHWD(HWDGeneratorBase):
    def _set_up(self, polynomial_order, max_tree_depth, scale_inputs=True):
        self._momentum_matrix_generator = NoCrossPolynomialMomentMatrixGeneratorTwoD(polynomial_order,max_tree_depth, scale_inputs)

class ReducedTwoDPolynomialHWD(HWDGeneratorBase):
    def _set_up(self, polynomial_order, max_tree_depth, scale_inputs=True):
        self._momentum_matrix_generator = ReducedPolynomialMomentMatrixGeneratorTwoD(polynomial_order,max_tree_depth, scale_inputs)

class ReducedTwoDPolynomialROHWD(RO_HWDGeneratorBase):
    def _set_up(self, polynomial_order, max_tree_depth, scale_inputs=True):
        self._momentum_matrix_generator = ReducedPolynomialMomentMatrixGeneratorTwoD(polynomial_order,max_tree_depth, scale_inputs)

class ThreeDPolynomialHWD(HWDGeneratorBase):
    def _set_up(self, polynomial_order, max_tree_depth, scale_inputs=True):
        self._momentum_matrix_generator = PolynomialMomentMatrixGeneratorThreeD(polynomial_order,max_tree_depth, scale_inputs)


