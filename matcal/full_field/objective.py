"""
The objective module contains the classes related to objectives. 
This includes the metric functions, the base objectives, specialized 
objectives, objective sets. User facing classes only include metric functions
and objectives.
"""

from matcal.core.objective import Objective
from matcal.core.qoi_extractor import (DataSpecificExtractorWrapper, 
                                       ReferenceKeyedExtractorWrapper,
                                       StateSpecificExtractorWrapper)
from matcal.core.utilities import check_item_is_correct_type

from matcal.full_field.data_importer import mesh_file_to_skeleton
from matcal.full_field.qoi_extractor import (ExternalVirtualPowerExtractor, 
                                             FieldTimeInterpolatorExtractor, 
                                             FlattenFieldDataExtractor,
                                             HWDColocatingExperimentSurfaceExtractor, 
                                             HWDExperimentSurfaceExtractor, 
                                             HWDPolynomialSimulationSurfaceExtractor, 
                                             InternalVirtualPowerExtractor, 
     MeshlessSimToExpInterpolatorExtractor,
     MeshlessSpaceInterpolatorExtractor)
from matcal.full_field.TwoDimensionalFieldGrid import MeshSkeleton


from matcal.core.logger import initialize_matcal_logger
logger = initialize_matcal_logger(__name__)


class InterpolatedFullFieldObjective(Objective):
    """Interpolated full-field objective for calibration studies.

    Compares simulation and experimental point-cloud data extracted from
    2D surfaces using GMLS spatial interpolation and linear time
    interpolation.

    Two modes of operation are supported:

    1. **Mesh-file mode** (``fem_mesh_file`` is a path string):
       The simulation target grid is derived from the mesh file at
       construction time.  Experimental data is interpolated onto the
       mesh grid (exp → sim direction), and simulation data is
       node-mapped/time-interpolated for comparison.

    2. **Deferred mode** (``fem_mesh_file=None``):
       No mesh file is required.  The simulation coordinate grid is
       discovered from the first simulation result.  Simulation data is
       interpolated onto the experiment grid (sim → exp direction) with
       a lazily-constructed GMLS mapper. This eliminates the need for a
       pre-existing mesh file and avoids node-remapping issues.
    """

    _class_name = "InterpolatedFullFieldObjective"

    def __init__(
        self,
        fem_mesh_file=None,
        *dependent_fields: str,
        time_variable: str = "time",
        fem_surface: str | None = None,
    ) -> None:
        """Initialise the interpolated full-field objective.

        Parameters
        ----------
        fem_mesh_file : str or None
            Path to the mesh JSON file used for simulation, or ``None``
            to defer initialisation until the first simulation result is
            available.
        *dependent_fields : str
            Names of the dependent field(s) of interest (e.g. ``"T"``).
        time_variable : str
            Name of the time field parameterising frames of data.
        fem_surface : str or None
            Surface name to extract from the mesh when the simulation
            returns more than just the required surface.  Only used in
            mesh-file mode.
        """
        super().__init__(*dependent_fields)
        self._mesh_file = fem_mesh_file
        self._mesh_surface = fem_surface
        self._time_variable = time_variable
        self._polynomial_order = 1
        self._search_radius_multiplier = 2.75

        if fem_mesh_file is not None:
            self._deferred = False
            mesh_skeleton = mesh_file_to_skeleton(fem_mesh_file, fem_surface)
            sim_qoi_extractor = FieldTimeInterpolatorExtractor(
                mesh_skeleton, time_variable
            )
            self.set_simulation_qoi_extractor(sim_qoi_extractor)
        else:
            self._deferred = True
            # Simulation extractor will be created in
            # data_specific_initialization once experiment coords are known.

        self.set_as_large_data_sets_objective()

    def set_interpolation_parameters(
        self, polynomial_order: int, search_radius_multiplier: float
    ) -> None:
        """Set GMLS interpolation meta-parameters.

        It is recommended that if much extrapolation is expected, the
        polynomial order be kept low (<=2).

        Parameters
        ----------
        polynomial_order : int
            Polynomial order used by GMLS.
        search_radius_multiplier : float
            Multiplier past the minimum required neighbours.  Must be
            greater than 1.  Larger values smooth the interpolation;
            smaller values allow sharper changes.
        """
        self._polynomial_order = polynomial_order
        self._search_radius_multiplier = search_radius_multiplier

    def data_specific_initialization(self, exp_data_collection) -> None:
        """Incorporate experimental data into the objective.

        In mesh-file mode this creates per-data-set GMLS extractors that
        interpolate experiment data onto the simulation mesh.

        In deferred mode (no mesh file) this creates a
        :class:`~matcal.full_field.qoi_extractor.MeshlessSimToExpInterpolatorExtractor`
        as the simulation extractor and a simple flatten extractor for
        the experiment side.

        Parameters
        ----------
        exp_data_collection : DataCollection
            Experimental data that this objective will evaluate.
        """
        if self._deferred:
            self._init_deferred_extractors(exp_data_collection)
        else:
            self._init_mesh_file_extractors(exp_data_collection)

    # ------------------------------------------------------------------
    # Mesh-file mode helpers (original behaviour)
    # ------------------------------------------------------------------

    def _init_mesh_file_extractors(self, exp_data_collection) -> None:
        """Create experiment extractors that interpolate exp → sim grid."""
        exp_extractor = DataSpecificExtractorWrapper()
        for state in exp_data_collection:
            for current_data in exp_data_collection[state]:
                field_extractor = self._create_field_extractor(
                    self._mesh_file,
                    self._mesh_surface,
                    current_data.skeleton,
                    self._time_variable,
                )
                exp_extractor.add(current_data, field_extractor)
        self.set_experiment_qoi_extractor(exp_extractor)

    def _create_field_extractor(
        self, fem_mesh_file, fem_surface, cloud_skeleton, time_variable
    ):
        fem_skeleton = mesh_file_to_skeleton(fem_mesh_file, fem_surface)
        field_extractor = MeshlessSpaceInterpolatorExtractor(
            cloud_skeleton.spatial_coords[:, :2],
            fem_skeleton.spatial_coords[:, :2],
            time_variable,
            self._polynomial_order,
            self._search_radius_multiplier,
        )
        return field_extractor

    # ------------------------------------------------------------------
    # Deferred mode helpers (no mesh file)
    # ------------------------------------------------------------------

    def _init_deferred_extractors(self, exp_data_collection) -> None:
        """Set up extractors for deferred (sim→exp) interpolation.

        Experiment extractor: flattens experiment data (stays on exp grid).
        Simulation extractor: builds GMLS (sim → exp) on first eval.
        """
        # Experiment side: pass-through flatten
        self.set_experiment_qoi_extractor(FlattenFieldDataExtractor())

        # Simulation side: one extractor per experiment data item
        sim_extractor = ReferenceKeyedExtractorWrapper()
        for state in exp_data_collection:
            for current_data in exp_data_collection[state]:
                exp_coords = current_data.skeleton.spatial_coords[:, :2]
                extractor = MeshlessSimToExpInterpolatorExtractor(
                    target_coords=exp_coords,
                    time_field=self._time_variable,
                    poly_order=self._polynomial_order,
                    search_radius_multiplier=self._search_radius_multiplier,
                )
                sim_extractor.add(current_data, extractor)
        self.set_simulation_qoi_extractor(sim_extractor)


class PolynomialHWDObjective(Objective):
    """
    The PolynomialHWDObjective class handles the calculation of 
    the residual vector and merit functions when
    comparing point data extracted from 2D surfaces for a study evaluation set. 
    See :meth:`~matcal.core.study_base.StudyBase.add_evaluation_set`.

    PolynomialHWDObjective defines its objective in a latent space. 
    The latent space represents the raw point data 
    by weights of different basis modes. These modes represent different 
    patterns the data can take over
    different parts of the domain. This method can support compression of 
    the full-field data, allowing it to have 
    a much smaller memory foot print than may be required by other methods. 

    These modes are formed by performing a QR factorization on a moment matrix. 
    This factorization provides a basis matrix(Q)
    and a change of basis matrix(R). These matrices are used to convert the raw 
    data tot the latent space and to map different latent spaces to each other. 

    This method generates its moment matrix by generating polynomials across 
    different tiers of subdividing the spatial domain. 
    each tier is a binary split of the last tier, creating a domain decomposition 
    tree, where depth d of the tree has 2^d subdivisions within it. 
    The deeper the tree the more local information can be captured by the objective.

    Currently the default behavior for this objective involves a colocation 
    step of the experimental data to the simulation data points. 
    This step may not be necessary in the future as partitioning 
    methods for HWD become better.
    """
    _class_name = "PolynomialHWDObjective"
    def __init__(self, target_coords,  *dependent_fields,  time_variable:str="time", max_depth:int=6, polynomial_order:int=8):        
        """
        Specify the geometry and fields of interest for the full-filed objective.
        For full-field objectives it is expected that each frame 
        of data is parameterized by an independent time field. 
        While it does not need to be called time, some field must 
        characterize the order of each frame of full-field data. 
        A frame of data consists of the field values 
        (temperature, displacement, etc.) of all the 
        relevant points for the data set.
        This method allows the user to specify these fields for the objective. 
        This objective will use polynomial HWD to compare 
        experimental data and simulation data in a latent space.
        To align the data points in time the simulation
        data will interpolated in time to match the experimental time stamps. 

        There are two meta-parameters that define the latent 
        space definition, polynomial order and maximum tree depth.
        polynomial order changes the polynomial order used to 
        generate the moment matrix, and the maximum tree depth dictates how
        deep of a binary tree can be generated for a given domain. 

        :param target_coords: two-dimensional array containing 
            the points to be colocated to to generate a 
            more consistent set of basis modes. If None is passed 
            colocation will be skipped. non-colocated HWD 
            is still in beta at this time an may have unpredictable performance. 
        :type independent_field: numpy.array or None

        :param dependent_fields: the dependent fields for the objective.
        :type dependent_fields: list(str)

        :param time_variable: The name of the time field 
            used to parameterize the frames of full-field data
        :type time_variable: str

        :param max_depth: Specify the maximum depth of binary tree to make. 
            Value must be 0 or greater.
            Note that the number of subdivisions grow geometrically, 
            and that depths greater than 7 can 
            take a long time to process. 
        :type fem_surface: int

        :param polynomial_order: Specify the order of polynomial used 
            to generate the moment matrix. value must be 0 or greater.
        :type polynomial_order: int
        """
        super().__init__(*dependent_fields)
        self._max_depth = max_depth
        self._polynomial_order = polynomial_order
        self._time_variable = time_variable
        self._surface_name = None
        self._target_coords = None
        self._colocate_exp_data = False
        if target_coords is None:
            logger.info("WARNING:: Using not co-located HWD, this is pre-beta. "
                        "Pass a mesh file to use co-location.")
        else:
            self._set_colocation(target_coords, max_depth=max_depth, 
                                 polynomial_order=polynomial_order)
        self.set_as_large_data_sets_objective()

    def _set_colocation(self, target_coords, max_depth:int=6, 
                        polynomial_order:int=8):
        self._target_coords = target_coords
        self._polynomial_order = polynomial_order
        self._max_depth  = max_depth
        self._colocate_exp_data = True

    def data_specific_initialization(self, exp_data_collection):
        """
        Public method used by this class to correctly incorporate experimental 
        data into the initialization of the objective. 
        This is a method meant for use inside of MatCal and 
        is not intended to be used by users.

        :param exp_data_collection: the :class:`~matcal.core.data.DataCollection` 
            containing the relevant experimental data that this 
            objective will evaluate. 
        :type exp_data_collection: :class:`~matcal.core.data.DataCollection`
        """
        sim_extractor = StateSpecificExtractorWrapper()
        exp_extractor = StateSpecificExtractorWrapper()
        for state in exp_data_collection.states:
            first_data = exp_data_collection[state][0]
            if self._colocate_exp_data:
                state_extractors = self._set_up_colocated(first_data)
            else:
                state_extractors = self._set_up_mapping(first_data)
            state_sim_field_extractor, state_exp_field_extractor = state_extractors
            sim_extractor.add(first_data, state_sim_field_extractor)
            exp_extractor.add(first_data, state_exp_field_extractor)
        self.set_simulation_qoi_extractor(sim_extractor)
        self.set_experiment_qoi_extractor(exp_extractor)

    def _set_up_mapping(self, first_data):
        state_sim_field_extractor = HWDPolynomialSimulationSurfaceExtractor(first_data.skeleton, 
                                                                            self._max_depth, 
                                                                            self._polynomial_order, 
                                                                            self._time_variable)
        self._add_surface(state_sim_field_extractor)
        state_exp_field_extractor = HWDExperimentSurfaceExtractor(state_sim_field_extractor)
        return state_sim_field_extractor,state_exp_field_extractor

    def _set_up_colocated(self, first_data):
        sim_skeleton = self._get_simulation_points()
        state_sim_field_extractor = HWDPolynomialSimulationSurfaceExtractor(sim_skeleton, 
                                                                            self._max_depth, 
                                                                            self._polynomial_order, 
                                                                            self._time_variable)
        self._add_surface(state_sim_field_extractor)
        state_exp_field_extractor = HWDColocatingExperimentSurfaceExtractor(state_sim_field_extractor,
                                                                            first_data.skeleton.spatial_coords[:,:2], 
                                                                            sim_skeleton.spatial_coords[:,:2])
        return state_sim_field_extractor, state_exp_field_extractor

    def _add_surface(self, state_sim_field_extractor):
        if self._surface_name is not None:
            state_sim_field_extractor.extract_cloud_from_mesh(self._surface_name)

    def _get_simulation_points(self):
        if not isinstance(self._target_coords, str):
            sim_skeleton = MeshSkeleton()
            if self._surface_name is not None:
                sp = self._target_coords.surfaces[self._surface_name]
                sim_skeleton.spatial_coords = self._target_coords.spatial_coords[sp, :]
            else:
                sim_skeleton.spatial_coords = self._target_coords.spatial_coords
            
        else:
            sim_skeleton = mesh_file_to_skeleton(self._target_coords, self._surface_name)
        return sim_skeleton

    def extract_data_from_mesh_surface(self, surface_name):
        self._surface_name = surface_name


class MechanicalVFMObjective(Objective):
    """
    The MechanicalVFMObjective class handles the calculation of 
    the residual vector and merit functions when
    calibrating a material model using the :ref:`Virtual Fields Method`
    and full-field displacement data from 
    an experiment. It must be combined with one of MatCal's
    VFM models which require a :class:`~matcal.full_field.data.FieldData` class
    object for the boundary condition data. 
    The data passed to :meth:`~matcal.core.study_base.StudyBase.add_evaluation_set`
    along with this type of objective can be of type :class:`~matcal.full_field.data.FieldData`
    or :class:`~matcal.core.data.Data` since the external virtual power 
    calculation only uses global data.
    For the internal virtual power calculation, the virtual field 
    functions used are

    .. math::
        
        \\mathbf{v}^*_X=\\cos\\frac{\\pi\\bar{Y}}{h}


    .. math::
        
        \\mathbf{v}^*_Y=\\frac{2\\bar{Y}+h}{2h}

    where :math:`Y` is the direction of loading, 
    :math:`\\bar{Y}` is the centered position 
    of the current point of interest 
    in the reference configuration, 
    and :math:`h` is the total height of 
    the data.

    """
    _class_name = "MechanicalVFMObjective"

    def __init__(self, time_field="time", load_field="load"):
        """
        Optionally specify the time and load field names
        that are required to be in the experiment :class:`~matcal.full_field.data.FieldData`
        class for this objective.
        These are assumed to be "time" and "load" by default. 

        :param time_field: the name of the time field in the data.
        :type independent_field: str

        :param load_field: the name of the load field in the data.
        :type load_field: str

        :raises Objective.TypeError: If the wrong types are passed into the constructor.
        """

        super().__init__('virtual_power')
        check_item_is_correct_type(time_field, str, "time_field")
        check_item_is_correct_type(load_field, str, "load_field")
        
        self._time_field = time_field
        self._load_field = load_field
        self._setup_qoi_extractors()
        self.set_as_large_data_sets_objective()

    def _setup_qoi_extractors(self):
        self.set_experiment_qoi_extractor(
            ExternalVirtualPowerExtractor(self._time_field, 
                                          self._load_field))

        self.set_simulation_qoi_extractor(
            InternalVirtualPowerExtractor(self._time_field))

    def _confirm_simulation_fields(self, simulation_data):
        required_fields = ['virtual_power']
        self._check_required_fields_are_in_data(
            simulation_data, required_fields)

    def _confirm_experiment_fields(self, exp_data):
        required_fields = ['virtual_power']
        self._check_required_fields_are_in_data(exp_data, required_fields)

    def virtual_velocity_function(self):
        return self._velocity_function

    def virtual_velocity_gradient_function(self):
        return self._velocity_gradient_function
