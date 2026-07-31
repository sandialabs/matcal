**********
Surrogates
**********

For complex physics simulations or characterization tests, the cost of model
evaluations can make direct calibration, optimization, uncertainty
quantification, or parameter studies impractical. Surrogate models provide a
lower-cost approximation of an expensive model response. Once trained, a
surrogate can often be evaluated many times at a small fraction of the cost of
the original model.

A surrogate model approximates a relationship of the form

.. math::

   y = f(x),

where :math:`x` is a vector of input parameters and :math:`y` is a model
response. In MatCal, :math:`x` is usually a vector of calibration or design
parameters, such as material parameters, geometric dimensions, or boundary
condition values. The response :math:`y` may be a single scalar value, but it is
often a vector-valued response, such as a force-displacement curve, a temperature
history, or a spatial field sampled at many locations.

MatCal provides two broad surrogate-modeling workflows:

* **Pretrained surrogates**, built from an existing set of model evaluations.
  These are created with :class:`matcal.core.surrogates.SurrogateGenerator`.

* **Adaptive surrogates**, where MatCal chooses new training samples
  iteratively based on surrogate error estimates and sampling rules. These are
  implemented in :mod:`matcal.core.adaptive_surrogates`.

Pretrained Surrogates
=====================

Pretrained surrogates are built from a fixed set of completed model evaluations.
The user first runs a sampling study, such as a Halton study, Latin-hypercube
study, or parameter study, and then passes the resulting
:class:`matcal.core.study_base.StudyResults` object to
:class:`matcal.core.surrogates.SurrogateGenerator`.

These surrogates are called "pretrained" here because their training data are
already available before surrogate construction begins. The surrogate generator
does not choose new sample points. It only fits a predictive model to the
supplied training data.

Response Dimensionality Reduction with PCA
------------------------------------------

Theory
^^^^^^

Many MatCal model responses are vectors rather than single scalar values. For
example, a model may return displacement at many spatial locations, temperature
over time, or force as a function of displacement. If the response vector has
many entries, directly training a separate surrogate for every response
component can be expensive and difficult.

MatCal uses principal component analysis (PCA) to reduce the dimension of large
response vectors.

Suppose a model is evaluated at :math:`n` different parameter points. At each
parameter point, the model returns a response vector with :math:`m` values. The
training responses can be arranged into a matrix

.. math::

   Y =
   \begin{bmatrix}
   y_{11} & y_{12} & \cdots & y_{1m} \\
   y_{21} & y_{22} & \cdots & y_{2m} \\
   \vdots & \vdots & \ddots & \vdots \\
   y_{n1} & y_{n2} & \cdots & y_{nm}
   \end{bmatrix}
   \in \mathbb{R}^{n \times m}.

Each row corresponds to one model evaluation. Each column corresponds to one
response location, time step, or other quantity of interest.

PCA represents this response matrix in a lower-dimensional basis. Conceptually,
the response at a parameter point is approximated as

.. math::

   y(x) \approx \bar{y}
   + a_1(x) \phi_1
   + a_2(x) \phi_2
   + \cdots
   + a_p(x) \phi_p,

where:

* :math:`\bar{y}` is the mean response;
* :math:`\phi_1, \phi_2, \ldots, \phi_p` are PCA basis vectors, also called
  modes or principal components;
* :math:`a_1(x), a_2(x), \ldots, a_p(x)` are scalar coefficients, also called
  PCA amplitudes or latent variables;
* :math:`p` is the number of retained PCA modes.

The important idea is that the surrogate does not need to predict every entry
of the full response vector directly. Instead, it predicts the smaller set of
PCA amplitudes

.. math::

   a(x) =
   \begin{bmatrix}
   a_1(x) & a_2(x) & \cdots & a_p(x)
   \end{bmatrix}.

After these amplitudes are predicted, the full response vector is reconstructed
using the PCA basis.

PCA is usually computed using a singular value decomposition. If the centered
and scaled response matrix is denoted by :math:`\tilde{Y}`, then

.. math::

   \tilde{Y} = U \Sigma V^T.

Here:

* :math:`U` contains left singular vectors;
* :math:`\Sigma` contains singular values;
* :math:`V` contains right singular vectors;
* the columns of :math:`V` define response-space basis directions.

The singular values indicate how much variation is captured by each mode.
Large singular values correspond to dominant response patterns. Small singular
values correspond to weaker patterns.

If only the first :math:`p` modes are retained, the approximation becomes

.. math::

   \tilde{Y} \approx U_p \Sigma_p V_p^T.

For users not familiar with this notation, the main interpretation is:

* PCA finds the dominant shapes or patterns in the response data.
* Each model response is approximated as a weighted combination of those
  dominant patterns.
* The surrogate predicts the weights instead of predicting the full response
  directly.

This is useful when the response vector is large but most of its variation can
be described by a small number of dominant patterns.

Implementation Details
^^^^^^^^^^^^^^^^^^^^^^

In MatCal's implementation, PCA is used automatically only when the number of
quantities of interest, or response features, is greater than 15. If the
response has 15 or fewer features, MatCal skips PCA and uses the native response
features directly. This avoids unnecessary decomposition for small responses.

The scikit-learn implementation of PCA is used by MatCal :cite:`scikit-learn`.

MatCal also scales parameters and responses internally before training. The
generated surrogate returns predictions in the original physical response space,
so users do not need to manually undo the scaling or PCA reconstruction.

SurrogateGenerator Implementation Details
-----------------------------------------

Pretrained surrogates are built with
:class:`matcal.core.surrogates.SurrogateGenerator`.

A typical workflow is:

#. Run a MatCal study to generate training data.
#. Create a :class:`~matcal.core.surrogates.SurrogateGenerator` from the study
   results.
#. Select the response fields to model with
   :meth:`~matcal.core.surrogates.SurrogateGenerator.set_fields_of_interest`.
#. Configure PCA behavior with
   :meth:`~matcal.core.surrogates.SurrogateGenerator.set_PCA_details`.
#. Generate and save the surrogate with
   :meth:`~matcal.core.surrogates.SurrogateGenerator.generate`.

The basic mathematical structure is

.. math::

   x \xrightarrow{\text{regressor}} \hat{a}(x)
   \xrightarrow{\text{PCA reconstruction}} \hat{y}(x),

where :math:`x` is the parameter vector, :math:`\hat{a}(x)` is the predicted
latent response vector, and :math:`\hat{y}(x)` is the final predicted physical
response.

If PCA is not used, the structure is simpler:

.. math::

   x \xrightarrow{\text{regressor}} \hat{y}(x).

The main configuration options are:

``surrogate_type``
    Controls how regressors are assigned to response components.

    ``"PCA Multiple Regressors"``
        Builds one regressor for each latent coordinate, or for each response
        component when PCA is not used. This is the default. In equation form,
        MatCal trains separate maps

        .. math::

           \hat{a}_j(x) = g_j(x),
           \qquad j = 1, \ldots, p.

        Each :math:`g_j` is a separate regression model. This can be flexible
        and accurate because each latent coordinate is learned independently.

    ``"PCA Monolithic Regressor"``
        Builds one multi-output regressor that predicts the full latent vector,
        or the full response vector when PCA is not used. In equation form,
        MatCal trains one map

        .. math::

           \hat{a}(x) = g(x).

        This can be more compact, but performance depends on the selected
        regressor and problem.

``regressor_type``
    Controls the machine-learning model used to map model parameters to latent
    response coordinates. Implemented options are:

    * ``"Gaussian Process"``
    * ``"RBF"``
    * ``"Random Forest"``

``interpolation_field`` and ``interpolation_locations``
    These options are used for curve-like responses. MatCal interpolates each
    training response onto a common set of independent-variable locations before
    training the surrogate. For scalar responses, users commonly provide a
    one-point independent-variable array.

``training_fraction``
    Controls how MatCal splits supplied data into training and test data. If
    ``training_fraction=1.0``, then separate ``test_eval_info`` must be provided
    so that the surrogate can still be scored on data not used for training.

``decomp_var``
    Controls how many PCA modes are retained when PCA is active. For example,
    ``decomp_var=0.99`` keeps enough PCA modes to explain 99% of the response
    variance. An integer value can also be supplied to request a fixed number of
    components. This option only affects fields for which PCA is actually used.
    MatCal skips PCA for fields with 15 or fewer response features.

The generated surrogate can be evaluated using positional parameters, keyword
parameters, a parameter dictionary, or a batch array. For batch arrays, use
``batch_evaluate=True``.

Gaussian Process Surrogates
---------------------------

Theory
^^^^^^

A Gaussian process (GP) surrogate treats the unknown model response as a random
function. Instead of assuming a fixed polynomial or neural-network form, a GP
defines a probability distribution over possible functions. The final prediction
is based on functions that are consistent with the observed training data.

The pretrained PCA/GP surrogate workflow used in MatCal follows the general
approach described in :cite:`pca_gp_surrogates`: high-dimensional response data
may first be reduced with PCA, then Gaussian-process regressors are trained to
predict the PCA amplitudes from the model parameters. The predicted amplitudes
are finally reconstructed into the original physical response space.

Suppose the training data are

.. math::

   X =
   \begin{bmatrix}
   x_1^T \\
   x_2^T \\
   \vdots \\
   x_n^T
   \end{bmatrix},
   \qquad
   y =
   \begin{bmatrix}
   y_1 \\
   y_2 \\
   \vdots \\
   y_n
   \end{bmatrix},

where :math:`x_i` is the parameter vector for training sample :math:`i`, and
:math:`y_i` is the corresponding scalar response or latent PCA coordinate.

A GP assumes that the responses are jointly Gaussian:

.. math::

   y \sim \mathcal{N}(m(X), K(X, X)).

Here:

* :math:`m(X)` is the mean function evaluated at the training points;
* :math:`K(X, X)` is a covariance matrix;
* the covariance matrix is built from a kernel function.

The kernel function defines how strongly two points are related. A common
kernel is the radial-basis-function kernel,

.. math::

   k(x, x')
   =
   \sigma_f^2
   \exp
   \left(
   -\frac{1}{2}
   \sum_{j=1}^{d}
   \frac{(x_j - x'_j)^2}{\ell_j^2}
   \right),

where:

* :math:`x` and :math:`x'` are two parameter points;
* :math:`d` is the number of parameters;
* :math:`\sigma_f^2` controls the overall response variance;
* :math:`\ell_j` is a length scale for parameter :math:`j`.

The length scale :math:`\ell_j` is important. If :math:`\ell_j` is large, the
response is assumed to change slowly in parameter direction :math:`j`. If
:math:`\ell_j` is small, the response is allowed to change rapidly in that
direction.

For a new point :math:`x_*`, the GP prediction has a mean and variance. A common
form for the predictive mean is

.. math::

   \hat{y}(x_*)
   =
   k_*^T
   \left(
   K + \sigma_n^2 I
   \right)^{-1}
   y,

where:

* :math:`K` is the covariance matrix between all training points;
* :math:`k_*` is the vector of covariances between the new point and each
  training point;
* :math:`\sigma_n^2 I` is a small noise or regularization term;
* :math:`I` is the identity matrix.

This equation says that the prediction is a weighted combination of the training
responses. The weights are determined by the kernel-based similarity between
the new point and the training data.

The GP predictive variance is commonly written as

.. math::

   \mathrm{Var}[\hat{y}(x_*)]
   =
   k(x_*, x_*)
   -
   k_*^T
   \left(
   K + \sigma_n^2 I
   \right)^{-1}
   k_*.

The variance is small near well-sampled regions and larger in regions far from
training data. This uncertainty estimate is one reason GP models are commonly
used in adaptive sampling.

Implementation Details
^^^^^^^^^^^^^^^^^^^^^^

MatCal's Gaussian-process pretrained surrogates use
``sklearn.gaussian_process.GaussianProcessRegressor``.

If the user does not provide a kernel, MatCal uses a default kernel of the form

.. math::

   k(x, x')
   =
   C \, k_\mathrm{RBF}(x, x')

where:

* :math:`C` is a constant kernel and
* :math:`k_\mathrm{RBF}` is an RBF covariance kernel.

When the predicted response has more than 15 features, MatCal trains the GP
surrogate on PCA coordinates instead of directly on every response feature. For
15 or fewer features, the GP is trained directly on the scaled response
features.

Useful GP options include:

``n_restarts_optimizer``
    Number of times the kernel hyperparameter optimization is restarted.

``alpha``
    Additional diagonal regularization added to the covariance matrix.

``normalize_y``
    Whether scikit-learn normalizes the training targets before fitting.

``kernel``
    A user-specified scikit-learn kernel.

GPs are often accurate for smooth deterministic simulation data, but training can
become expensive for large training sets because standard GP training requires
operations on the dense covariance matrix.

Radial Basis Function Surrogates
--------------------------------

Theory
^^^^^^

A radial basis function (RBF) surrogate approximates a response using functions
that depend on distance from training points. The basic scalar RBF interpolant
has the form

.. math::

   \hat{y}(x)
   =
   \sum_{i=1}^{n}
   w_i \, \varphi(\|x - x_i\|),

where:

* :math:`x` is the point where a prediction is needed;
* :math:`x_i` is training point :math:`i`;
* :math:`w_i` is a fitted weight;
* :math:`\varphi(r)` is a radial basis function;
* :math:`r = \|x - x_i\|` is the distance between :math:`x` and :math:`x_i`.

The word "radial" means the basis function depends only on distance, not on
direction. Points at the same distance from a training point have the same basis
function value.

Common radial basis functions include Gaussian, multiquadric, inverse
multiquadric, cubic, and thin-plate spline functions. For example, a Gaussian
RBF has the form

.. math::

   \varphi(r) = \exp\left(-(\epsilon r)^2\right),

where :math:`\epsilon` controls the width of the basis function.

The weights :math:`w_i` are chosen so that the surrogate matches or approximates
the training responses. In exact interpolation, the surrogate satisfies

.. math::

   \hat{y}(x_i) = y_i,
   \qquad i = 1, \ldots, n.

RBF surrogates are intuitive: the prediction at a new point is assembled from
distance-weighted influence functions centered at the known training points.
Nearby points generally influence the prediction more strongly than distant
points.

Implementation Details
^^^^^^^^^^^^^^^^^^^^^^

MatCal implements RBF pretrained surrogates with
``scipy.interpolate.RBFInterpolator`` through a scikit-learn-like wrapper.

The MatCal regressor type is:

.. code-block:: python

   regressor_type="RBF"

By default, MatCal uses a local RBF interpolator with ``neighbors=50``. This
means that each prediction uses a nearby subset of training samples rather than
all training samples. This can reduce prediction cost for large training sets.

Users can change this behavior by passing RBF keyword arguments through the
:class:`~matcal.core.surrogates.SurrogateGenerator`.

For example:

.. code-block:: python

   surrogate_generator = mc.SurrogateGenerator(
       study_results,
       regressor_type="RBF",
       neighbors=25,
   )

When the response has more than 15 features, MatCal applies PCA before training
the RBF model. For smaller responses, the RBF model predicts the scaled native
response features directly.

Because the RBF implementation does not provide predictive standard deviations,
uncertainty-based metrics such as negative log predictive density are not
meaningful for RBF surrogates.

Random Forest Surrogates
------------------------

Theory
^^^^^^

A random forest surrogate is an ensemble of decision trees. A single decision
tree partitions the parameter space into regions and assigns a simple prediction
to each region. For regression, the prediction in a terminal region, or leaf, is
often the average of the training responses in that leaf.

A single decision tree prediction can be written conceptually as

.. math::

   \hat{y}_\mathrm{tree}(x)
   =
   \sum_{\ell=1}^{L}
   c_\ell \, I(x \in R_\ell),

where:

* :math:`R_\ell` is leaf region :math:`\ell`;
* :math:`c_\ell` is the response value assigned to that leaf;
* :math:`I(x \in R_\ell)` is an indicator function that equals 1 if :math:`x`
  is in region :math:`R_\ell` and 0 otherwise;
* :math:`L` is the number of leaf regions.

A random forest averages many such trees:

.. math::

   \hat{y}(x)
   =
   \frac{1}{N_\mathrm{trees}}
   \sum_{b=1}^{N_\mathrm{trees}}
   \hat{y}_b(x),

where :math:`\hat{y}_b(x)` is the prediction from tree :math:`b`.

Each tree is trained on a randomized version of the data and often considers
random subsets of parameters when splitting nodes. This randomness causes the
trees to make different errors. Averaging many trees reduces variance and
usually improves robustness compared with a single tree.

For users unfamiliar with tree models, the practical interpretation is:

* each tree asks a sequence of yes/no questions about the input parameters;
* the final answer places the point into a region of parameter space;
* the tree predicts a response for that region;
* the forest prediction is the average of many tree predictions.

Random forests can be useful for nonlinear and irregular responses. However,
they are not smooth interpolants. Their predictions are often piecewise constant
or only weakly smooth, so they may not be ideal when a smooth response surface
is required.

Implementation Details
^^^^^^^^^^^^^^^^^^^^^^

MatCal's random forest pretrained surrogates use
``sklearn.ensemble.RandomForestRegressor``.

The MatCal regressor type is:

.. code-block:: python

   regressor_type="Random Forest"

Additional keyword arguments are passed directly to scikit-learn. For example:

.. code-block:: python

   surrogate_generator = mc.SurrogateGenerator(
       study_results,
       regressor_type="Random Forest",
       n_estimators=100,
       random_state=123,
   )

When the response has more than 15 features, MatCal applies PCA and trains the
random forest in the reduced latent space. For 15 or fewer response features,
the random forest predicts the scaled response features directly.

Random forests do not provide the same predictive standard deviation interface
as Gaussian-process regressors. Therefore, uncertainty-based diagnostics such as
negative log predictive density are not generally meaningful for this regressor
type.

Adaptive Surrogates
===================

Adaptive surrogate studies build a surrogate iteratively. Instead of requiring
the user to provide all training samples in advance, the adaptive study selects
new parameter points based on the current surrogate, model responses, and an
adaptive sampling strategy.

The adaptive surrogate workflow is useful when each model evaluation is
expensive and the user wants to concentrate samples in regions that are difficult
to approximate.

MatCal currently implements two adaptive surrogate strategies:

* Voronoi adaptive surrogate studies.
* Sparse-grid adaptive surrogate studies.

AdaptiveSurrogate Container
---------------------------

Both adaptive workflows store their results in an
:class:`matcal.core.adaptive_surrogates.AdaptiveSurrogate` object, or a subclass
of it.

The adaptive surrogate object stores:

* the retained surrogate models;
* the test parameter points used to score the surrogates;
* the test responses;
* RMSE history;
* maximum absolute error history;
* global :math:`R^2` score history;
* sample-count history;
* metadata records for each adaptive batch.

The RMSE is computed as

.. math::

   \mathrm{RMSE}
   =
   \sqrt{
   \frac{1}{N}
   \sum_{i=1}^{N}
   \left(
   y_i - \hat{y}_i
   \right)^2
   },

where :math:`y_i` is a reference test value, :math:`\hat{y}_i` is the surrogate
prediction, and :math:`N` is the total number of scalar comparisons.

The maximum absolute error is

.. math::

   E_\mathrm{max}
   =
   \max_i
   |y_i - \hat{y}_i|.

The global :math:`R^2` score is

.. math::

   R^2
   =
   1
   -
   \frac{
   \sum_i (y_i - \hat{y}_i)^2
   }{
   \sum_i (y_i - \bar{y})^2
   },

where :math:`\bar{y}` is the mean reference response. An :math:`R^2` value near
1 indicates excellent agreement. A value near 0 indicates that the surrogate is
not much better than predicting the mean response.

To avoid very large saved surrogate files, MatCal does not need to retain every
surrogate object generated during adaptive training. The retention behavior is
controlled with:

.. code-block:: python

   study.set_surrogate_storage_options(
       best_n_surrogates=1,
       save_every_n_batches=None,
       score_metric="max_error",
   )

Voronoi Adaptive Surrogates
---------------------------

Theory
^^^^^^

Voronoi adaptive sampling divides the parameter space into cells around the
current training samples. MatCal's Voronoi adaptive surrogate capability is
based on the cross-validation and K-fold cross-validation Voronoi adaptive
sampling concepts described in :cite:`voronoi_adaptive_surrogates`.
A Voronoi cell associated with sample :math:`s_i` is

.. math::

   V_i
   =
   \left\{
   x :
   \|x - s_i\| \leq \|x - s_j\|,
   \quad
   \text{for all } j \neq i
   \right\}.

In words, :math:`V_i` is the region of parameter space closer to sample
:math:`s_i` than to any other sample. The Voronoi cells form a partition of the
parameter domain.

The adaptive sampling idea is:

#. Estimate which existing samples are associated with large surrogate error.
#. Select the Voronoi cell associated with one of those high-error samples.
#. Add a new sample in that cell, far from the existing sample.

The point "far from the existing sample" promotes exploration within the selected
cell. The high-error criterion promotes exploitation of difficult regions.

A simple leave-one-out cross-validation error for sample :math:`s_i` is

.. math::

   e_i^\mathrm{LOO}
   =
   \left|
   y(s_i) - \hat{y}_{S \setminus s_i}(s_i)
   \right|.

Here:

* :math:`S` is the full set of training samples;
* :math:`S \setminus s_i` means the training set with sample :math:`s_i`
  removed;
* :math:`\hat{y}_{S \setminus s_i}` is the surrogate trained without sample
  :math:`s_i`;
* :math:`e_i^\mathrm{LOO}` measures how poorly the surrogate predicts sample
  :math:`s_i` when that sample is not included in training.

If :math:`e_i^\mathrm{LOO}` is large, then sample :math:`s_i` is important for
the surrogate, and its surrounding region may need more samples.

Full leave-one-out cross validation can be expensive because it requires fitting
one surrogate for each training sample. The K-fold cross-validation Voronoi
method reduces this cost.

In K-fold cross validation, the training samples are divided into :math:`K`
groups, or folds. For fold :math:`k`, a surrogate is trained without that fold
and tested on the samples in that fold. A fold error can be written as

.. math::

   e_k^\mathrm{KF}
   =
   \sum_{s_j \in \mathrm{fold}\ k}
   \left|
   y(s_j)
   -
   \hat{y}_{S \setminus \mathrm{fold}\ k}(s_j)
   \right|.

This error measures the aggregate effect of removing an entire fold. The
highest-error folds are treated as regions of interest. Leave-one-out
cross-validation can then be performed only on samples in those selected folds,
rather than on all samples.

After a high-error sample is selected, the next point is chosen inside that
sample's Voronoi cell. Conceptually,

.. math::

   x_\mathrm{new}
   =
   \arg\max_{x \in V_i}
   \|x - s_i\|.

This means that the new point is the point in the selected Voronoi cell that is
farthest from the existing sample :math:`s_i`. This encourages the new sample to
add information that is different from what is already known.

Implementation Details
^^^^^^^^^^^^^^^^^^^^^^

Voronoi adaptive surrogates are implemented by
:class:`matcal.core.adaptive_surrogates.VoronoiAdaptiveSurrogateStudy`.

The internal surrogate model is built using
:class:`matcal.core.surrogates.SurrogateGenerator`. Therefore, the Voronoi
adaptive surrogate can use the same pretrained surrogate machinery described
above, including:

* Gaussian-process regressors;
* RBF regressors;
* random forest regressors;
* PCA response reduction for responses with more than 15 features.

The default and most common choice is a Gaussian-process regressor.

Important user-facing options include:

``set_number_of_initial_samples``
    Sets the number of initial samples before adaptive refinement begins.

``set_cross_validation_options``
    Configures the K-fold and leave-one-out cross-validation procedure.

    Useful options include:

    * ``nsplits``: number of K-fold splits;
    * ``nmax_folds``: number of highest-error folds retained;
    * ``nmax_loo``: number of leave-one-out candidates retained, or ``"all"``;
    * ``cv_metric``: response-space error metric;
    * ``batch_size``: number of new samples requested per adaptive batch.

``set_voronoi_sampling_options``
    Configures the Voronoi geometry and candidate reduction behavior.

    Useful options include:

    * ``voronoi_type="full"`` for a full Voronoi tessellation;
    * ``voronoi_type="local"`` for a local nearest-neighbor tessellation;
    * ``iterative_updates=True`` to update the tessellation after each new
      point in a batch.

``set_surrogate_options``
    Passes options to :class:`~matcal.core.surrogates.SurrogateGenerator` and
    the selected regressor.

For example:

.. code-block:: python

   study.set_surrogate_options(
       regressor_type="Gaussian Process",
       n_restarts_optimizer=10,
       alpha=1.0e-8,
       normalize_y=True,
       decomp_var=0.99,
   )

Cross-validation errors are computed in native response space for deterministic
metrics such as ``"rmse"``, ``"mae"``, ``"sum_abs"``, and ``"nrmse"``. The
special ``"nlpd"`` option uses Gaussian-process latent-space uncertainty
diagnostics instead.

* ``"rmse"``;
* ``"mae"`` or ``"abs"``;
* ``"sum_abs"``;
* ``"nrmse"``;
* ``"nlpd"``: Gaussian-process latent-space negative log predictive density.
  This option requires ``regressor_type="Gaussian Process"`` because it depends
  on predictive standard deviations.

The ``"sum_abs"`` option is closest to the KFCV-Voronoi error expression used
in :cite:`voronoi_adaptive_surrogates`.

Voronoi Batch Sampling
^^^^^^^^^^^^^^^^^^^^^^

The Voronoi adaptive study can add more than one point per adaptive batch. A
naive batch approach can cluster new samples in neighboring high-error cells.
MatCal supports an iterative batch update strategy: after a new batch point is
chosen, the Voronoi tessellation can be updated before choosing the next point.

Conceptually, for a batch of size :math:`B`, MatCal repeats the following steps:

.. math::

   x_\mathrm{new}^{(b)}
   =
   \arg\max_{x \in V_{i_b}}
   \|x - s_{i_b}\|,
   \qquad b = 1, \ldots, B,

with the Voronoi cells updated after each selected point when
``iterative_updates=True``. This tends to spread new batch points more
effectively than selecting all points from the original tessellation.

Sparse-Grid Adaptive Surrogates
-------------------------------

Theory
^^^^^^

Sparse-grid adaptive surrogates approximate a response using a structured set of
basis functions over the parameter space. They are based on the idea that many
smooth multidimensional functions can be approximated accurately without using a
full tensor-product grid.

For one parameter, an interpolating approximation may be written as

.. math::

   \hat{f}(x)
   =
   \sum_{i=1}^{n}
   c_i \psi_i(x),

where :math:`\psi_i(x)` are basis functions and :math:`c_i` are coefficients.

For multiple parameters, a tensor-product basis function has the form

.. math::

   \Psi_{\boldsymbol{i}}(x)
   =
   \psi_{i_1}(x_1)
   \psi_{i_2}(x_2)
   \cdots
   \psi_{i_d}(x_d),

where:

* :math:`d` is the number of parameters;
* :math:`x = (x_1, x_2, \ldots, x_d)` is a parameter vector;
* :math:`\boldsymbol{i} = (i_1, i_2, \ldots, i_d)` is a multi-index.

A full tensor grid uses many combinations of basis functions in every
dimension. The number of points can grow rapidly with dimension. For example,
if :math:`n` points are used in each of :math:`d` dimensions, the full grid has

.. math::

   n^d

points. This exponential growth is often called the curse of dimensionality.

A sparse grid avoids using all tensor-product combinations. Instead, it selects
a smaller set of multi-indices that are expected to be most important. A generic
sparse-grid approximation can be written as

.. math::

   \hat{f}(x)
   =
   \sum_{\boldsymbol{i} \in \mathcal{I}}
   c_{\boldsymbol{i}}
   \Psi_{\boldsymbol{i}}(x),

where :math:`\mathcal{I}` is the selected sparse index set.

Adaptive sparse grids refine this index set incrementally. The algorithm
estimates which candidate basis functions or subspaces are likely to improve the
approximation most, then adds corresponding samples and basis terms.

For users not familiar with sparse-grid notation, the main idea is:

* a full grid samples every combination of points in every direction;
* a sparse grid samples only the most useful combinations;
* adaptive sparse grids add resolution where the approximation appears to need
  it most.

Sparse grids are often effective for smooth functions. For discontinuous or
highly localized responses, local piecewise bases can be more stable than global
polynomial-like bases.

Implementation Details
^^^^^^^^^^^^^^^^^^^^^^

Sparse-grid adaptive surrogates are implemented by
:class:`matcal.core.adaptive_surrogates.SparseGridAdaptiveSurrogateStudy`.

This workflow uses PyApprox's adaptive sparse-grid functionality rather than
MatCal's PCA/GP surrogate machinery. Therefore, the sparse-grid surrogate does
not use the ``SurrogateGenerator`` PCA decomposition and does not use MatCal's
Gaussian-process, RBF, or random forest regressors.

The sparse-grid basis is configured with:

.. code-block:: python

   study.set_sparse_grid_basis(
       basis_type="piecewise",
       piecewise_degree=2,
   )

Available basis options are:

``"lagrange"``
    A global Lagrange basis using nested Clenshaw-Curtis-type points. This is
    often effective for smooth responses.

``"piecewise"``
    A local piecewise-polynomial basis. This can be more stable for nonsmooth
    responses than a global polynomial-like approximation. Supported polynomial
    degrees are 1, 2, and 3.

The adaptive sparse-grid limits are configured with:

.. code-block:: python

   study.set_sparse_grid_adaptivity_limits(
       max_level=20,
       pnorm=1.0,
   )

The sparse-grid study stores its adaptive history in a
:class:`matcal.core.adaptive_surrogates.SparseGridAdaptiveSurrogate`, which is a
subclass of :class:`~matcal.core.adaptive_surrogates.AdaptiveSurrogate`. This
wrapper converts MatCal-style parameter inputs into the orientation expected by
PyApprox and packages sparse-grid predictions into the same field-based format
used by other MatCal surrogates.

Because sparse-grid surrogates are not built with
:class:`~matcal.core.surrogates.SurrogateGenerator`, the following pretrained
surrogate options do not apply:

* ``regressor_type``;
* Gaussian-process settings;
* RBF settings;
* random forest settings;
* PCA ``decomp_var``.

Choosing a Surrogate Workflow
=============================

Use a pretrained surrogate when:

* a training data set already exists;
* the desired sample locations are known in advance;
* the model is inexpensive enough to sample with a standard design;
* the goal is to build a reusable approximation from fixed data.

Use a Voronoi adaptive surrogate when:

* model evaluations are expensive;
* important response behavior is localized in the parameter space;
* the user wants MatCal to concentrate samples in high-error regions;
* a GP, RBF, or random forest response surrogate is desired.

Use a sparse-grid adaptive surrogate when:

* the response is smooth or moderately nonsmooth;
* a structured adaptive approximation is appropriate;
* PyApprox is available;
* the user wants an adaptive method that does not rely on Gaussian-process
  regression.

.. include:: surrogate_verification/index.rst
   :start-after: :orphan:

.. include:: surrogate_examples/index.rst
   :start-after: :orphan: