In general a model :math:`\mathbf{m}` has parameters :math:`\boldsymbol{\theta}` and produces 
outputs :math:`\mathbf{y}` as a function of inputs :math:`\mathbf{x}`:

.. math::
   :label: eq:model

   \mathbf{y} = \mathbf{m}(\mathbf{x}; \boldsymbol{\theta}) \ .

Observed data :math:`\mathsf{D}` = ( \mathbf{x}, \mathbf{y} ) can be related to the model 
by adding mechanisms for discrepancies 

.. math::

   \mathbf{y} = \mathbf{m}(\mathbf{x}; \boldsymbol{\theta} + \boldsymbol{\xi}) + \boldsymbol{\eta}

where we typically have data that displays (small) uncorrelated measurement 
error :math:`\boldsymbol{\eta}` and (larger) model inconsistencies which we associate with :math:`\boldsymbol{\xi}`.
Both :math:`\boldsymbol{\eta}` and :math:`\boldsymbol{\xi}` are considered random variables; :math:`\boldsymbol{\eta}` is an 
aleatoric (irreducible, stochastic) source of discrepancy and :math:`\boldsymbol{\xi}`
is an epistemic (reducible with additional data).
In the simplest case we assume that both :math:`\boldsymbol{\eta}` and :math:`\boldsymbol{\xi}` are 
normally distributed so we just need to determine their mean and variance.

We make the common assumption that the measurement noise is mean-zero, independent, 
identically distributed (IID) and normal :math:`\boldsymbol{\eta} \sim \mathcal{N}(\mathbf{0},\sigma^2 \mathbf{I})`.
Furthermore we will take the hyperparameter :math:`\sigma` as known, either from prior 
knowledge of the testing machine or extracted from the high-frequency noise in the data :math:`\mathbf{y}`.

For the random variable :math:`\boldsymbol{\theta} + \boldsymbol{\xi}` we take its mean 
to be the least squares solution

.. math::

   \boldsymbol{\theta}^* 
   = \operatorname{argmin}_{\boldsymbol{\theta}} \| \mathbf{d} - \mathbf{y} \|^2_{\mathsf{W}}

due its connection with the maximum likelihood estimate.
Hence, we take :math:`\boldsymbol{\theta} = \boldsymbol{\theta}^*`.
To obtain the variance of :math:`\boldsymbol{\theta} + \boldsymbol{\xi}` we linearize the model 

.. math::

   \mathbf{m}(\mathbf{x}; \boldsymbol{\theta}) \approx 
   \mathbf{m}(\mathbf{x}; \boldsymbol{\theta}^*) + \underbrace{\mathsf{G}(\mathbf{x})}_{\boldsymbol{\partial}_{\boldsymbol{\theta}} \mathbf{m}} \boldsymbol{\theta}

at the optimum, so that 

.. math::

   \mathbf{y} - \mathbf{m}(\mathbf{x}; \boldsymbol{\theta}^*) = 
   \mathsf{G}(\mathbf{x}) \boldsymbol{\xi} + \boldsymbol{\eta}

Since we assume both :math:`\boldsymbol{\xi}` and :math:`\boldsymbol{\eta}` are normal, the variances are related by

.. math::

   \boldsymbol{\Sigma}_{\mathbf{y}} = \mathsf{G} \boldsymbol{\Sigma}_{\boldsymbol{\theta}} \mathsf{G}^T + \sigma^2 \mathbf{I}

We solve for the covariance of the parameters :math:`\boldsymbol{\Sigma}_{\boldsymbol{\theta}}` using a bit of linear algebra.
First

.. math::

   \underbrace{\boldsymbol{\Sigma}_{\mathbf{y}} - \sigma^2 \mathbf{I}}_{\mathsf{A}} = \mathsf{G} \boldsymbol{\Sigma}_{\boldsymbol{\theta}} \mathsf{G}^T 

then substitute Cholesky factorization of :math:`\mathsf{A} = \mathsf{L} \mathsf{L}^T`

.. math::
   :nowrap:

   \begin{eqnarray}
   \mathsf{L} \mathsf{L}^T  &=& \mathsf{G} \boldsymbol{\Sigma}_{\boldsymbol{\theta}} \mathsf{G}^T  \\
   \mathbf{I}  &=& \mathsf{L}^{-1} \mathsf{G} \boldsymbol{\Sigma}_{\boldsymbol{\theta}} \mathsf{G}^T \mathsf{L}^{-T}  
   \end{eqnarray}

Finally use the singular value decomposition of :math:`\mathsf{L}^{-1} \mathsf{G} = \mathsf{U} \mathsf{S} \mathsf{V}^T` so that

.. math::
   :nowrap:

   \begin{eqnarray}
   \mathbf{I}  &=& \mathsf{U} \mathsf{S} \mathsf{V}^T \boldsymbol{\Sigma}_{\boldsymbol{\theta}} \mathsf{V} \mathsf{S} \mathsf{U}^T \\
   \mathbf{I}  &=& \mathsf{S} \mathsf{V}^T \boldsymbol{\Sigma}_{\boldsymbol{\theta}} \mathsf{V} \mathsf{S}  \\
   \mathsf{V} \mathsf{S}^{-2} \mathsf{V}^T  &=& \boldsymbol{\Sigma}_{\boldsymbol{\theta}} 
   \end{eqnarray}