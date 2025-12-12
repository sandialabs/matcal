In general a model :math:`\mathbf{m}` has parameters :math:`\boldsymbol{\theta}` and produces outputs :math:`\mathbf{y}` as a function of inputs :math:`\mathbf{x}`:

.. math::
   \mathbf{y} = \mathbf{m}(\mathbf{x}; \boldsymbol{\theta}) \ .

Observed data :math:`\mathsf{D}` = ( :math:`\mathbf{x}`, :math:`\mathbf{y}` ) can be related to the model by adding mechanisms for discrepancies, for instance:

.. math::
   \mathbf{y} = \mathbf{m}(\mathbf{x}; \boldsymbol{\theta}^*) + \boldsymbol{\epsilon}

where the last term accounts for discrepancies between the model and the data, or 

.. math::
   \mathbf{y} = \mathbf{m}(\mathbf{x}; \bar{\boldsymbol{\theta}} + \boldsymbol{\xi}) + \boldsymbol{\eta}

where we split the potential discrepancies into two sources: one internal to the model and the other external to the model.
The first form will be called the classic uncertainty formulation and the second the embedded formulation.

The classic form has the benefit of simplicity, while the embedded form allows for the uncertainty to be ascribed to external, uncorrelated noise and variations that may have parametric trends.
The classic form is easily interpretable: :math:`\boldsymbol{\theta}^*` are the uncertain best fit parameters and :math:`\boldsymbol{\epsilon}` are the discrepancies between the data and model that due to aleatoric (irreducible, stochastic) and epistemic (reducible with additional data) aspects of the data. 
For this form, typically :math:`\boldsymbol{\theta}^*` will converge with increasing (self-consistent) data.

The embedded form is motivated by the fact that we usually have experimental data that displays (small) measurement noise :math:`\boldsymbol{\eta}` and (larger) model inconsistencies which we associate with :math:`\boldsymbol{\xi}`.
Both :math:`\boldsymbol{\eta}` and :math:`\boldsymbol{\xi}` are considered random variables; :math:`\boldsymbol{\eta}` is an aleatoric (irreducible, stochastic) source of discrepancy and :math:`\boldsymbol{\xi}` is an epistemic (reducible with additional data) error.
The variable :math:`\boldsymbol{\theta}` is the mean of the random variable  :math:`\boldsymbol{\theta} + \boldsymbol{\xi}` and can be considered the best estimate of parameters :math:`\bar{\boldsymbol{\theta}}`.
We associate :math:`\boldsymbol{\eta}` with uncorrelated measurement noise and :math:`\boldsymbol{\xi}` with parameter uncertainty due to limited data.
In the simplest case we assume that both :math:`\boldsymbol{\eta}` and :math:`\boldsymbol{\xi}` are normally distributed so we just need to determine their mean and variance.

In both cases, we make the common, simplifying assumption that the external  noise is mean-zero, independent, identically distributed (IID) and normal  :math:`\boldsymbol{\epsilon}` or  :math:`\boldsymbol{\eta} \sim \mathcal{N}(\mathbf{0},\sigma^2 \mathbf{I})`, :math:`\sigma^2` is the variance of the measurement noise.
Furthermore we will take the hyperparameter :math:`\sigma` as known, either from prior knowledge of the testing machine or extracted from the high-frequency variation in the data :math:`\mathbf{y}` that is uncorrelated with its mean trends.

=========================================================================
Uncertainty quantification
=========================================================================
Bayes rule is a basis for determining the uncertainties in the parameters as reprsented by the random variables in the discrepancy terms:

.. math::
  p(\boldsymbol{\theta} | \mathsf{D}) =
  \frac{p(\mathsf{D} | \boldsymbol{\theta})} 
       {p(\mathsf{D})}  p(\boldsymbol{\theta})

where :math:`p(\boldsymbol{\theta} | \mathsf{D} )` is the posterior distribution of the parameters, :math:`p(\boldsymbol{\theta} | \mathsf{D})` is the parameter likelihood given the data, :math:`p(\boldsymbol{\theta} | \mathsf{D})` is the prior distribution of the parameters, and the evidence :math:`p(\mathsf{D})` is a normalizing factor.

If both the likelihood and the prior are Gaussian then the posterior is also Gaussian. likewise for the likelihood being Gaussian and the prior being an improper uniform (i.e. a constant).
With these assumption the parameter posterior is characterized by a mean and a variance.
In general, obtaining the moments of the posterior in a Bayesian calibration is difficult because the model is non-linear in its parameters.
Here we use an estimation method derived from the Hessian-based Laplace approximation. 

-------------------------------------------------------------------------
Classic case
-------------------------------------------------------------------------

If we make the assumption that we have an uniformative prior and focus on the classic case, it sufficies to examine the likelihood:

.. math::
  L(\boldsymbol{\theta}, \mathsf{D}) \equiv p( \mathsf{D} | \boldsymbol{\theta} ) = \mathcal{N}( \mathbf{y}; \mathbf{m}(\mathbf{x};\boldsymbol{\theta}) , \sigma^2 \mathbf{I})

since it is equal the posterior in this case; it is normal from the previous assumption:

.. math::
  \boldsymbol{\epsilon} = \mathbf{y} - \mathbf{m}(\mathbf{x};\boldsymbol{\theta}) \sim \mathcal{N}(\boldsymbol{\epsilon}; \mathbf{0}, \sigma^2 \mathbf{I})

The log likelihood:

.. math::
  -\log L = \frac{1}{2 \sigma^2} \| \underbrace{ \mathbf{y} - \mathbf{m}(\mathbf{x};\boldsymbol{\theta}) }_{\mathbf{R}} \|^2 + n_{\boldsymbol{\theta}} \log( \sigma \sqrt{2\pi} )

has a term that is a variance-weighted mean squared error between the data and the model predictions plus a term that penalizes parameter complexity (:math:`n_{\boldsymbol{\theta}}` is the number of parameters) and variance.
Here :math:`\mathbf{R}` are the residuals.

The mean can be taken to be the maximum likelihood estimate :math:`\boldsymbol{\theta}^*` which is determined by the optimum of the weighted mean squared error.

Laplace's method approximates the log likelihood by a truncated Taylor series:

.. math::
  -\log L \approx 
  - \underbrace{\log L |_{\boldsymbol{\theta} = {\boldsymbol{\theta}}^*} }_\text{constant} 
  - \underbrace{\partial_{\boldsymbol{\theta}} \log L |_{\boldsymbol{\theta} = {\boldsymbol{\theta}}^*} }_{\mathbf{g}=\mathbf{0}} \Delta \boldsymbol{\theta} 
  - \frac{1}{2} \underbrace{\partial_{\boldsymbol{\theta}} \partial_{\boldsymbol{\theta}} \log L |_{\boldsymbol{\theta} = {\boldsymbol{\theta}}^*} }_{\mathbf{H}} \Delta \boldsymbol{\theta} \otimes \Delta \boldsymbol{\theta} 

where the gradient is zero since :math:`{\boldsymbol{\theta}}^*` is an optimum and :math:`\mathbf{H}` is the Hessian.

The gradient of the likelihood:

.. math::
  \mathbf{g} \equiv \partial_{\boldsymbol{\theta}} \log L 
  = - \frac{1}{\sigma^2}  \mathbf{R} \cdot \underbrace{\partial_{\boldsymbol{\theta}} \mathbf{m}(\mathbf{x},\boldsymbol{\theta})}_{\mathsf{G}}

and the Hessian 

.. math::
  \mathbf{H} \equiv \partial_{\boldsymbol{\theta}} \partial_{\boldsymbol{\theta}} \log L 
  = \frac{1}{\sigma^2} \left(  \mathsf{G} \otimes \mathsf{G} + \mathsf{R} \cdot \partial_{\boldsymbol{\theta}}  \partial_{\boldsymbol{\theta}}  \mathbf{m}(\mathbf{x}; \boldsymbol{\theta}) \right)
  \approx \frac{1}{\sigma^2} \left(  \mathsf{G} \otimes \mathsf{G} \right)

involve the model sensitivities :math:`\mathsf{G} \equiv \boldsymbol{\partial}_{\boldsymbol{\theta}} \mathbf{m}`.
The last approximation is valid when the residuals are close to zero :math:`\mathsf{R}`.
Finally the Hessian can be used as the precision/inverse covariance matrix to obtain an approximate parameter posterior distribution:

.. math::
  p( \boldsymbol{\theta} | \mathsf{D} ) = \mathcal{N}\left( \boldsymbol{\theta} ; \boldsymbol{\theta}^*, \mathsf{H}^{-1} \right)

Note the scaling of the covariance depends on the user provided :math:`\sigma^2`.
Also an extension to a Gaussian prior is straightforward.


-------------------------------------------------------------------------
Embedded case
-------------------------------------------------------------------------

With the embedded case, for the random variable :math:`\boldsymbol{\theta} + \boldsymbol{\xi}`, we take its mean :math:`\bar{\boldsymbol{\theta}}` to be the least squares solution :math:`\boldsymbol{\theta}^*`
due to its connection with the maximum likelihood estimate.
To obtain the variance of :math:`\boldsymbol{\theta} + \boldsymbol{\xi}`, we linearize the model 

.. math::

   \mathbf{m}(\mathbf{x}; \boldsymbol{\theta}) \approx 
   \mathbf{m}(\mathbf{x}; \boldsymbol{\theta}^*) + \underbrace{\mathsf{G}(\mathbf{x})}_{\boldsymbol{\partial}_{\boldsymbol{\theta}} \mathbf{m}} \, \Delta \boldsymbol{\theta}

at the optimum :math:`\boldsymbol{\theta}^*`, so that 

.. math::

   \mathbf{y} - \mathbf{m}(\mathbf{x}; \boldsymbol{\theta}^*) = 
   \mathsf{G}(\mathbf{x}) \boldsymbol{\xi} + \boldsymbol{\eta} .

Since we assume both :math:`\boldsymbol{\xi}` and :math:`\boldsymbol{\eta}` are normal and independent, the variances are related by

.. math::

   \boldsymbol{\Sigma}_{\mathbf{y}} = \mathsf{G} \boldsymbol{\Sigma}_{\boldsymbol{\theta}} \mathsf{G}^T + \sigma^2 \mathbf{I} ,

due to the independence of the two Gaussian random variables and the linearity of the transform from the parameters to the observable output.

We can solve for the covariance of the parameters :math:`\boldsymbol{\Sigma}_{\boldsymbol{\theta}}` using linear algebra.
First, move the characterized measurement noise

.. math::

   \underbrace{\boldsymbol{\Sigma}_{\mathbf{y}} - \sigma^2 \mathbf{I}}_{\mathsf{A}} = \mathsf{G} \boldsymbol{\Sigma}_{\boldsymbol{\theta}} \mathsf{G}^T 
   = \mathsf{L} \mathsf{L}^T  

then substitute Cholesky factorization of :math:`\mathsf{A} = \mathsf{L} \mathsf{L}^T` to obtain

.. math::
   \mathbf{I}  = \mathsf{L}^{-1} \mathsf{G} \boldsymbol{\Sigma}_{\boldsymbol{\theta}} \mathsf{G}^T \mathsf{L}^{-T} .

Finally, use the singular value decomposition of :math:`\mathsf{L}^{-1} \mathsf{G} = \mathsf{U} \mathsf{S} \mathsf{V}^T` so that

.. math::
   \mathbf{I}  = \mathsf{U} \mathsf{S} \mathsf{V}^T \boldsymbol{\Sigma}_{\boldsymbol{\theta}} \mathsf{V} \mathsf{S} \mathsf{U}^T 
    = \mathsf{S} \mathsf{V}^T \boldsymbol{\Sigma}_{\boldsymbol{\theta}} \mathsf{V} \mathsf{S}  
    \Rightarrow
   \boldsymbol{\Sigma}_{\boldsymbol{\theta}} =
   \mathsf{V} \mathsf{S}^{-2} \mathsf{V}^T .

This provides a guess for the needed parameter convariance :math:`\boldsymbol{\Sigma}_{\boldsymbol{\theta}}`, given the assumptions, that is usually sufficient for uncertainty quantification, i.e. an ensemble of push-forwards of the parameter distribution through the model 

.. math::
  \{ \mathbf{y}_a = \mathbf{m}(\mathbf{x} ; \boldsymbol{\theta}_a )  \ \ | \ \
  \boldsymbol{\theta}_a \sim \mathcal{N}(\bar{\boldsymbol{\theta}}, \boldsymbol{\Sigma}_{\boldsymbol{\theta}} )
  \}_{a=1}^{n}

that usually covers the calibration data.

To accommodate more of the potential deviations from the assumptions, we can take the linear algebra solution as a initial guess for the optimization of the log posterior

.. math::
   \boldsymbol{\Sigma}_{\boldsymbol{\theta}}= \text{argmin}_{\boldsymbol{\Sigma}_{\boldsymbol{\theta}}} \left( -\log p(\boldsymbol{\theta} | \mathsf{D}) \right)  = \text{argmin}{\boldsymbol{\Sigma}_{\boldsymbol{\theta}}} \left( n_r \log( \det \boldsymbol{\Sigma}_{\boldsymbol{\theta}}) + \text{tr} ( \mathsf{R}  \boldsymbol{\Sigma}_{\boldsymbol{\theta}}^{-1} \mathsf{R}) \right)

given :math:`\boldsymbol{\theta} = \boldsymbol{\theta}^*`.
This optimization minimizes the mean squared error (the second term) with respect to :math:`\boldsymbol{\Sigma}_{\boldsymbol{\theta}}`, regularized by first term which penalizes overly broad uncertainty and complexity.
Here :math:`n_r` is the number of realizations of the data.


