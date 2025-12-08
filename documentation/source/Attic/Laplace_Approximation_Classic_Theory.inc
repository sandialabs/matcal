
Bayes rule

.. math:: p({\boldsymbol{\theta}}| \mathbf{d}) = \frac{p(\mathbf{d}| {\boldsymbol{\theta}})}{p(\mathbf{d})} p({\boldsymbol{\theta}})

with data :math:`\mathbf{d}= (\mathbf{x}, \mathbf{y})`, parameters
:math:`{\boldsymbol{\theta}}`, prior :math:`p({\boldsymbol{\theta}})`,
likelihood

.. math::

   L({\boldsymbol{\theta}},\mathbf{d}) \equiv p(\mathbf{d}| {\boldsymbol{\theta}}) 

model

.. math::

   \label{eq:noise}
   \mathbf{y}= M(\mathbf{x}; {\boldsymbol{\theta}}) + \epsilon

and noise/discrepancy
:math:`\epsilon \sim \mathcal{N}(\mathbf{0},\sigma^2\mathbf{I})`. For
MLE
:math:`p({\boldsymbol{\theta}}| \mathbf{d}) \sim L({\boldsymbol{\theta}},\mathbf{d})`.

A multivariate normal distribution consistent with the additive
homoskedastic noise, Eq. (`[eq:noise] <#eq:noise>`__), is a standard
choice for the likelihood:

.. math::

   \label{eq:likelihood}
   L({\boldsymbol{\theta}},\mathbf{d}) = \left(\frac{1}{\sigma \sqrt{2 \pi}}\right)^n \prod_i
   \exp\left( - \frac{(d_i - M(x_i; {\boldsymbol{\theta}}))^2}{2\sigma^2} \right) \ ,

where :math:`i\in(1,n)` and :math:`n` is the number of data samples.
This log-likelihood is a rescaling of the generalized least squares
error (:math:`\Phi`) with an additive complexity term

.. math::

   -\log L =  
   \frac{1}{2\sigma^2}\underbrace{\sum_i  (d_i - M(x_i; {\boldsymbol{\theta}}))^2}_{\Phi}
   + \underbrace{n \log\left({\sigma \sqrt{2 \pi}}\right)}_{n (\log\sigma + \log \sqrt{2\pi})}

where :math:`\sigma^2` is a estimate of the variance of the noise
:math:`{\boldsymbol{\epsilon}}`. This can be recast with a variance
weighted residual,
:math:`\mathbf{r}= (\mathbf{d}- M(\mathbf{x},{\boldsymbol{\theta}}))/\sigma`,
as

.. math::

   -\log L =  
   \frac{1}{2} \sum_i r_i^2
   + n \log\left({\sqrt{2 \pi}}\right)  \ .

If :math:`\sigma` is calibrated it will be affected by the model
discrepancy as well as the measurement noise. With infinite data, no
noise, and significant discrepancy, the posterior converges to the Dirac
delta and variance of the noise accommodates the discrepancy.

Laplace’s method approximates the log likelihood by a Taylor series

.. math::

   \label{eq:logL}
   -\log L \approx 
   - \underbrace{\log L |_{{\boldsymbol{\theta}}= \check{{\boldsymbol{\theta}}}} }_\text{constant}
   - \underbrace{\partial_{{\boldsymbol{\theta}}} \log L |_{{\boldsymbol{\theta}}= \check{{\boldsymbol{\theta}}}} }_{\mathbf{g}=\mathbf{0}} \Delta {\boldsymbol{\theta}}
   - \frac{1}{2} \underbrace{\partial_{{\boldsymbol{\theta}}} \partial_{{\boldsymbol{\theta}}} \log L |_{{\boldsymbol{\theta}}= \check{{\boldsymbol{\theta}}}} }_{\mathbf{H}} \Delta {\boldsymbol{\theta}}\otimes \Delta {\boldsymbol{\theta}}

where :math:`\check{{\boldsymbol{\theta}}}` is a maximum likelihood
estimate from an optimizer. The gradient of the likelihood

.. math::

   \label{eq:g}
   \mathbf{g}\equiv \partial_{{\boldsymbol{\theta}}} \log L 
   = \frac{1}{2\sigma^2} \partial_{{\boldsymbol{\theta}}} \Phi
   = - \frac{1}{\sigma^2} \sum_i
    \underbrace{(d_i - M(x_i,{\boldsymbol{\theta}}))}_{\varepsilon_i} \cdot \partial_{{\boldsymbol{\theta}}} M(x_i,{\boldsymbol{\theta}}) 
   \ ,

where :math:`\varepsilon_i` are the errors/residuals. The Hessian of the
likelihood

.. math::

   \label{eq:H}
   \mathbf{H}\equiv \partial_{{\boldsymbol{\theta}}} \partial_{{\boldsymbol{\theta}}} \log L 
   = \frac{1}{2\sigma^2} \partial_{{\boldsymbol{\theta}}} \partial_{{\boldsymbol{\theta}}} \Phi
   = - \frac{1}{\sigma^2} \sum_i \left[
     \partial_{{\boldsymbol{\theta}}} M(x_i,{\boldsymbol{\theta}}) \otimes \partial_{{\boldsymbol{\theta}}} M(x_i,{\boldsymbol{\theta}}) 
   - (d_i - M(x_i,{\boldsymbol{\theta}})) \cdot \partial_{{\boldsymbol{\theta}}} \partial_{{\boldsymbol{\theta}}} M(x_i,{\boldsymbol{\theta}}) 
   \right]

is a rescaling of the least squares Hessian and it composed of a
linearization of the model :math:`M` and its second derivatives with
respect to the parameters. The second term is not positive definite and
it the only term that depends on model error. QUESTION is the first term
rank deficient? connect with Gauss-Newton, Fisher info matrix. QUESTION
Since the approximation doesn’t depend on the data it can’t capture the
convergence of the posterior to a delta with increasing data? Some
practitioners drop the second term and make the approximation

.. math::

   \label{eq:Happrox}
   \mathbf{H}\approx 
    - \frac{1}{\sigma^2} \sum_i \left[
     \partial_{{\boldsymbol{\theta}}} M(x_i,{\boldsymbol{\theta}}) \otimes \partial_{{\boldsymbol{\theta}}} M(x_i,{\boldsymbol{\theta}}) 
   \right]

using only the gradient model sensitivities at the MLE/MAP point. It is
only negligible if the error/discrepancy goes to zero. The Hessian can
be used as the inverse covariance (precision) in the normalized
multivariate Gaussian:

.. math::

   L \approx \frac{ |\mathbf{H}|^{1/2} }{ (2\pi)^{n/2}} \exp\left( -\frac{1}{2} ({\boldsymbol{\theta}}- \check{{\boldsymbol{\theta}}}) \mathbf{H}({\boldsymbol{\theta}}- \check{{\boldsymbol{\theta}}}) \right) 
   = \mathcal{N}_{\boldsymbol{\theta}}(\check{{\boldsymbol{\theta}}}, \mathbf{H}^{-1})

which we can assume is the posterior for a non-informative prior. (The
constant in `[eq:logL] <#eq:logL>`__ becomes a prefactor that is make
irrelevant by normalization.) If we have a Gaussian prior
:math:`p({\boldsymbol{\theta}}) \sim \mathcal{N}_{\boldsymbol{\theta}}(\bar{{\boldsymbol{\mu}}},\bar{{\boldsymbol{\Sigma}}})`
we can obtain the posterior

.. math::

   p({\boldsymbol{\theta}}| \mathbf{d}) \sim p(\mathbf{d}| {\boldsymbol{\theta}}) p(\mathbf{d})  = \mathcal{N}_{\boldsymbol{\theta}}(\check{{\boldsymbol{\theta}}}, \mathbf{H}^{-1}) \, \mathcal{N}_{\boldsymbol{\theta}}(\bar{{\boldsymbol{\mu}}},\bar{{\boldsymbol{\Sigma}}}) 
   =  \mathcal{N}_{\boldsymbol{\theta}}( \hat{{\boldsymbol{\mu}}}, \hat{{\boldsymbol{\Sigma}}})

where
:math:`\hat{{\boldsymbol{\Sigma}}} = (\mathbf{H}+ \bar{{\boldsymbol{\Sigma}}}^{-1})^{-1}`
and
:math:`\hat{{\boldsymbol{\mu}}} = \hat{{\boldsymbol{\Sigma}}} ( \mathbf{H}\check{{\boldsymbol{\mu}}} + \bar{{\boldsymbol{\Sigma}}}^{-1} \bar{{\boldsymbol{\mu}}})`.
In this case :math:`\hat{{\boldsymbol{\mu}}}` is the maximum a
posteriori estimate, :math:`\check{{\boldsymbol{\mu}}}` is the maximum
likelihood estimate, and :math:`\bar{{\boldsymbol{\mu}}}` is the prior
mean.

Since the dependence of the Laplace approximation of the likelihood /
posterior is parametric on :math:`\sigma` we can sample
:math:`L({\boldsymbol{\theta}})`, push these samples forward through the
model and compare variance of the samples to the data and then adjust
the presumed noise :math:`\sigma` to cover the data variance and model
discrepancy. This, in turn, will affect the parameter covariance
:math:`\mathbf{H}^{-1} \sim \sigma^2 \mathbf{I}`. We define “cover” in
the sense that the posterior predictive responses overlap with the
training data by:

.. math::

   \operatorname{cover}(y,y^*) = \frac{ \max( \min(y,y^*) - \max(y,y^*), 0 ) }
                                      { ( \max(y^*) - \min(y^*) ) }

Estimate :math:`\sigma^2` by matching the estimated variance of the
response to that of the data. The variance of the data wrt to the ML
respinse :math:`\hat{\mathbf{y}}`

.. math:: \varsigma^2 = \max {\operatorname{var}}(\mathbf{d}- \hat{\mathbf{y}})

The model response can be linearized, as in the Laplace approximation,

.. math:: \mathbf{y}- \hat{\mathbf{y}} = M(\mathbf{x}; {\boldsymbol{\theta}}) - \hat{\mathbf{y}} \approx \underbrace{\partial_{\boldsymbol{\theta}}M(\mathbf{x};}_{\mathbf{M}^T} \hat{{\boldsymbol{\theta}}}) \Delta {\boldsymbol{\theta}}

If
:math:`{\boldsymbol{\theta}}\sim \mathcal{N}(\hat{{\boldsymbol{\theta}}},{\boldsymbol{\Sigma}})`
then

.. math::

   {\operatorname{var}}(\mathbf{y}-\hat{\mathbf{y}}) = \mathbf{M}{\boldsymbol{\Sigma}}\mathbf{M}^T
   \approx - \sigma^2  \mathbf{M}\mathbf{M}^T \mathbf{M}\mathbf{M}^T

since

.. math:: {\boldsymbol{\Sigma}}= - \sigma^2 \mathbf{H}^{-1} \approx - \sigma^2 \mathbf{M}^T \mathbf{M}

Match using the max eigenvalue :math:`\lambda^{-2}` of
:math:`\mathbf{H}`

.. math:: \sigma^2 = \lambda^2 \varsigma^2

So

.. math:: {\boldsymbol{\Sigma}}= - \varsigma^2 \left[ \lambda^2 \mathbf{M}^T \mathbf{M}\right]

or, alternately, normalize by mean eigenvalue

.. math:: {\boldsymbol{\Sigma}}= - \varsigma^2 \frac{1}{\mathbf{M}\cdot \mathbf{M}}  \lambda^2 \mathbf{M}^T \mathbf{M}

