To estimate the variance of the measurement noise :math:`\sigma^2` we typically make an empirical estimate (quick) or use a point-estimate/least-squares fit and calculate the variance of the residuals (better).

To form the parameter sensitivity Jacobian :math:`\mathsf{G}` in the model linearization we use a finite difference stencil to evaluate the model with centered (accurate) or one-sided (cheap) differences


To keep the log posterior optimization well-conditioned we reparameterize the parameter covariance as

.. math::
  \left[ \boldsymbol{\Sigma} \right]_{ij} = \sqrt{\varsigma^2_{i}}\left[\boldsymbol{\Xi}\right]_{ij} \sqrt{\varsigma^2_{j}}

with the diagonal of the covariance matrix :math:`\varsigma_i^2 = \Sigma_{ii}` and the correlation :math:`\Xi_{ij}` with off-diagonal components :math:`\rho_{ij}`, which is  reparameterization as: :math:`\varsigma^2 \leftarrow \exp(\sigma^2)` and :math:`\rho \leftarrow \operatorname{arctanh}(\rho)`.

