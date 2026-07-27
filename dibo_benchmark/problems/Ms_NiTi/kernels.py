import torch
from gpytorch.kernels import (
    AdditiveKernel,
    ScaleKernel,
    MaternKernel,
    PeriodicKernel,
    RBFKernel,
)
import gpytorch


def kernel_simplify_additive():

    k1 = ScaleKernel(
        RBFKernel(
            active_dims=torch.tensor([0]),
            lengthscale_prior=gpytorch.priors.LogNormalPrior(0.0, 1.0),
            lengthscale_constraint=gpytorch.constraints.GreaterThan(2.5e-2),
        )
    )
    k2 = ScaleKernel(
        RBFKernel(
            ard_num_dims=2,
            active_dims=torch.tensor([1, 2]),
            lengthscale_prior=gpytorch.priors.LogNormalPrior(0.0, 1.0),
            lengthscale_constraint=gpytorch.constraints.GreaterThan(2.5e-2),
        )
    )
    return AdditiveKernel(k1, k2)
