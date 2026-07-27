import torch
from gpytorch.kernels import (
    AdditiveKernel,
    ScaleKernel,
    MaternKernel,
    PeriodicKernel,
    RBFKernel,
    PolynomialKernel,
)
import gpytorch


def kernel_2HWP_additive():

    k1 = ScaleKernel(
        RBFKernel(
            ard_num_dims=3,
            active_dims=torch.tensor([0, 2, 4]),
            lengthscale_prior=gpytorch.priors.LogNormalPrior(0.0, 1.0),
            lengthscale_constraint=gpytorch.constraints.GreaterThan(2.5e-2),
        )
    )
    k2 = ScaleKernel(
        RBFKernel(
            ard_num_dims=3,
            active_dims=torch.tensor([1, 3, 4]),
            lengthscale_prior=gpytorch.priors.LogNormalPrior(0.0, 1.0),
            lengthscale_constraint=gpytorch.constraints.GreaterThan(2.5e-2),
        )
    )
    return AdditiveKernel(k1, k2)


def kernel_1HWP():

    k1 = ScaleKernel(
        RBFKernel(
            active_dims=torch.tensor([1, 3, 4]),
            lengthscale_prior=gpytorch.priors.LogNormalPrior(0.0, 1.0),
            lengthscale_constraint=gpytorch.constraints.GreaterThan(2.5e-2),
        )
    )
    return k1
