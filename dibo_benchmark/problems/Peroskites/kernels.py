import torch
from gpytorch.kernels import (
    AdditiveKernel,
    ScaleKernel,
    MaternKernel,
    PeriodicKernel,
    RBFKernel,
)
from gpytorch.kernels import ScaleKernel, PolynomialKernel
import gpytorch


def kernel_polyscale_additive():

    k1 = ScaleKernel(
        PolynomialKernel(
            power=1,
            offset_prior=gpytorch.priors.LogNormalPrior(0.0, 1.0),
        )
    )
    k2 = ScaleKernel(
        RBFKernel(
            ard_num_dims=2,
            lengthscale_prior=gpytorch.priors.LogNormalPrior(0.0, 1.0),
            lengthscale_constraint=gpytorch.constraints.GreaterThan(2.5e-2),
        )
    )
    return AdditiveKernel(k1, k2)
