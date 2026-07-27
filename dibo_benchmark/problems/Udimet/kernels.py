import torch
from gpytorch.kernels import (
    AdditiveKernel,
    ScaleKernel,
    MaternKernel,
    PeriodicKernel,
    RBFKernel,
    LinearKernel,
    ProductKernel,
)
from gpytorch.kernels import ScaleKernel, PolynomialKernel
import gpytorch



def kernel_ys_optimization():
    return ScaleKernel(
        MaternKernel(
            nu=2.5,
            ard_num_dims=9,
            lengthscale_prior=gpytorch.priors.LogNormalPrior(-0.2, 0.8),
            lengthscale_constraint=gpytorch.constraints.GreaterThan(7.5e-2),
        ),
        outputscale_prior=gpytorch.priors.LogNormalPrior(0.0, 1.0),
    )
