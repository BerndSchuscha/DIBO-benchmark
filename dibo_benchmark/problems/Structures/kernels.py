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


def kernel_linear_matern():
    k1 = ScaleKernel(
        LinearKernel(
            variance_prior=gpytorch.priors.LogNormalPrior(0.0, 1.0),
        )
    )
    k2 = ScaleKernel(
        MaternKernel(
            nu=2.5,
            ard_num_dims=13,
            lengthscale_prior=gpytorch.priors.LogNormalPrior(0.0, 1.0),
            lengthscale_constraint=gpytorch.constraints.GreaterThan(2.5e-2),
        )
    )
    return AdditiveKernel(k1, k2)

def kernel_poly_rbf_product():
    k1 = ScaleKernel(
        PolynomialKernel(
            power=2,
            offset_prior=gpytorch.priors.LogNormalPrior(0.0, 1.0),
        )
    )
    k2 = ScaleKernel(
        RBFKernel(
            ard_num_dims=13,
            lengthscale_prior=gpytorch.priors.LogNormalPrior(0.0, 1.0),
            lengthscale_constraint=gpytorch.constraints.GreaterThan(2.5e-2),
        )
    )
    return ProductKernel(k1, k2)
