import torch
from gpytorch.kernels import (
    AdditiveKernel,
    ScaleKernel,
    MaternKernel,
    PeriodicKernel,
)
from gpytorch.kernels import ScaleKernel, PolynomialKernel
import gpytorch


def kernel_simplify_poly2():
    return ScaleKernel(
        PolynomialKernel(
            power=2,
            offset_prior=gpytorch.priors.LogNormalPrior(0.0, 1.0),
        )
    )
