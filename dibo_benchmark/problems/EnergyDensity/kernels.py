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


def kernel_poly2():
    return ScaleKernel(
        PolynomialKernel(
            power=1,
            offset_prior=gpytorch.priors.LogNormalPrior(0.0, 1.0),
        )
    )
