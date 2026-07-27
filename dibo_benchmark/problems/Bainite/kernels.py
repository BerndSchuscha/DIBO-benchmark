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

    k1 = ScaleKernel(
        PolynomialKernel(
            power=1,
            offset_prior=gpytorch.priors.LogNormalPrior(0.0, 1.0),
        )
    )

    k2 = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=7))

    return k1 + k2
