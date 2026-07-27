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


def kernel_additive():
    k_fe = ScaleKernel(gpytorch.kernels.RBFKernel(active_dims=[0]))
    k_co = ScaleKernel(gpytorch.kernels.RBFKernel(active_dims=[1]))
    k_ni = ScaleKernel(gpytorch.kernels.RBFKernel(active_dims=[2]))

    return k_fe + k_co + k_ni
