import torch
from pibob.models.analytical_model import AnalyticalCore, PoolDescriptorCore
import torch
import torch.nn.functional as F
from pibob.problems.Bainite.Bainite_Problem import Bainite_Problem

problem = Bainite_Problem()


class Bainitecore(PoolDescriptorCore):
    N_PAR = 2
    INIT_LOC = [0, 1]
    INIT_SCALE = [10, 10]
    PAR_INIT = [0, 1]
    SIG1_INIT = 40

    def __init__(self, descriptor_name=""):
        super().__init__(problem=problem, descriptor_name=descriptor_name)

    def phys_fun(self, x, par):
        A, B = par[..., 0], par[..., 1]
        d = self.lookup_descriptor(x)
        return A + d * B


class core_linear7(AnalyticalCore):
    """
    7D linear model
    par = [bias, w1, w2, w3, w4, w5, w6, w7]
    """

    N_PAR = 8
    INIT_LOC = [0.0] * 8
    PAR_INIT = [0.0] * 8
    INIT_SCALE = [10.0] * 8
    SIG1_INIT = 100.0

    def phys_fun(self, x, par):
        bias = par[..., 0]
        weights = par[..., 1:]  # shape (..., 7)

        # Linear model: y = bias + sum_i w_i * x_i
        y = bias + torch.sum(weights * x, dim=-1)

        return y


class core_linear1D(AnalyticalCore):
    """
    1D linear model
    par = [A, B]
    """

    N_PAR = 2
    INIT_LOC = [0.0, 0.0]
    PAR_INIT = [0.0, 0.0]
    INIT_SCALE = [100.0, 100.0]
    SIG1_INIT = 100.0

    def phys_fun(self, x, par):
        A, B = par[..., 0], par[..., 1]
        y = A + B * x[..., -1]
        return y
