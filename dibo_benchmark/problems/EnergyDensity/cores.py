import torch
from pibob.models.analytical_model import AnalyticalCore, PoolDescriptorCore
import torch
import torch.nn.functional as F
from pibob.problems.EnergyDensity.EnergyDensity_Problem import EnergyDensity_Problem

problem = EnergyDensity_Problem()


class EngnergyCore(PoolDescriptorCore):
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
    
class core_linear(AnalyticalCore):


    N_PAR = 6
    INIT_LOC = [0.0] * 6
    PAR_INIT = [0.0] * 6
    INIT_SCALE = [100.0] * 6
    SIG1_INIT = 100.0

    def phys_fun(self, x, par):
        bias = par[..., 0]
        weights = par[..., 1:]  # shape (..., 7)

        # Linear model: y = bias + sum_i w_i * x_i
        y = bias + torch.sum(weights * x, dim=-1)

        return y
