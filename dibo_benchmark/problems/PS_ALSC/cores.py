import torch
from pibob.models.analytical_model import AnalyticalCore, PoolDescriptorCore
import torch
import torch.nn.functional as F
from pibob.problems.PS_ALSC.PS_ALSC_Problem import PS_ALSC_Problem

problem = PS_ALSC_Problem()


class ALSCCore(PoolDescriptorCore):
    N_PAR = 2
    INIT_LOC = [0, -1]
    INIT_SCALE = [10, 10]
    PAR_INIT = [0, -1]
    SIG1_INIT = 40

    def __init__(self, descriptor_name="M_W_recover_cor_0"):
        super().__init__(problem=problem, descriptor_name=descriptor_name)

    def phys_fun(self, x, par):
        A, B = par[..., 0], par[..., 1]
        d = self.lookup_descriptor(x)
        return A + d * B


class ALSCCore1(AnalyticalCore):


    N_PAR = 6
    INIT_LOC = [0.00, 0.0, 0.0, 0.0, 0.0, 00.00]
    INIT_SCALE = [500.0, 500.0, 500.0, 500.0, 500.0, 500.0]
    PAR_INIT = [0.00, 0.0, 0.0, 0.0, 0.0, 00.00]
    SIG1_INIT = 20

    def phys_fun(self, x, par):
        A, B, C, D, E, F = (
            par[..., 0],
            par[..., 1],
            par[..., 2],
            par[..., 3],
            par[..., 4],
            par[..., 5],
        )
        y = (
            A
            + x[..., 0] * B
            + x[..., 1] * C
            + x[..., 2] * D
            + x[..., 3] * E
            + x[..., 4] * F
        )
        return y

