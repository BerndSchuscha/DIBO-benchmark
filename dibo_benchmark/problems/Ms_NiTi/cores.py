import torch
from pibob.models.analytical_model import AnalyticalCore
import torch
import torch.nn.functional as F

# ============================================================
# Branin variants
# ============================================================


class core1(AnalyticalCore):


    N_PAR = 2
    INIT_LOC = [250, 0]
    INIT_SCALE = [500, 500]
    PAR_INIT = [250, 0]
    SIG1_INIT = 40

    def phys_fun(self, x, par):
        A, B = par[..., 0], par[..., 1]
        return A + (x[..., 0] - 0.5) * B


class core2(AnalyticalCore):
    """
    Valley residual model
    par = [logA, B, C]
    """

    N_PAR = 4
    INIT_LOC = [330, 0.0, 0.0, 1.0]
    PAR_INIT = [330, 0.0, 0.0, 1.0]
    INIT_SCALE = [500.0, 500.0, 500.0, 5.0]
    SIG1_INIT = 20

    def phys_fun(self, x, par):
        A, B, C, logD = par[..., 0], par[..., 1], par[..., 2], par[..., 3]
        D = torch.exp(logD)
        y = A + x[..., 0] * B + C * D ** (x[..., 0] - 0.5)
        return y


class core3(AnalyticalCore):
    """
    Valley residual model
    par = [logA, B, C]
    """

    N_PAR = 4
    INIT_LOC = [330, 0.0, 0.0, 0.0]
    PAR_INIT = [330, 0.0, 0.0, 0.0]
    INIT_SCALE = [500.0, 500.0, 500.0, 500.0]
    SIG1_INIT = 20

    def phys_fun(self, x, par):
        A, B, C, D = par[..., 0], par[..., 1], par[..., 2], par[..., 3]

        y = A + x[..., 0] * B + C / x[..., 2] + D * torch.sqrt(-x[..., 2])
        return y
