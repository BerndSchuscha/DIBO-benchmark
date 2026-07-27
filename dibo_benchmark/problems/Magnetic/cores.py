import math
import torch
from torch import nn
from pibob.models.analytical_model import AnalyticalCore


class _FeCoNiBase(AnalyticalCore):
    def __init__(self, problem):
        super().__init__()
        self.problem = problem

    def _safe_log(self, x, eps=1e-12):
        return torch.log(torch.clamp(x, min=eps))

    def _safe_exp(self, x):
        return torch.exp(torch.clamp(x, min=-50.0, max=50.0))

    def _positive_width(self, x):
        return torch.nn.functional.softplus(x) + 1e-8

    def _bounded_asym(self, z, alpha):
        return alpha * torch.tanh(z)

    def _split_inputs(self, x):
        Fe = x[..., 0]
        Co = x[..., 1]
        Ni = x[..., 2]

        FeCo = Fe * Co
        log_Fe_over_Ni = self._safe_log(Fe) - self._safe_log(Ni)

        return {
            "Fe": Fe,
            "Co": Co,
            "Ni": Ni,
            "FeCo": FeCo,
            "log_Fe_over_Ni": log_Fe_over_Ni,
        }


class KerrFeCoNiAsym(_FeCoNiBase):
    """
    Kerr response:
    peaked in Fe-Co interaction and composition ratio.

    par = [K0, logK1, mu_mix, mu_ratio, logs_mix, logs_ratio, alpha]
    """

    N_PAR = 7
    INIT_LOC = [0.0, 0.0, -2.0, 0.0, -0.5, -0.5, 0.0]
    PAR_INIT = [0.0, 0.0, -2.0, 0.0, -0.5, -0.5, 0.0]
    INIT_SCALE = [5.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0]
    SIG1_INIT = 20

    def __init__(self, problem):
        super().__init__(problem)

    def phys_fun(self, x, par):
        K0, logK1, mu_mix, mu_ratio, logs_mix, logs_ratio, alpha = par.unbind(-1)

        v = self._split_inputs(x)

        s_mix = self._positive_width(logs_mix)
        s_ratio = self._positive_width(logs_ratio)
        K1 = self._safe_exp(logK1)

        log_mix = self._safe_log(v["FeCo"])
        ratio = v["log_Fe_over_Ni"]

        z_mix = (log_mix - mu_mix) / s_mix
        z_ratio = (ratio - mu_ratio) / s_ratio

        exponent = (
            -0.5 * z_mix**2 - 0.5 * z_ratio**2 + self._bounded_asym(z_ratio, alpha)
        )

        return K0 + K1 * self._safe_exp(exponent)


class CoercivityFeCoNiAsym(_FeCoNiBase):
    """
    Coercivity response:
    peaked in Fe-Co interaction and composition ratio,
    but with asymmetry primarily along the interaction axis.

    par = [H0, logH1, mu_mix, mu_ratio, logs_mix, logs_ratio, alpha]
    """

    N_PAR = 7
    INIT_LOC = [0.0, 0.0, -2.0, 0.0, -0.5, -0.5, 0.0]
    PAR_INIT = [0.0, 0.0, -2.0, 0.0, -0.5, -0.5, 0.0]
    INIT_SCALE = [5.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0]
    SIG1_INIT = 20

    def __init__(self, problem):
        super().__init__(problem)

    def phys_fun(self, x, par):
        H0, logH1, mu_mix, mu_ratio, logs_mix, logs_ratio, alpha = par.unbind(-1)

        v = self._split_inputs(x)

        s_mix = self._positive_width(logs_mix)
        s_ratio = self._positive_width(logs_ratio)
        H1 = self._safe_exp(logH1)

        log_mix = self._safe_log(v["FeCo"])
        ratio = v["log_Fe_over_Ni"]

        z_mix = (log_mix - mu_mix) / s_mix
        z_ratio = (ratio - mu_ratio) / s_ratio

        exponent = -0.5 * z_mix**2 - 0.5 * z_ratio**2 + self._bounded_asym(z_mix, alpha)

        return H0 + H1 * self._safe_exp(exponent)


class LinearFeCoNi4(AnalyticalCore):
    """
    Minimal 4-parameter linear model.

    Uses only:
        Fe, Co, Ni

    par = [b0, bFe, bCo, bNi]
    """

    N_PAR = 4
    INIT_LOC = [0.0, 0.0, 0.0, 0.0]
    PAR_INIT = [0.0, 0.0, 0.0, 0.0]
    INIT_SCALE = [1.0, 1.0, 1.0, 1.0]
    SIG1_INIT = 20

    def phys_fun(self, x, par):
        b0, bFe, bCo, bNi = par.unbind(-1)

        Fe, Co, Ni = x[..., 0], x[..., 1], x[..., 2]
        y = b0 + bFe * Fe + bCo * Co + bNi * Ni

        return y
