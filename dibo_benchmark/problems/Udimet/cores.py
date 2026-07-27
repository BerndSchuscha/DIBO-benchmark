import math
import torch
from pibob.models.analytical_model import AnalyticalCore


class YieldStrengthPowerUdimet(AnalyticalCore):
    """
    Power-law analytical model for ys.

    Assumes x passed to phys_fun is normalized to [0, 1]
    using normalize(..., bounds=problem.bounds).

    Raw input order:
    [
        primary_mean_radius,
        primary_minimal_radius,
        primary_maximal_radius,
        primary_rsd,
        secondary_mean_radius,
        secondary_minimal_radius,
        secondary_maximal_radius,
        secondary_rsd,
        T1,
    ]

    Model:
        ys = exp(logA)
             * p_mean^a_p
             * p_rsd^a_prsd
             * s_mean^a_s
             * s_rsd^a_srsd
             * (p_mean / s_mean)^a_ratio
             * T1^a_T

    par = [logA, a_p, a_prsd, a_s, a_srsd, a_ratio, a_T]
    """

    N_PAR = 7
    INIT_LOC = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    PAR_INIT = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    INIT_SCALE = [3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    SIG1_INIT = 20

    LOG_EPS = 1e-12
    MAX_LOGARG = 1e30
    EXP_CLAMP_MIN = -40.0
    EXP_CLAMP_MAX = 40.0

    def __init__(self, problem):
        super().__init__()
        bounds = problem.bounds
        self.register_buffer("lb", bounds[0].clone())
        self.register_buffer("ub", bounds[1].clone())
        self.register_buffer("span", (bounds[1] - bounds[0]).clone())

    def _to_physical(self, x):
        return self.lb + x * self.span

    def _safe_log(self, z):
        return torch.log(torch.clamp(z, min=self.LOG_EPS, max=self.MAX_LOGARG))

    def _safe_exp(self, z):
        return torch.exp(torch.clamp(z, min=self.EXP_CLAMP_MIN, max=self.EXP_CLAMP_MAX))

    def _split_inputs(self, x):
        eps = self.LOG_EPS
        xp = self._to_physical(x)

        p_mean = torch.clamp(xp[..., 0], min=eps)
        p_rsd = torch.clamp(xp[..., 3], min=eps)
        s_mean = torch.clamp(xp[..., 4], min=eps)
        s_rsd = torch.clamp(xp[..., 7], min=eps)
        T1 = torch.clamp(xp[..., 8], min=eps)

        ratio = torch.clamp(p_mean / s_mean, min=eps)

        return {
            "xp": xp,
            "p_mean": p_mean,
            "p_rsd": p_rsd,
            "s_mean": s_mean,
            "s_rsd": s_rsd,
            "ratio": ratio,
            "T1": T1,
        }

    def phys_fun(self, x, par):
        logA, a_p, a_prsd, a_s, a_srsd, a_ratio, a_T = par.unbind(-1)
        v = self._split_inputs(x)

        log_mu = (
            logA
            + a_p * self._safe_log(v["p_mean"])
            + a_prsd * self._safe_log(v["p_rsd"])
            + a_s * self._safe_log(v["s_mean"])
            + a_srsd * self._safe_log(v["s_rsd"])
            + a_ratio * self._safe_log(v["ratio"])
            + a_T * self._safe_log(v["T1"])
        )
        return self._safe_exp(log_mu)


class YieldStrengthStrengtheningUdimet(AnalyticalCore):
    """
    Strengthening-style analytical model for ys.

    Model:
        ys =
            A
            + b_p / sqrt(p_mean)
            + b_s / sqrt(s_mean)
            + b_ratio * log(p_mean / s_mean)
            + b_prsd * log(1 + p_rsd)
            + b_srsd * log(1 + s_rsd)
            - b_T * T1

    par = [A, b_p, b_s, b_ratio, b_prsd, b_srsd, b_T]
    """

    N_PAR = 7
    INIT_LOC = [500.0, 10.0, 10.0, 0.0, 0.0, 0.0, 0.1]
    PAR_INIT = [500.0, 10.0, 10.0, 0.0, 0.0, 0.0, 0.1]
    INIT_SCALE = [300.0, 50.0, 50.0, 20.0, 20.0, 20.0, 1.0]
    SIG1_INIT = 20

    LOG_EPS = 1e-12
    MAX_LOGARG = 1e30

    def __init__(self, problem):
        super().__init__()
        bounds = problem.bounds
        self.register_buffer("lb", bounds[0].clone())
        self.register_buffer("ub", bounds[1].clone())
        self.register_buffer("span", (bounds[1] - bounds[0]).clone())

    def _to_physical(self, x):
        return self.lb + x * self.span

    def _safe_log(self, z):
        return torch.log(torch.clamp(z, min=self.LOG_EPS, max=self.MAX_LOGARG))

    def _split_inputs(self, x):
        eps = self.LOG_EPS
        xp = self._to_physical(x)

        p_mean = torch.clamp(xp[..., 0], min=eps)
        p_rsd = torch.clamp(xp[..., 3], min=eps)
        s_mean = torch.clamp(xp[..., 4], min=eps)
        s_rsd = torch.clamp(xp[..., 7], min=eps)
        T1 = xp[..., 8]

        ratio = torch.clamp(p_mean / s_mean, min=eps)

        return {
            "xp": xp,
            "p_mean": p_mean,
            "p_rsd": p_rsd,
            "s_mean": s_mean,
            "s_rsd": s_rsd,
            "ratio": ratio,
            "T1": T1,
        }

    def phys_fun(self, x, par):
        A, b_p, b_s, b_ratio, b_prsd, b_srsd, b_T = par.unbind(-1)
        v = self._split_inputs(x)

        mu = (
            A
            + b_p / torch.sqrt(v["p_mean"])
            + b_s / torch.sqrt(v["s_mean"])
            + b_ratio * self._safe_log(v["ratio"])
            + b_prsd * self._safe_log(1.0 + v["p_rsd"])
            + b_srsd * self._safe_log(1.0 + v["s_rsd"])
            - b_T * v["T1"]
        )
        return mu


class FgammaPoly3T1(AnalyticalCore):
    """
    3rd-order polynomial model for fgamma using only T1.

    Assumes x passed to phys_fun is normalized to [0, 1]
    using normalize(..., bounds=problem.bounds).

    Raw input order:
    [
        primary_mean_radius,
        primary_minimal_radius,
        primary_maximal_radius,
        primary_rsd,
        secondary_mean_radius,
        secondary_minimal_radius,
        secondary_maximal_radius,
        secondary_rsd,
        T1,
    ]

    Model:
        fgamma = a0 + a1*T1 + a2*T1^2 + a3*T1^3

    par = [a0, a1, a2, a3]
    """

    N_PAR = 4
    INIT_LOC = [0.1, 0.0, 0.0, 0.0]
    PAR_INIT = [0.1, 0.0, 0.0, 0.0]
    INIT_SCALE = [0.2, 1e-3, 1e-6, 1e-9]
    SIG1_INIT = 0.05

    def __init__(self, problem):
        super().__init__()
        bounds = problem.bounds
        self.register_buffer("lb", bounds[0].clone())
        self.register_buffer("ub", bounds[1].clone())
        self.register_buffer("span", (bounds[1] - bounds[0]).clone())

    def _to_physical(self, x):
        return self.lb + x * self.span

    def phys_fun(self, x, par):
        a0, a1, a2, a3 = par.unbind(-1)

        xp = self._to_physical(x)
        T1 = xp[..., 8]

        mu = a0 + a1 * T1 + a2 * T1**2 + a3 * T1**3
        return mu


class core_linear(AnalyticalCore):

    N_PAR = 10
    INIT_LOC = [0.0] * 10
    PAR_INIT = [0.0] * 10
    INIT_SCALE = [100.0] * 10
    SIG1_INIT = 100.0

    def phys_fun(self, x, par):
        bias = par[..., 0]
        weights = par[..., 1:]  # shape (..., 7)

        # Linear model: y = bias + sum_i w_i * x_i
        y = bias + torch.sum(weights * x, dim=-1)

        return y
