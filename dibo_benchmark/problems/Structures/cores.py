import math
import torch
from pibob.models.analytical_model import AnalyticalCore


class _ALSBase(AnalyticalCore):
    """
    Base class for ALS analytical cores.

    Assumes x passed to phys_fun is normalized to [0, 1] using
    botorch.utils.transforms.normalize(..., bounds=problem.bounds).

    Descriptor order:
    [TargetHeight, WallThickness, AveragePerimeter,
     x1, x2, x3, x4, x5, x6, x7, x8,
     Modulus, PlateauStrength]
    """

    LOG_EPS = 1e-12
    MIN_WIDTH = 5e-2
    MAX_LOGARG = 1e30
    EXP_CLAMP_MIN = -40.0
    EXP_CLAMP_MAX = 40.0
    ASYM_SCALE = 2.0

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

    def _positive_width(self, raw_log_s):
        # exp(raw) but never too small
        return torch.clamp(torch.exp(raw_log_s), min=self.MIN_WIDTH)

    def _bounded_asym(self, z, alpha):
        # bounded odd asymmetry term; cannot run away
        # scale controls how quickly tanh saturates
        return alpha * torch.tanh(z / self.ASYM_SCALE)

    def _split_inputs(self, x):
        eps = self.LOG_EPS
        xp = self._to_physical(x)

        h = torch.clamp(xp[..., 0], min=eps)
        t = torch.clamp(xp[..., 1], min=eps)
        P = torch.clamp(xp[..., 2], min=eps)
        shape = xp[..., 3:11]
        E = torch.clamp(xp[..., 11], min=eps)
        sig_p = torch.clamp(xp[..., 12], min=eps)

        d = torch.clamp(P / (2.0 * torch.pi), min=eps)
        d_over_t = torch.clamp(d / t, min=eps)
        omega = torch.clamp(h / torch.sqrt(torch.clamp(d * t, min=eps)), min=eps)

        return {
            "xp": xp,
            "h": h,
            "t": t,
            "P": P,
            "shape": shape,
            "E": E,
            "sig_p": sig_p,
            "d": d,
            "d_over_t": d_over_t,
            "omega": omega,
        }


# ---------------------------------------------------------------------
# BASIC PAIR: BOTH HAVE N_PAR = 6
# ---------------------------------------------------------------------


class CriticalStressALS(_ALSBase):
    """
    Analytical power-law model for critical stress.

    par = [logA, a_sigp, a_dt, a_omega, a_E, a_h]
    """

    N_PAR = 6
    INIT_LOC = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    PAR_INIT = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    INIT_SCALE = [3.0, 1.0, 0.5, 0.5, 0.5, 0.5]
    SIG1_INIT = 20

    def __init__(self, problem):
        super().__init__(problem)

    def phys_fun(self, x, par):
        logA, a_sigp, a_dt, a_omega, a_E, a_h = par.unbind(-1)
        v = self._split_inputs(x)

        log_mu = (
            logA
            + a_sigp * self._safe_log(v["sig_p"])
            + a_dt * self._safe_log(v["d_over_t"])
            + a_omega * self._safe_log(v["omega"])
            + a_E * self._safe_log(v["E"])
            + a_h * self._safe_log(v["h"])
        )
        return self._safe_exp(log_mu)


class CriticalEfficiencyALS(_ALSBase):
    """
    Analytical peaked model for critical efficiency.

    par = [K0, logK1, mu_dt, mu_omega, logs_dt, logs_omega]
    """

    N_PAR = 6
    INIT_LOC = [0.2, 0.0, 0.0, 0.0, -0.5, -0.5]
    PAR_INIT = [0.2, 0.0, 0.0, 0.0, -0.5, -0.5]
    INIT_SCALE = [0.5, 1.5, 1.0, 1.0, 0.5, 0.5]
    SIG1_INIT = 20

    def __init__(self, problem):
        super().__init__(problem)

    def phys_fun(self, x, par):
        K0, logK1, mu_dt, mu_omega, logs_dt, logs_omega = par.unbind(-1)

        v = self._split_inputs(x)

        K1 = self._safe_exp(logK1)
        s_dt = self._positive_width(logs_dt)
        s_omega = self._positive_width(logs_omega)

        log_dt = self._safe_log(v["d_over_t"])
        log_omega = self._safe_log(v["omega"])

        z_dt = (log_dt - mu_dt) / s_dt
        z_omega = (log_omega - mu_omega) / s_omega

        exponent = -0.5 * z_dt**2 - 0.5 * z_omega**2
        return K0 + K1 * self._safe_exp(exponent)


# ---------------------------------------------------------------------
# SHAPE-ADDITIVE PAIR: BOTH HAVE N_PAR = 15
# ---------------------------------------------------------------------


class CriticalStressALSShape(_ALSBase):
    """
    Critical stress with geometry physics + linear shape correction.

    par = [logA, a_sigp, a_dt, a_omega, a_E, a_h, b1, ..., b8, c0]
    """

    N_PAR = 15
    INIT_LOC = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0] + [0.0] * 8 + [0.0]
    PAR_INIT = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0] + [0.0] * 8 + [0.0]
    INIT_SCALE = [3.0, 1.0, 0.5, 0.5, 0.5, 0.5] + [0.25] * 8 + [0.5]
    SIG1_INIT = 20

    def __init__(self, problem):
        super().__init__(problem)

    def phys_fun(self, x, par):
        logA, a_sigp, a_dt, a_omega, a_E, a_h = par[..., 0:6].unbind(-1)
        b = par[..., 6:14]
        c0 = par[..., 14]

        v = self._split_inputs(x)

        log_base = (
            logA
            + a_sigp * self._safe_log(v["sig_p"])
            + a_dt * self._safe_log(v["d_over_t"])
            + a_omega * self._safe_log(v["omega"])
            + a_E * self._safe_log(v["E"])
            + a_h * self._safe_log(v["h"])
        )
        base = self._safe_exp(log_base)

        # mild additive correction; tanh prevents runaway from shape coefficients
        shape_term = 0.1 * torch.tanh(c0 + torch.sum(v["shape"] * b, dim=-1))

        # preserve positivity
        return torch.clamp(base * (1.0 + shape_term), min=self.LOG_EPS)


class CriticalEfficiencyALSShape(_ALSBase):
    """
    Critical efficiency with peaked geometry physics + linear shape correction.

    par = [K0, logK1, mu_dt, mu_omega, logs_dt, logs_omega, b1, ..., b8, c0]
    """

    N_PAR = 15
    INIT_LOC = [0.2, 0.0, 0.0, 0.0, -0.5, -0.5] + [0.0] * 8 + [0.0]
    PAR_INIT = [0.2, 0.0, 0.0, 0.0, -0.5, -0.5] + [0.0] * 8 + [0.0]
    INIT_SCALE = [0.5, 1.5, 1.0, 1.0, 0.5, 0.5] + [0.25] * 8 + [0.5]
    SIG1_INIT = 20

    def __init__(self, problem):
        super().__init__(problem)

    def phys_fun(self, x, par):
        K0, logK1, mu_dt, mu_omega, logs_dt, logs_omega = par[..., 0:6].unbind(-1)
        b = par[..., 6:14]
        c0 = par[..., 14]

        v = self._split_inputs(x)

        K1 = self._safe_exp(logK1)
        s_dt = self._positive_width(logs_dt)
        s_omega = self._positive_width(logs_omega)

        log_dt = self._safe_log(v["d_over_t"])
        log_omega = self._safe_log(v["omega"])

        z_dt = (log_dt - mu_dt) / s_dt
        z_omega = (log_omega - mu_omega) / s_omega

        exponent = -0.5 * z_dt**2 - 0.5 * z_omega**2
        base = K0 + K1 * self._safe_exp(exponent)

        shape_term = 0.1 * torch.tanh(c0 + torch.sum(v["shape"] * b, dim=-1))
        return base * (1.0 + shape_term)


# ---------------------------------------------------------------------
# SHAPE-MULTIPLICATIVE STRESS MODEL: N_PAR = 15
# ---------------------------------------------------------------------


class CriticalStressALSShapeMul(_ALSBase):
    """
    Critical stress:
    power-law geometry/material scaling
    multiplied by a positive bounded shape correction.

    par = [logA, a_sigp, a_dt, a_omega, a_E, a_h, c0, c1, ..., c8]
    """

    N_PAR = 15
    INIT_LOC = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0] + [0.0] * 9
    PAR_INIT = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0] + [0.0] * 9
    INIT_SCALE = [3.0, 1.0, 0.5, 0.5, 0.5, 0.5] + [0.25] * 9
    SIG1_INIT = 20

    def __init__(self, problem):
        super().__init__(problem)

    def phys_fun(self, x, par):
        logA, a_sigp, a_dt, a_omega, a_E, a_h = par[..., 0:6].unbind(-1)
        c = par[..., 6:15]

        v = self._split_inputs(x)

        log_base = (
            logA
            + a_sigp * self._safe_log(v["sig_p"])
            + a_dt * self._safe_log(v["d_over_t"])
            + a_omega * self._safe_log(v["omega"])
            + a_E * self._safe_log(v["E"])
            + a_h * self._safe_log(v["h"])
        )
        base = self._safe_exp(log_base)

        shape_raw = c[..., 0] + torch.sum(v["shape"] * c[..., 1:9], dim=-1)

        # bounded multiplicative factor in roughly [exp(-0.5), exp(0.5)]
        shape_factor = self._safe_exp(0.5 * torch.tanh(shape_raw))

        return base * shape_factor


# ---------------------------------------------------------------------
# ASYMMETRIC PAIR: BOTH HAVE N_PAR = 7
# ---------------------------------------------------------------------


class CriticalStressALSAsym(_ALSBase):
    """
    Critical stress:
    power-law model with bounded asymmetric omega contribution.

    par = [logA, a_sigp, a_dt, a_omega, a_E, a_h, alpha]
    """

    N_PAR = 7
    INIT_LOC = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    PAR_INIT = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    INIT_SCALE = [3.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5]
    SIG1_INIT = 20

    def __init__(self, problem):
        super().__init__(problem)

    def phys_fun(self, x, par):
        logA, a_sigp, a_dt, a_omega, a_E, a_h, alpha = par.unbind(-1)
        v = self._split_inputs(x)

        log_sigp = self._safe_log(v["sig_p"])
        log_dt = self._safe_log(v["d_over_t"])
        log_omega = self._safe_log(v["omega"])
        log_E = self._safe_log(v["E"])
        log_h = self._safe_log(v["h"])

        # bounded odd asymmetry, no cubic runaway
        omega_term = a_omega * log_omega + self._bounded_asym(log_omega, alpha)

        log_mu = (
            logA
            + a_sigp * log_sigp
            + a_dt * log_dt
            + omega_term
            + a_E * log_E
            + a_h * log_h
        )
        return self._safe_exp(log_mu)


class CriticalEfficiencyALSAsym(_ALSBase):
    """
    Critical efficiency:
    peaked log-space model with bounded asymmetric omega response.

    par = [K0, logK1, mu_dt, mu_omega, logs_dt, logs_omega, alpha]
    """

    N_PAR = 7
    INIT_LOC = [0.2, 0.0, 0.0, 0.0, -0.5, -0.5, 0.0]
    PAR_INIT = [0.2, 0.0, 0.0, 0.0, -0.5, -0.5, 0.0]
    INIT_SCALE = [0.5, 1.5, 1.0, 1.0, 0.5, 0.5, 0.5]
    SIG1_INIT = 20

    def __init__(self, problem):
        super().__init__(problem)

    def phys_fun(self, x, par):
        K0, logK1, mu_dt, mu_omega, logs_dt, logs_omega, alpha = par.unbind(-1)

        v = self._split_inputs(x)

        s_dt = self._positive_width(logs_dt)
        s_omega = self._positive_width(logs_omega)
        K1 = self._safe_exp(logK1)

        log_dt = self._safe_log(v["d_over_t"])
        log_omega = self._safe_log(v["omega"])

        z_dt = (log_dt - mu_dt) / s_dt
        z_omega = (log_omega - mu_omega) / s_omega

        # still asymmetric, but bounded; exponent stays controlled
        exponent = (
            -0.5 * z_dt**2 - 0.5 * z_omega**2 + self._bounded_asym(z_omega, alpha)
        )

        return K0 + K1 * self._safe_exp(exponent)


class core_linear(AnalyticalCore):


    N_PAR = 14
    INIT_LOC = [0.0] * 14
    PAR_INIT = [0.0] * 14
    INIT_SCALE = [100.0] * 14
    SIG1_INIT = 100.0

    def phys_fun(self, x, par):
        bias = par[..., 0]
        weights = par[..., 1:]  # shape (..., 7)

        # Linear model: y = bias + sum_i w_i * x_i
        y = bias + torch.sum(weights * x, dim=-1)

        return y
