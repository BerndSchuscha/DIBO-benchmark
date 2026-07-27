import torch
from pibob.models.analytical_model import AnalyticalCore
import torch
import torch.nn.functional as F

# ============================================================
# Branin variants
# ============================================================


class BraninVar1(AnalyticalCore):
    """
    y = A * Branin(x) + B
    par = [logA, B]
    """

    N_PAR = 2
    INIT_LOC = [1.0, 0.0]
    INIT_SCALE = [10.0, 10.0]
    PAR_INIT = [0.5, 0.0]
    SIG1_INIT= 20

    def phys_fun(self, x, par):
        A, B = par[..., 0], par[..., 1]

        x1 = 15.0 * x[..., 0] - 5.0
        x2 = 15.0 * x[..., 1]

        b = 5.1 / (4.0 * torch.pi**2)
        c = 5.0 / torch.pi
        t = 1.0 / (8.0 * torch.pi)

        branin = (
            (x2 - b * x1**2 + c * x1 - 6.0) ** 2
            + 10.0 * (1.0 - t) * torch.cos(x1)
            + 10.0
        )
        return -(A * branin + B)


class BraninVar2(AnalyticalCore):
    """
    Valley residual model
    par = [logA, B, C]
    """

    N_PAR = 3
    INIT_LOC = [0.0, 0.0, 10.0]
    PAR_INIT = [0.0, 0.0, 10.0]
    INIT_SCALE = [1.0, 10.0, 10.0]
    SIG1_INIT= 20

    def phys_fun(self, x, par):
        logA, B, C = par[..., 0], par[..., 1], par[..., 2]
        A = torch.exp(logA)

        x1 = 15.0 * x[..., 0] - 5.0
        x2 = 15.0 * x[..., 1]

        b = 5.1 / (4.0 * torch.pi**2)
        c = 5.0 / torch.pi

        valley = x2 - b * x1**2 + c * x1 - 6.0
        y = A * valley**2 + C * torch.cos(x1) + B
        return -y


class BraninVar3(AnalyticalCore):
    """
    Exact Branin + linear + quadratic ridge
    par = [logA, B, L1, L2, log_eps]
    """

    N_PAR = 5
    INIT_LOC = [0.0, 0.0, 0.0, 0.0, -10.0]
    PAR_INIT = [0.0, 0.0, 0.0, 0.0, -10.0]
    INIT_SCALE = [10.0, 10.0, 10.0, 10.0, 10.0]
    SIG1_INIT= 20

    def phys_fun(self, x, par):
        logA, B, L1, L2, log_eps = (
            par[..., 0],
            par[..., 1],
            par[..., 2],
            par[..., 3],
            par[..., 4],
        )

        A = torch.exp(logA)
        eps = torch.exp(log_eps)

        x1 = 15.0 * x[..., 0] - 5.0
        x2 = 15.0 * x[..., 1]

        b = 5.1 / (4.0 * torch.pi**2)
        c = 5.0 / torch.pi
        t = 1.0 / (8.0 * torch.pi)

        branin = (
            (x2 - b * x1**2 + c * x1 - 6.0) ** 2
            + 10.0 * (1.0 - t) * torch.cos(x1)
            + 10.0
        )

        extra = L1 * x1 + L2 * x2 + eps * (x2 - 7.5) ** 2
        return -(A * branin + B + extra)


class BraninVar4(AnalyticalCore):
    """
    Exact Branin + interior-only perturbations
    par = [logA, B, P1, P2, P3]
    """

    N_PAR = 5
    INIT_LOC = [0.0, 0.0, 0.0, 0.0, 0.0]
    PAR_INIT = [0.0, 0.0, 0.0, 0.0, 0.0]
    INIT_SCALE = [10.0, 10.0, 10.0, 10.0, 10.0]
    SIG1_INIT= 20

    def phys_fun(self, x, par):
        logA, B, P1, P2, P3 = (
            par[..., 0],
            par[..., 1],
            par[..., 2],
            par[..., 3],
            par[..., 4],
        )
        A = torch.exp(logA)

        x1p = 15.0 * x[..., 0] - 5.0
        x2p = 15.0 * x[..., 1]

        b = 5.1 / (4.0 * torch.pi**2)
        c = 5.0 / torch.pi
        t = 1.0 / (8.0 * torch.pi)

        branin = (
            (x2p - b * x1p**2 + c * x1p - 6.0) ** 2
            + 10.0 * (1.0 - t) * torch.cos(x1p)
            + 10.0
        )

        s1 = x[..., 0] * (1.0 - x[..., 0])
        s2 = x[..., 1] * (1.0 - x[..., 1])

        extra = (
            P1 * (s1 * s2)
            + P2 * (s1 * torch.cos(2.0 * torch.pi * x[..., 0]))
            + P3 * (s2 * (x[..., 1] - 0.5))
        )

        return -(A * branin + B + extra)


# ============================================================
# Currin variants
# ============================================================


class CurrinVar1(AnalyticalCore):
    """
    y = A * Currin(x) + B
    par = [logA, B]
    """

    N_PAR = 2
    INIT_LOC = [0.0, 0.0]
    INIT_SCALE = [10.0, 10.0]
    PAR_INIT = [0.0, 0.0]
    SIG1_INIT= 2

    def phys_fun(self, x, par):
        logA, B = par[..., 0], par[..., 1]
        A = torch.exp(logA)

        x1 = x[..., 0]
        x2 = x[..., 1].clamp_min(1e-6)

        num = 2300.0 * x1**3 + 1900.0 * x1**2 + 2092.0 * x1 + 60.0
        den = 100.0 * x1**3 + 500.0 * x1**2 + 4.0 * x1 + 20.0

        currin = (1.0 - torch.exp(-1.0 / (2.0 * x2))) * (num / den)
        return -(A * currin + B)


class CurrinVar2(AnalyticalCore):
    """
    Currin with learnable gate sharpness
    par = [k, logA, B]
    """

    N_PAR = 3
    INIT_LOC = [1.0, 0.0, 0.0]
    PAR_INIT = [1.0, 0.0, 0.0]
    INIT_SCALE = [1.0, 10.0, 10.0]
    SIG1_INIT= 2

    def phys_fun(self, x, par):
        k_raw, logA, B = par[..., 0], par[..., 1], par[..., 2]

        # Enforce k > 0 (prevents exp overflow in the gate)
        k = F.softplus(k_raw) + 1e-8

        # Prevent exp overflow for A (pick a max appropriate for your scale)
        logA_clamped = logA.clamp(max=50.0)  # exp(50) ~ 5e21 (already huge)
        A = torch.exp(logA_clamped)

        x1 = x[..., 0]
        x2 = x[..., 1].clamp_min(1e-6)

        num = 2300.0 * x1**3 + 1900.0 * x1**2 + 2092.0 * x1 + 60.0
        den = 100.0 * x1**3 + 500.0 * x1**2 + 4.0 * x1 + 20.0

        # Stable gate computation
        z = -k / (2.0 * x2)  # z <= 0 always now
        z = z.clamp(min=-80.0)  # exp(-80) ~ 1.8e-35, safe in float32 too
        gate = 1.0 - torch.exp(z)

        currin = gate * (num / den)
        return -(A * currin + B)


class CurrinVar3(AnalyticalCore):
    """
    Exact Currin + linear terms
    par = [logA, B, L1, L2, log_x2eps]
    """

    N_PAR = 5
    INIT_LOC = [0.0, 0.0, 0.0, 0.0, -14.0]
    PAR_INIT = [0.0, 0.0, 0.0, 0.0, -14.0]
    INIT_SCALE = [10.0, 10.0, 10.0, 10.0, 10.0]
    SIG1_INIT= 2

    def phys_fun(self, x, par):
        logA, B, L1, L2, log_x2eps = (
            par[..., 0],
            par[..., 1],
            par[..., 2],
            par[..., 3],
            par[..., 4],
        )

        A = torch.exp(logA)
        x2eps = torch.exp(log_x2eps)

        x1 = x[..., 0]
        x2 = x[..., 1] + x2eps

        num = 2300.0 * x1**3 + 1900.0 * x1**2 + 2092.0 * x1 + 60.0
        den = 100.0 * x1**3 + 500.0 * x1**2 + 4.0 * x1 + 20.0

        currin = (1.0 - torch.exp(-1.0 / (2.0 * x2))) * (num / den)
        return -(A * currin + B + L1 * x1 + L2 * x2)


class CurrinVar4(AnalyticalCore):
    """
    Exact Currin + interior-only perturbations
    par = [logA, B, P1, P2, P3]
    """

    N_PAR = 5
    INIT_LOC = [0.0, 0.0, 0.0, 0.0, 0.0]
    PAR_INIT = [0.0, 0.0, 0.0, 0.0, 0.0]
    INIT_SCALE = [10.0, 10.0, 10.0, 10.0, 10.0]
    SIG1_INIT= 2

    def phys_fun(self, x, par):
        logA, B, P1, P2, P3 = (
            par[..., 0],
            par[..., 1],
            par[..., 2],
            par[..., 3],
            par[..., 4],
        )
        A = torch.exp(logA)

        x1 = x[..., 0]
        x2 = x[..., 1].clamp_min(1e-6)

        num = 2300.0 * x1**3 + 1900.0 * x1**2 + 2092.0 * x1 + 60.0
        den = 100.0 * x1**3 + 500.0 * x1**2 + 4.0 * x1 + 20.0

        currin = (1.0 - torch.exp(-1.0 / (2.0 * x2))) * (num / den)

        s1 = x1 * (1.0 - x1)
        s2 = x2 * (1.0 - x2)

        extra = (
            P1 * (s1 * s2)
            + P2 * (s2 * torch.log1p(9.0 * x2))
            + P3 * (s1 * (2.0 * x1 - 1.0))
        )

        return -(A * currin + B + extra)


class BraninVar5(AnalyticalCore):
    """
    y = A * Branin(x) + B
    par = [logA, B]
    """

    N_PAR = 5
    INIT_LOC = [1.0, 1.0, 1.0, 10.0, 10.0]
    PAR_INIT = [1.0, 1.0, 1.0, 10.0, 10.0]
    INIT_SCALE = [10.0, 10.0, 10.0, 10.0, 10.0]
    SIG1_INIT= 20

    def phys_fun(self, x, par):
        A, B, C, D, E = par[..., 0], par[..., 1], par[..., 2], par[..., 3]

        x1 = 15.0 * x[..., 0] - 5.0
        x2 = 15.0 * x[..., 1]

        b = 5.1 / (4.0 * torch.pi**2)
        c = 5.0 / torch.pi
        t = 1.0 / (8.0 * torch.pi)

        branin = (
            (x2 - b * A * x1**2 + c * B * x1 - 6.0 * C) ** 2
            + D * (1.0 - t) * torch.cos(x1)
            + E
        )
        return branin


class CurrinVar5(AnalyticalCore):
    """
    Exact Currin + interior-only perturbations
    par = [logA, B, P1, P2, P3]
    """

    N_PAR = 5
    INIT_LOC = [1.0, 2.0, 1.0, 2300.0]
    PAR_INIT = [1.0, 2.0, 1.0, 2300.0]
    INIT_SCALE = [10.0, 10.0, 10.0, 10000.0]
    SIG1_INIT= 2

    def phys_fun(self, x, par):
        A, B, C, D = (
            par[..., 0],
            par[..., 1],
            par[..., 2],
            par[..., 3],
        )

        x1 = x[..., 0]
        x2 = x[..., 1].clamp_min(1e-6)

        num = D * x1**3 + 1900.0 * x1**2 + 2092.0 * x1 + 60.0
        den = 100.0 * x1**3 + 500.0 * x1**2 + 4.0 * x1 + 20.0

        currin = (1.0 - A * torch.exp(-1.0 / (B * x2))) * (num / den) * C

        return currin
    
class core_linear2(AnalyticalCore):
    """
    7D linear model
    par = [bias, w1, w2, w3, w4, w5, w6, w7]
    """

    N_PAR = 3
    INIT_LOC = [0.0] * 3
    PAR_INIT = [0.0] * 3
    INIT_SCALE = [100.0] * 3
    SIG1_INIT = 100.0

    def phys_fun(self, x, par):
        bias = par[..., 0]
        weights = par[..., 1:]  # shape (..., 7)

        # Linear model: y = bias + sum_i w_i * x_i
        y = bias + torch.sum(weights * x, dim=-1)

        return y
