import torch
from pibob.models.analytical_model import AnalyticalCore
import torch.nn.functional as F
from pibob.models.analytical_model import AnalyticalCore


# ============================================================
# Order-1 cores
# ============================================================


class NanoSizeLinearCore(AnalyticalCore):
    """
    Order-1 size model
    par = [B, L1, L2, L3, L4]
    """

    N_PAR = 5
    INIT_LOC = [0.0, 0.0, 0.0, 0.0, 0.0]
    PAR_INIT = [0.0, 0.0, 0.0, 0.0, 0.0]
    INIT_SCALE = [5.0, 5.0, 5.0, 5.0, 5.0]
    SIG1_INIT = 2

    def phys_fun(self, x, par):
        B, L1, L2, L3, L4 = (
            par[..., 0],
            par[..., 1],
            par[..., 2],
            par[..., 3],
            par[..., 4],
        )

        x1, x2, x3, x4 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]

        const = 19.36549
        return const + B + L1 * x1 + L2 * x2 + L3 * x3 + L4 * x4


class NanoPolyLinearCore(AnalyticalCore):
    """
    Order-1 polydispersity model
    par = [B, L1, L2, L3, L4]
    """

    N_PAR = 5
    INIT_LOC = [0.0, 0.0, 0.0, 0.0, 0.0]
    PAR_INIT = [0.0, 0.0, 0.0, 0.0, 0.0]
    INIT_SCALE = [5.0, 5.0, 5.0, 5.0, 5.0]
    SIG1_INIT = 2

    def phys_fun(self, x, par):
        B, L1, L2, L3, L4 = (
            par[..., 0],
            par[..., 1],
            par[..., 2],
            par[..., 3],
            par[..., 4],
        )

        x1, x2, x3, x4 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]

        const = 19.6114239
        return const + B + L1 * x1 + L2 * x2 + L3 * x3 + L4 * x4


# ============================================================
# 8-parameter cores
# grouped quadratic + directional drift
# ============================================================


class NanoSizeCore8(AnalyticalCore):
    """
    Size model with 8 parameters
    par = [B, S_lin, S_int, S_quad, D1, D2, D3, D4]

    - S_lin scales the known first-order structure
    - S_int scales the known interaction structure
    - S_quad scales the known quadratic structure
    - D1..D4 add flexible linear drift
    """

    N_PAR = 8
    INIT_LOC = [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    PAR_INIT = [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    INIT_SCALE = [5.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]
    SIG1_INIT = 2

    def phys_fun(self, x, par):
        B, S_lin, S_int, S_quad, D1, D2, D3, D4 = [par[..., i] for i in range(8)]

        x1, x2, x3, x4 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]

        const = 19.36549

        lin = (
            -0.2797 * x1
            + 1.56885 * x2
            + 3.5447 * x3
            + 1.82225 * x4
        )

        inter = (
            -1.1978 * x1 * x2
            - 1.66594 * x1 * x3
            - 1.62873 * x1 * x4
            - 0.02003 * x2 * x3
            - 0.001268 * x2 * x4
            - 0.35086 * x3 * x4
        )

        quad = (
            +0.3914 * x1**2
            + 0.52265 * x2**2
            - 0.81701 * x3**2
            - 2.74921 * x4**2
        )

        drift = D1 * x1 + D2 * x2 + D3 * x3 + D4 * x4

        return const + B + S_lin * lin + S_int * inter + S_quad * quad + drift


class NanoPolyCore8(AnalyticalCore):
    """
    Polydispersity model with 8 parameters
    par = [B, S_lin, S_int, S_quad, D1, D2, D3, D4]

    - S_lin scales the known first-order structure
    - S_int scales the known interaction structure
    - S_quad scales the known quadratic structure
    - D1..D4 add flexible linear drift
    """

    N_PAR = 8
    INIT_LOC = [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    PAR_INIT = [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    INIT_SCALE = [5.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]
    SIG1_INIT = 2

    def phys_fun(self, x, par):
        B, S_lin, S_int, S_quad, D1, D2, D3, D4 = [par[..., i] for i in range(8)]

        x1, x2, x3, x4 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]

        const = 19.6114239

        lin = (
            +1.0313718 * x1
            + 1.48527 * x2
            + 1.7991534 * x3
            - 4.1983899 * x4
        )

        inter = (
            +1.4263262 * x1 * x2
            - 0.4279443 * x1 * x3
            - 1.3865203 * x1 * x4
            - 1.051601 * x2 * x3
            - 2.06380 * x2 * x4
            - 2.476674 * x3 * x4
        )

        quad = (
            -0.4497319 * x1**2
            - 1.8040123 * x2**2
            - 3.8699325 * x3**2
            - 2.6148 * x4**2
        )

        drift = D1 * x1 + D2 * x2 + D3 * x3 + D4 * x4

        return const + B + S_lin * lin + S_int * inter + S_quad * quad + drift