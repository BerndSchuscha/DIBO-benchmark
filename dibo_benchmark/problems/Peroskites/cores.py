import torch
from pibob.models.analytical_model import AnalyticalCore
import torch.nn.functional as F

# ============================================================
# Order-1 cores
# ============================================================


class DoublePerovskiteOVacLinearCore(AnalyticalCore):
    """
    Linear analytical model for oxygen vacancy formation energy
    in La_x Sr_(1-x) Fe_y Co_(1-y) O6.

    par = [B, L1, L2, L3]
    model = const + B + L1*nB + L2*t + L3*sigma_A2
    """

    N_PAR = 4
    INIT_LOC = [0.0, 0.0, 0.0, 0.0]
    PAR_INIT = [0.0, 0.0, 0.0, 0.0]
    INIT_SCALE = [5.0, 5.0, 5.0, 5.0]
    SIG1_INIT = 2

    # choose one consistent Shannon-radii convention
    R_LA = 1.36   # A-site, e.g. CN=12
    R_SR = 1.44   # A-site, e.g. CN=12
    R_FE = 0.645  # B-site, e.g. CN=6
    R_CO = 0.61   # B-site, e.g. CN=6
    R_O = 1.40

    def phys_fun(self, x, par):
        B, L1, L2, L3 = (
            par[..., 0],
            par[..., 1],
            par[..., 2],
            par[..., 3],
        )

        xA = x[..., 0]  # La fraction
        yB = x[..., 1]  # Fe fraction

        r_a = xA * self.R_LA + (1.0 - xA) * self.R_SR
        r_b = yB * self.R_FE + (1.0 - yB) * self.R_CO

        # average B-site oxidation state
        n_b = 4.0 - 0.5 * xA

        # tolerance factor
        t = (r_a + self.R_O) / (torch.sqrt(torch.tensor(2.0, device=x.device, dtype=x.dtype)) * (r_b + self.R_O))

        # A-site radius variance
        sigma_a2 = xA * (self.R_LA - r_a) ** 2 + (1.0 - xA) * (self.R_SR - r_a) ** 2

        const = 0.0
        return const + B + L1 * n_b + L2 * t + L3 * sigma_a2

    
class DoublePerovskiteHullLinearCore(AnalyticalCore):
    """
    Linear analytical model for energy above hull
    in La_x Sr_(1-x) Fe_y Co_(1-y) O6.

    par = [B, L1, L2, L3]
    model = const + B + L1*t + L2*sigma_B2 + L3*S_conf
    """

    N_PAR = 4
    INIT_LOC = [0.0, 0.0, 0.0, 0.0]
    PAR_INIT = [0.0, 0.0, 0.0, 0.0]
    INIT_SCALE = [5.0, 5.0, 5.0, 5.0]
    SIG1_INIT = 2

    R_LA = 1.36
    R_SR = 1.44
    R_FE = 0.645
    R_CO = 0.61
    R_O = 1.40

    @staticmethod
    def safe_xlogx(z, eps=1e-12):
        zc = torch.clamp(z, min=eps)
        return torch.where(z > 0, z * torch.log(zc), torch.zeros_like(z))

    def phys_fun(self, x, par):
        B, L1, L2, L3 = (
            par[..., 0],
            par[..., 1],
            par[..., 2],
            par[..., 3],
        )

        xA = x[..., 0]  # La fraction
        yB = x[..., 1]  # Fe fraction

        r_a = xA * self.R_LA + (1.0 - xA) * self.R_SR
        r_b = yB * self.R_FE + (1.0 - yB) * self.R_CO

        # tolerance factor
        t = (r_a + self.R_O) / (torch.sqrt(torch.tensor(2.0, device=x.device, dtype=x.dtype)) * (r_b + self.R_O))

        # B-site radius variance
        sigma_b2 = yB * (self.R_FE - r_b) ** 2 + (1.0 - yB) * (self.R_CO - r_b) ** 2

        # configurational entropy, dimensionless
        s_conf = -(
            self.safe_xlogx(xA)
            + self.safe_xlogx(1.0 - xA)
            + self.safe_xlogx(yB)
            + self.safe_xlogx(1.0 - yB)
        )

        const = 0.0
        return const + B + L1 * t + L2 * sigma_b2 + L3 * s_conf