import torch
import torch.nn as nn


class DPTransformVac_nB(nn.Module):

    is_one_to_many = True
    dim = 3

    def forward(self, x):
        xA = x[..., 0]
        yB = x[..., 1]

        nB = 4.0 - xA

        return torch.stack(
            [
                xA,
                yB,
                nB,
            ],
            dim=-1,
        )


class DPTransformVac_t(nn.Module):

    is_one_to_many = True
    dim = 3

    R_LA = 1.36
    R_SR = 1.44
    R_FE = 0.645
    R_CO = 0.61
    R_O = 1.40

    def forward(self, x):
        xA = x[..., 0]
        yB = x[..., 1]

        rA = xA * self.R_LA + (1 - xA) * self.R_SR
        rB = yB * self.R_FE + (1 - yB) * self.R_CO

        t = (rA + self.R_O) / (
            torch.sqrt(torch.tensor(2.0, device=x.device, dtype=x.dtype))
            * (rB + self.R_O)
        )

        return torch.stack(
            [
                xA,
                yB,
                t,
            ],
            dim=-1,
        )


class DPTransformVac_sigmaB2(nn.Module):

    is_one_to_many = True
    dim = 3

    R_FE = 0.645
    R_CO = 0.61

    def forward(self, x):
        xA = x[..., 0]
        yB = x[..., 1]

        rB = yB * self.R_FE + (1 - yB) * self.R_CO

        sigmaB2 = yB * (self.R_FE - rB) ** 2 + (1 - yB) * (self.R_CO - rB) ** 2

        return torch.stack(
            [
                xA,
                yB,
                sigmaB2,
            ],
            dim=-1,
        )


class DPTransformHull_t(nn.Module):

    is_one_to_many = True
    dim = 3

    R_LA = 1.36
    R_SR = 1.44
    R_FE = 0.645
    R_CO = 0.61
    R_O = 1.40

    def forward(self, x):
        xA = x[..., 0]
        yB = x[..., 1]

        rA = xA * self.R_LA + (1 - xA) * self.R_SR
        rB = yB * self.R_FE + (1 - yB) * self.R_CO

        t = (rA + self.R_O) / (
            torch.sqrt(torch.tensor(2.0, device=x.device, dtype=x.dtype))
            * (rB + self.R_O)
        )

        return torch.stack(
            [
                xA,
                yB,
                t,
            ],
            dim=-1,
        )


class DPTransformHull_Sconf(nn.Module):

    is_one_to_many = True
    dim = 3

    @staticmethod
    def safe_xlogx(z, eps=1e-12):
        zc = torch.clamp(z, min=eps)
        return torch.where(z > 0, z * torch.log(zc), torch.zeros_like(z))

    def forward(self, x):
        xA = x[..., 0]
        yB = x[..., 1]

        Sconf = -(
            self.safe_xlogx(xA)
            + self.safe_xlogx(1 - xA)
            + self.safe_xlogx(yB)
            + self.safe_xlogx(1 - yB)
        )

        return torch.stack(
            [
                xA,
                yB,
                Sconf,
            ],
            dim=-1,
        )


class DPTransformHull_sigmaB2(nn.Module):

    is_one_to_many = True
    dim = 3

    R_FE = 0.645
    R_CO = 0.61

    def forward(self, x):
        xA = x[..., 0]
        yB = x[..., 1]

        rB = yB * self.R_FE + (1 - yB) * self.R_CO

        sigmaB2 = yB * (self.R_FE - rB) ** 2 + (1 - yB) * (self.R_CO - rB) ** 2

        return torch.stack(
            [
                xA,
                yB,
                sigmaB2,
            ],
            dim=-1,
        )
