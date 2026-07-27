import torch
from torch import nn


class FeCoNiMinimalPhysicsTransform(nn.Module):
    """
    Adds only two physically meaningful descriptors:

    1) FeCo interaction → pair interaction (cluster expansion)
    2) log(Fe / Ni)     → compositional geometry (simplex-aware)

    Output:
    [Fe, Co, Ni, FeCo, log_Fe_over_Ni]
    """

    is_one_to_many = True
    dim = 5

    def __init__(self, problem):
        super().__init__()
        bounds = problem.bounds
        self.register_buffer("lb", bounds[0].clone())
        self.register_buffer("ub", bounds[1].clone())
        self.register_buffer("span", (bounds[1] - bounds[0]).clone())

    def _to_physical(self, x):
        return self.lb + x * self.span

    def forward(self, x):
        eps = 1e-12

        xp = self._to_physical(x)

        Fe = torch.clamp(xp[..., 0], min=eps)
        Co = torch.clamp(xp[..., 1], min=eps)
        Ni = torch.clamp(xp[..., 2], min=eps)

        # 🔹 interaction (physics: mixing)
        FeCo = Fe * Co

        # 🔹 log-ratio (physics: compositional geometry)
        log_Fe_over_Ni = torch.log(Fe / Ni)

        return torch.stack(
            [
                Fe,
                Co,
                Ni,
                FeCo,
                log_Fe_over_Ni,
            ],
            dim=-1,
        )
