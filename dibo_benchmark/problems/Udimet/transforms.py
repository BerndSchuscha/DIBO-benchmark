import torch
import torch.nn as nn


class UdimetWidthTransformCompact(nn.Module):
    """
    Input x is assumed normalized with:
        normalize(X, bounds=problem.bounds)

    Output:
    [
        log_primary_mean,
        log_primary_relative_width,
        log_secondary_mean,
        log_secondary_relative_width,
        log_primary_over_secondary,
        T1,
    ]
    """

    is_one_to_many = True
    dim = 6

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

        p_mean = torch.clamp(xp[..., 0], min=eps)
        p_min  = xp[..., 1]
        p_max  = xp[..., 2]

        s_mean = torch.clamp(xp[..., 4], min=eps)
        s_min  = xp[..., 5]
        s_max  = xp[..., 6]

        T1 = xp[..., 8]

        p_width = torch.clamp((p_max - p_min) / p_mean, min=eps)
        s_width = torch.clamp((s_max - s_min) / s_mean, min=eps)

        return torch.cat(
            [
                torch.log(p_mean).unsqueeze(-1),
                torch.log(p_width).unsqueeze(-1),
                torch.log(s_mean).unsqueeze(-1),
                torch.log(s_width).unsqueeze(-1),
                torch.log(p_mean / s_mean).unsqueeze(-1),
                T1.unsqueeze(-1),
            ],
            dim=-1,
        )
    

class UdimetMinimalTransform(nn.Module):
    """
    Input x is assumed normalized with:
        normalize(X, bounds=problem.bounds)

    Output:
    [
        log_primary_mean,
        log_secondary_mean,
        log_primary_over_secondary,
        T1,
    ]
    """

    is_one_to_many = True
    dim = 4

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

        p_mean = torch.clamp(xp[..., 0], min=eps)
        s_mean = torch.clamp(xp[..., 4], min=eps)
        T1 = xp[..., 8]

        return torch.cat(
            [
                torch.log(p_mean).unsqueeze(-1),
                torch.log(s_mean).unsqueeze(-1),
                torch.log(p_mean / s_mean).unsqueeze(-1),
                T1.unsqueeze(-1),
            ],
            dim=-1,
        )