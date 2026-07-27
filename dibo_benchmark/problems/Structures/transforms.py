import torch
import torch.nn as nn


class ALSGeometryTransform(nn.Module):
    """
    Input x is assumed normalized with:
        normalize(X, bounds=problem.bounds)

    Physical descriptor order:
    [TargetHeight, WallThickness, AveragePerimeter,
     x1, x2, x3, x4, x5, x6, x7, x8,
     Modulus, PlateauStrength]

    Output:
    [
        h, t, P,
        d, d_over_t, omega,
        log_h, log_t, log_P, log_d_over_t, log_omega,
        x1, ..., x8,
        x_mean, x_std,
        E, sig_p,
        log_E, log_sig_p
    ]
    """

    is_one_to_many = True
    dim = 25

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

        h = torch.clamp(xp[..., 0], min=eps)
        t = torch.clamp(xp[..., 1], min=eps)
        P = torch.clamp(xp[..., 2], min=eps)

        shape = xp[..., 3:11]

        E = torch.clamp(xp[..., 11], min=eps)
        sig_p = torch.clamp(xp[..., 12], min=eps)

        d = P / (2.0 * torch.pi)
        d_over_t = torch.clamp(d / t, min=eps)
        omega = torch.clamp(h / torch.sqrt(d * t), min=eps)

        log_h = torch.log(h)
        log_t = torch.log(t)
        log_P = torch.log(P)
        log_d_over_t = torch.log(d_over_t)
        log_omega = torch.log(omega)

        log_E = torch.log(E)
        log_sig_p = torch.log(sig_p)

        x_mean = torch.mean(shape, dim=-1)
        x_std = torch.std(shape, dim=-1, unbiased=False)

        return torch.cat(
            [
                h.unsqueeze(-1),
                t.unsqueeze(-1),
                P.unsqueeze(-1),
                d.unsqueeze(-1),
                d_over_t.unsqueeze(-1),
                omega.unsqueeze(-1),
                log_h.unsqueeze(-1),
                log_t.unsqueeze(-1),
                log_P.unsqueeze(-1),
                log_d_over_t.unsqueeze(-1),
                log_omega.unsqueeze(-1),
                shape,
                x_mean.unsqueeze(-1),
                x_std.unsqueeze(-1),
                E.unsqueeze(-1),
                sig_p.unsqueeze(-1),
                log_E.unsqueeze(-1),
                log_sig_p.unsqueeze(-1),
            ],
            dim=-1,
        )


class ALSGeometryTransformCompact(nn.Module):
    """
    Input x is assumed normalized with:
        normalize(X, bounds=problem.bounds)

    Output:
    [
        d_over_t,
        omega,
        log_d_over_t,
        log_omega,
        x1, ..., x8,
        log_Modulus,
        log_PlateauStrength
    ]
    """

    is_one_to_many = True
    dim = 14

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

        h = torch.clamp(xp[..., 0], min=eps)
        t = torch.clamp(xp[..., 1], min=eps)
        P = torch.clamp(xp[..., 2], min=eps)

        shape = xp[..., 3:11]

        E = torch.clamp(xp[..., 11], min=eps)
        sig_p = torch.clamp(xp[..., 12], min=eps)

        d = P / (2.0 * torch.pi)
        d_over_t = torch.clamp(d / t, min=eps)
        omega = torch.clamp(h / torch.sqrt(d * t), min=eps)

        return torch.cat(
            [
                d_over_t.unsqueeze(-1),
                omega.unsqueeze(-1),
                torch.log(d_over_t).unsqueeze(-1),
                torch.log(omega).unsqueeze(-1),
                shape,
                torch.log(E).unsqueeze(-1),
                torch.log(sig_p).unsqueeze(-1),
            ],
            dim=-1,
        )
