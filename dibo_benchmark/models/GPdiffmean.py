import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import gpytorch


class FrozenAnalyticalMean(gpytorch.means.Mean):
    """
    Mean(x) = core(x, par_hat) with par_hat frozen (not optimized in GP stage).

    Notes:
    - par_hat stored as a buffer so it moves with .to(device/dtype) and is not trainable.
    - forward returns shape (N,) (typical for single-output ExactGP).
    """

    def __init__(self, core, par_hat: torch.Tensor):
        super().__init__()
        self.core = core
        self.register_buffer("par_hat", par_hat.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.core(x, self.par_hat)
        # Normalize to (N,) for single-output GP (ExactGP expects train_y: (N,))
        if m.ndim >= 2 and m.shape[-1] == 1:
            m = m.squeeze(-1)
        return m


def _as_1d_y(y: torch.Tensor) -> torch.Tensor:
    """
    Normalize Y to shape (N,) for ExactGP training.
    """
    if y.ndim == 2 and y.shape[-1] == 1:
        return y.squeeze(-1)
    if y.ndim != 1:
        raise ValueError(f"Expected Y shape (N,) or (N,1), got {tuple(y.shape)}")
    return y


def _infer_device_dtype(
    X: torch.Tensor, Y: torch.Tensor
) -> Tuple[torch.device, torch.dtype]:
    if X.device != Y.device:
        # You can either raise or silently move Y; raising is safer in library code.
        raise ValueError(
            f"X.device ({X.device}) != Y.device ({Y.device}). Move tensors to same device."
        )
    if X.dtype != Y.dtype:
        raise ValueError(
            f"X.dtype ({X.dtype}) != Y.dtype ({Y.dtype}). Cast tensors to same dtype."
        )
    return X.device, X.dtype


def fit_mean_only(
    core,
    X: torch.Tensor,
    Y: torch.Tensor,
    steps: int = 2500,
    lr: float = 2e-2,
    learn_sigma: bool = True,
    sigma_init: float = 0.2,
    jitter: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """
    Stage 1: deterministic fit of analytical mean to Y.

    Model:
        Y ~ Normal(m(X; par), sigma)   (if learn_sigma=True)
    or:
        minimize MSE(Y, m(X; par))     (if learn_sigma=False)

    Args
    ----
    core:
        Must provide:
          - init_loc: Tensor[n_par]
          - __call__(X, par) -> mean values
    X:
        Tensor[N, d] or Tensor[N] (core decides)
    Y:
        Tensor[N] or Tensor[N,1]
    steps, lr:
        Optimizer settings
    learn_sigma:
        If True, learns a single homoscedastic sigma via log_sigma parameter.
    sigma_init:
        Initial value for sigma (only if learn_sigma=True)
    jitter:
        Small positive number added to sigma for numerical stability.

    Returns
    -------
    dict with:
      - "par_hat": Tensor[n_par]
      - "sigma_hat": Tensor[] (scalar) if learn_sigma=True
    """
    device, dtype = _infer_device_dtype(X, Y)
    Y_1d = _as_1d_y(Y)

    init_loc = torch.as_tensor(core.init_loc, device=device, dtype=dtype).clone()
    par = nn.Parameter(init_loc)

    params = [par]
    log_sigma: Optional[nn.Parameter] = None
    if learn_sigma:
        log_sigma = nn.Parameter(
            torch.tensor(math.log(float(sigma_init)), device=device, dtype=dtype)
        )
        params.append(log_sigma)

    opt = torch.optim.Adam(params, lr=float(lr))

    for _ in range(int(steps)):
        opt.zero_grad(set_to_none=True)

        m = core(X, par)
        if m.ndim >= 2 and m.shape[-1] == 1:
            m = m.squeeze(-1)

        if m.shape != Y_1d.shape:
            raise ValueError(
                f"Mean shape {tuple(m.shape)} does not match Y shape {tuple(Y_1d.shape)}. "
                "Ensure core(X, par) returns (N,) or (N,1) matching Y."
            )

        if learn_sigma:
            sigma = torch.exp(log_sigma) + float(jitter)
            loss = -torch.distributions.Normal(m, sigma).log_prob(Y_1d).mean()
        else:
            loss = (Y_1d - m).pow(2).mean()

        loss.backward()
        opt.step()

    out = {"par_hat": par.detach().clone()}
    if learn_sigma:
        out["sigma_hat"] = torch.exp(log_sigma.detach()).clone()
    return out


def make_frozen_mean(core, par_hat: torch.Tensor) -> FrozenAnalyticalMean:
    """
    Convenience helper to build a FrozenAnalyticalMean module.
    """
    return FrozenAnalyticalMean(core, par_hat)
