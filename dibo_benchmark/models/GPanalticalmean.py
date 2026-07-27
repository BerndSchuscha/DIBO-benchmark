import gpytorch
import torch
import torch.nn as nn
from pibob.models.analytical_model import AnalyticalCore

class GPAnalyticalMean(gpytorch.means.Mean):
    """
    GP mean = core(x, par) with learnable par.
    Takes ONLY AnalyticalCore; core must provide n_par and init_loc.
    """
    def __init__(self, core: AnalyticalCore):
        super().__init__()
        self.core = core

        if not hasattr(core, "n_par"):
            raise AttributeError(f"{core.__class__.__name__} must define 'n_par'")
        if not hasattr(core, "init_loc"):
            raise AttributeError(f"{core.__class__.__name__} must define 'init_loc'")

        init = torch.as_tensor(core.init_loc, dtype=torch.float32).clone()
        if init.ndim != 1 or init.numel() != int(core.n_par):
            raise ValueError(
                f"{core.__class__.__name__}: init_loc must be 1D with length n_par={core.n_par}"
            )

        self.par = nn.Parameter(init)

    @property
    def n_params(self) -> int:
        return self.par.numel()

    @property
    def init_values(self) -> torch.Tensor:
        return self.par.detach().clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.core(x, self.par)
    