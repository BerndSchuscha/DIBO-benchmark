from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
import torch

class WeightedAcq(AcquisitionFunction):
    def __init__(self, base_acqf: AcquisitionFunction, g, h):
        super().__init__(model=base_acqf.model)
        self.base_acqf = base_acqf
        self.g = g
        self.h = h

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: (..., q, d)
        Returns: (...,) same shape as base_acqf(X)
        """
        base_val = self.base_acqf(X)      # qEHVI(X) → shape (...,)
        gx = self.g(X)                    # shape (...,)
        hx = self.h(X)                    # shape (...,)
        return base_val * gx + hx