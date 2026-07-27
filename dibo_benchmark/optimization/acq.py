from __future__ import annotations
import torch

from botorch.utils.transforms import normalize
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)

from botorch.sampling.index_sampler import IndexSampler
import torch

def build_qlognehvi(
    model,
    train_X: torch.Tensor,
    ref_point: torch.Tensor,
    mc_samples: int,
):
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))
    X_baseline = train_X


    return qLogNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,
        X_baseline=X_baseline,
        sampler=sampler,
        prune_baseline=True,
        cache_pending=True,
    )
