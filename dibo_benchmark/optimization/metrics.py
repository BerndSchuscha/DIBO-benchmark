from __future__ import annotations
import torch
from typing import List, Tuple

from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated


def hypervolume_trace(
    Y_true: torch.Tensor,
    ref_point: torch.Tensor,
    n_init: int,
    n_batch: int,
    batch_size: int,
) -> Tuple[List[int], List[float]]:
    hv = Hypervolume(ref_point=ref_point)
    hs, ns = [], []

    for t in range(n_init-1):
        hs.append(float(0))
        ns.append(int(0))

    for t in range(n_batch + 1):
        n = n_init + t * batch_size
        Yt = Y_true[:n]
        pareto_Y = Yt[is_non_dominated(Yt)]
        hs.append(float(hv.compute(pareto_Y)))
        ns.append(int(n - n_init))
        

    return ns, hs
