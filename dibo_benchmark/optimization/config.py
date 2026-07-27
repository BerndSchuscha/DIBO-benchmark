from __future__ import annotations
from dataclasses import dataclass
import torch
from botorch.models.model import Model
from typing import Any, Dict, Sequence, Optional, Literal, Callable

ModelKind = Literal["gpr", "gpr_input_tf", "gpr_mean", "identity", "pyro_vi"]


@dataclass(frozen=True)
class ObjModelSpec:
    kind: ModelKind
    Knowledge: Optional = None
    kwargs: Dict[str, Any] = None
    # passed to fit_gpytorch_mll
    fit_options: Optional[Dict[str, Any]] = None
    input_transform: Optional[Callable] = None

    def __post_init__(self):
        object.__setattr__(self, "kwargs", self.kwargs or {})


@dataclass
class DiscreteMOBOConfig:
    bounds: torch.Tensor  # (2, d)
    ref_point: torch.Tensor  # (m,) objective space, assumes maximization
    models: Sequence[ObjModelSpec]
    noise_se: Optional[torch.Tensor] = (
        None  # (m,) additive Gaussian noise per objective
    )
    batch_size: int = 1
    n_init: int = 10
    n_batch: int = 90

    mc_samples: int = 128
    use_known_noise: bool = True  # if noise_se provided: pass yvar into SingleTaskGP
    knowledge: Any = None
    place: str = "surrogate"
    eval_batch_size: int = 1000


models = {
    "gpr": ObjModelSpec(
        kind="gpr",
        fit_options={"maxiter": 200},
        input_transform=None,
    ),
    "gpr_input_tf": ObjModelSpec(
        kind="gpr_input_tf",
        fit_options={"maxiter": 200},
        input_transform=None,
    ),
    "pyro_svi": ObjModelSpec(
        kind="pyro_svi",
        fit_options={"maxiter": 200},
        input_transform=None,
    ),
    "gpr_mean": ObjModelSpec(
        kind="gpr_mean",
        input_transform=None,
        fit_options={"maxiter": 200},
    ),
    "gpr_phystransform": ObjModelSpec(
        kind="gpr", fit_options={"maxiter": 200}, input_transform=None
    ),
}
