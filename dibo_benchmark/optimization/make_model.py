from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple, Union

import torch

from botorch.utils.transforms import normalize
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.model import Model  # base type
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood


# ---------- Typed signatures ----------
# Builds a full multi-objective BoTorch model (any type)
ModelFactory = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], bool], Model
]
# args: (train_X, train_Y, bounds, noise_se, use_known_noise) -> model

# Builds ONE objective model from (train_X, yi, bounds, noise_se, use_known_noise)
ObjModelFactory = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], bool], Model
]
# args: (train_X, yi, bounds, noise_se_i, use_known_noise) -> model


def _wrap_as_modellist(model: Model, m: int) -> Model:
    """
    If user gives a single model for multi-output, we accept it as-is.
    If they give a ModelListGP already, return it.
    """
    if isinstance(model, ModelListGP):
        return model
    return model  # could be a batched multi-output GP etc.


def make_model(
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    bounds: torch.Tensor,
    noise_se: Optional[torch.Tensor] = None,
    use_known_noise: bool = True,
    models: Optional[Sequence[Union[Model, ObjModelFactory]]] = None,
) -> Tuple[SumMarginalLogLikelihood, Model]:
    """
    Flexible initializer.

    Default behavior (no model/models/model_factory provided):
      - builds ModelListGP of SingleTaskGPs (one GP per objective)
      - uses known noise via yvar if noise_se is provided and use_known_noise=True
      - models:
    Returns:
      (mll, model)
    """
    m = train_Y.shape[-1]

    # user-provided list of models OR per-objective factories

    if len(models) != m:
        raise ValueError(
            f"`models` length must match number of objectives m={m}, got {len(models)}."
        )
    print(models)
    built: list[Model] = []
    for i, mi in enumerate(models):
        yi = train_Y[..., i : i + 1]

        # pick per-objective noise if provided
        noise_i = None
        if noise_se is not None:
            # allow scalar, (m,), or broadcastable
            if noise_se.numel() == 1:
                noise_i = noise_se.reshape(1)
            else:
                noise_i = noise_se[i].reshape(1)

        if callable(mi):
            # per-objective factory
            print(train_X, yi, bounds, noise_i, use_known_noise)
            built_model = mi(train_X, yi, bounds, noise_i, use_known_noise)
            built.append(built_model)
        else:
            # pre-built Model
            built.append(mi)

    user_model = ModelListGP(*built)
    mll = SumMarginalLogLikelihood(user_model.likelihood, user_model)
    return mll, user_model
