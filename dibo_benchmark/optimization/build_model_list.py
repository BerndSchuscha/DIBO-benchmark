from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Literal, Optional, Union

import torch
from botorch.utils.transforms import normalize
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.input import Normalize as InputNormalize
from gpytorch.means import ConstantMean, LinearMean
from botorch.models.model import Model
from pibob.models.identity_model import IdentityModel
from pibob.models.pyro_model import (
    PyroVIGaussianWrapper,
    PyroAnalyticalMean,
    PyroAnalyticalRegressor,
    fit_svi,
)
from pibob.optimization.config import ObjModelSpec
from pibob.problems.Knowledge import KnowledgeKind
import gpytorch
from gpytorch.likelihoods.gaussian_likelihood import FixedNoiseGaussianLikelihood
from pibob.models.MyInputTransform import MyInputTransform
from pibob.models.GPdiffmean import make_frozen_mean, fit_mean_only


def build_model_list(
    train_X: torch.Tensor,
    train_Y: torch.Tensor,  # (n, m)
    noise_se: Optional[torch.Tensor],
    use_known_noise: bool,
    specs: Sequence[ObjModelSpec],  # length m
) -> List[Model]:
    """Construct per-objective models for multi-objective Bayesian optimization.

    This function builds one model per objective according to a list of
    objective model specifications. Each objective can be modeled using
    different model types (e.g., exact GPs, identity mappings, or Pyro
    variational models).

    Inputs are normalized internally unless an input-transform-based model
    is requested. Known observation noise can optionally be injected into
    GP models via fixed observation variances.

    Args:
        train_X: Training inputs with shape ``(n, d)``, where ``n`` is the
            number of observations and ``d`` is the input dimensionality.
        train_Y: Training objective values with shape ``(n, m)``, where
            ``m`` is the number of objectives.
        noise_se: Optional noise standard deviation. May be a scalar tensor
            or a tensor of shape ``(m,)`` specifying per-objective noise.
        use_known_noise: If True and ``noise_se`` is provided, the noise
            variance is treated as known and passed to GP models via
            ``train_Yvar``.
        specs: Sequence of objective model specifications of length ``m``.
            Each specification determines the model type and configuration
            for one objective.

    Returns:
        A list of models, one per objective, in the same order as ``specs``.

    Raises:
        ValueError: If the number of specifications does not match the number
            of objectives, or if required parameters for a model type are
            missing.
        KeyError: If an unknown model kind is encountered.

    Notes:
        Supported model kinds include:
        - ``"gpr"``: Standard exact GP with constant mean.
        - ``"gpr_mean"``: Exact GP with configurable mean function
          (constant or linear).
        - ``"gpr_input_tf"``: Exact GP using an input transform for
          normalization instead of external normalization.
        - ``"identity"``: Identity model that returns inputs unchanged.
        - ``"pyro_vi"``: Pyro variational model wrapped for BO usage.


    """
    m = train_Y.shape[-1]
    if len(specs) != m:
        raise ValueError(f"Need {m} specs, got {len(specs)}")

    d = train_X.shape[-1]
    train_Xn = train_X

    models: List[Model] = []
    for i, spec in enumerate(specs):
        yi = train_Y[..., i : i + 1]

        yvar = None
        if (noise_se is not None) and use_known_noise:
            # supports scalar or per-objective noise
            se_i = noise_se if noise_se.numel() == 1 else noise_se[i]
            yvar = torch.full_like(yi, se_i.pow(2))

        kw = dict(spec.kwargs)

        if spec.kind == "gpr":

            # base kernel (yours)
            base_kernel = gpytorch.kernels.RBFKernel(
                train_X.shape[-1],
                lengthscale_prior=gpytorch.priors.LogNormalPrior(0.0, 1.0),
                lengthscale_constraint=gpytorch.constraints.GreaterThan(2.5e-2),
            )

            # wrap with ScaleKernel -> adds outputscale
            covar = gpytorch.kernels.ScaleKernel(base_kernel)

            # ----- constant mean initialized to data mean -----
            mean_module = gpytorch.means.ConstantMean()

            y_mean = yi.mean()
            y_std = yi.std(unbiased=False).clamp_min(1e-12)  # avoid 0
            # mean_module.initialize(constant=y_mean)
            # covar.initialize(outputscale=y_std.pow(2))  # outputscale is variance
            likelihood = FixedNoiseGaussianLikelihood(
                noise=yvar, learn_additional_noise=True
            )

            mi = SingleTaskGP(
                train_Xn,
                yi,
                train_Yvar=yvar,
                likelihood=likelihood,
                mean_module=mean_module,
                covar_module=covar,
            )

        elif spec.kind is KnowledgeKind.CUSTOM_MEAN:
            # base kernel (yours)
            base_kernel = gpytorch.kernels.RBFKernel(
                ard_num_dims=train_X.shape[-1],
                lengthscale_prior=gpytorch.priors.LogNormalPrior(0.0, 1.0),
                lengthscale_constraint=gpytorch.constraints.GreaterThan(2.5e-2),
            )

            # wrap with ScaleKernel -> adds outputscale
            covar = gpytorch.kernels.ScaleKernel(base_kernel)
            likelihood = FixedNoiseGaussianLikelihood(
                noise=yvar, learn_additional_noise=True
            )

            mi = SingleTaskGP(
                train_Xn,
                yi,
                train_Yvar=yvar,
                covar_module=covar,
                likelihood=likelihood,
                outcome_transform=None,
                mean_module=spec.Knowledge,
                **kw,
            )
        elif spec.kind is KnowledgeKind.DESCRIPTOR:
            is_one_to_many = False
            transform_module = spec.Knowledge
            kdim = train_X.shape[-1]
            if hasattr(transform_module, "is_one_to_many"):
                is_one_to_many = transform_module.is_one_to_many
                kdim = transform_module.dim

            input_tf = MyInputTransform(
                transform_module=transform_module, is_one_to_many=is_one_to_many
            )
            # base kernel (yours)
            base_kernel = gpytorch.kernels.RBFKernel(
                kdim,
                lengthscale_prior=gpytorch.priors.LogNormalPrior(0.0, 1.0),
                lengthscale_constraint=gpytorch.constraints.GreaterThan(2.5e-2),
            )

            # wrap with ScaleKernel -> adds outputscale
            covar = gpytorch.kernels.ScaleKernel(base_kernel)

            # ----- constant mean initialized to data mean -----
            mean_module = gpytorch.means.ConstantMean()

            # mean_module.initialize(constant=y_mean)
            # covar.initialize(outputscale=y_std.pow(2))  # outputscale is variance
            likelihood = FixedNoiseGaussianLikelihood(
                noise=yvar, learn_additional_noise=True
            )
            mi = SingleTaskGP(
                train_Xn,
                yi,
                train_Yvar=yvar,
                likelihood=likelihood,
                mean_module=mean_module,
                input_transform=input_tf,
                covar_module=covar,
            )

        elif spec.kind == "identity":
            # IdentityModel outputs d dims (same as X). Usually only sensible if your "objective"
            # is a function of X that matches that shape; otherwise don't use this for scalar objectives.
            mi = IdentityModel(train_Xn, yi, **kw)

        elif spec.kind is KnowledgeKind.PYRO:
            # You must provide pyro_model + guide via kwargs, OR build them here.
            core = spec.Knowledge
            core.NOISE_LEVEL = noise_se[i]
            regressor = PyroAnalyticalRegressor(PyroAnalyticalMean(core))
            guide, losses = fit_svi(
                regressor,
                train_Xn,
                yi,
                num_steps=spec.fit_options["maxiter"],
                lr=1e-2,
                show_progress=False,
            )
            range_y = yi.max() - yi.min()
            mi = PyroVIGaussianWrapper(
                regressor=regressor, guide=guide, jitter=range_y * 1e-3
            )

        elif spec.kind == KnowledgeKind.DELTA:
            stage1 = fit_mean_only(spec.Knowledge, train_Xn, yi, learn_sigma=True)
            par_hat = stage1["par_hat"]
            mean_module = make_frozen_mean(spec.Knowledge, par_hat)

            base_kernel = gpytorch.kernels.RBFKernel(
                ard_num_dims=train_X.shape[-1],
                lengthscale_prior=gpytorch.priors.LogNormalPrior(0.0, 1.0),
                lengthscale_constraint=gpytorch.constraints.GreaterThan(2.5e-2),
            )
            # wrap with ScaleKernel -> adds outputscale
            covar = gpytorch.kernels.ScaleKernel(base_kernel)

            likelihood = FixedNoiseGaussianLikelihood(
                noise=yvar, learn_additional_noise=True
            )

            mi = SingleTaskGP(
                train_Xn,
                yi,
                train_Yvar=yvar,
                covar_module=covar,
                likelihood=likelihood,
                outcome_transform=None,
                mean_module=mean_module,
                **kw,
            )

        elif spec.kind == KnowledgeKind.KERNEL:
            covar = spec.Knowledge

            likelihood = FixedNoiseGaussianLikelihood(
                noise=yvar, learn_additional_noise=True
            )
            mean_module = gpytorch.means.ConstantMean()

            mi = SingleTaskGP(
                train_Xn,
                yi,
                train_Yvar=yvar,
                likelihood=likelihood,
                mean_module=mean_module,
                covar_module=covar,
            )

        else:
            raise KeyError(spec.kind)

        models.append(mi)

    return models
