from __future__ import annotations
import torch
from typing import Tuple

from botorch.fit import fit_gpytorch_mll
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated
import pandas as pd
import gpytorch
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
import time
from contextlib import contextmanager


from pibob.optimization.CombinedGaussianIndependentModel import (
    CombinedGaussianIndependentModel,
)
from pibob.optimization.config import DiscreteMOBOConfig
from pibob.optimization.build_model_list import build_model_list


def compute_hv(Y: torch.Tensor, ref_point: torch.Tensor) -> float:
    """
    Y: (n, m) objective values (assumed maximization)
    ref_point: (m,)
    Returns scalar hypervolume.
    """
    hv = Hypervolume(ref_point=ref_point)
    nd_mask = is_non_dominated(Y)
    pareto_Y = Y[nd_mask]
    return float(hv.compute(pareto_Y))


@contextmanager
def timer(name, log_dict):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    log_dict[name] = log_dict.get(name, 0.0) + dt


def _is_exact_gp_model(m) -> bool:
    """Check whether a model is an Exact GPyTorch GP.

    This function identifies models that inherit from
    ``gpytorch.models.ExactGP`` and expose a likelihood attribute,
    which is required for ExactMarginalLogLikelihood-based training.

    Args:
        m: A model instance.

    Returns:
        True if the model is an ExactGP with a likelihood attribute,
        False otherwise.
    """
    return isinstance(m, gpytorch.models.ExactGP) and hasattr(m, "likelihood")


def has_non_gp_specs(specs) -> bool:
    """Check whether any objective model specifications are non-GP.

    This helper inspects the ``kind`` attribute of each specification
    and flags model types that are not standard GPyTorch exact GPs,
    such as identity mappings or Pyro variational models.

    Args:
        specs: Iterable of objective model specifications. Each spec
            is expected to expose a ``kind`` attribute.

    Returns:
        True if at least one specification corresponds to a non-GP
        model type (e.g., ``"identity"``, ``"pyro_vi"``), False otherwise.
    """
    bad = {"identity", "pyro_vi"}
    return any(getattr(s, "kind", None) in bad for s in specs)


import torch
import math
from typing import Optional

# ---------- Mahalanobis-RMSE ----------


@torch.no_grad()
def rmse_mahalanobis(
    mu: torch.Tensor,
    y: torch.Tensor,
    *,
    cov: Optional[torch.Tensor] = None,
    shrinkage: float = 1e-6,
) -> torch.Tensor:
    """
    sqrt( mean_n (e_n^T Σ^{-1} e_n) ), e_n = (mu_n - y_n)
    Shapes: mu,y = (N, M), cov = (M, M)
    """
    mu = mu.reshape(mu.shape[0], -1)
    y = y.reshape(y.shape[0], -1)
    assert mu.shape == y.shape, f"mu {mu.shape} vs y {y.shape}"
    N, M = y.shape

    device, dtype = y.device, y.dtype

    if cov is None:
        yc = y - y.mean(dim=0, keepdim=True)
        denom = max(N - 1, 1)
        cov = (yc.T @ yc) / denom
    else:
        cov = cov.to(device=device, dtype=dtype)

    cov = cov + shrinkage * torch.eye(M, device=device, dtype=dtype)

    # Cholesky solve for stability
    L = torch.linalg.cholesky(cov)  # Σ = L L^T
    e = mu - y  # (N, M)

    z = torch.linalg.solve_triangular(L, e.T, upper=False)  # (M, N)
    d2 = (z**2).sum(dim=0)  # (N,)
    return torch.sqrt(d2.mean())


# ---------- other metrics (your style) ----------


@torch.no_grad()
def rmse_2d(mu: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    err2 = (mu - y).pow(2).sum(dim=-1)
    return torch.sqrt(err2.mean())


@torch.no_grad()
def rmse_per_objective(mu: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(((mu - y) ** 2).mean(dim=0))


@torch.no_grad()
def nlpd_gaussian_diag(
    mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor, eps: float = 1e-12
):
    sigma = sigma.clamp_min(eps)
    var = sigma**2
    two_pi = torch.tensor(2.0 * math.pi, device=mu.device, dtype=mu.dtype)
    nlp = 0.5 * (torch.log(two_pi * var) + (y - mu) ** 2 / var)
    return nlp.sum(dim=-1).mean()


@torch.no_grad()
def gaussian_interval_coverage_joint(
    mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor, z: float
):
    lo = mu - z * sigma
    hi = mu + z * sigma
    return ((y >= lo) & (y <= hi)).all(dim=-1).float().mean()


# ---------- evaluator with Mahalanobis RMSE built in ----------


@torch.no_grad()
def evaluate_from_tensors_chunked(
    model,
    X: torch.Tensor,  # (N, d)
    y: torch.Tensor,  # (N, 2) or (N, 1, 2)
    device: torch.device,
    cfg=DiscreteMOBOConfig,
    *,
    batch_size: int = 4096,
    yvar_eval: Optional[torch.Tensor] = None,  # (N, 2) or (N, 1, 2) in ORIGINAL units
    eps: float = 1e-12,
    # --- new options ---
    maha_cov: Optional[
        torch.Tensor
    ] = None,  # (2,2) if you want fixed covariance (e.g., from train set)
    maha_shrinkage: float = 1e-6,
):
    """
    2D-output evaluation. BoTorch posterior assumed with mean/variance.
    Computes:
      - rmse_2d (Euclidean in objective space)
      - rmse per objective
      - rmse_mahalanobis (with optional fixed covariance)
      - nlpd (diag)
      - joint coverage
    """
    model.eval()

    # ---- shape y -> (N, 2) ----
    y2 = y
    if y2.ndim == 3 and y2.shape[1] == 1:

        y2 = y2[:, 0, :]
    y2 = y2.reshape(-1, y2.shape[-1])
    assert y2.shape[-1] == 2, f"Expected 2 outputs, got y.shape={tuple(y.shape)}"

    N = X.shape[0]

    mu_list, sigma_list = [], []

    # Outcome transform scaling (if present)
    ot = getattr(model, "outcome_transform", None)
    if ot is not None and hasattr(ot, "stdvs"):
        std = ot.stdvs.squeeze().detach().reshape(-1)  # (2,)
        if std.numel() != 2:
            raise ValueError(
                f"outcome_transform.stdvs has {std.numel()} elems; expected 2."
            )
    else:
        std = None
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        xb = X[start:end].to(device, non_blocking=True)
        observation_noise = None
        if yvar_eval is not None:
            nb = yvar_eval[start:end]
            if nb.ndim == 3 and nb.shape[1] == 1:
                nb = nb[:, 0, :]
            nb = nb.to(device, non_blocking=True)

            # if model has outcome transform, BoTorch posterior uses transformed space
            if std is not None:
                nb = nb / (std.to(device) ** 2)

            observation_noise = nb.unsqueeze(-2)  # (batch, 1, 2)
        post = model.posterior(xb, observation_noise=observation_noise)

        mu = post.mean  # typically (batch, 1, 2)
        var = post.variance  # typically (batch, 1, 2)

        mu = mu.squeeze(-2)  # -> (batch, 2)
        sigma = torch.sqrt(var.clamp_min(eps)).squeeze(-2)

        mu_list.append(torch.atleast_2d(mu.detach().cpu()))
        sigma_list.append(torch.atleast_2d(sigma.detach().cpu()))

    mu_all = torch.cat(mu_list, dim=0)  # (N, 2) on CPU
    sigma_all = torch.cat(sigma_list, dim=0)  # (N, 2) on CPU
    y_all = y2.detach().cpu()  # (N, 2) on CPU

    # ---- metrics ----
    rmse_obj = rmse_per_objective(mu_all, y_all)

    hv = Hypervolume(ref_point=cfg.ref_point)

    nd_mask_model = is_non_dominated(mu_all)
    nd_mask_true = is_non_dominated(y_all)
    pareto_Y_model = y_all[nd_mask_model]
    pareto_Y_true = y_all[nd_mask_true]

    metrics = {
        "hv_model": float(hv.compute(pareto_Y_model)),
        "hv_true": float(hv.compute(pareto_Y_true)),
        "rmse_2d": rmse_2d(mu_all, y_all).item(),
        "rmse_obj0": rmse_obj[0].item(),
        "rmse_obj1": rmse_obj[1].item(),
        # NEW: Mahalanobis RMSE (uses maha_cov if provided, else estimated from y_all)
        "rmse_mahalanobis": rmse_mahalanobis(
            mu_all, y_all, cov=maha_cov, shrinkage=maha_shrinkage
        ).item(),
        "nlpd_diag_2d": nlpd_gaussian_diag(mu_all, sigma_all, y_all, eps=eps).item(),
        "cov68_joint": gaussian_interval_coverage_joint(
            mu_all, sigma_all, y_all, z=1.0
        ).item(),
        "cov95_joint": gaussian_interval_coverage_joint(
            mu_all, sigma_all, y_all, z=1.96
        ).item(),
        "mean_sigma_obj0": sigma_all[:, 0].mean().item(),
        "mean_sigma_obj1": sigma_all[:, 1].mean().item(),
    }

    return metrics


def get_model_performance(
    obj: torch.tensor,
    X_set: torch.Tensor,
    cfg: DiscreteMOBOConfig,
) -> pd.DataFrame:
    """Run discrete multi-objective Bayesian optimization (MOBO).

    This routine performs batch-wise MOBO over a finite candidate set.
    It supports both random selection and qLogNEHVI-based acquisition
    with greedy batch construction. Mixed objective models are handled,
    including exact GPs, identity objectives, and Pyro VI models.

    For purely exact-GP objectives, a fast joint training path using
    a summed marginal log likelihood is used. If non-GP objectives
    are present, exact GP objectives are trained individually and
    combined into a ``ModelListGP``.

    Args:
        obj: Objective callable defining the (possibly noisy)
            multi-objective evaluation.
        X_set: Tensor of shape ``(N, d)`` containing the discrete
            candidate set in original (unnormalized) input space.
        cfg: Discrete MOBO configuration object. This must define,
            among others, batch sizes, model specifications,
            noise levels, and acquisition parameters.
        method: Selection strategy. Must be either:
            - ``"qlognehvi"``: Greedy batch selection using qLogNEHVI.
            - ``"random"``: Uniform random selection from unused points.

    Returns:
        A tuple ``(train_X, train_Y, train_Y_true)`` where:
            - train_X: Tensor of all evaluated inputs, shape ``(n_eval, d)``.
            - train_Y: Observed (noisy) objective values,
              shape ``(n_eval, m)``.
            - train_Y_true: Noise-free objective values,
              shape ``(n_eval, m)``.

    Raises:
        AssertionError: If ``method`` is not one of the supported options.
    """

    device = X_set.device
    n_set = X_set.shape[0]

    # Precompute normalized candidate set (big speed win)

    used_mask = torch.zeros(n_set, dtype=torch.bool, device=device)

    init_idx = range(cfg.n_init)
    used_mask[init_idx] = True

    train_X = X_set[init_idx]
    train_Y, train_Y_true = obj.evaluate(init_idx, cfg.noise_se)
    results = []
    for t in range(cfg.n_batch):

        tlog = {}  # per-iteration timing

        with timer("total_iter", tlog):

            # ---------------- model build ----------------
            per_obj_models = build_model_list(
                train_X=train_X,
                train_Y=train_Y,
                noise_se=cfg.noise_se,
                use_known_noise=cfg.use_known_noise,
                specs=cfg.models,
            )

            # ---------------- fitting ----------------
            for i, (mi, spec) in enumerate(zip(per_obj_models, cfg.models)):
                if _is_exact_gp_model(mi):
                    mll_i = ExactMarginalLogLikelihood(mi.likelihood, mi)
                    fit_gpytorch_mll(mll_i, options=cfg.models[i].fit_options)
                elif getattr(spec, "kind", None) == "identity":
                    mi.set_lookup_table(
                        X_set,
                        obj.Y_table[..., i : i + 1],
                        torch.full_like(obj.Y_table[..., i : i + 1], cfg.noise_se[i]),
                    )

            model = CombinedGaussianIndependentModel(*per_obj_models)

            idx = torch.arange(obj.Y_table.shape[0], device=obj.Y_table.device)
            Y_noisy, Y_true = obj.evaluate(idx, cfg.noise_se)
            Y_true = Y_true

            yvar_eval = (cfg.noise_se.to(X_set)).expand(X_set.shape[0], -1).clone()
            result = evaluate_from_tensors_chunked(
                model=model,
                device=X_set.device,
                X=X_set[~used_mask, :],
                y=Y_true[~used_mask],
                batch_size=1000,
                cfg=cfg,
                yvar_eval=yvar_eval,
            )
        result["iter"] = t
        print(
            f"[iter {t:03d}] obj {i} | "
            f"HV_Ratio: {result['hv_ratio']:.4f} | "
            f"HV_model: {result['hv_model']:.4f} | "
            f"HV_true: {result['hv_true']:.4f} | "
            f"RMSE0: {result['rmse_obj0']:.4f} | "
            f"RMSE1: {result['rmse_obj1']:.4f} | "
            f"RMSE_M: {result['rmse_mahalanobis']:.4f} | "
            f"NLPD: {result['nlpd_diag_2d']:.4f} | "
            f"C68: {result['cov68_joint']:.3f} | "
            f"C95: {result['cov95_joint']:.3f} | "
            f"σ̄0: {result['mean_sigma_obj0']:.4f} | "
            f"σ̄1: {result['mean_sigma_obj1']:.4f}"
        )

        results.append(result)

        new_idx = torch.tensor([t + cfg.n_init])
        used_mask[new_idx] = True

        # ---------------- evaluation ----------------
        new_X = X_set[new_idx]
        new_Y, new_Y_true = obj.evaluate(new_idx, cfg.noise_se)

        # ---------------- data update ----------------
        train_X = torch.cat([train_X, new_X], dim=0)
        train_Y = torch.cat([train_Y, new_Y], dim=0)
        train_Y_true = torch.cat([train_Y_true, new_Y_true], dim=0)

    df = pd.DataFrame(results)

    return df, per_obj_models
