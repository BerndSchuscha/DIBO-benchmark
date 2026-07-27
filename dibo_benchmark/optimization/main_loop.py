from __future__ import annotations
import torch
from typing import Tuple

from botorch.fit import fit_gpytorch_mll

from pibob.optimization.config import DiscreteMOBOConfig
from pibob.optimization.selection import (
    select_qlognehvi_indices_greedy,
)
from pibob.optimization.build_model_list import build_model_list
import gpytorch
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood


from pibob.optimization.CombinedGaussianIndependentModel import (
    CombinedGaussianIndependentModel,
)

import time
from contextlib import contextmanager

from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated
from pibob.optimization.get_model_performance import evaluate_from_tensors_chunked
import pandas as pd
from pibob.optimization import ObjModelSpec


def compute_hv(Y: torch.Tensor, ref_point: torch.Tensor) -> float:
    """
    Y: (n, m) objective values (assumed maximization)
    ref_point: (m,)
    Returns scalar hypervolume.
    """
    hv = Hypervolume(ref_point=ref_point)
    nd_mask = is_non_dominated(Y)
    pareto_Y = Y[nd_mask]
    return float(hv.compute(pareto_Y)), nd_mask.sum().item()


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


def main_loop(
    obj: torch.tensor,
    X_set: torch.Tensor,
    cfg: DiscreteMOBOConfig,
    method: str = "qlognehvi",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    assert method in {"qlognehvi", "random"}

    device = X_set.device
    n_set = X_set.shape[0]

    # Precompute normalized candidate set (big speed win)
    Xn_set = X_set

    used_mask = torch.zeros(n_set, dtype=torch.bool, device=device)

    init_idx = range(cfg.n_init)
    used_mask[init_idx] = True

    train_X = X_set[init_idx]
    train_Y, train_Y_true = obj.evaluate(init_idx, cfg.noise_se)
    hv_true_hist = []  # optional: if you track noise-free objective separately
    per_obj_models_extra = []
    model_spec = []
    model_extra = None
    results = []
    for t in range(cfg.n_batch):

        tlog = {}  # per-iteration timing

        with timer("total_iter", tlog):

            # ---------------- model build ----------------
            with timer("build_models", tlog):
                if cfg.place == "surrogate":
                    per_obj_models = build_model_list(
                        train_X=train_X,
                        train_Y=train_Y,
                        noise_se=cfg.noise_se,
                        use_known_noise=cfg.use_known_noise,
                        specs=cfg.models,
                    )
                elif cfg.place == "prior":
                    per_obj_models = build_model_list(
                        train_X=train_X,
                        train_Y=train_Y,
                        noise_se=cfg.noise_se,
                        use_known_noise=cfg.use_known_noise,
                        specs=cfg.models,
                    )

                    model_spec = [
                        ObjModelSpec(kind="gpr", fit_options={"maxiter": 1000}),
                        ObjModelSpec(kind="gpr", fit_options={"maxiter": 1000}),
                    ]

                    per_obj_models_extra = build_model_list(
                        train_X=train_X,
                        train_Y=train_Y,
                        noise_se=cfg.noise_se,
                        use_known_noise=cfg.use_known_noise,
                        specs=model_spec,
                    )

                elif cfg.place == "regularisation":
                    per_obj_models = build_model_list(
                        train_X=train_X,
                        train_Y=train_Y,
                        noise_se=cfg.noise_se,
                        use_known_noise=cfg.use_known_noise,
                        specs=cfg.models,
                    )

                    model_spec = [
                        ObjModelSpec(kind="gpr", fit_options={"maxiter": 1000}),
                        ObjModelSpec(kind="gpr", fit_options={"maxiter": 1000}),
                    ]

                    per_obj_models_extra = build_model_list(
                        train_X=train_X,
                        train_Y=train_Y,
                        noise_se=cfg.noise_se,
                        use_known_noise=cfg.use_known_noise,
                        specs=model_spec,
                    )

            # ---------------- fitting ----------------
            with timer("fit_models", tlog):

                for i, (mi, spec) in enumerate(zip(per_obj_models, cfg.models)):
                    if _is_exact_gp_model(mi):
                        mll_i = ExactMarginalLogLikelihood(mi.likelihood, mi)
                        fit_gpytorch_mll(mll_i, options=cfg.models[i].fit_options)
                    elif getattr(spec, "kind", None) == "identity":
                        mi.set_lookup_table(
                            X_set,
                            obj.Y_table[..., i : i + 1],
                            torch.full_like(
                                obj.Y_table[..., i : i + 1], cfg.noise_se[i]
                            ),
                        )

                model = CombinedGaussianIndependentModel(*per_obj_models)

                if cfg.place in ["prior", "regularisation"]:
                    for i, (mi, spec) in enumerate(
                        zip(per_obj_models_extra, model_spec)
                    ):
                        mll_i = ExactMarginalLogLikelihood(mi.likelihood, mi)
                        fit_gpytorch_mll(mll_i, options=model_spec[i].fit_options)

                    model_extra = CombinedGaussianIndependentModel(
                        *per_obj_models_extra
                    )
            # ---------------- acquisition ----------------
            with timer("acquisition", tlog):
                if method == "random":
                    new_idx = torch.tensor([t + cfg.n_init])
                    used_mask[new_idx] = True
                else:
                    new_idx = select_qlognehvi_indices_greedy(
                        model=model,
                        model_extra=model_extra,
                        train_X=train_X,
                        train_Y=train_Y,
                        Xn_set=Xn_set,
                        used_mask=used_mask,
                        ref_point=cfg.ref_point,
                        q=cfg.batch_size,
                        mc_samples=cfg.mc_samples,
                        eval_batch_size=cfg.eval_batch_size,
                        place=cfg.place,
                        iteration=t,
                    )
            with timer("model_testing_model_eval", tlog):
                idx = torch.arange(obj.Y_table.shape[0], device=obj.Y_table.device)
                _, Y_true = obj.evaluate(idx, cfg.noise_se)

            with timer("model_testing_metric_eval", tlog):
                yvar_eval = (cfg.noise_se.to(X_set)).expand(X_set.shape[0], -1).clone()
                result = evaluate_from_tensors_chunked(
                    model=model,
                    device=X_set.device,
                    X=X_set[~used_mask, :],
                    y=Y_true[~used_mask],
                    batch_size=cfg.eval_batch_size,
                    cfg=cfg,
                    yvar_eval=yvar_eval,
                )
                result["iter"] = t
            results.append(result)

            # ---------------- evaluation ----------------
            with timer("evaluate_objective", tlog):
                new_X = X_set[new_idx]
                new_Y, new_Y_true = obj.evaluate(new_idx, cfg.noise_se)

            # ---------------- data update ----------------
            with timer("update_dataset", tlog):
                train_X = torch.cat([train_X, new_X], dim=0)
                train_Y = torch.cat([train_Y, new_Y], dim=0)
                train_Y_true = torch.cat([train_Y_true, new_Y_true], dim=0)

            # ---------------- hypervolume ----------------
            with timer("hypervolume", tlog):
                # Use observed (noisy) or true (noise-free) depending on what you want to report
                # train_Y is typically noisy; train_Y_true is noise-free in your code.
                hv_true_val, N_par = compute_hv(train_Y_true, cfg.ref_point)
                hv_true_hist.append(hv_true_val)

            if used_mask.sum().item() == used_mask.numel() - 10:
                print("All candidates exhausted → stopping BO.")
                break

        # ---------------- pretty print ----------------
        df = pd.DataFrame(results)
        print(
            f"{method} | iter {t+1:02d}/{cfg.n_batch} | n={train_X.shape[0]} | "
            f"HV_true={hv_true_hist[-1]:.4g} | N_par={N_par:d} | "
            f"build={tlog['build_models']:.3f}s | "
            f"fit={tlog['fit_models']:.3f}s | "
            f"acq={tlog['acquisition']:.3f}s | "
            f"testing model={tlog['model_testing_model_eval']:.3f}s | "
            f"testing metric={tlog['model_testing_metric_eval']:.3f}s | "
            f"eval={tlog['evaluate_objective']:.3f}s | "
            f"total={tlog['total_iter']:.3f}s"
        )

    return train_X, train_Y, train_Y_true, model, df
