from __future__ import annotations
import torch
from typing import Optional
from pibob.optimization.acq import build_qlognehvi
from torch import nn
from torch import Tensor
from torch.distributions.normal import Normal
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.utils.multi_objective.pareto import is_non_dominated
import matplotlib.pyplot as plt


class CombinedAcqf(nn.Module):
    def __init__(self, iteration, acqf1, acqf2=None, acqf3=None, w2=10.0, w3=0.2):
        super().__init__()
        self.acqf1 = acqf1
        self.acqf2 = acqf2
        self.acqf3 = acqf3
        self.w2 = float(w2)
        self.w3 = float(w3)
        self.iteration = float(iteration)

    def forward(self, X):
        y1 = self.acqf1(X).exp()
        # multiplicative factor y2
        if self.acqf2 is None:
            y2 = torch.ones_like(y1)
        else:
            a2 = self.acqf2(X)
            # guard: if someone passed an acqf that returns None
            if a2 is None:
                y2 = torch.ones_like(y1)
            else:
                # y2 = a2.view(-1)
                power = 10 / (self.iteration + 10.0)
                y2 = a2.pow(power).view(-1)

        # additive term y3
        if self.acqf3 is None:
            y3 = torch.zeros_like(y1)  # additive identity
        else:
            a3 = self.acqf3(X)
            if a3 is None:
                y3 = torch.zeros_like(y1)
            else:
                scale = self.w3 * (10.0 / (10.0 + self.iteration))
                y3 = a3 * scale
        y = y1 * y2 + y3

        return y

    def set_X_pending(self, X_pending: torch.Tensor | None = None) -> None:
        """BoTorch-compatible: set pending points for greedy batch selection."""
        self.X_pending = X_pending

        # Propagate to sub-acquisition functions if they support it
        if hasattr(self.acqf1, "set_X_pending"):
            self.acqf1.set_X_pending(X_pending)
        if self.acqf2 is not None and hasattr(self.acqf2, "set_X_pending"):
            self.acqf2.set_X_pending(X_pending)
        if self.acqf3 is not None and hasattr(self.acqf3, "set_X_pending"):
            self.acqf3.set_X_pending(X_pending)


_normal = Normal(0.0, 1.0)


class MultiObjectivePIProduct(AcquisitionFunction):
    """
    π_MO(x) = ∏_{k=1}^K (1 - p_k(x))
    p_k(x)  = ∏_{i=1}^M Φ((y_i^(k) - μ_i(x)) / σ_eff,i(x))

    Assumes independent Gaussian objectives (fits CombinedGaussianIndependentModel).
    """

    def __init__(
        self,
        model,
        Y_all: Tensor,
        pareto_Y: Tensor,  # [K, M] (non-dominated points; in "maximize all" convention)
        sigma_eps: float = 1e-12,
        use_observation_noise: bool = False,  # optional: include obs noise in σ_eff
    ):
        super().__init__(model=model)
        self.register_buffer("pareto_Y", pareto_Y)
        self.register_buffer("Y_all", Y_all)
        self.sigma_eps = float(sigma_eps)
        self.use_observation_noise = bool(use_observation_noise)

    def forward(self, X: Tensor) -> Tensor:
        """
        X: [..., q, d] (BoTorch convention)
        returns: [..., q] (PI per point). If you optimize q=1, it's [..., 1].
        """
        posterior = self.model.posterior(X)
        mu = posterior.mean  # [..., q, M]
        var = posterior.variance  # [..., q, M]

        # Optionally include observation noise if available (depends on model/posterior)
        if self.use_observation_noise and hasattr(
            posterior, "observation_noise_variance"
        ):
            var = var + posterior.observation_noise_variance

        pareto_range = torch.abs(
            self.Y_all.max(dim=0).values - self.Y_all.min(dim=0).values
        )  # [M]
        extra_noise = 0.2 * pareto_range  # [M]
        extra_noise = extra_noise.view(*([1] * (var.ndim - 1)), -1)
        var = var + extra_noise**2

        sigma = var.clamp_min(self.sigma_eps).sqrt()  # [..., q, M]

        # Broadcast pareto points: [K, ..., q, M]
        z = (self.pareto_Y[:, None, None, :] - mu.unsqueeze(0)) / sigma.unsqueeze(0)

        # log p_k = sum_i log Φ(z)
        log_cdf = torch.log(_normal.cdf(z).clamp_min(self.sigma_eps))  # [K, ..., q, M]
        log_p_k = log_cdf.sum(dim=-1)  # [K, ..., q]
        p_k = log_p_k.exp().clamp(max=1.0 - self.sigma_eps)  # [K, ..., q]

        # log π = sum_k log(1 - p_k)  (still the same product, just stable)
        log_pi = torch.log1p(-p_k).sum(dim=0)  # [..., q]
        return log_pi.exp()


@torch.no_grad()
def _argmax_acq_over_candidates_chunked(
    acqf,
    Xn_cand: torch.Tensor,  # (Ncand, d)
    eval_batch_size: int = 256,
) -> tuple[int, torch.Tensor]:
    """
    Evaluate acqf over ALL candidates in chunks, return:
      - best_local_idx (int) w.r.t Xn_cand
      - best_value (tensor scalar)
    """
    Ncand = Xn_cand.shape[0]
    best_val = None
    best_idx = None

    # loop over chunks
    for start in range(0, Ncand, eval_batch_size):
        end = min(start + eval_batch_size, Ncand)
        X_chunk = Xn_cand[start:end].unsqueeze(1)  # (chunk, 1, d)
        vals = acqf(X_chunk).view(-1)  # (chunk,)

        # local best in chunk
        chunk_best_val, chunk_best_rel = torch.max(vals, dim=0)
        chunk_best_idx = start + int(chunk_best_rel.item())

        if best_val is None or chunk_best_val > best_val:
            best_val = chunk_best_val
            best_idx = chunk_best_idx

    return int(best_idx), best_val


@torch.no_grad()
def select_random_indices(
    used_mask: torch.Tensor,
    q: int,
) -> torch.Tensor:
    """Uniform random selection from remaining indices."""
    cand = torch.where(~used_mask)[0]
    pick = cand[torch.randperm(cand.numel(), device=used_mask.device)[:q]]
    return pick


@torch.no_grad()
def select_qlognehvi_indices_greedy(
    model,
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    Xn_set: torch.Tensor,  # (N, d) normalized candidate set
    used_mask: torch.Tensor,
    ref_point: torch.Tensor,
    q: int,
    mc_samples: int,
    eval_batch_size: int = 256,  # <-- chunk size for evaluating ALL candidates
    place: str = "surrogate",
    iteration: int = 1,
    model_extra: Optional = None,
) -> torch.Tensor:
    """
    Greedy batch construction from a discrete set.
    Each step evaluates qLogNEHVI over ALL remaining candidates, but in chunks.
    Returns indices into the candidate set.
    """

    acqf1, acqf2, acqf3 = None, None, None
    if place == "surrogate":
        acqf1 = build_qlognehvi(
            model=model,
            train_X=train_X,
            ref_point=ref_point,
            mc_samples=mc_samples,
        )
    if place == "regularisation":
        acqf1 = build_qlognehvi(
            model=model_extra,
            train_X=train_X,
            ref_point=ref_point,
            mc_samples=mc_samples,
        )

        acqf3 = build_qlognehvi(
            model=model,
            train_X=train_X,
            ref_point=ref_point,
            mc_samples=mc_samples,
        )

    if place == "prior":
        nd_mask = is_non_dominated(train_Y)
        pareto_Y = train_Y[nd_mask]

        acqf1 = build_qlognehvi(
            model=model_extra,
            train_X=train_X,
            ref_point=ref_point,
            mc_samples=mc_samples,
        )

        acqf2 = MultiObjectivePIProduct(
            model=model,
            Y_all=train_Y,
            pareto_Y=pareto_Y,
            use_observation_noise=True,
        )

    acqf = CombinedAcqf(acqf1=acqf1, acqf2=acqf2, acqf3=acqf3, iteration=iteration)

    chosen = []
    pending_Xn = None
    for _ in range(q):
        acqf.set_X_pending(pending_Xn)

        cand_idx = torch.where(~used_mask)[0]
        Xn_cand = Xn_set[cand_idx]  # (Ncand, d)

        best_local, _ = _argmax_acq_over_candidates_chunked(
            acqf=acqf,
            Xn_cand=Xn_cand,
            eval_batch_size=eval_batch_size,
        )
        best_idx = cand_idx[best_local]
        chosen.append(best_idx)
        used_mask[best_idx] = True

        x_best_n = Xn_set[best_idx : best_idx + 1]  # (1, d)
        x_best_pending = x_best_n.unsqueeze(0)  # (1, 1, d)

        if pending_Xn is None:
            pending_Xn = x_best_pending  # (1, 1, d)
        else:
            pending_Xn = torch.cat([pending_Xn, x_best_pending], dim=1)  # (1, k+1, d)

    return torch.stack(chosen, dim=0)
