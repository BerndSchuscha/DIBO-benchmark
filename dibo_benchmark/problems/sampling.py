from __future__ import annotations

from pathlib import Path
import json
import torch
from torch.quasirandom import SobolEngine
import torch
from torch.quasirandom import SobolEngine
import numpy as np


def sobol_samples(bounds: torch.Tensor, n: int, seed: int) -> torch.Tensor:
    """
    bounds: (2, d) tensor
    returns: (n, d) Sobol points scaled to bounds
    """
    bounds = bounds.to(dtype=torch.double)
    lo, hi = bounds[0], bounds[1]
    d = lo.numel()
    if seed is not None:
        eng = SobolEngine(dimension=d, scramble=True, seed=int(seed))
    else:
        eng = SobolEngine(dimension=d)
    u = eng.draw(n).to(dtype=torch.double)  # in [0,1]^d
    return lo + (hi - lo) * u


def grid_points(bounds: torch.Tensor, n: int) -> torch.Tensor:
    """
    bounds: (2, d) tensor
    returns: (n_per_dim**d, d) full Cartesian grid in bounds

    Warning: grows exponentially with d.
    """
    n_per_dim = int((n) ** (1 / bounds[:, 0].size()[0]))
    bounds = bounds.to(dtype=torch.double)
    lo, hi = bounds[0], bounds[1]
    d = lo.numel()

    axes = [
        torch.linspace(lo[i], hi[i], n_per_dim, dtype=torch.double) for i in range(d)
    ]
    mesh = torch.meshgrid(*axes, indexing="ij")
    grid = torch.stack([m.reshape(-1) for m in mesh], dim=-1)  # (N, d)
    return grid


def _check_truncated_simplex_feasible(
    lb: torch.Tensor, ub: torch.Tensor, tol: float = 1e-12
) -> bool:
    lb = torch.as_tensor(lb, dtype=torch.double)
    ub = torch.as_tensor(ub, dtype=torch.double)
    if (lb < -tol).any() or (ub > 1 + tol).any() or (lb > ub + tol).any():
        return False
    return (lb.sum() <= 1 + tol) and (ub.sum() >= 1 - tol)


def _sobol_to_simplex_order_stats(u: torch.Tensor) -> torch.Tensor:
    """
    u: (n, d-1) in [0,1]
    returns: (n, d) uniform on simplex via sorted uniforms + differences
    """
    n, d_minus_1 = u.shape
    u_sorted, _ = torch.sort(u, dim=-1)
    z0 = torch.zeros((n, 1), dtype=u.dtype, device=u.device)
    z1 = torch.ones((n, 1), dtype=u.dtype, device=u.device)
    cuts = torch.cat([z0, u_sorted, z1], dim=-1)  # (n, d+1)
    x = cuts[:, 1:] - cuts[:, :-1]  # (n, d)
    return x


def sobol_truncated_simplex(
    n: int,
    lb: torch.Tensor,
    ub: torch.Tensor,
    *,
    seed: int = 0,
    batch: int = 4096,
    max_draws: int = 2_000_000,
    dtype=torch.double,
    device=None,
) -> torch.Tensor:
    """
    Generate n points on the simplex with box bounds using a Sobol sequence + rejection.

    Returns:
        X: (n, d) with sum=1 and lb<=X<=ub.

    Notes:
      - Deterministic given (seed, n, batch) because SobolEngine is deterministic.
      - If acceptance is very low (tiny feasible region), this may be slow; then you need
        a polytope sampler (hit-and-run) or a different construction.
    """
    lb = torch.as_tensor(lb, dtype=dtype, device=device)
    ub = torch.as_tensor(ub, dtype=dtype, device=device)
    d = lb.numel()

    if not _check_truncated_simplex_feasible(lb, ub):
        raise ValueError("Infeasible bounds: intersection (simplex ∩ box) is empty.")

    eng = SobolEngine(dimension=d - 1, scramble=True, seed=int(seed))

    kept = []
    draws = 0
    kept_n = 0

    while kept_n < n:
        if draws >= max_draws:
            raise RuntimeError(
                f"Reached max_draws={max_draws} without enough accepted points "
                f"({kept_n}/{n}). Feasible region likely too small."
            )

        u = eng.draw(batch).to(dtype=dtype)
        if device is not None:
            u = u.to(device)

        X = _sobol_to_simplex_order_stats(u)  # (batch, d), sum=1

        mask = (X >= lb).all(dim=-1) & (X <= ub).all(dim=-1)
        Xk = X[mask]
        if Xk.numel():
            kept.append(Xk)
            kept_n += Xk.shape[0]

        draws += batch

    return torch.cat(kept, dim=0)[:n]


def save_precalc(
    out_dir: str | Path,
    *,
    bounds: torch.Tensor,  # (2, d)
    n_sobol: int,
    sobol_seeds: list[int],  # e.g. length 10
    n_grid_per_dim: int,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # grid
    X_grid = grid_points(bounds, n_grid_per_dim)
    torch.save(X_grid, out_dir / "grid.pt")

    # sobol batches
    for i, seed in enumerate(sobol_seeds):
        X = sobol_samples(bounds, n_sobol, seed=seed)
        torch.save(X, out_dir / f"sobol_{i:02d}.pt")

    # metadata (handy for debugging / provenance)
    meta = {
        "bounds": bounds.detach().cpu().tolist(),
        "dim": int(bounds.shape[1]),
        "n_sobol": int(n_sobol),
        "sobol_seeds": [int(s) for s in sobol_seeds],
        "n_grid_per_dim": int(n_grid_per_dim),
        "grid_size": int(X_grid.shape[0]),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
