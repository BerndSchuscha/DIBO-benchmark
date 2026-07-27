import torch
import torch.nn as nn
from pibob.models.identity_model import IdentityModel
from pibob.problems.load_points import load_dataset
import pibob
from pibob.problems.Ms_NiTi.Ms_NiTi_Problem import Ms_NiTi_Problem
from pathlib import Path
from botorch.utils.transforms import normalize
# =========================
# Currin transforms
# =========================


class Transform1(nn.Module):
    def forward(self, x):
        x1 = x[..., 0]
        x2 = 1 / (x[..., 1]+3)
        x3 = torch.sqrt(-x[..., 2])

        y = torch.stack([x1, x2, x3], dim=-1)
        return y


class Transform2(nn.Module):
    is_one_to_many = True
    dim = 4

    def __init__(self, strict=True):
        super().__init__()
        self._strict = strict

        problem = Ms_NiTi_Problem()
        outdir = Path(
            pibob.PROJECT_ROOT, "results", problem.__class__.__name__, "datapoints"
        )
        df = load_dataset(
            base_dir=outdir,
            input_names=problem.input_names,
            objective_names=["cfrac"],
            primary_sobol_id=0,
        )

        self.X_pool = df.all.X
        self.Y_pool = df.all.Y

        self.X_pool = normalize(self.X_pool, bounds=problem.bounds)

        # Build lookup dictionary once
        self._index = {
            self._row_key(self.X_pool[i]): i for i in range(self.X_pool.shape[0])
        }

    def _row_key(self, row):
        # Example: convert tensor row to a hashable tuple
        return tuple(row.detach().cpu().tolist())

    def forward(self, x):
        x1 = x[..., 0]
        x2 = 1 / (x[..., 1]+3)
        x3 = torch.sqrt(-x[..., 2])

        rows = x.reshape(-1, x.shape[-1])

        idx = []
        for i in range(rows.shape[0]):
            k = self._row_key(rows[i])
            j = self._index.get(k, None)
            if j is None:
                if self._strict:
                    raise ValueError(
                        i, k, "Queried x not in pool (exact match required)"
                    )
                j = -1
            idx.append(j)

        idx = torch.tensor(idx, device=x.device, dtype=torch.long)

        y4 = self.Y_pool.to(x.device)[idx]
        y4 = y4.reshape(x.shape[:-1])

        y4 = x1 - 4 / 7 * y4 / (1 - y4)

        y = torch.stack([x1, x2, x3, y4], dim=-1)
        return y
