import torch
import torch.nn as nn
from botorch.models.transforms.input import InputTransform
from pathlib import Path
from pibob.problems.load_points import load_dataset
import pibob
from botorch.utils.transforms import normalize
from collections.abc import Sequence


class MyInputTransform(InputTransform):
    def __init__(
        self,
        transform_module: nn.Module,
        *,
        transform_on_train: bool = True,
        transform_on_eval: bool = True,
        transform_on_fantasize: bool = True,
        is_one_to_many: bool = False,
    ):
        super().__init__()
        self.tf = transform_module

        # these must be attributes (not passed to super)
        self.transform_on_train = transform_on_train
        self.transform_on_eval = transform_on_eval
        self.transform_on_fantasize = transform_on_fantasize
        self.is_one_to_many = is_one_to_many

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        # X shape: (..., n, d)
        return self.tf(X)

    def untransform(self, X: torch.Tensor) -> torch.Tensor:
        # identity is fine unless you truly need an inverse
        return X

    def equals(self, other) -> bool:
        return isinstance(other, MyInputTransform)


class PoolDescriptorTransform(nn.Module):
    is_one_to_many = True

    def __init__(
        self,
        problem,
        descriptor_name,
    ):
        super().__init__()

        self.problem = problem

        # Accept either a single string or a list/tuple of strings
        if isinstance(descriptor_name, str):
            self.descriptor_names = [descriptor_name]
            self._single_output = True
        elif isinstance(descriptor_name, Sequence) and all(
            isinstance(d, str) for d in descriptor_name
        ):
            self.descriptor_names = list(descriptor_name)
            self._single_output = False
        else:
            raise TypeError("descriptor_name must be a string or a sequence of strings")

        outdir = Path(
            pibob.PROJECT_ROOT,
            "results",
            problem.__class__.__name__,
            "datapoints",
        )
        df = load_dataset(
            base_dir=outdir,
            input_names=problem.input_names,
            objective_names=self.descriptor_names,
            primary_sobol_id=0,
        )

        self.X_pool = normalize(df.all.X, bounds=problem.bounds)

        # Keep Y as 2D: [n_pool, n_descriptors]
        Y_pool = df.all.Y
        if Y_pool.ndim == 1:
            Y_pool = Y_pool.unsqueeze(-1)
        self.Y_pool = Y_pool

        self._index = {
            self._row_key(self.X_pool[i]): i for i in range(self.X_pool.shape[0])
        }

    def _row_key(self, row):
        return tuple(row.detach().cpu().tolist())

    def lookup_descriptor(self, x):
        rows = x.reshape(-1, x.shape[-1])

        idx = []
        for i in range(rows.shape[0]):
            k = self._row_key(rows[i])
            j = self._index.get(k)

            if j is None:
                raise ValueError(i, k, "Queried x not in pool (exact match required)")

            idx.append(j)

        idx = torch.tensor(idx, device=x.device, dtype=torch.long)
        y = self.Y_pool.to(x.device)[idx]  # shape: [n_rows, n_descriptors]

        if self._single_output:
            return y.squeeze(-1).reshape(x.shape[:-1])

        return y.reshape(*x.shape[:-1], len(self.descriptor_names))
