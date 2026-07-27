import torch
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
import gpytorch
import torch
from botorch.models.model import Model
import torch


def _mvn_diag(var: torch.Tensor) -> torch.Tensor:
    # var: (..., q) -> (..., q, q)
    return torch.diag_embed(var)


class IdentityModel(Model):
    def __init__(
        self,
        X_train_init: torch.Tensor,
        Y_train_init: torch.Tensor,
        *,
        strict_pool_lookup=True,
    ):
        super().__init__()
        if X_train_init.dim() == 1:
            X_train_init = X_train_init.unsqueeze(0)
        if Y_train_init.dim() == 1:
            Y_train_init = Y_train_init.unsqueeze(0)

        self._d = X_train_init.shape[-1]
        self._m = Y_train_init.shape[-1]
        self._strict = bool(strict_pool_lookup)

        self._device = X_train_init.device
        self._dtype = X_train_init.dtype

        self.X_pool = None
        self.mean_table = None
        self.sigma_table = None
        self._index = None

    @property
    def num_outputs(self) -> int:
        return self._m

    def set_lookup_table(
        self, X_pool: torch.Tensor, mean_table: torch.Tensor, sigma_table: torch.Tensor
    ):
        X_pool = X_pool.to(self._device, self._dtype)
        mean_table = mean_table.to(self._device, self._dtype)
        sigma_table = sigma_table.to(self._device, self._dtype)

        assert X_pool.shape[1] == self._d
        assert mean_table.shape[1] == self._m
        assert mean_table.shape == sigma_table.shape
        assert X_pool.shape[0] == mean_table.shape[0]

        keys = [self._row_key(X_pool[i]) for i in range(X_pool.shape[0])]
        index = {k: i for i, k in enumerate(keys)}
        assert len(index) == len(keys), "duplicate rows in X_pool"

        self.X_pool = X_pool
        self.mean_table = mean_table
        self.sigma_table = sigma_table
        self._index = index

    def posterior(self, X, output_indices=None, **kwargs):
        # X: (..., q, d)
        *batch, q, d = X.shape
        assert d == self._d
        assert self.mean_table is not None

        X2 = X.reshape(-1, d)
        idx = []
        for i in range(X2.shape[0]):
            k = self._row_key(X2[i])
            j = self._index.get(k, None)
            if j is None:
                if self._strict:
                    raise ValueError("Queried X not in pool (exact match required)")
                j = -1
            idx.append(j)
        idx = torch.tensor(idx, device=X.device, dtype=torch.long)

        mu = self.mean_table.index_select(0, idx.to(self._device)).to(
            X.device, X.dtype
        )  # (B*q, m)
        sd = self.sigma_table.index_select(0, idx.to(self._device)).to(
            X.device, X.dtype
        )  # (B*q, m)
        mu = mu.reshape(*batch, q, self._m)  # (..., q, m)
        var = (sd**2).reshape(*batch, q, self._m)  # (..., q, m)

        # IMPORTANT: make a Multitask MVN so BoTorch sees q correctly
        mean_t = mu.transpose(-1, -2)  # (..., m, q)
        var_t = var.transpose(-1, -2)  # (..., m, q)
        cov_t = _mvn_diag(var_t)  # (..., m, q, q)

        batch_mvn = gpytorch.distributions.MultivariateNormal(mean_t, cov_t)
        mtmvn = gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            batch_mvn, task_dim=-1
        )
        post = GPyTorchPosterior(mtmvn)
        if output_indices is not None:
            post = post._index_output(output_indices)
        return post

    @staticmethod
    def _row_key(xrow: torch.Tensor) -> bytes:
        return xrow.detach().cpu().contiguous().numpy().tobytes()

    def predict(self, X, output_indices=None, **kwargs):
        # X: (..., q, d)
        *batch, q, d = X.shape
        assert d == self._d
        assert self.mean_table is not None

        X2 = X.reshape(-1, d)
        idx = []
        for i in range(X2.shape[0]):
            k = self._row_key(X2[i])
            j = self._index.get(k, None)
            if j is None:
                if self._strict:
                    raise ValueError("Queried X not in pool (exact match required)")
                j = -1
            idx.append(j)
        idx = torch.tensor(idx, device=X.device, dtype=torch.long)

        mu = self.mean_table.index_select(0, idx.to(self._device)).to(
            X.device, X.dtype
        )  # (B*q, m)
        return mu
