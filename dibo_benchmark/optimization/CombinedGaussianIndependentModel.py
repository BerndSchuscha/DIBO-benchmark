import torch
import gpytorch
import botorch

from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal


def _add_jitter_and_symmetrize(cov: torch.Tensor, jitter: float) -> torch.Tensor:
    # cov: (..., q, q) or (..., m, q, q)
    cov = 0.5 * (cov + cov.transpose(-1, -2))  # enforce symmetry
    eye = torch.eye(cov.size(-1), device=cov.device, dtype=cov.dtype)
    cov = cov + jitter * eye  # broadcast over batch dims
    return cov

def _make_psd(cov: torch.Tensor, rel_jitter: float = 1e-6, max_tries: int = 3) -> torch.Tensor:
    """Symmetrize and add relative jitter until Cholesky succeeds; eigh-clamp as last resort.

    cov: (..., n, n)
    """
    cov = 0.5 * (cov + cov.transpose(-1, -2))
    diag_mean = (
        cov.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True).clamp_min(1e-12)
    )  # (..., 1)
    eye = torch.eye(cov.shape[-1], dtype=cov.dtype, device=cov.device)

    jitter = rel_jitter
    for _ in range(max_tries):
        cov_j = cov + (jitter * diag_mean).unsqueeze(-1) * eye
        _, info = torch.linalg.cholesky_ex(cov_j)
        if (info == 0).all():
            return cov_j
        jitter *= 10.0

    # Last resort: eigenvalue floor at rel_jitter * scale
    evals, evecs = torch.linalg.eigh(cov)
    evals = evals.clamp_min(rel_jitter * diag_mean)
    return evecs @ (evals.unsqueeze(-1) * evecs.transpose(-1, -2))


def _to_batch_mvn(
    mvn: gpytorch.distributions.Distribution,
    jitter: float = 1e-6,
) -> MultivariateNormal:
    """
    Convert a gpytorch distribution into a *batch* MultivariateNormal where tasks live in
    a batch dimension:

      mean: (..., m, q)
      cov : (..., m, q, q)
    """
    if isinstance(mvn, MultivariateNormal) and not isinstance(
        mvn, MultitaskMultivariateNormal
    ):
        mean = mvn.mean  # (..., q)
        cov = (
            mvn.lazy_covariance_matrix.to_dense()
            if hasattr(mvn, "lazy_covariance_matrix")
            else mvn.covariance_matrix
        )  # (..., q, q)
        cov = _make_psd(cov, rel_jitter=jitter)

        mean_b = mean.unsqueeze(-2)  # (..., 1, q)
        cov_b = cov.unsqueeze(-3)  # (..., 1, q, q)
        return MultivariateNormal(mean_b, cov_b)

    if isinstance(mvn, MultitaskMultivariateNormal):
        mean = mvn.mean  # (..., q, m)

        cov_full = mvn.covariance_matrix  # (..., q*m, q*m)
        q = mean.shape[-2]
        m = mean.shape[-1]
        interleaved = getattr(mvn, "_interleaved", True)

        if interleaved:
            cov_5 = cov_full.reshape(
                *cov_full.shape[:-2], q, m, q, m
            )  # (..., q, m, q, m)
            cov_task = cov_5.diagonal(dim1=-3, dim2=-1)  # (..., q, q, m)
        else:
            cov_5 = cov_full.reshape(
                *cov_full.shape[:-2], m, q, m, q
            )  # (..., m, q, m, q)
            cov_task = cov_5.diagonal(dim1=-4, dim2=-2)  # (..., q, q, m)

        cov_task = cov_task.permute(*range(cov_task.dim() - 3), -1, -3, -2).contiguous()
        # cov_task: (..., m, q, q)

        cov_task = _make_psd(cov_task, rel_jitter=jitter)

        mean_b = mean.transpose(-1, -2).contiguous()  # (..., m, q)
        return MultivariateNormal(mean_b, cov_task)

    raise TypeError(f"Unsupported distribution type: {type(mvn)}")

def _to_batch_mvn_old(
    mvn: gpytorch.distributions.Distribution,
    jitter: float = 1e-6,
) -> MultivariateNormal:
    """
    Convert a gpytorch distribution into a *batch* MultivariateNormal where tasks live in
    a batch dimension:

      mean: (..., m, q)
      cov : (..., m, q, q)
    """
    if isinstance(mvn, MultivariateNormal) and not isinstance(
        mvn, MultitaskMultivariateNormal
    ):
        mean = mvn.mean  # (..., q)

        # Prefer lazy -> jitter -> dense (more stable than densify first)
        if hasattr(mvn, "lazy_covariance_matrix"):
            cov = mvn.lazy_covariance_matrix.add_jitter(
                jitter
            ).to_dense()  # (..., q, q)
        else:
            cov = mvn.covariance_matrix  # (..., q, q)
            cov = _add_jitter_and_symmetrize(cov, jitter)

        mean_b = mean.unsqueeze(-2)  # (..., 1, q)
        cov_b = cov.unsqueeze(-3)  # (..., 1, q, q)
        return MultivariateNormal(mean_b, cov_b)

    if isinstance(mvn, MultitaskMultivariateNormal):
        mean = mvn.mean  # (..., q, m)

        cov_full = mvn.covariance_matrix  # (..., q*m, q*m)
        q = mean.shape[-2]
        m = mean.shape[-1]
        interleaved = getattr(mvn, "_interleaved", True)

        if interleaved:
            cov_5 = cov_full.reshape(
                *cov_full.shape[:-2], q, m, q, m
            )  # (..., q, m, q, m)
            cov_task = cov_5.diagonal(dim1=-3, dim2=-1)  # (..., q, q, m)
        else:
            cov_5 = cov_full.reshape(
                *cov_full.shape[:-2], m, q, m, q
            )  # (..., m, q, m, q)
            cov_task = cov_5.diagonal(dim1=-4, dim2=-2)  # (..., q, q, m)

        cov_task = cov_task.permute(*range(cov_task.dim() - 3), -1, -3, -2).contiguous()
        # cov_task: (..., m, q, q)

        cov_task = _add_jitter_and_symmetrize(cov_task, jitter)

        mean_b = mean.transpose(-1, -2).contiguous()  # (..., m, q)
        return MultivariateNormal(mean_b, cov_task)

    raise TypeError(f"Unsupported distribution type: {type(mvn)}")


class CombinedGaussianIndependentModel(Model):
    def __init__(self, a: Model, b: Model):
        super().__init__()
        self.a = a
        self.b = b
        self._m = a.num_outputs + b.num_outputs

    @property
    def num_outputs(self):
        return self._m

    def posterior(self, X, output_indices=None, observation_noise=None, **kwargs):
        if observation_noise is None:

            pa = self.a.posterior(X, **kwargs).distribution
            pb = self.b.posterior(X, **kwargs).distribution
        else:
            pa = self.a.posterior(
                X, observation_noise=observation_noise[:, :, 0], **kwargs
            ).distribution
            pb = self.b.posterior(
                X, observation_noise=observation_noise[:, :, 1], **kwargs
            ).distribution

        ba = _to_batch_mvn(pa)  # mean (..., m_a, q)
        bb = _to_batch_mvn(pb)  # mean (..., m_b, q)

        mean = torch.cat([ba.mean, bb.mean], dim=-2)  # (..., m_total, q)
        cov = torch.cat(
            [ba.covariance_matrix, bb.covariance_matrix], dim=-3
        )  # (..., m_total, q, q)

        bmvn = gpytorch.distributions.MultivariateNormal(mean, cov)
        mtmvn = gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            bmvn, task_dim=-1
        )
        post = GPyTorchPosterior(mtmvn)
        if output_indices is not None:
            post = post._index_output(output_indices)
        return post
