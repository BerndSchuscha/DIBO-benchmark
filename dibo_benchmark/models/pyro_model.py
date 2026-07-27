import torch
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample, PyroParam
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.infer.autoguide import AutoMultivariateNormal
from pyro.optim import Adam
import gpytorch
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from pyro.distributions import constraints
import torch
import pyro
import pyro.distributions as dist
import torch.nn.functional as F
from pyro.nn import PyroModule, PyroParam
from pyro.distributions import constraints
from pyro.infer import Predictive
import os
import re
import torch


class PyroAnalyticalMean(PyroModule):
    """
    Takes ONLY an AnalyticalCore.
    AnalyticalCore is responsible for providing:
      - n_par (int)
      - init_loc (Tensor[n_par])
      - init_scale (Tensor[n_par])
      - noise_level (float)
      - sig1_init (float)
      - forward(x, par)
    """

    def __init__(self, core):
        super().__init__()
        self.core = core

        # ---- strict contract checks ----
        required = ["n_par", "init_loc", "init_scale", "noise_level", "sig1_init"]
        for name in required:
            if not hasattr(core, name):
                raise AttributeError(f"{core.__class__.__name__} must define '{name}'")

        n_par = int(core.n_par)

        init_loc = torch.as_tensor(core.init_loc, dtype=torch.float32)
        init_scale = torch.as_tensor(core.init_scale, dtype=torch.float32)

        if init_loc.ndim != 1 or init_scale.ndim != 1:
            raise ValueError("core.init_loc and core.init_scale must be 1D tensors.")

        if init_loc.numel() != n_par or init_scale.numel() != n_par:
            raise ValueError(
                f"Size mismatch: n_par={n_par}, "
                f"init_loc={init_loc.numel()}, init_scale={init_scale.numel()}."
            )

        if not torch.all(init_scale > 0):
            raise ValueError("core.init_scale must be strictly positive (std devs).")

        self.par = PyroSample(dist.Normal(init_loc, init_scale).to_event(1))

        self.noise_level = float(core.noise_level)

        # ✅ positive parameter
        self.sig1 = PyroParam(
            torch.tensor(float(core.sig1_init), dtype=torch.float32),
            constraint=constraints.positive,
        )

    def _sigma(self, f):
        sig0_t = torch.as_tensor(self.noise_level, device=f.device, dtype=f.dtype)
        return torch.sqrt(sig0_t**2 + self.sig1**2)

    def forward(self, x):
        return self.core(x, self.par)


class PyroAnalyticalRegressor(PyroModule):
    def __init__(self, mean_module):
        super().__init__()
        self.mean_module = mean_module

    def _ensure_f_shape(self, f, X):
        if f.ndim == X.ndim - 1:  # (b,q) or (N,)
            f = f.unsqueeze(-1)  # (b,q,1) or (N,1)
        return f

    def model(self, X, Y=None):
        pyro.module("mean_module", self.mean_module)  # ✅ THIS LINE IS REQUIRED
        f = self._ensure_f_shape(self.mean_module(X), X)
        sigma = self.mean_module._sigma(f)  # ✅ pass f

        if Y is not None:
            Y = torch.as_tensor(Y, device=f.device, dtype=f.dtype)
            if Y.ndim == f.ndim - 1:
                Y = Y.unsqueeze(-1)

        data_n = f.shape[-2] if f.ndim >= 3 else f.shape[0]
        with pyro.plate("data", data_n):
            pyro.sample("obs", dist.Normal(f, sigma).to_event(1), obs=Y)

        return f

    def predictive(
        self,
        X,
        guide,
        num_samples: int,
        noisy: torch.tensor = None,
        return_sigma: bool = False,
    ):
        def _predict_model(X_):
            f = self._ensure_f_shape(self.mean_module(X_), X_)
            sigma = self.mean_module._sigma(f)  # ✅ use mean_module

            pyro.sample("f", dist.Delta(f).to_event(2 if f.ndim >= 3 else 1))
            if return_sigma:
                pyro.sample("sigma", dist.Delta(sigma))
            if noisy is not None:
                pyro.sample(
                    "y",
                    dist.Normal(f, torch.sqrt(sigma**2 + noisy**2)).to_event(
                        2 if f.ndim >= 3 else 1
                    ),
                )
            return f

        sites = ["f"]
        sites.append("y")
        if return_sigma:
            sites.append("sigma")

        return Predictive(
            _predict_model,
            guide=guide,
            num_samples=int(num_samples),
            return_sites=tuple(sites),
        )(X)


def fit_svi(
    regressor: PyroAnalyticalRegressor,
    X_train,
    Y_train,
    num_steps=2000,
    lr=0.02,
    seed=0,
    show_progress=False,
):
    pyro.set_rng_seed(seed)
    pyro.clear_param_store()

    guide = AutoMultivariateNormal(regressor.model)
    optim = Adam({"lr": lr})
    svi = SVI(regressor.model, guide, optim, loss=Trace_ELBO())

    losses = []
    for t in range(num_steps):
        loss = svi.step(X_train, Y_train)
        losses.append(loss)
        if show_progress and (t % 200 == 0):
            print(f"step {t:5d}  loss {loss:.3f}")
    return guide, torch.tensor(losses)


class PyroVIGaussianWrapper(Model):
    def __init__(
        self, regressor, guide, num_outputs=1, num_mc_samples=512, jitter=1e-4
    ):
        super().__init__()
        self.regressor = regressor
        self.guide = guide
        self._m = int(num_outputs)
        self.S = int(num_mc_samples)
        self.jitter = float(jitter)

    @property
    def num_outputs(self):
        return self._m

    def posterior1(self, X, output_indices=None, observation_noise=False, **kwargs):
        # If observation_noise=True, moment-match y; else moment-match f
        out = self.regressor.predictive(
            X,
            guide=self.guide,
            num_samples=self.S,
            noisy=observation_noise,
            return_sigma=False,
        )

        key = "y" if observation_noise else "f"
        samples = out[key]  # (S, b, q, m) expected for BO

        # normalize: if somehow (S, N, m), treat as (S, 1, N, m)
        if samples.ndim == 3:
            samples = samples.unsqueeze(1)

        if output_indices is not None:
            samples = samples[..., output_indices]

        mu = samples.mean(dim=0)  # (b,q,m)
        centered = samples - mu.unsqueeze(0)  # (S,b,q,m)

        cov_bmqq = torch.einsum("sbqm,sbkm->bmqk", centered, centered) / float(self.S)
        eye = torch.eye(cov_bmqq.size(-1), device=cov_bmqq.device, dtype=cov_bmqq.dtype)
        cov_bmqq = cov_bmqq + self.jitter * eye  # (b,m,q,q)

        mean_mq = mu.transpose(-1, -2).contiguous()  # (b,m,q)

        bmvn = gpytorch.distributions.MultivariateNormal(mean_mq, cov_bmqq)
        mtmvn = gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            bmvn, task_dim=-1
        )
        return GPyTorchPosterior(mtmvn)

    def posterior(self, X, output_indices=None, observation_noise=None, **kwargs):
        out = self.regressor.predictive(
            X,
            guide=self.guide,
            num_samples=self.S,
            noisy=observation_noise,
            return_sigma=False,
        )

        key = "y" if observation_noise is not None else "f"
        samples = out[key]

        if output_indices is not None:
            samples = samples[..., output_indices]

        if samples.ndim == 3:
            # samples: (S, q, m)  -> no batch
            mu = samples.mean(dim=0)  # (q, m)
            centered = samples - mu.unsqueeze(0)  # (S, q, m)

            cov_mqq = torch.einsum("sqm,skm->mqk", centered, centered) / float(
                self.S
            )  # (m, q, q)
            eye = torch.eye(
                cov_mqq.size(-1), device=cov_mqq.device, dtype=cov_mqq.dtype
            )
            cov_mqq = cov_mqq + self.jitter * eye

            mean_mq = mu.transpose(-1, -2).contiguous()  # (m, q)

            try:
                mvn = gpytorch.distributions.MultivariateNormal(
                    mean_mq, cov_mqq
                )  # batch=(m), event=(q)
            except RuntimeError as e:
                msg = str(e)

                # This is only executed on failures, so overhead is negligible across thousands of calls.
                res = dump_failed_mvn(
                    mean_mq=mean_mq,
                    cov_mqq=cov_mqq,
                    prefix="pyro_post",
                    outdir="mvn_fail_dumps",
                    err_msg=msg,
                )

                if res is not None:
                    m_bad, k, path = res
                    # Keep the print short so logs don't explode:
                    print(
                        f"[MVN FAIL] batch/task index m_bad={m_bad}, leading minor k={k}, dump={path}"
                    )

                raise  # re-raise original error

            mtmvn = gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
                mvn, task_dim=-1  # only batch dim is tasks here
            )
            return GPyTorchPosterior(mtmvn)

        elif samples.ndim == 4:

            # samples: (S, b, q, m)
            mu = samples.mean(dim=0)  # (b, q, m)
            centered = samples - mu.unsqueeze(0)  # (S, b, q, m)

            cov_bmqq = torch.einsum("sbqm,sbkm->bmqk", centered, centered) / float(
                self.S
            )  # (b, m, q, q)
            eye = torch.eye(
                cov_bmqq.size(-1), device=cov_bmqq.device, dtype=cov_bmqq.dtype
            )
            cov_bmqq = cov_bmqq + self.jitter * eye

            mean_bmq = mu.transpose(-1, -2).contiguous()  # (b, m, q)
            mvn = gpytorch.distributions.MultivariateNormal(
                mean_bmq, cov_bmqq
            )  # batch=(b,m), event=(q)
            mtmvn = gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
                mvn, task_dim=-1
            )
            return GPyTorchPosterior(mtmvn)

        else:
            raise RuntimeError(f"Unexpected samples shape: {tuple(samples.shape)}")


_MINOR_RE = re.compile(r"leading minor of order (\d+)")


def dump_failed_mvn(
    mean_mq, cov_mqq, prefix="mvn_fail", outdir="mvn_fail_dumps", err_msg=None
):
    """
    mean_mq: (m, q)
    cov_mqq: (m, q, q)

    Fast path:
      - run cholesky_ex ONCE on the batch
      - if all PD: return None (no I/O)
    Failure path:
      - find first failing batch index m_bad from info
      - parse leading minor order k from err_msg if provided
      - dump mean_mq[m_bad] and cov_mqq[m_bad] (+ optional kxk block) to a .pt file
      - return (m_bad, k, dump_path)
    """
    # cholesky_ex returns info>0 where PD fails
    # NOTE: this does the same O(m*q^3) as the MVN would do anyway;
    # on success it adds essentially one extra cholesky call, so you should
    # only call this inside an exception handler (recommended below).
    L, info = torch.linalg.cholesky_ex(cov_mqq)

    if torch.all(info == 0):
        return None

    # first failing batch element
    m_bad = int((info != 0).nonzero(as_tuple=False)[0].item())

    k = None
    if err_msg:
        m = _MINOR_RE.search(err_msg)
        if m:
            k = int(m.group(1))

    os.makedirs(outdir, exist_ok=True)
    dump = {
        "m_bad": m_bad,
        "k_minor": k,
        "mean_mq_bad": mean_mq[m_bad].detach().cpu(),
        "cov_mqq_bad": cov_mqq[m_bad].detach().cpu(),
        "cov_diag_bad": torch.diagonal(cov_mqq[m_bad]).detach().cpu(),
        "info": info.detach().cpu(),
    }
    if k is not None:
        dump["cov_minor_kxk"] = cov_mqq[m_bad, :k, :k].detach().cpu()

    dump_path = os.path.join(
        outdir, f"{prefix}_m{m_bad}_k{k if k is not None else 'NA'}.pt"
    )
    torch.save(dump, dump_path)
    return m_bad, k, dump_path
