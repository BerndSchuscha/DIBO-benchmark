import math
import pytest
import torch

# --- your package imports (adjust paths to your project) ---
from pibob.models.utils import FitCfg
from pibob.models.old.features import PhiIdentity, PhiDescriptor, PhiCustom
from pibob.models.old.kernels import rbf_kernel_factory, custom_additive_factory
from pibob.models.old.means import PhysMean
from pibob.models.variants import GPVariant, GPVariantCfg, PyroPlusGP
from pibob.models.old.pyro_fit import PyroPhysFitter  # delta/MAP version

# If you put the Auto guides fitter in another module, import it here:
# from gp_pyro_stack.pyro_fit_auto import PyroPhysFitterAutoNormal, PyroPhysFitterMVN

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro.infer.autoguide import AutoNormal, AutoMultivariateNormal


# ----------------------------
# Helpers: synthetic dataset
# ----------------------------
def make_synth_data(N=80, seed=0, noise=0.03):
    torch.manual_seed(seed)
    # X_raw = [t, T, x]
    t = torch.rand(N) * 10.0  # 0..10
    T = 300.0 + torch.rand(N) * 200.0  # 300..500
    x = torch.randn(N)  # feature

    X = torch.stack([t, T, x], dim=1)

    # ground truth physics params
    a_true = torch.tensor(1.4, dtype=torch.float64)
    b_true = torch.tensor(1200.0, dtype=torch.float64)
    c_true = torch.tensor(-0.6, dtype=torch.float64)

    mu = a_true * torch.sqrt(torch.clamp(t, min=0.0)) * torch.exp(-b_true / T) + c_true * x

    # add structured non-physics residual + noise
    resid = 0.15 * torch.sin(0.7 * t) + 0.07 * torch.cos(0.03 * T)
    y = mu + resid + noise * torch.randn(N, dtype=torch.float64)

    return X.to(torch.float64), y.to(torch.float64)


def assert_predict_ok(pred_mean, pred_std, N):
    assert pred_mean.shape == (N,)
    assert pred_std.shape == (N,)
    assert torch.isfinite(pred_mean).all()
    assert torch.isfinite(pred_std).all()
    # std should be non-negative
    assert (pred_std >= 0).all()


# ----------------------------
# Pyro fitters for guides
# ----------------------------
class PyroFitterAutoNormal:
    def __init__(self, steps=500, lr=1e-2, warm_start=True):
        self.steps = steps
        self.lr = lr
        self.warm_start = warm_start
        self._guide = None

    def fit(self, X_raw, y):
        if not self.warm_start:
            pyro.clear_param_store()
            self._guide = None

        def model(X, y=None):
            a = pyro.sample("a", dist.LogNormal(0.0, 0.5))
            b = pyro.sample("b", dist.LogNormal(0.0, 0.5))
            c = pyro.sample("c", dist.Normal(0.0, 1.0))
            sigma = pyro.sample("sigma", dist.LogNormal(-2.0, 0.5))
            mu = (
                a * torch.sqrt(torch.clamp(X[:, 0], min=0.0)) * torch.exp(-b / X[:, 1])
                + c * X[:, 2]
            )
            with pyro.plate("data", X.shape[0]):
                pyro.sample("obs", dist.Normal(mu, sigma), obs=y)

        if self._guide is None:
            self._guide = AutoNormal(model)

        svi = SVI(model, self._guide, Adam({"lr": self.lr}), loss=Trace_ELBO())
        for _ in range(self.steps):
            svi.step(X_raw, y)

        # deterministic mean function uses posterior median as point estimate
        est = self._guide.median(X_raw)
        a_hat, b_hat, c_hat = est["a"].detach(), est["b"].detach(), est["c"].detach()

        # return a MeanModel-like callable (minimal)
        class MeanFn(torch.nn.Module):
            def forward(self, X):
                t, T, x = X[:, 0], X[:, 1], X[:, 2]
                return a_hat * torch.sqrt(torch.clamp(t, min=0.0)) * torch.exp(-b_hat / T) + c_hat * x

        return MeanFn()


class PyroFitterAutoMVN(PyroFitterAutoNormal):
    def fit(self, X_raw, y):
        if not self.warm_start:
            pyro.clear_param_store()
            self._guide = None

        def model(X, y=None):
            a = pyro.sample("a", dist.LogNormal(0.0, 0.5))
            b = pyro.sample("b", dist.LogNormal(0.0, 0.5))
            c = pyro.sample("c", dist.Normal(0.0, 1.0))
            sigma = pyro.sample("sigma", dist.LogNormal(-2.0, 0.5))
            mu = (
                a * torch.sqrt(torch.clamp(X[:, 0], min=0.0)) * torch.exp(-b / X[:, 1])
                + c * X[:, 2]
            )
            with pyro.plate("data", X.shape[0]):
                pyro.sample("obs", dist.Normal(mu, sigma), obs=y)

        if self._guide is None:
            self._guide = AutoMultivariateNormal(model)

        svi = SVI(model, self._guide, Adam({"lr": self.lr}), loss=Trace_ELBO())
        for _ in range(self.steps):
            svi.step(X_raw, y)

        est = self._guide.median(X_raw)
        a_hat, b_hat, c_hat = est["a"].detach(), est["b"].detach(), est["c"].detach()

        class MeanFn(torch.nn.Module):
            def forward(self, X):
                t, T, x = X[:, 0], X[:, 1], X[:, 2]
                return a_hat * torch.sqrt(torch.clamp(t, min=0.0)) * torch.exp(-b_hat / T) + c_hat * x

        return MeanFn()


# ----------------------------
# Tests: GP variants
# ----------------------------
def test_gp_normal():
    X, y = make_synth_data(N=60, seed=1)

    case = GPVariant(GPVariantCfg(
        phi=PhiIdentity(),
        kernel_factory=rbf_kernel_factory,
        kernel_meta={},
        mean_mode="zero",
        fit_cfg=FitCfg(lr=0.15, steps=120),
    ))

    case.fit(X, y)
    pred = case.predict(X)
    assert_predict_ok(pred.mean, pred.std, X.shape[0])


def test_gp_transformed():
    X, y = make_synth_data(N=60, seed=2)

    case = GPVariant(GPVariantCfg(
        phi=PhiDescriptor(),
        kernel_factory=rbf_kernel_factory,
        kernel_meta={},
        mean_mode="zero",
        fit_cfg=FitCfg(lr=0.15, steps=120),
    ))

    case.fit(X, y)
    pred = case.predict(X)
    assert_predict_ok(pred.mean, pred.std, X.shape[0])


def test_gp_imprinting_custom_mean():
    X, y = make_synth_data(N=60, seed=3)

    case = GPVariant(GPVariantCfg(
        phi=PhiDescriptor(),
        kernel_factory=rbf_kernel_factory,
        kernel_meta={},
        mean_mode="imprint",
        mean_model=PhysMean(),  # torch-learned mean, trained via GP optimization
        fit_cfg=FitCfg(lr=0.15, steps=160),
    ))

    case.fit(X, y)
    pred = case.predict(X)
    assert_predict_ok(pred.mean, pred.std, X.shape[0])


def test_gp_custom_kernel():
    X, y = make_synth_data(N=60, seed=4)

    # For your PhiCustom: output dim = dim_x + 2.
    # Here X_raw has x-dim=1 -> PhiCustom(1) => [x, sqrt(t), 1/T]
    phi = PhiCustom(dim_x=1)

    # cut=2 means k1 on [x, sqrt(t)], k2 on [1/T]
    case = GPVariant(GPVariantCfg(
        phi=phi,
        kernel_factory=custom_additive_factory,
        kernel_meta={"cut": 2},
        mean_mode="zero",
        fit_cfg=FitCfg(lr=0.15, steps=140),
    ))

    case.fit(X, y)
    pred = case.predict(X)
    assert_predict_ok(pred.mean, pred.std, X.shape[0])


# ----------------------------
# Tests: Pyro-only variants
# ----------------------------
def test_pyro_only_delta_map():
    X, y = make_synth_data(N=60, seed=5)
    fitter = PyroPhysFitter(steps=400, lr=2e-2)  # your Delta/MAP fitter
    mean_model = fitter.fit(X, y)
    mu = mean_model(X)
    assert mu.shape == (X.shape[0],)
    assert torch.isfinite(mu).all()


def test_pyro_only_autonormal():
    X, y = make_synth_data(N=60, seed=6)
    fitter = PyroFitterAutoNormal(steps=400, lr=2e-2, warm_start=False)
    mean_model = fitter.fit(X, y)
    mu = mean_model(X)
    assert mu.shape == (X.shape[0],)
    assert torch.isfinite(mu).all()


def test_pyro_only_automvn():
    X, y = make_synth_data(N=60, seed=7)
    fitter = PyroFitterAutoMVN(steps=500, lr=2e-2, warm_start=False)
    mean_model = fitter.fit(X, y)
    mu = mean_model(X)
    assert mu.shape == (X.shape[0],)
    assert torch.isfinite(mu).all()


# ----------------------------
# Tests: Hybrid Pyro + GP
# ----------------------------
@pytest.mark.parametrize("pyro_fitter", [
    PyroPhysFitter(steps=300, lr=2e-2),                  # Delta/MAP
    PyroFitterAutoNormal(steps=300, lr=2e-2, warm_start=False),
    PyroFitterAutoMVN(steps=350, lr=2e-2, warm_start=False),
])
def test_hybrid_pyro_plus_gp(pyro_fitter):
    X, y = make_synth_data(N=70, seed=10)

    gp_cfg = GPVariantCfg(
        phi=PhiDescriptor(),
        kernel_factory=rbf_kernel_factory,
        kernel_meta={},
        mean_mode="imprint",     # overridden internally, but fine
        mean_model=None,         # set by hybrid
        fit_cfg=FitCfg(lr=0.15, steps=120),
    )

    hybrid = PyroPlusGP(pyro_fitter=pyro_fitter, gp_cfg=gp_cfg, outer_iters=2)
    hybrid.fit(X, y)
    pred = hybrid.predict(X)
    assert_predict_ok(pred.mean, pred.std, X.shape[0])
