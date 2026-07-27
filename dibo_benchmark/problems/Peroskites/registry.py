from pibob.problems.Knowledge import Knowledge
from pibob.problems.ExclusiveBank import ExclusiveBank

from pibob.problems.Peroskites.kernels import kernel_polyscale_additive
from pibob.problems.Peroskites.cores import (
    DoublePerovskiteOVacLinearCore,
    DoublePerovskiteHullLinearCore,
)
from pibob.problems.Peroskites.Peroskites_Problem import Peroskites_Problem
from pibob.problems.Peroskites.transforms import (
    DPTransformVac_nB,
    DPTransformVac_t,
    DPTransformVac_sigmaB2,
    DPTransformHull_t,
    DPTransformHull_Sconf,
    DPTransformHull_sigmaB2,
)


def attach_knowledge(
    obj, *, descriptor_bank, kernel_bank, eq_bank, active_kind="custom_mean"
):
    obj.Knowledge = Knowledge(
        descriptor_bank=descriptor_bank,
        kernel_bank=kernel_bank,
        eq_bank=eq_bank,
        active_kind=active_kind,
    )
    return obj


def noise_profile(level: str):
    # replace with your actual numbers
    if level == "low":
        return [0.001, 0.001]
    if level == "medium":
        return [0.06, 1]
    if level == "high":
        return [0.15, 3]
    raise ValueError(level)


def make_problem():
    obj = Peroskites_Problem()
    obj.noise_profile = noise_profile

    descriptor_bank = ExclusiveBank(
        [
            [DPTransformHull_Sconf(), DPTransformVac_nB()],
            [DPTransformHull_sigmaB2(), DPTransformVac_sigmaB2()],
            [DPTransformHull_t(), DPTransformVac_t()],
        ],
        active=0,
    )

    kernel_bank = ExclusiveBank(
        [
            [kernel_polyscale_additive(), kernel_polyscale_additive()],
        ],
        active=0,
    )

    eq_bank = ExclusiveBank(
        [
            [DoublePerovskiteHullLinearCore(), DoublePerovskiteOVacLinearCore()],
        ],
        active=0,
    )

    return attach_knowledge(
        obj,
        descriptor_bank=descriptor_bank,
        kernel_bank=kernel_bank,
        eq_bank=eq_bank,
        active_kind="custom_mean",
    )
