from pibob.problems.Knowledge import Knowledge
from pibob.problems.ExclusiveBank import ExclusiveBank

from pibob.problems.EnergyDensity.cores import EngnergyCore, core_linear
from pibob.problems.EnergyDensity.EnergyDensity_Problem import EnergyDensity_Problem
from pibob.problems.EnergyDensity.transforms import EnergyTransform
from pibob.problems.EnergyDensity.kernels import kernel_poly2


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
        return [0.1, 0.01]
    if level == "medium":
        return [0.9, 0.5]
    if level == "high":
        return [2.7, 0.1]
    raise ValueError(level)


def make_problem():
    obj = EnergyDensity_Problem()
    obj.noise_profile = noise_profile

    descriptor_bank = ExclusiveBank(
        [
            [EnergyTransform(f"M_W_recover_cor_{i}"), EnergyTransform(f"M_eff_{i}")]
            for i in range(10)
        ],
        active=0,
    )

    kernel_bank = ExclusiveBank(
        [
            [kernel_poly2(), kernel_poly2()],
        ],
        active=0,
    )

    eq_bank = ExclusiveBank(
        [
            [EngnergyCore(f"M_W_recover_cor_{i}"), EngnergyCore(f"M_eff_{i}")]
            for i in range(10)
        ]
        + [[core_linear(), core_linear()]],
        active=0,
    )

    return attach_knowledge(
        obj,
        descriptor_bank=descriptor_bank,
        kernel_bank=kernel_bank,
        eq_bank=eq_bank,
        active_kind="custom_mean",
    )
