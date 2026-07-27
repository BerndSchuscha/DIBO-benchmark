from pibob.problems.Knowledge import Knowledge
from pibob.problems.ExclusiveBank import ExclusiveBank

from pibob.problems.Magnetic.cores import (
    LinearFeCoNi4,
    KerrFeCoNiAsym,
    CoercivityFeCoNiAsym,
)
from pibob.problems.Magnetic.Magnetic_Problem import Magnetic_Problem
from pibob.problems.Magnetic.transforms import FeCoNiMinimalPhysicsTransform
from pibob.problems.Magnetic.kernels import kernel_additive


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
        return [0.01, 0.1]
    if level == "medium":
        return [0.025, 0.25]
    if level == "high":
        return [0.05, 0.5]
    raise ValueError(level)


def make_problem():
    obj = Magnetic_Problem()
    obj.noise_profile = noise_profile

    descriptor_bank = ExclusiveBank(
        [
            [
                FeCoNiMinimalPhysicsTransform(problem=Magnetic_Problem()),
                FeCoNiMinimalPhysicsTransform(problem=Magnetic_Problem()),
            ]
        ],
        active=0,
    )

    kernel_bank = ExclusiveBank(
        [
            [kernel_additive(), kernel_additive()],
        ],
        active=0,
    )

    eq_bank = ExclusiveBank(
        [
            [LinearFeCoNi4(), LinearFeCoNi4()],
            [
                KerrFeCoNiAsym(problem=Magnetic_Problem()),
                CoercivityFeCoNiAsym(problem=Magnetic_Problem()),
            ],
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
