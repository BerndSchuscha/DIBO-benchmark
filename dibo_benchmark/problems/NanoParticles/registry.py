from pibob.problems.Knowledge import Knowledge
from pibob.problems.ExclusiveBank import ExclusiveBank

from pibob.problems.NanoParticles.kernels import kernel_simplify_poly2
from pibob.problems.NanoParticles.cores import (
    NanoSizeLinearCore,
    NanoPolyLinearCore,
    NanoSizeCore8,
    NanoPolyCore8,
)
from pibob.problems.NanoParticles.NanoParticles_Problem import NanoParticles_Problem
from pibob.problems.NanoParticles.transforms import NanoDescriptorTransform_2


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
        return [0.05, 0.05]
    if level == "medium":
        return [0.25, 0.25]
    if level == "high":
        return [1, 1]
    raise ValueError(level)


def make_problem():
    obj = NanoParticles_Problem()
    obj.noise_profile = noise_profile

    descriptor_bank = ExclusiveBank(
        [
            [NanoDescriptorTransform_2(),NanoDescriptorTransform_2()],
        ],
        active=0,
    )

    kernel_bank = ExclusiveBank(
        [
            [kernel_simplify_poly2(),kernel_simplify_poly2()],
        ],
        active=0,
    )

    eq_bank = ExclusiveBank(
        [
            [NanoSizeLinearCore(),NanoPolyLinearCore()],
            [NanoSizeCore8(),NanoPolyCore8()],
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
