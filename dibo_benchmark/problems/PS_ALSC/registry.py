from pibob.problems.Knowledge import Knowledge
from pibob.problems.ExclusiveBank import ExclusiveBank

from pibob.problems.PS_ALSC.cores import ALSCCore, ALSCCore1
from pibob.problems.PS_ALSC.PS_ALSC_Problem import PS_ALSC_Problem
from pibob.problems.PS_ALSC.transforms import ALSCTransform, ALSCTransform1
from pibob.problems.PS_ALSC.kernels import kernel_2HWP_additive, kernel_1HWP


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
        return [10, 0.0005]
    if level == "medium":
        return [20, 0.001]
    if level == "high":
        return [30, 0.002]
    raise ValueError(level)


def make_problem():
    obj = PS_ALSC_Problem()
    obj.noise_profile = noise_profile

    descriptor_bank = ExclusiveBank(
        [[ALSCTransform(), ALSCTransform()]]
        + [
            [ALSCTransform1(f"M_sigma_pc_{i}"), ALSCTransform1(f"M_cfrac_{i}")]
            for i in range(10)
        ],
        active=0,
    )

    kernel_bank = ExclusiveBank(
        [
            [kernel_1HWP(), kernel_1HWP()],
            [kernel_2HWP_additive(), kernel_2HWP_additive()],
        ],
        active=0,
    )

    eq_bank = ExclusiveBank(
        [[ALSCCore1(), ALSCCore1()]]
        + [[ALSCCore(f"M_sigma_pc_{i}"), ALSCCore(f"M_cfrac_{i}")] for i in range(10)],
        active=0,
    )

    return attach_knowledge(
        obj,
        descriptor_bank=descriptor_bank,
        kernel_bank=kernel_bank,
        eq_bank=eq_bank,
        active_kind="custom_mean",
    )
