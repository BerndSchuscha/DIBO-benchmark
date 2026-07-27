from botorch.test_functions.multi_objective import BraninCurrin
from pibob.problems.Knowledge import Knowledge
from pibob.problems.ExclusiveBank import ExclusiveBank

from pibob.problems.Ms_NiTi.kernels import kernel_simplify_additive
from pibob.problems.Ms_NiTi.cores import core1, core2, core3
from pibob.problems.Ms_NiTi.Ms_NiTi_Problem import Ms_NiTi_Problem
from pibob.problems.Ms_NiTi.transforms import Transform1, Transform2


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
        return [5.0, 0.001]
    if level == "medium":
        return [10, 0.001]
    if level == "high":
        return [15.0, 0.001]
    raise ValueError(level)


def make_problem():
    obj = Ms_NiTi_Problem()
    obj.noise_profile = noise_profile

    descriptor_bank = ExclusiveBank(
        [
            [Transform1()],
            [Transform2()],
        ],
        active=0,
    )

    kernel_bank = ExclusiveBank(
        [
            [kernel_simplify_additive()],
        ],
        active=0,
    )

    eq_bank = ExclusiveBank(
        [
            [core1()],
            [core2()],
            [core3()],
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
