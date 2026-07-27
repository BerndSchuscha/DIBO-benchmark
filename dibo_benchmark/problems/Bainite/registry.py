from pibob.problems.Knowledge import Knowledge
from pibob.problems.ExclusiveBank import ExclusiveBank

from pibob.problems.Bainite.cores import Bainitecore, core_linear7, core_linear1D
from pibob.problems.Bainite.Bainite_Problem import Bainite_Problem
from pibob.problems.Bainite.transforms import (
    AddDescriptorsYs,
    AddDescriptorsYs1,
    AddDescriptorsUEL,
    AddDescriptorsUEL1,
)
from pibob.problems.Bainite.kernels import kernel_poly2


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
        return [20, 0.1]
    if level == "medium":
        return [50, 1]
    if level == "high":
        return [100, 2]
    raise ValueError(level)


def make_problem():
    obj = Bainite_Problem()
    obj.noise_profile = noise_profile

    descriptor_bank = ExclusiveBank(
        [
            [
                AddDescriptorsYs1(descriptor_name=f"M_Ys_{i}"),
                AddDescriptorsUEL1(descriptor_name=f"M_UEL_{i}"),
            ]
            for i in range(10)
        ]
        + [[AddDescriptorsYs(), AddDescriptorsUEL()]],
        active=0,
    )

    kernel_bank = ExclusiveBank(
        [
            [kernel_poly2(), kernel_poly2()],
        ],
        active=0,
    )

    eq_bank = ExclusiveBank(
        [[Bainitecore(f"M_Ys_{i}"), Bainitecore(f"M_UEL_{i}")] for i in range(10)]
        + [[core_linear7(), core_linear7()], [core_linear1D(), core_linear1D()]],
        active=0,
    )

    return attach_knowledge(
        obj,
        descriptor_bank=descriptor_bank,
        kernel_bank=kernel_bank,
        eq_bank=eq_bank,
        active_kind="custom_mean",
    )
