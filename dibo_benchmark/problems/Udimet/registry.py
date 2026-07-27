from pibob.problems.Knowledge import Knowledge
from pibob.problems.ExclusiveBank import ExclusiveBank

from pibob.problems.Udimet.kernels import kernel_ys_optimization
from pibob.problems.Udimet.cores import (
    YieldStrengthPowerUdimet,
    YieldStrengthStrengtheningUdimet,
    FgammaPoly3T1,
    core_linear,
)

from pibob.problems.Udimet.Udimet_Problem import Udimet_Problem
from pibob.problems.Udimet.transforms import (
    UdimetMinimalTransform,
    UdimetWidthTransformCompact,
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
        return [5, 0.005]
    if level == "medium":
        return [25, 0.025]
    if level == "high":
        return [50, 0.05]
    raise ValueError(level)


def make_problem():
    obj = Udimet_Problem()
    obj.noise_profile = noise_profile

    descriptor_bank = ExclusiveBank(
        [
            [
                UdimetMinimalTransform(problem=Udimet_Problem()),
                UdimetWidthTransformCompact(problem=Udimet_Problem()),
            ],
            [
                UdimetMinimalTransform(problem=Udimet_Problem()),
                UdimetWidthTransformCompact(problem=Udimet_Problem()),
            ],
        ],
        active=0,
    )

    kernel_bank = ExclusiveBank(
        [[kernel_ys_optimization(), kernel_ys_optimization()]],
        active=0,
    )

    eq_bank = ExclusiveBank(
        [
            [
                YieldStrengthPowerUdimet(problem=Udimet_Problem()),
                FgammaPoly3T1(problem=Udimet_Problem()),
            ],
            [
                YieldStrengthStrengtheningUdimet(problem=Udimet_Problem()),
                FgammaPoly3T1(problem=Udimet_Problem()),
            ],
            [core_linear(), core_linear()],
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
