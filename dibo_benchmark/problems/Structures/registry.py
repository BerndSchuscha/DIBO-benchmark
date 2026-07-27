from pibob.problems.Knowledge import Knowledge
from pibob.problems.ExclusiveBank import ExclusiveBank

from pibob.problems.Structures.kernels import (
    kernel_linear_matern,
    kernel_poly_rbf_product,
)
from pibob.problems.Structures.cores import (
    CriticalStressALS,
    CriticalEfficiencyALS,
    CriticalStressALSShape,
    CriticalEfficiencyALSShape,
    CriticalStressALSAsym,
    CriticalEfficiencyALSAsym,
    core_linear,
)
from pibob.problems.Structures.Structures_Problem import Structures_Problem
from pibob.problems.Structures.transforms import (
    ALSGeometryTransform,
    ALSGeometryTransformCompact,
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
        return [0.5, 0.025]
    if level == "high":
        return [1, 0.05]
    raise ValueError(level)


def make_problem():
    obj = Structures_Problem()
    obj.noise_profile = noise_profile

    descriptor_bank = ExclusiveBank(
        [
            [
                ALSGeometryTransform(problem=Structures_Problem()),
                ALSGeometryTransform(problem=Structures_Problem()),
            ],
            [
                ALSGeometryTransformCompact(problem=Structures_Problem()),
                ALSGeometryTransformCompact(problem=Structures_Problem()),
            ],
        ],
        active=0,
    )

    kernel_bank = ExclusiveBank(
        [
            [kernel_linear_matern(), kernel_linear_matern()],
            [kernel_poly_rbf_product(), kernel_poly_rbf_product()],
        ],
        active=0,
    )

    eq_bank = ExclusiveBank(
        [
            [
                CriticalStressALSAsym(problem=Structures_Problem()),
                CriticalEfficiencyALSAsym(problem=Structures_Problem()),
            ],
            [
                CriticalStressALSShape(problem=Structures_Problem()),
                CriticalEfficiencyALSShape(problem=Structures_Problem()),
            ],
            [
                CriticalStressALS(problem=Structures_Problem()),
                CriticalEfficiencyALS(problem=Structures_Problem()),
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
