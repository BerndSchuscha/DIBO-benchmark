from botorch.test_functions.multi_objective import BraninCurrin
from pibob.problems.Knowledge import Knowledge
from pibob.problems.ExclusiveBank import ExclusiveBank
from pibob.problems.BraninCurrin.transforms import (
    BraninDescriptorTransform_1,
    BraninDescriptorTransform_2,
    BraninDescriptorTransform_3,
    CurrinDescriptorTransform_1,
    CurrinDescriptorTransform_2,
)
from pibob.problems.BraninCurrin.kernels import kernel_simplify_additive
from pibob.problems.BraninCurrin.cores import (
    BraninVar1,
    CurrinVar1,
    BraninVar2,
    BraninVar3,
    BraninVar4,
    CurrinVar2,
    CurrinVar3,
    CurrinVar4,
    core_linear2,
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
        return [2.0, 0.1]
    if level == "medium":
        return [15.19, 0.63]
    if level == "high":
        return [40.0, 2.0]
    raise ValueError(level)


def make_problem():
    obj = BraninCurrin(negate=True)
    obj.noise_profile = noise_profile
    obj.input_names = ["x1", "x2"]
    obj.objective_names = ["branin", "currin"]
    obj.constraint_names = []
    obj.obj_ismodeled = [True, True]

    descriptor_bank = ExclusiveBank(
        [
            [BraninDescriptorTransform_1(), CurrinDescriptorTransform_1()],
            [BraninDescriptorTransform_2(), CurrinDescriptorTransform_2()],
            [BraninDescriptorTransform_3(), CurrinDescriptorTransform_2()],
        ],
        active=0,
    )

    kernel_bank = ExclusiveBank(
        [
            [kernel_simplify_additive(), kernel_simplify_additive()],
        ],
        active=0,
    )

    eq_bank = ExclusiveBank(
        [
            [BraninVar1(), CurrinVar1()],
            [BraninVar2(), CurrinVar2()],
            [BraninVar3(), CurrinVar3()],
            [BraninVar4(), CurrinVar4()],
            [core_linear2(), core_linear2()],
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
