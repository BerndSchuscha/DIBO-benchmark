import torch
from torch import Tensor

from pibob.problems.base_problem import base_problem


class Structures_Problem(base_problem):
    r""" """

    dim: int = 13
    num_objectives: int = 2
    num_constraints: int = 0
    input_names: list = [
        "TargetHeight",
        "WallThickness",
        "AveragePerimeter",
        "x1",
        "x2",
        "x3",
        "x4",
        "x5",
        "x6",
        "x7",
        "x8",
        "Modulus",
        "PlateauStrength",
    ]
    objective_names: list = ["CriticalStress", "CriticalEfficiency"]
    obj_ismodeled = [True, True]
    constraint_names: list = []
    is_minimization: bool = True  # True => minimize all objectives by default
    _bounds: list[tuple[float, float]] = [
        (0, 4.1455e01),
        (0, 1.0000e00),
        (0, 4.9653e02),
        (0, 3.8903e02),
        (0, 1.2000e00),
        (0, 1.1376e00),
        (0, 9.9682e-01),
        (0, 9.9682e-01),
        (0, 6.2832e00),
        (0, 3.1416e00),
        (0, 1.5789e-01),
        (0, 1.5470e03),
        (0, 6.8881e01),
    ]  # length = dim
    _con_bounds: list[tuple[float, float]] = []  # length = num_constraints
    _ref_point: [list[float]] = (9.1316e-02, 5.0400e-05)  # length = num_objectives
    _max_hv: float | None = 100000

    # Optional indices (default empty)
    continuous_inds: list[int] = list(range(dim))
    discrete_inds: list[int] = []
    categorical_inds: list[int] = []

    _check_grad_at_opt: bool = True

    # comp , time, temperature

    testpoint = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    def _eval_model(self, X: Tensor):
        return 0, 0
