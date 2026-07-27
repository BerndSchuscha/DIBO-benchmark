import torch
from torch import Tensor

from pibob.problems.base_problem import base_problem


class Bainite_Problem(base_problem):
    r""" """

    dim: int = 7
    num_objectives: int = 2
    num_constraints: int = 0
    input_names: list = ["C", "Si", "Mn", "Cr", "Al", "V", "T"]
    objective_names: list = ["Ys", "UEL"]
    obj_ismodeled = [True, True]
    constraint_names: list = []
    is_minimization: bool = True  # True => minimize all objectives by default
    _bounds: list[tuple[float, float]] = [
        (0.3, 0.8),
        (1, 4),
        (0, 3),
        (0, 4),
        (0, 2),
        (0, 0.5),
        (250, 425),
    ]  # length = dim
    _con_bounds: list[tuple[float, float]] = []  # length = num_constraints
    _ref_point: [list[float]] = (1000, 2.6)  # length = num_objectives
    _max_hv: float | None = 0.3092941129872422

    # Optional indices (default empty)
    continuous_inds: list[int] = list(range(dim))
    discrete_inds: list[int] = []
    categorical_inds: list[int] = []

    _check_grad_at_opt: bool = True

    # comp , time, temperature

    testpoint = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    def _eval_model(self, X: Tensor):

        return 0, 1
