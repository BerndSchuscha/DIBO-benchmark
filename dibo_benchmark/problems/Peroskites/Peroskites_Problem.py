import torch
from torch import Tensor

from pibob.problems.base_problem import base_problem


class Peroskites_Problem(base_problem):
    r""" """

    dim: int = 2
    num_objectives: int = 2
    num_constraints: int = 0
    input_names: list = ["Sr / La", "Co / Fe"]
    objective_names: list = ["e_above_hull", "Vacancy Formation Energy"]
    obj_ismodeled = [True, True]
    constraint_names: list = []
    is_minimization: bool = True  # True => minimize all objectives by default
    _bounds: list[tuple[float, float]] = [
        (0, 1),
        (0, 1),
    ]  # length = dim
    _con_bounds: list[tuple[float, float]] = []  # length = num_constraints
    _ref_point: [list[float]] = (-0.2, -0.5)  # length = num_objectives
    _max_hv: float | None = 100000

    # Optional indices (default empty)
    continuous_inds: list[int] = list(range(dim))
    discrete_inds: list[int] = []
    categorical_inds: list[int] = []

    _check_grad_at_opt: bool = True

    # comp , time, temperature

    testpoint = [0.1, 0.1]

    def _eval_model(self, X: Tensor):
        return 0, 0
