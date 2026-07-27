import torch
from torch import Tensor

from pibob.problems.base_problem import base_problem


class Magnetic_Problem(base_problem):
    r""" """

    dim: int = 3
    num_objectives: int = 2
    num_constraints: int = 0
    input_names: list = ["Fe", "Co", "Ni"]
    objective_names: list = ["Kerr", "Coercivity"]
    obj_ismodeled = [True, True]
    constraint_names: list = []
    is_minimization: bool = True  # True => minimize all objectives by default
    _bounds: list[tuple[float, float]] = [
        (0, 100),
        (0, 100),
        (0, 100),
    ]  # length = dim
    _con_bounds: list[tuple[float, float]] = []  # length = num_constraints
    _ref_point: [list[float]] = (0.2, -4)  # length = num_objectives
    _max_hv: float | None = 1000

    # Optional indices (default empty)
    continuous_inds: list[int] = list(range(dim))
    discrete_inds: list[int] = []
    categorical_inds: list[int] = []

    _check_grad_at_opt: bool = True

    # comp , time, temperature

    testpoint = [0.1, 0.1, 0.1]

    def _eval_model(self, X: Tensor):

        return 0, 1
