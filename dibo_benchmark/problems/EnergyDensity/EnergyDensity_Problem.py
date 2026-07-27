import torch
from torch import Tensor

from pibob.problems.base_problem import base_problem


class EnergyDensity_Problem(base_problem):
    r""" """

    dim: int = 5
    num_objectives: int = 2
    num_constraints: int = 0
    input_names: list = ["den", "interval", "hff", "hfr", "vsr"]
    objective_names: list = ["W_recover_cor", "eff"]
    obj_ismodeled = [True, True]
    constraint_names: list = []
    is_minimization: bool = True  # True => minimize all objectives by default
    _bounds: list[tuple[float, float]] = [
        (0, 1),
        (0, 1),
        (0, 1),
        (0, 1),
        (0, 1),
    ]  # length = dim
    _con_bounds: list[tuple[float, float]] = []  # length = num_constraints
    _ref_point: [list[float]] = (12, 0.8)  # length = num_objectives
    _max_hv: float | None = 0.3092941129872422

    # Optional indices (default empty)
    continuous_inds: list[int] = list(range(dim))
    discrete_inds: list[int] = []
    categorical_inds: list[int] = []

    _check_grad_at_opt: bool = True

    # comp , time, temperature

    testpoint = [0.1, 0.1, 0.1, 0.1, 0.1]

    def _eval_model(self, X: Tensor):

        return 0, 1
