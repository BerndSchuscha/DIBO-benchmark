import torch
from torch import Tensor

from pibob.problems.base_problem import base_problem


class Udimet_Problem(base_problem):
    r""" """

    dim: int = 9
    num_objectives: int = 2
    num_constraints: int = 0
    input_names: list = [
        "primary_mean_radius",
        "primary_minimal_radius",
        "primary_maximal_radius",
        "primary_rsd",
        "secondary_mean_radius",
        "secondary_minimal_radius",
        "secondary_maximal_radius",
        "secondary_rsd",
        "T1",
    ]
    objective_names: list = ["ys_700", "fgamma"]
    obj_ismodeled = [True, True]
    constraint_names: list = []
    is_minimization: bool = True  # True => minimize all objectives by default
    _bounds: list[tuple[float, float]] = [
        (0.5e-6, 200e-6),
        (1e-6, 40e-6),
        (20e-6, 200e-6),
        (0.1, 4),
        (0.5e-9, 200e-9),
        (1e-9, 40e-9),
        (20e-9, 200e-9),
        (0.1, 4),
        (1075, 1150),
    ]  # length = dim
    _con_bounds: list[tuple[float, float]] = []  # length = num_constraints
    _ref_point: [list[float]] = (540, 0)  # length = num_objectives
    _max_hv: float | None = 100000

    # Optional indices (default empty)
    continuous_inds: list[int] = list(range(dim))
    discrete_inds: list[int] = []
    categorical_inds: list[int] = []

    _check_grad_at_opt: bool = True

    # comp , time, temperature

    testpoint = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    def _eval_model(self, X: Tensor):
        return 0, 0
