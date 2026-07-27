import torch
from torch import Tensor

from pibob.problems.base_problem import base_problem


class Ms_NiTi_Problem(base_problem):
    r"""
    Martensite start minimization problem from:

    X has the order comp , time, temperature
    Cons have the order: volume fraciton of Ni4Ti3, mean inner particle distance
    Objectives: Ms, time
    """

    dim: int = 3
    num_objectives: int = 2
    num_constraints: int = 3
    input_names: list = ["xNi", "T", "t"]
    objective_names: list = ["Ms", "t"]
    obj_ismodeled = [True, False]
    constraint_names: list = ["cfrac", "mipd", "Ms"]
    is_minimization: bool = True  # True => minimize all objectives by default
    _bounds: list[tuple[float, float]] = [
        (0.5, 0.54),
        (650, 780),
        (0, 18000),
    ]  # length = dim
    _con_bounds: list[tuple[float, float]] = [
        (0.01, 0.5),
        (0, 5e-8),
        (0, 1000),
    ]  # length = num_constraints
    _ref_point: [list[float]] = (250, -3600)  # length = num_objectives
    _max_hv: float | None = 100000

    # Optional indices (default empty)
    continuous_inds: list[int] = list(range(dim))
    discrete_inds: list[int] = []
    categorical_inds: list[int] = []

    _check_grad_at_opt: bool = True

    # comp , time, temperature

    testpoint = [0.52, 660, 18000]

    def _eval_model(self, X: Tensor):

        return 0, 1
