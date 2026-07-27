import torch
from torch import Tensor

from pibob.problems.base_problem import base_problem


class PS_ALSC_Problem(base_problem):
    r"""
    Martensite start minimization problem from:

    X has the order comp , time, temperature
    Cons have the order: volume fraciton of Ni4Ti3, mean inner particle distance
    Objectives: Ms, time
    """

    dim: int = 5
    num_objectives: int = 2
    num_constraints: int = 0
    input_names: list = ["T1", "T2", "t1", "t2", "c"]
    obj_ismodeled = [True, True]
    objective_names: list = ["sigma_pc", "cfrac"]
    constraint_names: list = []
    is_minimization: bool = True  # True => minimize all objectives by default
    _bounds: list[tuple[float, float]] = [
        (300, 500),
        (300, 500),
        (1 / 60, 5),
        (1 / 60, 5),
        (0.001, 0.003),
    ]  # length = dim
    _con_bounds: list[tuple[float, float]] = []  # length = num_constraints

    _ref_point: [list[float]] = (10, -0.0117)  # length = num_objectives
    _max_hv: float | None = 100000

    # Optional indices (default empty)
    continuous_inds: list[int] = list(range(dim))
    discrete_inds: list[int] = []
    categorical_inds: list[int] = []

    _check_grad_at_opt: bool = True

    # comp , time, temperature

    testpoint = [300, 400, 1, 10, 0.002]

    def _eval_model(self, X: Tensor):

        return 0, 1
