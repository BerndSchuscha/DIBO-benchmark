import torch
from torch import Tensor

from pibob.problems.base_problem import base_problem


class NanoParticles_Problem(base_problem):
    r""" """

    dim: int = 4
    num_objectives: int = 2
    num_constraints: int = 0
    input_names: list = ["x1", "x2", "x3", "x4"]
    objective_names: list = ["r", "dis"]
    obj_ismodeled = [True, True]
    constraint_names: list = []
    is_minimization: bool = True  # True => minimize all objectives by default
    _bounds: list[tuple[float, float]] = [
        (0, 1),
        (0, 1),
        (0, 1),
        (0, 1),
    ]  # length = dim
    _con_bounds: list[tuple[float, float]] = []  # length = num_constraints
    _ref_point: [list[float]] = (20, -14.5)  # length = num_objectives
    _max_hv: float | None = 100000

    # Optional indices (default empty)
    continuous_inds: list[int] = list(range(dim))
    discrete_inds: list[int] = []
    categorical_inds: list[int] = []

    _check_grad_at_opt: bool = True

    # comp , time, temperature

    testpoint = [0.1, 0.1, 0.1, 0.1]

    def _eval_model(self, X: Tensor):
        x1 = X[..., 0]
        x2 = X[..., 1]
        x3 = X[..., 2]
        x4 = X[..., 3]

        y1 = (
            19.36549
            - 0.2797 * x1
            + 1.56885 * x2
            + 3.5447 * x3
            + 1.82225 * x4
            - 1.1978 * x1 * x2
            - 1.66594 * x1 * x3
            - 1.62873 * x1 * x4
            - 0.02003 * x2 * x3
            - 0.001268 * x2 * x4
            - 0.35086 * x3 * x4
            + 0.3914 * x1**2
            + 0.52265 * x2**2
            - 0.81701 * x3**2
            - 2.74921 * x4**2
        )

        y2 = -(
            19.6114239
            + 1.0313718 * x1
            + 1.48527 * x2
            + 1.7991534 * x3
            - 4.1983899 * x4
            + 1.4263262 * x1 * x2
            - 0.4279443 * x1 * x3
            - 1.3865203 * x1 * x4
            - 1.051601 * x2 * x3
            - 2.06380 * x2 * x4
            - 2.476674 * x3 * x4
            - 0.4497319 * x1**2
            - 1.8040123 * x2**2
            - 3.8699325 * x3**2
            - 2.6148 * x4**2
        )
        y = torch.stack([y1, y2], dim=-1)
        return y, 0
