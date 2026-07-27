from abc import abstractmethod

import torch
from botorch.test_functions.base import (
    ConstrainedBaseTestProblem,
    MultiObjectiveTestProblem,
)
from torch import Tensor


class base_problem(MultiObjectiveTestProblem, ConstrainedBaseTestProblem):
    r"""
    Problem class on BoTorch basis.

    Conventions:
    - Minimization if `is_minimization=True` (so we pass `negate=False`).
    - `_evaluate_true(X)` returns shape (..., M) for M objectives.
    - `_evaluate_slack_true(X)` returns shape (..., C) of constraint slacks,
      where POSITIVE slack means FEASIBLE.
    """

    # ---- To be set by subclass (before __init__) ----
    dim: int
    num_objectives: int
    num_constraints: int
    is_minimization: bool  # True => minimize all objectives by default
    _bounds: list[tuple[float, float]]  # length = dim
    _con_bounds: list[tuple[float, float]]  # length = num_constraints
    _ref_point: list[float]  # length = num_objectives
    _max_hv: float | None = None

    Model_bool_based = False

    # Optional indices (default empty)
    continuous_inds: list[int]
    discrete_inds: list[int]
    categorical_inds: list[int]

    testpoint: list

    _check_grad_at_opt: bool = True

    def __init__(
        self,
        noise_std: None | float | list[float] = None,
        constraint_noise_std: None | float | list[float] = None,
        dtype: torch.dtype = torch.double,
    ) -> None:
        # negate=False means minimization; negate=True means maximize.
        # If this problem is a minimization problem, DO NOT negate.
        super().__init__(
            noise_std=noise_std, negate=not self.is_minimization, dtype=dtype
        )
        self._cache = {"X": None, "Obj": None, "Con": None}
        # Register constraint bounds as a 2 x C tensor buffer
        # Input format: list of (low, high) per constraint.

        # print("I did change here some stuff --- base_problem")
        # con_bounds = torch.tensor(self._con_bounds, dtype=self.bounds.dtype).transpose(
        #    -1, -2
        # )
        # self.register_buffer("con_bounds", con_bounds)

        self.constraint_noise_std = constraint_noise_std

    @property
    def is_minimization_problem(self) -> bool:
        """Whether the problem is a minimization problem after accounting for `negate`."""
        return self.is_minimization

    def _evaluate_true(self, X: Tensor) -> Tensor:
        r"""
        Evaluate the (noise-free) objectives at X.

        Args:
            X: A `(..., d)`-dim tensor.

        Returns:
            Y: A `(..., M)`-dim tensor of objective values (M = num_objectives).
        """
        Obj, _ = self._get_or_compute(X)
        return Obj

    def _evaluate_slack_true(self, X: Tensor) -> Tensor:
        r"""
        Evaluate (noise-free) constraint slacks at X.

        Args:
            X: A `(..., d)`-dim tensor.

        Returns:
            S: A `(..., C)`-dim tensor of constraint slacks where
               **positive slack means FEASIBLE** (i.e., constraints satisfied).
        """
        _, Con = self._get_or_compute(X)
        return Con

    def _get_or_compute(self, X: Tensor):
        """
        Compute or retrieve cached results (y, z) for input X.

        This avoids recomputing expensive model evaluations when
        BoTorch calls `_evaluate_true` and `_evaluate_slack_true`
        sequentially with the same X.

        Returns
        -------
        y : Tensor
            Noise-free objective values (..., M)
        z : Tensor
            Auxiliary quantities (..., K) used for constraints
        """
        cache = self._cache

        # --- 1. Check if X matches cached X ---
        if (
            cache["X"] is not None
            and cache["X"].shape == X.shape
            and torch.allclose(cache["X"], X, atol=1e-12, rtol=1e-12)
            and not X.requires_grad
        ):
            # --- 2. Use cached results ---
            return cache["Obj"], cache["Con"]

        # --- 3. Otherwise compute fresh ---
        obj, con = self._eval_model(X)

        # --- 4. Store CPU copy for equality checks ---
        self._cache = {
            "X": X.detach().cpu(),  # small, safe to keep on CPU
            "Obj": obj,
            "Con": con,
        }

        return obj, con

    @abstractmethod
    def _eval_model(self, X: Tensor):
        pass
