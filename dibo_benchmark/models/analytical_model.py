import torch
import torch.nn as nn
from pathlib import Path
from pibob.problems.load_points import load_dataset
import pibob
from botorch.utils.transforms import normalize


class AnalyticalCore(nn.Module):
    """
    Core model template.
    Subclasses fill in class attributes + phys_fun.

    No init arguments.
    """

    # --- to be defined in subclasses ---
    N_PAR: int = None
    PAR_INIT = None  # list/tuple/tensor length N_PAR

    # optional metadata
    INIT_LOC = None
    INIT_SCALE = None
    NOISE_LEVEL: float = 0.0
    SIG1_INIT: float = 2.0

    def __init__(self):
        super().__init__()
        self._validate_spec()

    def _validate_spec(self):
        if self.N_PAR is None or not isinstance(self.N_PAR, int) or self.N_PAR <= 0:
            raise ValueError(
                f"{self.__class__.__name__}: set class attribute N_PAR (>0 int)."
            )

        if self.PAR_INIT is None:
            raise ValueError(
                f"{self.__class__.__name__}: set class attribute PAR_INIT."
            )

        par_init = torch.as_tensor(self.PAR_INIT, dtype=torch.float32)
        if par_init.numel() != self.N_PAR:
            raise ValueError(
                f"{self.__class__.__name__}: PAR_INIT has {par_init.numel()} elements, "
                f"expected N_PAR={self.N_PAR}."
            )

        # store init vectors as buffers (device/dtype friendly)
        self.register_buffer("par_init", par_init.clone())
        self.register_buffer(
            "init_loc",
            (
                torch.as_tensor(self.INIT_LOC, dtype=torch.float32)
                if self.INIT_LOC is not None
                else torch.empty(0)
            ),
        )
        self.register_buffer(
            "init_scale",
            (
                torch.as_tensor(self.INIT_SCALE, dtype=torch.float32)
                if self.INIT_SCALE is not None
                else torch.empty(0)
            ),
        )

        self.noise_level = float(self.NOISE_LEVEL)
        self.sig1_init = float(self.SIG1_INIT)

    @property
    def n_par(self) -> int:
        return int(self.N_PAR)

    def init_par(self, device=None, dtype=None) -> torch.Tensor:
        p = self.par_init
        if device is not None:
            p = p.to(device)
        if dtype is not None:
            p = p.to(dtype)
        return p.clone()

    # --- subclass must implement ---
    def phys_fun(self, x: torch.Tensor, par: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor, par: torch.Tensor) -> torch.Tensor:
        return self.phys_fun(x, par)


class MyUELCore(AnalyticalCore):
    N_PAR = 3
    PAR_INIT = [0.5, 1.2, 5.0]  # a,b,c
    NOISE_LEVEL = 0.05
    SIG1_INIT = 1.0
    INIT_LOC = None
    INIT_SCALE = None

    def phys_fun(self, x, par):
        a, b, c = par
        phi = x[..., 0]
        sigy = x[..., 1]
        return a * torch.log1p(phi) + b / sigy + c



class PoolDescriptorCore(AnalyticalCore):
    """
    AnalyticalCore with one extra descriptor loaded from the problem's datapoint pool.
    """

    def __init__(self, problem, descriptor_name, scale=10**8):
        super().__init__()
        self.problem = problem
        self.descriptor_name = descriptor_name
        self.scale = scale

        outdir = Path(
            pibob.PROJECT_ROOT,
            "results",
            problem.__class__.__name__,
            "datapoints",
        )

        df = load_dataset(
            base_dir=outdir,
            input_names=problem.input_names,
            objective_names=[descriptor_name],
            primary_sobol_id=0,
        )

        self.X_pool = normalize(df.all.X, bounds=problem.bounds)
        self.Y_pool = df.all.Y.squeeze(-1)

        # Quantize once on CPU
        self.X_pool_q = self._quantize(self.X_pool)

        # Faster dict: bytes key -> row index
        self._index = {
            self._row_key_from_quantized(self.X_pool_q[i]): i
            for i in range(self.X_pool_q.shape[0])
        }

        # cache Y_pool on devices
        self._Y_pool_cache = {}

    def _quantize(self, x: torch.Tensor) -> torch.Tensor:
        return torch.round(x.detach().cpu() * self.scale).to(torch.int64)

    def _row_key_from_quantized(self, row_q: torch.Tensor) -> bytes:
        return row_q.numpy().tobytes()

    def _row_key(self, row: torch.Tensor) -> bytes:
        row_q = self._quantize(row)
        return self._row_key_from_quantized(row_q)

    def _get_Y_pool_on_device(self, device: torch.device) -> torch.Tensor:
        device = torch.device(device)
        if device not in self._Y_pool_cache:
            self._Y_pool_cache[device] = self.Y_pool.to(device)
        return self._Y_pool_cache[device]

    def lookup_descriptor(self, x: torch.Tensor) -> torch.Tensor:
        rows = x.reshape(-1, x.shape[-1])

        # Quantize all rows at once
        rows_q = self._quantize(rows)

        idx = []
        for i in range(rows_q.shape[0]):
            k = self._row_key_from_quantized(rows_q[i])
            j = self._index.get(k)
            if j is None:
                raise ValueError(f"Row {i} not found in descriptor pool.")
            idx.append(j)

        idx = torch.tensor(idx, device=x.device, dtype=torch.long)
        y = self._get_Y_pool_on_device(x.device)[idx]
        return y.reshape(x.shape[:-1])
