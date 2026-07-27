from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple
import pandas as pd
import torch


@dataclass(frozen=True)
class LoadedPoints:
    X: torch.Tensor
    Y: Optional[torch.Tensor] = None

    @property
    def n(self) -> int:
        return int(self.X.shape[0])

    @property
    def d(self) -> int:
        return int(self.X.shape[1])

    @property
    def m(self) -> int:
        return 0 if self.Y is None else int(self.Y.shape[1])


@dataclass(frozen=True)
class LoadedDatasetWithFM:
    all: LoadedPoints  # everything (chosen sobol seed first)
    fm_primary: LoadedPoints  # only FM of that same chosen sobol seed (one file)


def _read_xy_csv(
    file: Path,
    *,
    input_names: Sequence[str],
    objective_names: Optional[Sequence[str]],
    dtype: torch.dtype,
    device: torch.device | str | None,
    drop_unnamed_index_cols: bool = True,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    df = pd.read_csv(file)

    if drop_unnamed_index_cols:
        df = df.loc[:, ~df.columns.str.startswith("Unnamed:")]

    # Handle empty dataframe
    if df.empty:
        X = torch.empty((0, len(input_names)), dtype=dtype, device=device)

        if objective_names is None:
            return X, None

        Y = torch.empty((0, len(objective_names)), dtype=dtype, device=device)
        return X, Y

    missing_x = [c for c in input_names if c not in df.columns]
    if missing_x:
        raise KeyError(f"{file.name}: missing input columns: {missing_x}")

    X = torch.as_tensor(
        df[list(input_names)].to_numpy(),
        dtype=dtype,
        device=device,
    )

    if objective_names is None:
        return X, None

    missing_y = [c for c in objective_names if c not in df.columns]
    if missing_y:
        raise KeyError(f"{file.name}: missing objective columns: {missing_y}")

    Y = torch.as_tensor(
        df[list(objective_names)].to_numpy(),
        dtype=dtype,
        device=device,
    )

    return X, Y


def load_dataset(
    base_dir: Path | str,
    *,
    primary_sobol_id: int = 0,  # index into sobol_seeds ordering
    FM_sobol_id: int = 0,  # index into sobol_seeds ordering
    input_names: Sequence[str],
    objective_names: Sequence[str] | None = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
    # what to load into the "all" dataset
    sobol_seeds: Iterable[int] = range(10),
    include_grid: bool = True,
    include_big_sobol: bool = True,  # sobol.csv (seed=None)
    # FM config (only primary FM will be loaded)
    fm_suffix: str = "_FM",
    require_primary_fm: bool = True,  # if False -> fm_primary can be empty
) -> LoadedDatasetWithFM:
    """
    Returns:
      - all: concatenation of datasets with primary sobol seed first
      - fm_primary: ONLY sobol_seedXX_FM.csv for the same chosen primary seed
    """
    base_dir = Path(base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"base_dir does not exist: {base_dir}")

    sobol_seeds_list = list(sobol_seeds)
    if not (0 <= primary_sobol_id < len(sobol_seeds_list)):
        raise ValueError(
            f"primary_sobol_id={primary_sobol_id} out of range for sobol_seeds={sobol_seeds_list}"
        )

    def sobol_file(seed: int) -> Path:
        return base_dir / f"sobol_seed{seed:02d}.csv"

    def fm_file(seed: int) -> Path:
        return base_dir / f"sobol_seed{seed:02d}{fm_suffix}.csv"

    primary_seed = sobol_seeds_list[primary_sobol_id]
    sobol_order = [primary_seed] + [s for s in sobol_seeds_list if s != primary_seed]

    # ---------- build ALL files list (primary sobol first) ----------
    all_files: list[Path] = [sobol_file(s) for s in sobol_order]

    if include_big_sobol:
        all_files.append(base_dir / "sobol.csv")

    if include_grid:
        all_files.append(base_dir / "grid.csv")

    # read ALL
    Xs: list[torch.Tensor] = []
    Ys: list[torch.Tensor] = []
    for f in all_files:
        if not f.exists():
            # print(f"missing expected file: {f}")
            continue  # ← VERY important
        Xi, Yi = _read_xy_csv(
            f,
            input_names=input_names,
            objective_names=objective_names,
            dtype=dtype,
            device=device,
        )
        if Xi.shape[0] == 0:
            continue

        Xs.append(Xi)
        if objective_names is not None:
            assert Yi is not None
            Ys.append(Yi)

    X_all = torch.cat(Xs, dim=0)
    Y_all = None if objective_names is None else torch.cat(Ys, dim=0)

    # ---------- read ONLY primary FM ----------
    f_fm = fm_file(FM_sobol_id)
    # if require_primary_fm and (not f_fm.exists()):
    # print(f"missing primary FM file: {f_fm}")

    if f_fm.exists():
        X_fm, Y_fm = _read_xy_csv(
            f_fm,
            input_names=input_names,
            objective_names=objective_names,
            dtype=dtype,
            device=device,
        )
        fm_primary = LoadedPoints(X=X_fm, Y=Y_fm)
    else:
        # best-effort empty return
        X_empty = torch.empty((0, len(input_names)), dtype=dtype, device=device)
        Y_empty = (
            None
            if objective_names is None
            else torch.empty((0, len(objective_names)), dtype=dtype, device=device)
        )
        fm_primary = LoadedPoints(X=X_empty, Y=Y_empty)

    return LoadedDatasetWithFM(
        all=LoadedPoints(X=X_all, Y=Y_all),
        fm_primary=fm_primary,
    )
