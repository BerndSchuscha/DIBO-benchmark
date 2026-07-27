from pibob.problems.PS_ALSC.PS_ALSC_Problem import PS_ALSC_Problem
from scipy.stats.qmc import Sobol
import numpy as np
import pibob
from pathlib import Path
import os
import pandas as pd
import os
from joblib import Parallel, delayed
from glob import glob
from pibob.problems.PS_ALSC.calc_point import calc_points
import time

problem = PS_ALSC_Problem()
import warnings

warnings.filterwarnings("ignore")


def sobol_sample(bounds, n):
    bounds = np.array(bounds)
    dim = len(bounds)

    engine = Sobol(d=dim, scramble=False)

    # Generate points in [0,1]^d
    X_unit = engine.random(n)

    # Scale to bounds
    lower = bounds[:, 0]
    upper = bounds[:, 1]

    return lower + (upper - lower) * X_unit


col_names = (
    ["Nr"] + problem.input_names + problem.objective_names + problem.constraint_names
)


OUTDIR = Path(pibob.PROJECT_ROOT, "results", problem.__class__.__name__, "datapoints")
os.makedirs(OUTDIR, exist_ok=True)

# Temp directory for per-process files
TMPDIR = OUTDIR / "tmp"
os.makedirs(TMPDIR, exist_ok=True)


# 2
problem_bounds = problem._bounds
N = 16384

X = sobol_sample(problem_bounds, n=N)


# Final merged file
main_path = OUTDIR / f"sobol_{N}_red.csv"


def eval_and_write_one(k):
    start = time.perf_counter()

    XX = X[k, :]

    objective = calc_points(XX)

    elapsed = time.perf_counter() - start

    print(
        " ".join(
            [str(k)]
            + [f"{v:.6f}" for v in XX]
            + [f"{v:.6f}" for v in objective]
            + [f"{elapsed:.6f}"]
        )
    )
    combined = np.concatenate(
        [
            np.array([k]),  # Nr
            XX,  # inputs
            objective,  # objectives
        ]
    )
    df_row = pd.DataFrame([combined], columns=col_names)

    tmp_path = TMPDIR / f"sobol_{N}_{k}.csv"

    # append row; create file if it doesn't exist yet
    df_row.to_csv(
        tmp_path,
        mode="a",
        header=not tmp_path.exists(),  # write header only on first write
        index=False,
    )


# Load existing main file if it exists
if main_path.exists():
    df_main = pd.read_csv(main_path)
    existing_nrs = set(df_main["Nr"].values.astype(int))
else:
    df_main = pd.DataFrame(columns=col_names)
    existing_nrs = set()
    df_main.to_csv(main_path)

# Decide which indices we still need
todo_indices = [k for k in range(N) if k not in existing_nrs]

# Run computations in parallel; each worker writes to its own temp CSV
Parallel(n_jobs=6)(delayed(eval_and_write_one)(k) for k in todo_indices)

df_main = pd.read_csv(main_path)
# Gather all temp files for this N
tmp_paths = sorted(TMPDIR.glob(f"sobol_{N}_red_*.csv"))

frames = [df_main]  # start with existing main data (maybe empty)
for p in tmp_paths:
    df_tmp = pd.read_csv(p)
    frames.append(df_tmp)

if frames:
    df_all = pd.concat(frames, ignore_index=True)
    # In case of duplicates in Nr (e.g. reruns), keep the last row
    df_all = df_all.drop_duplicates(subset="Nr", keep="last")
    df_all.to_csv(main_path, index=False)

    # Optional: clean up temp files
    for p in tmp_paths:
        p.unlink()


def sobol_sample1(bounds, n, seed):
    bounds = np.array(bounds)
    dim = len(bounds)

    engine = Sobol(d=dim, scramble=True, seed=seed)

    # Generate points in [0,1]^d
    X_unit = engine.random(n)

    # Scale to bounds
    lower = bounds[:, 0]
    upper = bounds[:, 1]

    return lower + (upper - lower) * X_unit


col_names = (
    ["Nr"] + problem.input_names + problem.objective_names + problem.constraint_names
)


OUTDIR = Path(pibob.PROJECT_ROOT, "results", problem.__class__.__name__, "datapoints")
os.makedirs(OUTDIR, exist_ok=True)

# Temp directory for per-process files
TMPDIR = OUTDIR / "tmp"
os.makedirs(TMPDIR, exist_ok=True)


# 2
problem_bounds = problem._bounds
N = 128
for seed in [0, 1, 2, 3, 4]:
    # Final merged file
    main_path = OUTDIR / f"sobol_seed{seed:02d}_raw.csv"

    X = sobol_sample1(problem_bounds, n=N, seed=seed)



    def eval_and_write_one(k):
        start = time.perf_counter()

        XX = X[k, :]

        objective = calc_points(XX)

        elapsed = time.perf_counter() - start

        print(
            " ".join(
                [str(k)]
                + [f"{v:.6f}" for v in XX]
                + [f"{v:.6f}" for v in objective]
                + [f"{elapsed:.6f}"]
            )
        )
        combined = np.concatenate(
            [
                np.array([k]),  # Nr
                XX,  # inputs
                objective,  # objectives
            ]
        )
        df_row = pd.DataFrame([combined], columns=col_names)

        tmp_path = TMPDIR / f"sobol_{N}_{k}.csv"

        # append row; create file if it doesn't exist yet
        df_row.to_csv(
            tmp_path,
            mode="a",
            header=not tmp_path.exists(),  # write header only on first write
            index=False,
        )

    # Load existing main file if it exists
    if main_path.exists():
        df_main = pd.read_csv(main_path)
        existing_nrs = set(df_main["Nr"].values.astype(int))
    else:
        df_main = pd.DataFrame(columns=col_names)
        existing_nrs = set()
        df_main.to_csv(main_path)

    # Decide which indices we still need
    todo_indices = [k for k in range(N) if k not in existing_nrs]

    # Run computations in parallel; each worker writes to its own temp CSV
    Parallel(n_jobs=6)(delayed(eval_and_write_one)(k) for k in todo_indices)

    df_main = pd.read_csv(main_path)
    # Gather all temp files for this N
    tmp_paths = sorted(TMPDIR.glob(f"sobol_{N}_red_*.csv"))

    frames = [df_main]  # start with existing main data (maybe empty)
    for p in tmp_paths:
        df_tmp = pd.read_csv(p)
        frames.append(df_tmp)

    if frames:
        df_all = pd.concat(frames, ignore_index=True)
        # In case of duplicates in Nr (e.g. reruns), keep the last row
        df_all = df_all.drop_duplicates(subset="Nr", keep="last")
        df_all.to_csv(main_path, index=False)

        # Optional: clean up temp files
        for p in tmp_paths:
            p.unlink()
