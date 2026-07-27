from pibob.problems.Ms_NiTi.Ms_NiTi_Problem_data import Ms_NiTi_Problem
import torch
from scipy.stats.qmc import Sobol
import numpy as np
import pibob
from pathlib import Path
import os
import pandas as pd
import os
from joblib import Parallel, delayed
from glob import glob

problem = Ms_NiTi_Problem()


def sobol_sample(bounds, n, seed=0, scramble=True):
    bounds = np.array(bounds)
    dim = len(bounds)

    if scramble == False:
        seed = None
    engine = Sobol(d=dim)

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
problem_bounds[2] = (0, 3600)
problem_bounds[1] = (650, 770)
problem_bounds[0] = (0.50, 0.545)
N = 2048 * 32

X = sobol_sample(problem_bounds, n=N, scramble=False, seed=44)

# Final merged file
main_path = OUTDIR / f"sobol_{N}_red.csv"


def eval_and_write_one(k):
    # compute for this index
    XX = X[k, :][None, :]
    XX = XX.copy()
    input_tensor = torch.from_numpy(XX)
    objective, constraints = problem._eval_model(input_tensor)
    print(k, input_tensor, objective, constraints)
    combined = np.concatenate(
        [
            np.array([k]),  # Nr
            input_tensor.numpy()[0],  # inputs
            objective.numpy()[0],  # objectives
            constraints.numpy()[0],  # constraints
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
