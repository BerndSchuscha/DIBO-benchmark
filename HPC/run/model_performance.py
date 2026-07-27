import os
import gc
import json
import time
import random
import traceback
from pathlib import Path
from itertools import product

import torch
import pandas as pd
from joblib import Parallel, delayed

import pibob
from pibob.problems.registry import PROBLEMS
from pibob.problems.load_points import load_dataset
from pibob.optimization import (
    DiscreteMOBOConfig,
    ObjModelSpec,
    Objective,
    main_loop,
    hypervolume_trace,
)
from botorch.utils.transforms import normalize
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from pibob.models.analytical_model import PoolDescriptorCore

# ============================================================
# GLOBAL SETTINGS
# ============================================================

TENSOR_KWARGS = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

MC_SAMPLES = 128
THREADS_PER_RUN = 6
CORES_PER_RUN = 6

PROBLEM_NAMES = [
    "branin_currin",
    "NanoParticles",
    "Ms_NiTi_Problem",
    "PS_ALSC_Problem",
    "EnergyDensity_Problem",
    "Peroskites_Problem",
    "Structures_Problem",
    "Udimet_Problem",
    "Magnetic_Problem",
    "Bainite_Problem",
]


# optional root marker for the whole sweep
GLOBAL_SWEEP_ROOT = Path(pibob.SAVE, "results", "global_shared_sweep_random")
GLOBAL_SWEEP_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================
# UTILITIES
# ============================================================


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def run_id_from_configuration(configuration) -> str:
    """
    Problem is intentionally NOT part of the run id.
    Problem separation is handled by different folders.
    """
    return "_".join(f"{v}" for k, v in sorted(configuration.items()))


def set_threads_per_run(n=6):
    """
    Must be called inside each worker process.
    """
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n)
    torch.set_num_threads(n)


def get_problem_paths(prob: str) -> dict:
    """
    All problem-specific folders live under results/<prob>/...
    """
    root_results = Path(pibob.SAVE, "results", prob)

    paths = {
        "problem_root": root_results,
        "datapoints": Path(pibob.PROJECT_ROOT, "results", prob, "datapoints"),
        "results": root_results / "results",
        "shared_root": root_results / "shared_sweep",
        "claims": root_results / "shared_sweep" / "claims",
        "done": root_results / "shared_sweep" / "done",
        "fail": root_results / "shared_sweep" / "fail",
    }

    for key, p in paths.items():
        if key != "datapoints":
            p.mkdir(parents=True, exist_ok=True)

    return paths


def done_path(prob: str, configuration: dict) -> Path:
    run_id = run_id_from_configuration(configuration)
    return get_problem_paths(prob)["done"] / f"cfg_{run_id}.json"


def claim_path(prob: str, configuration: dict) -> Path:
    run_id = run_id_from_configuration(configuration)
    return get_problem_paths(prob)["claims"] / f"cfg_{run_id}.claim"


def fail_path(prob: str, configuration: dict) -> Path:
    run_id = run_id_from_configuration(configuration)
    return get_problem_paths(prob)["fail"] / f"cfg_{run_id}.txt"


def result_trace_path(prob: str, configuration: dict) -> Path:
    run_id = run_id_from_configuration(configuration)
    return get_problem_paths(prob)["results"] / f"results_{run_id}.csv"


def result_model_path(prob: str, configuration: dict) -> Path:
    run_id = run_id_from_configuration(configuration)
    return get_problem_paths(prob)["results"] / f"modelresults_{run_id}.csv"


def try_claim(prob: str, configuration: dict) -> bool:
    """
    Atomic claim inside the problem-specific folder.
    """
    p = claim_path(prob, configuration)
    try:
        fd = os.open(str(p), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w") as f:
            f.write(f"pid={os.getpid()}\n")
            f.write(f"time={time.time()}\n")
            f.write(f"problem={prob}\n")
        return True
    except FileExistsError:
        return False


# ============================================================
# CONFIGURATION SWEEP
# ============================================================


def sweep_configs(problem):
    knowledge_types = [
        "descriptor",
        "custom_mean",
        "pyro",
        "delta",
        "kernel",
    ]
    noise_levels = ["low", "medium", "high"]
    places = ["surrogate"]
    kinds = ["gpr", "identity"] + knowledge_types
    numbers = list(range(0, 10))
    aqfs = ["random" ]

    configurations = {}
    k = 0

    for noise_level, place, kind, number, aqf in product(
        noise_levels, places, kinds, numbers, aqfs
    ):
        if kind in knowledge_types:
            n_options = len(problem.Knowledge._bank(kind).options)
            for knowledge_num in range(n_options):
                obj = type(problem.Knowledge._bank(kind).options[knowledge_num][0])
                if issubclass(obj, PoolDescriptorCore) and kind == "pyro":
                    continue
                configurations[k] = {
                    "noise_level": noise_level,
                    "place": place,
                    "kind": kind,
                    "number": number,
                    "knowledge_num": knowledge_num,
                    "aqf": aqf,
                }
                k += 1
        else:
            configurations[k] = {
                "noise_level": noise_level,
                "place": place,
                "kind": kind,
                "number": number,
                "knowledge_num": 1000,
                "aqf": aqf,
            }
            k += 1

    return configurations


def build_all_jobs(problem_names):
    """
    Flatten all problems into one global job list.
    Each element is (prob, key, configuration).
    """
    jobs = []
    for prob in problem_names:
        problem = PROBLEMS[prob]()
        configurations = sweep_configs(problem)
        for key, configuration in configurations.items():
            jobs.append((prob, key, configuration))
    return jobs


# ============================================================
# BUILD CONFIG
# ============================================================


def build_cfg(problem, configuration, datapoint_path: Path, tkwargs: dict):
    if configuration["knowledge_num"] != 1000:
        problem.Knowledge.set_active_kind(configuration["kind"])
        kind, know = problem.Knowledge.get_fns(configuration["knowledge_num"])
    else:
        kind = configuration["kind"]
        know = [None for _ in range(problem.num_objectives)]

    points = load_dataset(
        base_dir=datapoint_path,
        input_names=problem.input_names,
        objective_names=problem.objective_names,
        primary_sobol_id=configuration["number"],
    )

    X_set, Y_set = points.all.X, points.all.Y

    models = []
    j = 0
    for i in range(problem.num_objectives):
        if problem.obj_ismodeled[i]:
            models.append(
                ObjModelSpec(
                    kind=kind,
                    fit_options={"maxiter": 1000},
                    Knowledge=know[j],
                )
            )
            j = j + 1
        else:
            models.append(
                ObjModelSpec(
                    kind="identity",
                    fit_options={"maxiter": 1000},
                    Knowledge=None,
                )
            )

    cfg = DiscreteMOBOConfig(
        bounds=problem.bounds,
        ref_point=problem.ref_point.to(**tkwargs),
        noise_se=torch.tensor(
            problem.noise_profile(configuration["noise_level"]),
            **tkwargs,
        ),
        batch_size=1,
        n_init=2 * (problem.dim + 1),
        n_batch=min(X_set.shape[0] - 20, 100),
        mc_samples=MC_SAMPLES,
        eval_batch_size=1000,
        models=models,
        place=configuration["place"],
    )

    return cfg, X_set, Y_set


# ============================================================
# SINGLE RUN
# ============================================================


def run_one_config(prob: str, key: int, configuration: dict):
    cleanup()

    problem = PROBLEMS[prob]()
    paths = get_problem_paths(prob)

    cfg, X_set, Y_set = build_cfg(
        problem=problem,
        configuration=configuration,
        datapoint_path=paths["datapoints"],
        tkwargs=TENSOR_KWARGS,
    )

    X_set = normalize(X_set, bounds=problem.bounds)

    bd = DominatedPartitioning(ref_point=problem.ref_point, Y=Y_set)
    hv0 = bd.compute_hypervolume().item()

    obj = Objective(Y_set)
    train_X, train_Y, train_Y_true, _, results = main_loop(
        obj,
        X_set,
        cfg,
        method=configuration["aqf"],
    )

    results.to_csv(result_model_path(prob, configuration), index=False)

    ns_q_mean, hv_q_mean = hypervolume_trace(
        train_Y_true,
        cfg.ref_point,
        cfg.n_init,
        cfg.n_batch,
        cfg.batch_size,
    )

    df = {
        "ns_q_mean": ns_q_mean,
        "hv_q_mean": hv_q_mean,
    }

    Y = train_Y_true.detach().cpu().numpy()
    _, D = Y.shape
    for i in range(D):
        df[f"Y_{i}"] = Y[:, i]

    pd.DataFrame(df).to_csv(result_trace_path(prob, configuration), index=False)

    return {
        "problem": prob,
        "key": key,
        "config": configuration,
        "hv0": hv0,
        "hv_final": hv_q_mean[-1],
    }


# ============================================================
# GLOBAL STEALING WORKER
# ============================================================


def worker_steal_loop(worker_id: int, jobs):
    set_threads_per_run(THREADS_PER_RUN)

    jobs = list(jobs)

    while True:
        did_work = False

        for prob, key, configuration in jobs:
            if done_path(prob, configuration).exists():
                continue

            if not try_claim(prob, configuration):
                continue

            did_work = True
            print(
                f"[worker {worker_id}] CLAIMED problem={prob} key={key}",
                flush=True,
            )

            try:
                out = run_one_config(prob, key, configuration)
                done_path(prob, configuration).write_text(json.dumps(out, indent=2))
                print(
                    f"[worker {worker_id}] DONE problem={prob} key={key}",
                    flush=True,
                )

            except Exception:
                tb = traceback.format_exc()
                fail_path(prob, configuration).write_text(
                    f"problem={prob}\n"
                    f"key={key}\n"
                    f"config={repr(configuration)}\n\n"
                    f"{tb}"
                )
                print(
                    f"[worker {worker_id}] FAIL problem={prob} key={key}",
                    flush=True,
                )

        if not did_work:
            print(f"[worker {worker_id}] nothing left to do", flush=True)
            break

    return worker_id


# ============================================================
# MAIN
# ============================================================


def run_all_problems(problem_names, cores_per_run=CORES_PER_RUN):
    cleanup()

    jobs = build_all_jobs(problem_names)

    total_cores = os.cpu_count() or 1
    n_workers = max(1, total_cores // cores_per_run)

    print("===== GLOBAL MULTI-PROBLEM SWEEP =====", flush=True)
    print(f"problems: {problem_names}", flush=True)
    print(f"number of jobs: {len(jobs)}", flush=True)
    print(
        f"total_cores={total_cores}, cores_per_run={cores_per_run}, n_workers={n_workers}",
        flush=True,
    )

    stats = Parallel(
        n_jobs=n_workers,
        backend="loky",
        verbose=10,
    )(delayed(worker_steal_loop)(worker_id, jobs) for worker_id in range(n_workers))

    print("===== GLOBAL MULTI-PROBLEM SWEEP FINISHED =====", flush=True)
    return stats


if __name__ == "__main__":
    stats = run_all_problems(PROBLEM_NAMES, cores_per_run=CORES_PER_RUN)
