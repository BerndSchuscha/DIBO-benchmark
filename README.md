# PIBOB — Physics-Informed Bayesian Optimization Benchmark

**Benchmarking knowledge-integration strategies for multi-objective Bayesian optimization in materials design.**

PIBOB is the code base accompanying the paper

> *Knowledge integration strategies for Bayesian optimization in materials design*
> B. Schuscha et al., submitted to *Computational Materials Science* (2026).

It provides a suite of discrete multi-objective materials-design benchmark problems together with a modular framework for injecting prior physical knowledge into the Bayesian optimization (BO) loop — and for quantifying whether that knowledge actually helps.

<!-- Optional: drop your graphical abstract here
<p align="center">
  <img src="docs/figures/graphical_abstract.png" width="700"/>
</p>
-->

## What PIBOB studies

Physical knowledge about a materials system (empirical equations, descriptors, kernels) can be injected into a BO campaign in different **places**. PIBOB systematically compares three injection strategies:

| Strategy | Code name | Where the knowledge acts |
|---|---|---|
| **S** — Surrogate | `surrogate` | Inside the surrogate model (e.g., as a GP mean function or input transform) |
| **P** — Prior | `prior` | As a probabilistic prior in a hierarchical/Pyro model |
| **R** — Regularisation | `regularisation` | As a regularizing term steering the fit toward the physical model |

These are crossed with five **knowledge kinds** (`descriptor`, `kernel`, `custom_mean`, `pyro`, `delta`), three **observation-noise levels** (`low`, `medium`, `high`), and multiple random seeds, on ten benchmark problems — from the classic Branin–Currin test function to real materials data sets. Baselines are a plain GP surrogate (`gpr`) and an identity model. Optimization uses **qLogNEHVI** with greedy batch construction (BoTorch) over a finite candidate set.

Performance is evaluated via hypervolume traces (ΔCNHV against the plain-GP baseline) and surrogate-model quality (ΔR²), with non-parametric statistics (Friedman test with Holm correction) across problems and seeds.

## Benchmark problems

| Key | Domain |
|---|---|
| `branin_currin` | Synthetic bi-objective test function |
| `Bainite_Problem` | Carbide-free bainitic steel design |
| `Ms_NiTi_Problem` | Martensite-start temperature in NiTi shape-memory alloys |
| `NanoParticles` | Nanoparticle synthesis |
| `EnergyDensity_Problem` | Energy-storage materials |
| `PS_ALSC_Problem` | Precipitation-strengthened Al alloys |
| `Peroskites_Problem` | Perovskite materials |
| `Structures_Problem` | Architected structures |
| `Udimet_Problem` | Ni-base superalloy (Udimet) |
| `Magnetic_Problem` | Magnetic materials |

Each problem lives in `pibob/problems/<Name>/` with a common layout:

- `registry.py` — factory (`make_problem`) registering the problem in `pibob.problems.registry.PROBLEMS`
- `cores.py` — the physical knowledge cores (empirical equations, descriptor pools)
- `kernels.py` — problem-specific kernels
- `transforms.py` — input/output transforms
- `<Name>_Problem.py` — the BoTorch-style problem definition (objectives, constraints, bounds, reference point)

All problems subclass `pibob.problems.base_problem`, which builds on BoTorch's `MultiObjectiveTestProblem` / `ConstrainedBaseTestProblem`.

## Repository layout

```
PIBOB/
├── pibob/                  # Core package
│   ├── problems/           # Benchmark problems + Knowledge abstraction
│   ├── models/             # GP variants, analytical-mean GPs, Pyro models
│   ├── optimization/       # MOBO loop, config, acquisition, metrics
│   └── aqf/                # Weighted acquisition functions
├── HPC/run/                # Cluster sweep scripts (work-stealing scheme)
│   ├── full_sweep.py       # Full benchmark sweep (all problems × configs)
│   └── model_performance.py# Surrogate-quality evaluation sweep
├── examples/               # Notebooks: single runs, sweeps, evaluation/plots
├── tests/                  # pytest tests
├── docs/                   # Additional documentation
└── requirements/           # Environment definitions
```

## Installation

Python ≥ 3.10 is required.

```bash
git clone https://github.com/<user>/PIBOB.git
cd PIBOB
python -m venv .venv && source .venv/bin/activate
pip install -r requirements/pibo-dev.txt
pip install -e .
```

Core dependencies: [PyTorch](https://pytorch.org), [BoTorch](https://botorch.org), [GPyTorch](https://gpytorch.ai), [Pyro](https://pyro.ai), pandas, joblib, matplotlib.

## Quick start

Run a single BO campaign on one benchmark:

```python
import torch
from pibob.problems.registry import PROBLEMS
from pibob.optimization import DiscreteMOBOConfig, ObjModelSpec, Objective, main_loop

problem = PROBLEMS["branin_currin"]()

cfg = DiscreteMOBOConfig(
    bounds=problem.bounds,
    ref_point=torch.tensor(problem._ref_point, dtype=torch.double),
    models=[ObjModelSpec(kind="gpr", fit_options={"maxiter": 200})
            for _ in range(problem.num_objectives)],
    n_init=10,
    n_batch=90,
    batch_size=1,
)

# X_set: (N, d) discrete candidate pool; obj: Objective wrapping the problem
X, Y, trace = main_loop(obj, X_set, cfg, method="qlognehvi")
```

See `examples/example_notebook.ipynb` for a full end-to-end run and `examples/evaluate_runs*.ipynb` for the evaluation and plotting pipelines used in the paper.

## Reproducing the benchmark sweep

The full sweep (10 problems × 3 strategies × knowledge kinds × 3 noise levels × 10 seeds) is orchestrated by `HPC/run/full_sweep.py`. It uses a simple **file-based work-stealing scheme** so that any number of independent workers (e.g., SLURM array jobs) can share one job list without a scheduler database:

- each worker atomically **claims** a configuration (`shared_sweep/claims/`),
- writes a marker on completion (`done/`) or failure (`fail/`),
- results are stored per problem as `results/<problem>/results/results_<config>.csv`.

Workers can be added, killed, and restarted at any time; finished configurations are skipped automatically. Set the `SCRATCH` environment variable to redirect all outputs to a scratch file system on HPC clusters (see `pibob/__init__.py`).

```bash
python HPC/run/full_sweep.py          # start one worker
# launch as many workers as you like, e.g. as a SLURM array job
```

Surrogate-model quality (R² on held-out data along the BO trajectory) is evaluated separately with `HPC/run/model_performance.py`.

## Knowledge abstraction

Prior knowledge is organized in `pibob.problems.Knowledge`: each problem carries three *banks* (descriptor / kernel / equation) of knowledge options, of which exactly one *kind* is active per run. The sweep iterates over all available knowledge options per kind, so adding a new empirical model to a problem's `cores.py` automatically enrolls it in the benchmark.

## Citing

If you use PIBOB in your research, please cite (see also `CITATION.cff`):

```bibtex
@article{schuscha2026pibob,
  title   = {Knowledge integration strategies for Bayesian optimization in materials design},
  author  = {Schuscha, Bernd and others},
  journal = {Computational Materials Science},
  year    = {2026},
  note    = {submitted}
}
```

## Authors and acknowledgements

- **Bernd Schuscha** — Materials Center Leoben Forschung GmbH (MCL) / Montanuniversität Leoben ([bernd.schuscha@mcl.at](mailto:bernd.schuscha@mcl.at))

Developed at the Materials Center Leoben Forschung GmbH (MCL), Leoben, Austria.

## License

TBD — see `LICENSE` (to be added).
