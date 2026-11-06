# Guaranteed Robust Nonlinear MPC via Disturbance Feedback

Companion code for guaranteed robust nonlinear MPC via System Level Synthesis (SLS) tube MPC,
demonstrated on a **pendulum**, **quadrotor**, and **rocket**.

![Robust MPC Diagram](tintin_dark.png)

## Closed-loop rocket simulation

<p align="center">
  <img src="closed_loop_last_tube.gif" alt="Simulation Robust MPC" width="600"/>
</p>

## Citation

If you use this code, please cite:

> A. P. Leeman, J. Köhler, and M. N. Zeilinger, "Guaranteed Robust Nonlinear MPC via
> Disturbance Feedback," in *2026 IEEE International Conference on Robotics and Automation
> (ICRA)*, 2026. arXiv:2509.18760.

```bibtex
@inproceedings{leeman2026guaranteed,
  title     = {Guaranteed Robust Nonlinear {MPC} via Disturbance Feedback},
  author    = {Leeman, Antoine P. and K{\"o}hler, Johannes and Zeilinger, Melanie N.},
  booktitle = {2026 IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2026},
  eprint    = {2509.18760},
  archivePrefix = {arXiv},
  primaryClass  = {math.OC}
}
```

## Prerequisites

- **Python 3.11–3.13.** Check what `python3` points to before creating the venv — on many
  systems (including a stock macOS install) the default `python3` is **older** than 3.11 even
  when a newer version is installed elsewhere:

  ```bash
  python3 --version
  ```

  If that prints anything below 3.11, use an explicit interpreter instead of plain `python3`
  in the `Install` step below:
  - **macOS:** `brew install python@3.11`, then use `python3.11`.
  - **Debian/Ubuntu:** `sudo apt install python3.11 python3.11-venv`, then use `python3.11`.
  - **Windows:** install Python 3.11+ from [python.org](https://www.python.org/downloads/) or
    `winget install Python.Python.3.11`, then use `py -3.11` in place of `python3` below.
- For the fast (compiled) solver: **CMake** and a **C/C++ compiler**. The pure-Python
  fallback below needs neither.
  - **macOS:** `xcode-select --install`
  - **Debian/Ubuntu:** `sudo apt install build-essential cmake`
  - **Windows:** install the *Visual Studio Build Tools* (with the "Desktop development with
    C++" workload) and CMake.

## Install

```bash
python3 -m venv .venv              # use python3.11 / py -3.11 if python3 is older (see above)

# Activate the virtual environment:
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\Activate.ps1       # Windows PowerShell
# .venv\Scripts\activate.bat       # Windows cmd.exe

python -V                          # sanity check: should print 3.11-3.13 now that the venv is active
pip install -e .
```

> **Windows PowerShell:** if activation is blocked ("running scripts is disabled"), run
> `Set-ExecutionPolicy -Scope Process RemoteSigned` once in the same window, then activate.

> **`pip install` fails with a scipy/numpy "Requires-Python" resolution error?** That means the
> venv itself was created with a too-old Python — recreate it with the correct interpreter from
> the Prerequisites step above; `pip` can't fix this from inside an existing venv.

This installs a single command, `rnmpc`, that runs every experiment.

## Run

```bash
rnmpc build-fast pendulum      # codegen + compile the fast solver (once per system)
rnmpc pendulum --run           # run the closed-loop experiment (generates data)
rnmpc pendulum                 # no --run: plot the most recent result
```

Add `--smoke-test` for a quick, cheap end-to-end check (short horizon, few simulation steps),
or `--N <k>` to change the horizon.

> **The compiled solver is built for one specific horizon `N` at a time** (default `N=15`).
> `--smoke-test` runs use `N=3` internally, so `rnmpc build-fast pendulum` (the default, N=15)
> won't match `rnmpc pendulum --run --smoke-test` — pass `--smoke-test` to `build-fast` too, so
> both sides agree on `N`:
> ```bash
> rnmpc build-fast pendulum --smoke-test   # builds for N=3
> rnmpc pendulum --run --smoke-test        # runs at N=3
> ```
> The same rule applies to a custom `--N <k>`: pass the same `--N` to both `build-fast` and
> `--run`. A mismatch fails loudly and explains itself (it won't silently give wrong numbers).

| System           | Command                      | Output (under `results/`)              |
| ---------------- | ---------------------------- | -------------------------------------- |
| Pendulum         | `rnmpc pendulum --run`       | `pendulum_robust_closed_loop/`         |
| Quadrotor        | `rnmpc quadrotor --run`      | `quadrotor_robust_closed_loop/`        |
| Rocket           | `rnmpc rocket --run`         | `rockETH_robust_closed_loop/`          |
| Rocket (compare) | `rnmpc rocket-compare --run` | `rockETH_robust_vs_soft_closed_loop/`  |

## No compiler? It still runs — but you have to ask for it.

The paper's headline is solver *speed*, so the compiled solver is the default and `rnmpc`
will **not** silently downgrade to a slower backend. If `build-fast` hasn't been run (or no C
compiler/CMake is available), `rnmpc <system> --run` refuses with a clear message rather than
quietly switching solvers. Opt in explicitly with `--stock-osqp` to get the **same** control
results, just slower, on the pure-Python OSQP backend:

```bash
rnmpc pendulum --run --stock-osqp
```

Use the fallback only to try the code out; reproduce the paper's timing results with the
compiled solver.

## Reproducibility

Continuous integration (`.github/workflows/ci.yml`) installs the package and runs the test
suite on Linux, macOS, and Windows on every push — including a check that the pure-Python
backend (explicitly requested via `--stock-osqp`) reproduces the compiled solver's
trajectories.

**Local `pytest` doesn't exercise the compiled-solver equivalence tests.**
`tests/test_qp_jit.py` and `tests/test_backend_equivalence.py` compare the compiled OSQP codegen
path against a reference solver — they need a C compiler + CMake, so they're opt-in locally and
skip by default under a plain `pytest`. CI runs both on every push
(`.github/workflows/ci.yml`); to run them yourself, matching CI exactly:

```bash
RNMPC_QP_CODEGEN=1 pytest tests/test_qp_jit.py -q
RNMPC_FAST_TESTS=1 pytest tests/test_backend_equivalence.py -q
```

## Repository layout

- `dyn/` — the dynamical models (`pendulum.py`, `quadrotor.py`, `rocket.py`) and the shared
  `Model` / `LTI` / `LTV` base classes. The core ODEs live here. `ddyn(x, u)` integrates with
  the model's own `self.dt` (most models define 50 ms in their constructor) and is free of side
  effects — it no longer overwrites `model.dt`; models can set their own step (e.g. the reduced
  `RocketDirect` uses 25 Hz) and every solver component (Jacobians, nominal NLP, tube checks)
  uses it consistently.
- `solver/` — the SLS tube-MPC solver: SCP outer loop (`SCP_SLS_jit.py`), the fast inner
  Riccati/tube solver (`fast_SLS_jit.py`), and the QP front-end (`qp_jit.py`).
- `expe/` — the closed-loop experiment scripts, one per system.
- `tests/` — smoke tests and backend-equivalence checks.
