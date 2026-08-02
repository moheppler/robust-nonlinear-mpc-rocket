"""Solve-time summary across the pendulum/quadrotor/rocket closed-loop experiments.

Reads the t_jac/t_qp/t_riccati timing arrays each of those experiments already saves into its
closed-loop .npz, and plots a per-step breakdown for each system side by side.

`--run` additionally (re)builds and (re)runs all three systems first, via the `rnmpc` CLI as
subprocesses -- the compiled OSQP extension is a single global slot (see
solver/osqp_backend.py), so each system must be built+run in its own process rather than
importing each generate() function in-process (which would silently reuse whichever solver
was last built for the other systems).
"""
import argparse
import os
import subprocess
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from util.paths import REPO_ROOT, results_dir
from expe.main_pendulum_robust_closed_loop import _RESULT_NAME as _PENDULUM_RESULT_NAME
from expe.main_quadrotor_robust_closed_loop import _RESULT_NAME as _QUADROTOR_RESULT_NAME
from expe.main_rocket_robust_closed_loop import _RESULT_NAME as _ROCKET_RESULT_NAME

_RESULT_NAME = "solve_times_summary"

SYSTEMS = ["pendulum", "quadrotor", "rocket"]
SYSTEM_RESULT_NAMES = {
    "pendulum": _PENDULUM_RESULT_NAME,
    "quadrotor": _QUADROTOR_RESULT_NAME,
    "rocket": _ROCKET_RESULT_NAME,
}


def _cli(*args):
    subprocess.check_call([sys.executable, "-m", "rnmpc_cli", *args], cwd=str(REPO_ROOT))


def run_all(N: int | None = None, smoke_test: bool = False):
    for system in SYSTEMS:
        build_args = ["build-fast", system]
        run_args = [system, "--run"]
        if N is not None:
            build_args += ["--N", str(N)]
            run_args += ["--N", str(N)]
        if smoke_test:
            build_args += ["--smoke-test"]
            run_args += ["--smoke-test"]
        print(f"[solve-times] building fast solver for {system} ...")
        _cli(*build_args)
        print(f"[solve-times] running {system} closed loop ...")
        _cli(*run_args)


def _latest_npz(result_name: str, system: str):
    folder = results_dir(result_name)
    try:
        files = [f for f in os.listdir(folder) if f.endswith(".npz")]
    except FileNotFoundError:
        files = []
    if not files:
        raise FileNotFoundError(
            f"No closed-loop results found for '{system}' in {folder}. "
            f"Run it first: rnmpc {system} --run"
        )
    latest = max(files, key=lambda f: os.path.getctime(os.path.join(folder, f)))
    return np.load(os.path.join(folder, latest))


def _load_timings(result_name: str, system: str):
    data = _latest_npz(result_name, system)
    t_jac = np.asarray(data["t_jac"]).ravel()
    t_qp = np.asarray(data["t_qp"]).ravel()
    t_riccati = np.asarray(data["t_riccati"]).ravel()
    return t_jac, t_qp, t_riccati


def summarize():
    fig, axs = plt.subplots(1, len(SYSTEMS), figsize=(18, 5))
    for ax, system in zip(axs, SYSTEMS):
        t_jac, t_qp, t_riccati = _load_timings(SYSTEM_RESULT_NAMES[system], system)
        steps = np.arange(t_jac.size)

        ax.plot(steps, t_jac, label="t_jac", marker="o", markersize=3)
        ax.plot(steps, t_qp, label="t_qp", marker="o", markersize=3)
        ax.plot(steps, t_riccati, label="t_riccati", marker="o", markersize=3)

        total = t_jac + t_qp + t_riccati
        median = float(np.nanmedian(total))
        mx = float(np.nanmax(total))
        ax.set_title(f"{system} (median {median:.2f} ms, max {mx:.2f} ms)")
        ax.set_xlabel("Simulation step")
        ax.set_ylabel("Solve time [ms]")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    out_dir = results_dir(_RESULT_NAME)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_png = os.path.join(str(out_dir), f"solve_times_summary_{timestamp}.png")
    plt.savefig(out_png, dpi=200)
    print(f"Summary saved to {out_png}")
    plt.show()
    return out_png


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Solve-time summary across pendulum/quadrotor/rocket"
    )
    parser.add_argument("--run", action="store_true",
                         help="(Re)build and (re)run all three systems first")
    parser.add_argument("--N", type=int, default=None,
                         help="Horizon length (overrides each system's default if provided)")
    parser.add_argument("--smoke-test", action="store_true",
                         help="Short, cheap variant for all three systems")
    args = parser.parse_args()
    if args.run:
        run_all(args.N, smoke_test=args.smoke_test)
    else:
        summarize()
