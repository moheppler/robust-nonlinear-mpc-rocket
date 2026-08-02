"""End-to-end gate: the pure-Python OSQP backup must reproduce the compiled solver's control
results (only speed differs).

Runs each backend in its own subprocess (so the compiled `osqp_generated` extension is never
imported into the pytest process, avoiding cross-test interference), then compares the saved
trajectories. Opt-in via RNMPC_FAST_TESTS=1 and requires a C toolchain; skipped otherwise, so
it runs in the dedicated CI job but not in the default local suite.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

from util.paths import REPO_ROOT

pytestmark = pytest.mark.skipif(
    os.environ.get("RNMPC_FAST_TESTS") != "1",
    reason="set RNMPC_FAST_TESTS=1 (needs a C compiler + CMake) to run the fast-vs-stock gate",
)


def _cli(*args):
    subprocess.check_call([sys.executable, "-m", "rnmpc_cli", *args], cwd=str(REPO_ROOT))


def _latest_pendulum_traj():
    d = REPO_ROOT / "results" / "pendulum_robust_closed_loop"
    newest = max(d.glob("*.npz"), key=lambda p: p.stat().st_mtime)
    return np.load(newest)["state_trajectory"]


def test_fast_matches_stock_on_pendulum():
    _cli("build-fast", "pendulum", "--N", "3")

    _cli("pendulum", "--run", "--smoke-test")  # compiled fast backend
    fast = _latest_pendulum_traj()

    _cli("pendulum", "--run", "--smoke-test", "--stock-osqp")  # pure-Python backend
    stock = _latest_pendulum_traj()

    np.testing.assert_allclose(fast, stock, atol=1e-4, rtol=1e-5)
