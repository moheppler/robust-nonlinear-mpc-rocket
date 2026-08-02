"""Backend-equivalence tests: the pure-Python `osqp` backend vs the compiled `osqp_codegen`
extension must agree on primal/dual/cost, including after dynamics/bounds updates. A final test
also compares the compiled extension directly to a general-purpose solver (cvxpy + Clarabel) on
the identical QP data — the direct "compiled C-code == general solver" check.

The module fixture codegens + installs the compiled extension for the Integrator problem, so
these run in their own dedicated process (the shared `osqp_generated` module can only hold one
problem's shape at a time). Opt-in via RNMPC_QP_CODEGEN=1 (needs a C toolchain); skipped
otherwise.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

from tests.fixtures.integrator import Integrator

pytestmark = pytest.mark.skipif(
    os.environ.get("RNMPC_QP_CODEGEN") != "1",
    reason="set RNMPC_QP_CODEGEN=1 (needs a C compiler + CMake) to run the compiled-QP tests",
)


@pytest.fixture(scope="module", autouse=True)
def _integrator_codegen():
    """Codegen + install `osqp_generated` for the Integrator QP used below."""
    from solver.build_fast import patch_generated
    from solver.osqp_backend import CODEGEN_DIR
    from solver.qp_jit import QP

    m = Integrator()
    N = 3
    qp = QP(N, np.eye(m.nx), np.eye(m.nu), m, np.eye(m.nx), backend="osqp")
    qp.export_osqp_solver(str(CODEGEN_DIR), extension_name="osqp_generated")
    patch_generated(CODEGEN_DIR)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--use-pep517", str(CODEGEN_DIR)])
    yield


def _new_integrator():
    m = Integrator()
    m.replace_constraints(x_max=5, x_min=-5, u_max=3, u_min=-3, x_max_f=5, x_min_f=-5)
    return m


def test_qp_codegen():
    from solver.qp_jit import QP

    m = _new_integrator()
    Q, R, Qf, N = np.eye(m.nx), np.eye(m.nu), np.eye(m.nx), 3

    qp_osqp = QP(N, Q, R, m, Qf, backend="osqp", verbose=True)
    qp_cg = QP(N, Q, R, m, Qf, backend="osqp_codegen", codegen_module="osqp_generated", verbose=True)

    x0 = np.array([0.1, 0.0])
    sol_osqp = qp_osqp.solve(x0)
    sol_cg = qp_cg.solve(x0)
    assert sol_osqp["success"] and sol_cg["success"]

    dx = np.max(np.abs(sol_osqp["primal_vec"] - sol_cg["primal_vec"]))
    dy = np.max(np.abs(sol_osqp["dual_vec"] - sol_cg["dual_vec"]))
    dc = abs(sol_osqp["cost"] - sol_cg["cost"])
    assert dx <= 1e-3 and dy <= 1e-3 and dc <= 1e-6

    dq = np.zeros_like(np.array(qp_osqp.q_cost_lin).ravel())
    dq[:3] = [0.1, -0.05, 0.02]
    qp_osqp.add_q_cost_lin(dq)
    qp_cg.add_q_cost_lin(dq)

    sol_osqp2 = qp_osqp.solve(x0)
    sol_cg2 = qp_cg.solve(x0)
    dx2 = np.max(np.abs(sol_osqp2["primal_vec"] - sol_cg2["primal_vec"]))
    dy2 = np.max(np.abs(sol_osqp2["dual_vec"] - sol_cg2["dual_vec"]))
    dc2 = abs(sol_osqp2["cost"] - sol_cg2["cost"])
    assert dx2 <= 1e-3 and dy2 <= 1e-3 and dc2 <= 1e-6


def test_qp_codegen_update_A():
    from solver.qp_jit import QP

    m = _new_integrator()
    Q, R, Qf, N = np.eye(m.nx), np.eye(m.nu), np.eye(m.nx), 3

    qp_osqp = QP(N, Q, R, m, Qf, backend="osqp", verbose=True)
    qp_cg = QP(N, Q, R, m, Qf, backend="osqp_codegen", codegen_module="osqp_generated", verbose=True)

    x0 = np.array([1.0, -0.5])
    sol0_osqp = qp_osqp.solve(x0)
    sol0_cg = qp_cg.solve(x0)
    assert sol0_osqp["success"] and sol0_cg["success"]

    A2 = m.A.astype(np.float64, copy=True)
    B2 = m.B.astype(np.float64, copy=True)
    A2[0, 0] += 0.20
    A2[1, 1] += 0.10
    A2[0, 1] *= 1.50
    new_A_list = [A2 for _ in range(N)]
    new_B_list = [B2 for _ in range(N)]

    qp_osqp.update_dynamics(new_A_list, new_B_list)
    qp_cg.update_dynamics(new_A_list, new_B_list)

    sol1_osqp = qp_osqp.solve(x0)
    sol1_cg = qp_cg.solve(x0)
    assert sol1_osqp["success"] and sol1_cg["success"]

    dx = np.max(np.abs(sol1_osqp["primal_vec"] - sol1_cg["primal_vec"]))
    dy = np.max(np.abs(sol1_osqp["dual_vec"] - sol1_cg["dual_vec"]))
    dc = abs(sol1_osqp["cost"] - sol1_cg["cost"])
    assert dx <= 1e-3 and dy <= 1e-3 and dc <= 1e-4

    changed = np.max(np.abs(sol0_osqp["primal_vec"] - sol1_osqp["primal_vec"]))
    assert changed > 1e-6, "Perturbing A should change the optimal solution"


def test_qp_codegen_update_bounds():
    import casadi as ca
    from solver.qp_jit import QP

    m = _new_integrator()
    Q, R, Qf, N = np.eye(m.nx), np.eye(m.nu), np.eye(m.nx), 3

    qp_osqp = QP(N, Q, R, m, Qf, backend="osqp", verbose=True)
    qp_cg = QP(N, Q, R, m, Qf, backend="osqp_codegen", codegen_module="osqp_generated", verbose=True)

    x0 = np.array([0.2, -0.1])
    assert qp_osqp.solve(x0)["success"] and qp_cg.solve(x0)["success"]

    lbg0 = np.array(qp_osqp.lbg).astype(float).ravel()
    ubg0 = np.array(qp_osqp.ubg).astype(float).ravel()
    new_lbg = lbg0.copy()
    new_ubg = ubg0.copy()

    nx, ni, ni_f = m.nx, m.ni, m.ni_f
    M = N * (nx + ni) + ni_f
    ineq_mask = np.zeros(M, dtype=bool)
    base = 0
    for _ in range(N):
        base += nx
        ineq_mask[base:base + ni] = True
        base += ni
    ineq_mask[base:base + ni_f] = True

    finite_low = (~np.isneginf(lbg0)) & ineq_mask
    finite_high = (~np.isposinf(ubg0)) & ineq_mask
    delta = 3.0
    new_lbg[finite_low] = lbg0[finite_low] + delta
    new_ubg[finite_high] = ubg0[finite_high] - delta

    qp_osqp.lbg = ca.DM(new_lbg)
    qp_cg.lbg = ca.DM(new_lbg)
    qp_osqp.update_ubg(new_ubg)
    qp_cg.update_ubg(new_ubg)

    sol1_osqp = qp_osqp.solve(x0)
    sol1_cg = qp_cg.solve(x0)
    assert sol1_osqp["success"] and sol1_cg["success"]

    dx = np.max(np.abs(sol1_osqp["primal_vec"] - sol1_cg["primal_vec"]))
    dy = np.max(np.abs(sol1_osqp["dual_vec"] - sol1_cg["dual_vec"]))
    dc = abs(sol1_osqp["cost"] - sol1_cg["cost"])
    assert dx <= 1e-2 and dy <= 2e-2 and dc <= 1e-3


def test_qp_codegen_matches_clarabel():
    """The compiled C-code solves the *same* QP as a general-purpose solver (cvxpy + Clarabel).

    Compares the compiled `osqp_generated` solution to Clarabel on the identical QP data — primal,
    dual, and cost. Together with `test_qp_codegen` (compiled == stock) and
    `tests/test_qp_vs_general_solver.py` (stock == Clarabel), this closes compiled == Clarabel.
    """
    pytest.importorskip("cvxpy")
    from solver.qp_jit import QP
    from tests.oracles.qp_clarabel import solve_qp_reference

    m = _new_integrator()
    Q, R, Qf, N = np.eye(m.nx), np.eye(m.nu), np.eye(m.nx), 3
    qp_cg = QP(N, Q, R, m, Qf, backend="osqp_codegen", codegen_module="osqp_generated", verbose=True)

    for x0 in ([0.1, 0.0], [1.0, -0.5], [2.0, 2.0]):
        x0 = np.asarray(x0, dtype=float)
        sol_cg = qp_cg.solve(x0)
        sol_ref = solve_qp_reference(qp_cg, x0)
        assert sol_cg["success"] and sol_ref["success"]
        # Codegen bakes eps_abs/eps_rel = 1e-6 (+ polish); Clarabel is ~1e-8. 1e-4 leaves margin.
        assert np.max(np.abs(sol_cg["primal_vec"] - sol_ref["primal_vec"])) <= 1e-4
        assert np.max(np.abs(sol_cg["dual_vec"] - sol_ref["dual_vec"])) <= 1e-4
        assert abs(sol_cg["cost"] - sol_ref["cost"]) <= 1e-4


def test_qp_codegen_matches_stock_with_nonidentity_weights():
    """Regression test for a real trap found while optimizing qp_jit.py: the compiled
    extension's codegen-time P uses a generic placeholder Q/Qf (see build_fast.py), not
    necessarily this instance's actual weights. If the real P is never explicitly pushed once
    at runtime, the compiled solver silently solves the WRONG (codegen-time) problem. This test
    uses deliberately non-identity Q/Qf so that trap can't hide behind a coincidental match.
    """
    from solver.qp_jit import QP

    m = _new_integrator()
    Q = np.diag([3.0, 7.0])
    R = np.eye(m.nu) * 2.0
    Qf = np.diag([11.0, 13.0])
    N = 3

    qp_osqp = QP(N, Q, R, m, Qf, backend="osqp", verbose=True)
    qp_cg = QP(N, Q, R, m, Qf, backend="osqp_codegen", codegen_module="osqp_generated", verbose=True)

    x0 = np.array([0.3, -0.2])
    sol_osqp = qp_osqp.solve(x0)
    sol_cg = qp_cg.solve(x0)
    assert sol_osqp["success"] and sol_cg["success"]

    dx = np.max(np.abs(sol_osqp["primal_vec"] - sol_cg["primal_vec"]))
    dc = abs(sol_osqp["cost"] - sol_cg["cost"])
    assert dx <= 1e-3 and dc <= 1e-4
