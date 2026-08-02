"""QP-level equivalence: the nominal QP that ``solver.qp_jit`` hands OSQP must have the *same*
solution — primal, dual, and cost — as an independent general-purpose solver (cvxpy + Clarabel,
BSD; no MOSEK) on the identical QP data.

This is the locally-runnable half of "the compiled C-code == a general-purpose solver". It runs
the **stock** OSQP backend (no C toolchain needed) against Clarabel. The compiled backend is shown
equal to stock in ``tests/test_qp_jit.py`` (env-gated, CI), and that test also compares the
compiled backend directly to Clarabel; chaining the two gives ``compiled == Clarabel``.

Needs cvxpy (the ``[dev]`` extra); skipped if unavailable so CI stays green without it.
"""
import numpy as np
import pytest

pytest.importorskip("cvxpy")
if "CLARABEL" not in __import__("cvxpy").installed_solvers():
    pytest.skip("Clarabel not available in cvxpy", allow_module_level=True)

from tests.fixtures.integrator import Integrator
from tests.oracles.qp_clarabel import solve_qp_reference

# Stock OSQP (eps 1e-9) and Clarabel (interior point) both solve to high accuracy; observed
# agreement is ~1e-9. 1e-5 leaves ample margin across solver/OS versions.
PRIMAL_ATOL = 1e-5
DUAL_ATOL = 1e-5
COST_ATOL = 1e-5


def _assert_qp_equivalent(sol_osqp, sol_ref, label):
    assert sol_osqp.get("success"), f"{label}: stock OSQP did not solve"
    assert sol_ref.get("success"), f"{label}: Clarabel oracle did not solve"
    np.testing.assert_allclose(sol_osqp["primal_vec"], sol_ref["primal_vec"],
                               atol=PRIMAL_ATOL, rtol=0.0, err_msg=f"{label}: primal mismatch")
    np.testing.assert_allclose(sol_osqp["dual_vec"], sol_ref["dual_vec"],
                               atol=DUAL_ATOL, rtol=0.0, err_msg=f"{label}: dual mismatch")
    assert abs(sol_osqp["cost"] - sol_ref["cost"]) <= COST_ATOL, f"{label}: cost mismatch"


def _stock_qp(N, m, Qf_scale=1.0):
    """Build a stock-OSQP QP (force the pure-Python backend so no C toolchain is needed)."""
    import solver.osqp_backend as ob
    from solver.qp_jit import QP
    ob.set_force_stock(True)
    try:
        return QP(N, np.eye(m.nx), np.eye(m.nu), m, Qf_scale * np.eye(m.nx),
                  backend="osqp", verbose=False)
    finally:
        ob.set_force_stock(False)


@pytest.mark.parametrize("x0", [[0.1, 0.0], [1.0, -0.5], [2.0, 2.0], [-3.0, 1.0]])
def test_qp_matches_clarabel_lti(x0):
    """Integrator (LTI) nominal QP: stock OSQP == Clarabel (primal + dual + cost)."""
    m = Integrator()
    m.replace_constraints(5, -5, 3, -3, 5, -5)
    qp = _stock_qp(3, m)
    x0 = np.asarray(x0, dtype=float)
    _assert_qp_equivalent(qp.solve(x0), solve_qp_reference(qp, x0), f"Integrator x0={x0}")


@pytest.mark.parametrize("x0", [[0.1, 0.0], [1.0, -0.5]])
def test_qp_matches_clarabel_with_linear_cost(x0):
    """Adding a nonzero linear term q must not break primal/dual/cost equivalence."""
    m = Integrator()
    m.replace_constraints(5, -5, 3, -3, 5, -5)
    qp = _stock_qp(3, m)
    dq = np.zeros((m.nx + m.nu) * qp.N + m.nx)
    dq[:3] = [0.1, -0.05, 0.02]
    qp.add_q_cost_lin(dq)
    x0 = np.asarray(x0, dtype=float)
    _assert_qp_equivalent(qp.solve(x0), solve_qp_reference(qp, x0), f"Integrator+q x0={x0}")


@pytest.mark.parametrize("x0", [[0.2, 0.2, 0.2, 0.2], [0.5, -0.3, 0.1, 0.0]])
def test_qp_matches_clarabel_ltv(x0):
    """Pendulum (LTV) nominal QP: exercises the time-varying dynamics rows against Clarabel."""
    from dyn.pendulum import Pendulum
    from dyn.LTV import LTV
    m = Pendulum()
    m.E = 0.003 * np.eye(m.nx)
    m.dt = 0.05
    m.replace_constraints(10 * np.ones(m.nx), -10 * np.ones(m.nx),
                          5 * np.ones(m.nu), -5 * np.ones(m.nu),
                          10 * np.ones(m.nx), -10 * np.ones(m.nx))
    qp = _stock_qp(5, LTV(m, 5), Qf_scale=10.0)
    x0 = np.asarray(x0, dtype=float)
    _assert_qp_equivalent(qp.solve(x0), solve_qp_reference(qp, x0), f"Pendulum x0={x0}")
