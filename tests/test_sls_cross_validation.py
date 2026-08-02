"""Cross-validation of the production fast-SLS solvers against independent oracles.

Closes the open correctness gap (STATUS.md item #3): asserts the production SLS solvers produce
the same primal trajectory as two *independent* optimizers that solve the same problem a
different way:

  - ``tests.oracles.cvxpy_SLS.cvxpy_SLS``              -- the *linear* SLS problem as one convex
                                                         SOCP (cvxpy + Clarabel). Validates the
                                                         fast-SLS forward/backward solver
                                                         ``solver.fast_SLS_jit.fast_SLS``.
  - ``tests.oracles.casadi_nonlinear_sls.CasadiNonlinearSLS`` -- the *nonlinear* SLS problem as one
                                                         monolithic NLP (casadi + ipopt).
                                                         Validates the SCP solver
                                                         ``solver.SCP_SLS_jit.SCP_SLS``.

All three formulations use the same second-order-cone (2-norm) constraint tightening under unit
regularization (the OCP default ``Q_reg = R_reg = Q_reg_f = I``; see the convention note in
``cvxpy_SLS.initialize_solver``), so they converge to the same optimum -- the tests therefore use
the default (identity) regularization.

**We assert the primal trajectory (``primal_x`` / ``primal_u``), not the cost.** The trajectory is
the uniquely-determined, physically meaningful output and matches across solvers to ~1e-5 or
better. The reported *cost* is not comparable: the three implementations book-keep the tube
(system-response) cost differently -- e.g. ``fast_SLS_jit`` leaves ``cost_tube`` unfilled when it
converges in one iteration, and ``SCP_SLS`` accumulates cost across the SCP loop -- and the
system-response matrices that carry the tube cost are non-unique when the tube constraints are not
all active, so equal trajectories can carry different tube costs.

Gating: both tests need optional solvers absent on some machines (cvxpy + Clarabel from the
``[dev]`` extra / a working casadi-ipopt). They ``importorskip`` those so CI stays green where they
are unavailable. No MOSEK: Clarabel (BSD) ships with cvxpy and handles the SOCP tube constraints.
"""
import numpy as np
import pytest

from tests.fixtures.integrator import Integrator
from tests.fixtures.spring_damper import SpringDamper

# Trajectories agree to a few 1e-4 in practice; 2e-3 (absolute) leaves margin across solver/OS
# versions. rtol=0 so the bound is purely absolute (states/inputs here are O(1)).
PRIMAL_ATOL = 2e-3


def _assert_primal_match(sol_a, sol_b, label_a, label_b):
    assert sol_a.get("success"), f"{label_a} did not solve successfully"
    assert sol_b.get("success"), f"{label_b} did not solve successfully"
    for key in ("primal_x", "primal_u"):
        np.testing.assert_allclose(
            np.asarray(sol_a[key], dtype=float),
            np.asarray(sol_b[key], dtype=float),
            atol=PRIMAL_ATOL,
            rtol=0.0,
            err_msg=f"{key} mismatch: {label_a} vs {label_b}",
        )


# --------------------------------------------------------------------------------------------
# Test A -- fast-SLS vs cvxpy (linear SLS on the double-integrator LTI fixture)
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("x0", [[3.0, -2.0], [2.0, 2.0], [-3.0, 1.0]])
def test_fast_sls_matches_cvxpy_on_linear_model(x0):
    cp = pytest.importorskip("cvxpy")
    if "CLARABEL" not in cp.installed_solvers():
        pytest.skip("Clarabel solver not available in cvxpy")

    import solver.osqp_backend as ob
    from solver.fast_SLS_jit import fast_SLS
    from tests.oracles.cvxpy_SLS import cvxpy_SLS

    N = 5
    m = Integrator()
    Q = np.eye(m.nx)
    R = np.eye(m.nu)
    Qf = np.eye(m.nx)
    x0 = np.asarray(x0, dtype=float)

    ob.set_force_stock(True)
    try:
        fast = fast_SLS(N, Q, R, m, Qf)
        fast.verbose = False
        # fast_SLS_jit is a *delta* solver: its forward QP pins x(0) = -(argument) (see the
        # "x(0) + x0 = 0" convention in qp_jit). SCP drives it with (x_nom0 - x0); with a zero
        # nominal a standalone full solve for physical initial state x0 is therefore solve(-x0).
        sol_fast = fast.solve(-x0)
    finally:
        ob.set_force_stock(False)

    ref = cvxpy_SLS(N, Q, R, m, Qf)
    sol_ref = ref.solve(x0)

    _assert_primal_match(sol_fast, sol_ref, "fast_SLS_jit", "cvxpy_SLS")


# --------------------------------------------------------------------------------------------
# Test B -- nonlinear SLS vs ipopt (SCP on the nonlinear spring-damper fixture)
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("x0", [[5.0, -2.0], [-4.0, 2.0], [6.0, 0.0]])
def test_scp_sls_matches_ipopt_on_nonlinear_model(x0):
    pytest.importorskip("casadi")

    import solver.osqp_backend as ob
    from solver.SCP_SLS_jit import SCP_SLS
    from tests.oracles.casadi_nonlinear_sls import CasadiNonlinearSLS

    N = 5
    m = SpringDamper()
    Q = np.eye(m.nx)
    R = np.eye(m.nu)
    Qf = np.eye(m.nx)
    x0 = np.asarray(x0, dtype=float)

    ob.set_force_stock(True)
    try:
        scp = SCP_SLS(N, Q, R, m, Qf, verbose=False)
        sol_scp = scp.solve(x0)
    finally:
        ob.set_force_stock(False)

    oracle = CasadiNonlinearSLS(N, Q, R, m, Qf)
    oracle.verbose = False
    sol_ref = oracle.solve(x0)

    _assert_primal_match(sol_scp, sol_ref, "SCP_SLS_jit", "CasadiNonlinearSLS")
