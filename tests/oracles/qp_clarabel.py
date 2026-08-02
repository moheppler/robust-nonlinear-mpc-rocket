"""General-purpose QP oracle: rebuild the nominal QP that ``solver.qp_jit.QP`` hands OSQP and
solve it with cvxpy + Clarabel (open, BSD; replaces MOSEK).

This is the reference for the QP-level equivalence test: it consumes the *exact* data OSQP
receives — ``P = 2·P_mat_csc``, ``q = q_cost_lin``, the constraint rows of ``A_mat_csc`` with
bounds ``lbg``/``ubg``, and the ``x(0) = −x0`` pin — and returns the primal **and** dual solution
in OSQP's convention so both can be compared against the compiled/stock solver.

Dual convention. OSQP's multiplier ``y`` for ``l ≤ A z ≤ u`` satisfies the stationarity
``2P·z + q + Aᵀy = 0``. cvxpy returns per-constraint duals; mapping them back to one multiplier
per original row of ``A_mat_csc`` gives, from the KKT of ``min f s.t. g≤0, h=0``
(``∇f + Aeqᵀν + Aubᵀλ_ub − Albᵀλ_lb = 0``):

    y[eq_rows]  =  +ν       (equality / dynamics rows,   Aeq z == beq)
    y[ub_rows]  =  +λ_ub    (upper-only inequality rows,  Aub z <= u)
    y[lb_rows]  =  −λ_lb    (lower-only inequality rows,  Alb z >= l)

The overall sign of cvxpy's ``dual_value`` is pinned empirically against the stock OSQP backend
(see ``tests/test_qp_vs_general_solver.py``) and captured by ``_SIGN`` below.
"""
import numpy as np

# cvxpy 1.7 / Clarabel return duals with the same sign as the KKT derivation above; verified
# against stock OSQP on the Integrator QP (max|Δdual| ~1e-6). Flip to -1.0 if a future
# cvxpy/solver version inverts the convention (the equivalence test would catch it).
_SIGN = 1.0


def solve_qp_reference(qp, x0, solver="CLARABEL", eq_tol=1e-9):
    """Solve ``qp``'s current nominal QP with cvxpy+``solver``; return an OSQP-shaped result dict.

    ``qp`` is a built ``solver.qp_jit.QP``; ``x0`` pins ``x(0) = −x0`` exactly as ``qp.solve(x0)``.
    Returns keys: ``success``, ``primal_vec``, ``primal_x``, ``primal_u``, ``dual_vec``, ``cost``.
    """
    import cvxpy as cp

    nx = qp.m.nx
    A = qp.A_mat_csc.tocsr()
    m0, nv = A.shape

    P2 = np.asarray((2.0 * qp.P_mat_csc).todense(), dtype=float)
    P2 = 0.5 * (P2 + P2.T)  # numerically symmetric for quad_form
    q = np.asarray(qp.q_cost_lin, dtype=float).ravel()
    l = np.asarray(qp.lbg, dtype=float).ravel()
    u = np.asarray(qp.ubg, dtype=float).ravel()
    x0v = np.asarray(x0, dtype=float).ravel()

    fin_l = np.isfinite(l)
    fin_u = np.isfinite(u)
    eq = fin_l & fin_u & (np.abs(u - l) <= eq_tol)
    if np.any(fin_l & fin_u & ~eq):
        raise ValueError("qp_clarabel: genuinely double-bounded rows are not supported")
    ub = fin_u & ~fin_l  # upper-only inequalities
    lb = fin_l & ~fin_u  # lower-only inequalities

    y = cp.Variable(nv)
    obj = 0.5 * cp.quad_form(y, cp.psd_wrap(P2)) + q @ y

    cons = []
    con_eq = con_ub = con_lb = None
    if eq.any():
        con_eq = A[eq] @ y == u[eq]
        cons.append(con_eq)
    if ub.any():
        con_ub = A[ub] @ y <= u[ub]
        cons.append(con_ub)
    if lb.any():
        con_lb = A[lb] @ y >= l[lb]
        cons.append(con_lb)
    cons.append(y[:nx] == -x0v)  # x(0) = -x0, matching qp_jit

    prob = cp.Problem(cp.Minimize(obj), cons)
    prob.solve(solver=getattr(cp, solver))

    success = prob.status in ("optimal", "optimal_inaccurate")
    if not success or y.value is None:
        return {"success": False, "status": prob.status}

    yv = np.asarray(y.value, dtype=float).ravel()
    dual = np.zeros(m0)
    if con_eq is not None:
        dual[eq] = _SIGN * np.asarray(con_eq.dual_value, dtype=float).ravel()
    if con_ub is not None:
        dual[ub] = _SIGN * np.asarray(con_ub.dual_value, dtype=float).ravel()
    if con_lb is not None:
        dual[lb] = -_SIGN * np.asarray(con_lb.dual_value, dtype=float).ravel()

    primal_x, primal_u = qp.unpack_primal_nominal(yv)
    cost = float(yv @ (qp.P_mat_csc @ yv) + q @ yv)
    return {
        "success": True,
        "primal_vec": yv,
        "primal_x": primal_x,
        "primal_u": primal_u,
        "dual_vec": dual,
        "cost": cost,
    }
