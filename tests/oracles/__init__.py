"""Independent, general-purpose solver oracles for cross-validating the production
compiled-OSQP route. Test-only — not part of the shipped `solver/` package.

- `cvxpy_SLS`            -- the linear SLS problem as one convex SOCP (cvxpy + Clarabel).
- `casadi_nonlinear_sls` -- the nonlinear SLS problem as one monolithic NLP (casadi + ipopt).
- `qp_clarabel`         -- rebuilds the nominal QP that `solver.qp_jit` hands OSQP and solves it
                           with cvxpy + Clarabel (primal + dual), for the QP-level equivalence test.

No MOSEK. Clarabel ships with cvxpy (BSD); ipopt ships inside the casadi wheel.
"""
