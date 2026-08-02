"""Console-output checks for the unified verbose table (fast_SLS standalone + SCP_SLS-nested).
These are display/formatting checks, not numerical-correctness checks -- see CLAUDE.md on why
this repo otherwise has no test suite for solver correctness.
"""
import numpy as np
import pytest

from tests.fixtures.integrator import Integrator


def test_fast_sls_standalone_has_no_tab_indent_and_no_external_table_by_default(capsys):
    import solver.osqp_backend as ob
    from solver.fast_SLS_jit import fast_SLS

    m = Integrator()
    N, Q, R, Qf = 3, np.eye(m.nx), np.eye(m.nu), np.eye(m.nx)

    ob.set_force_stock(True)
    try:
        fast = fast_SLS(N, Q, R, m, Qf)
        assert fast._external_table is None
        assert fast._external_group_label is None
        assert not hasattr(fast, "_tab"), "the old tab_indent/_tab mechanism must be gone"

        fast.solve(np.array([1.0, -0.5]))
    finally:
        ob.set_force_stock(False)

    out = capsys.readouterr().out
    assert "\t" * 25 not in out, "no 25-tab indent block should ever be printed again"
    assert "it (SLS)" in out, "standalone fast_SLS should still print its own table header"


def test_scp_sls_does_not_force_disable_child_verbose():
    import solver.osqp_backend as ob
    from solver.SCP_SLS_jit import SCP_SLS
    from tests.fixtures.spring_damper import SpringDamper

    # SCP_SLS requires a CasADi-symbolic nonlinear Model (its jacobian setup calls
    # self.m.ddyn(x, u) with no time index); the LTI/LTV Integrator fixture used by the
    # standalone fast_SLS test above does not fit that shape (its ddyn takes a required
    # `k`), so SCP_SLS-based tests use SpringDamper instead -- same as
    # tests/test_sls_cross_validation.py.
    m = SpringDamper()
    N, Q, R, Qf = 3, np.eye(m.nx), np.eye(m.nu), np.eye(m.nx)

    ob.set_force_stock(True)
    try:
        scp = SCP_SLS(N, Q, R, m, Qf)
        assert scp.verbose is True
        assert scp.fast_SLS_solver.verbose is True, (
            "SCP_SLS must not force its child's verbose to False anymore"
        )
    finally:
        ob.set_force_stock(False)


def test_scp_sls_prints_one_merged_table_with_no_tab_indent(capsys):
    import solver.osqp_backend as ob
    from solver.SCP_SLS_jit import SCP_SLS
    from tests.fixtures.spring_damper import SpringDamper

    m = SpringDamper()
    N, Q, R, Qf = 5, np.eye(m.nx), np.eye(m.nu), np.eye(m.nx)

    ob.set_force_stock(True)
    try:
        scp = SCP_SLS(N, Q, R, m, Qf)
        sol = scp.solve(np.array([3.0, -1.5]))
    finally:
        ob.set_force_stock(False)

    assert sol.get('success') in (True, False)  # solved to completion either way
    out = capsys.readouterr().out
    assert "\t" * 25 not in out, "no 25-tab indent block should ever be printed again"
    assert "SCP it" in out and "SLS it" in out, "merged header must show both group columns"
    assert "->" in out, "at least one SCP summary row (closing a group) should be printed"
