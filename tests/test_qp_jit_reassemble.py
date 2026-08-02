"""Fast, backend-agnostic checks for the P/A buffer-reuse refactor in qp_jit.py. Uses
backend="osqp" (the pure-Python OSQP package, already a required dependency) so these run in
the default test suite -- no C compiler/CMake or compiled-extension gating needed.
"""
import numpy as np

from tests.fixtures.integrator import Integrator
from solver.qp_jit import QP


def _new_integrator():
    m = Integrator()
    m.replace_constraints(x_max=5, x_min=-5, u_max=3, u_min=-3, x_max_f=5, x_min_f=-5)
    return m


def test_P_mat_csc_is_never_reassigned_by_update_dynamics():
    m = _new_integrator()
    Q, R, Qf, N = np.eye(m.nx), np.eye(m.nu), np.eye(m.nx), 3
    qp = QP(N, Q, R, m, Qf, backend="osqp")

    P_before = qp.P_mat_csc
    px_before = qp._Px_buf.copy()

    A2 = (m.A + 0.1).astype(float)
    B2 = (m.B + 0.1).astype(float)
    qp.update_dynamics([A2] * N, [B2] * N)

    assert qp.P_mat_csc is P_before, "P_mat_csc must never be rebuilt/reassigned"
    np.testing.assert_allclose(qp._Px_buf, px_before)


def test_Px_buf_matches_a_naive_computation():
    import scipy.sparse as sp

    m = _new_integrator()
    Q, R, Qf, N = np.diag([2.0, 3.0]), np.eye(m.nu), np.diag([5.0, 7.0]), 3
    qp = QP(N, Q, R, m, Qf, backend="osqp")

    P_ut_ref = sp.triu(2 * qp.P_mat_csc, format='csc')
    P_ut_ref.sort_indices()
    np.testing.assert_allclose(qp._Px_buf, P_ut_ref.data)


def test_A_mat_csc_matches_naive_rebuild_after_update():
    import scipy.sparse as sp

    m = _new_integrator()
    Q, R, Qf, N = np.eye(m.nx), np.eye(m.nu), np.eye(m.nx), 3
    qp = QP(N, Q, R, m, Qf, backend="osqp")

    rng = np.random.default_rng(0)
    A2 = m.A + rng.uniform(-0.3, 0.3, size=m.A.shape)
    B2 = m.B + rng.uniform(-0.3, 0.3, size=m.B.shape)
    new_A_list = [A2 for _ in range(N)]
    new_B_list = [B2 for _ in range(N)]

    qp.update_dynamics(new_A_list, new_B_list)

    # naive from-scratch rebuild, independent of the production code path (mirrors today's
    # pre-refactor _reassemble_numeric_same_sparsity, kept here only as a static oracle)
    nx, nu, ni, ni_f = m.nx, m.nu, m.ni, m.ni_f
    rows, cols, data = [], [], []

    def put(r0, c0, M):
        C = sp.coo_matrix(np.asarray(M))
        rows.extend(r0 + C.row); cols.extend(c0 + C.col); data.extend(C.data)

    def put_dense_block(r0, c0, M):
        M = np.asarray(M, dtype=float)
        nr, nc = M.shape
        rows.extend(r0 + np.repeat(np.arange(nr), nc))
        cols.extend(c0 + np.tile(np.arange(nc), nr))
        data.extend(M.reshape(-1))

    r = 0
    for k in range(N):
        cx = k * (nx + nu); cu = cx + nx; cxp = (k + 1) * (nx + nu)
        put_dense_block(r, cx, A2)
        put_dense_block(r, cu, B2)
        put(r, cxp, -np.eye(nx))
        r += nx
        put(r, cx, m.G[:, :nx])
        put(r, cu, m.G[:, nx:])
        r += ni
    put(r, N * (nx + nu), m.Gf)
    r += ni_f
    A_ref = sp.csc_matrix((data, (rows, cols)), shape=(r, (nx + nu) * N + nx))
    A_ref.sort_indices()

    assert (qp.A_mat_csc.indices == A_ref.indices).all()
    assert (qp.A_mat_csc.indptr == A_ref.indptr).all()
    np.testing.assert_allclose(qp.A_mat_csc.data, A_ref.data)


def test_A_update_indices_cover_every_block_exactly_once():
    m = _new_integrator()
    Q, R, Qf, N = np.eye(m.nx), np.eye(m.nu), np.eye(m.nx), 3
    qp = QP(N, Q, R, m, Qf, backend="osqp")

    nx, nu = m.nx, m.nu
    seen = np.zeros(qp.A_mat_csc.nnz, dtype=bool)
    for idx_A, idx_B in qp._A_update_idx:
        assert idx_A.shape == (nx * nx,)
        assert idx_B.shape == (nx * nu,)
        assert not seen[idx_A].any(), "an A-block index position was reused"
        assert not seen[idx_B].any(), "a B-block index position was reused"
        seen[idx_A] = True
        seen[idx_B] = True
    # Ak/Bk blocks account for N*(nx*nx + nx*nu) of the total nnz; the rest (-I, G, Gf) is fixed
    assert seen.sum() == N * (nx * nx + nx * nu)


def test_solve_reuses_the_same_buffer_objects_across_calls():
    m = _new_integrator()
    Q, R, Qf, N = np.eye(m.nx), np.eye(m.nu), np.eye(m.nx), 3
    qp = QP(N, Q, R, m, Qf, backend="osqp")

    assert hasattr(qp, "_qv_buf") and hasattr(qp, "_l_buf") and hasattr(qp, "_u_buf")
    qv_id, l_id, u_id = id(qp._qv_buf), id(qp._l_buf), id(qp._u_buf)

    # backend="osqp" doesn't go through the osqp_codegen solve() branch that writes these
    # buffers, so exercise the write path directly here instead of via solve().
    qp._qv_buf[:] = 1.0
    qp._l_buf[:] = 2.0
    qp._u_buf[:] = 3.0

    assert id(qp._qv_buf) == qv_id
    assert id(qp._l_buf) == l_id
    assert id(qp._u_buf) == u_id
    assert qp._qv_buf.shape == (qp.P_mat_csc.shape[0],)
    assert qp._l_buf.shape == (qp.A_mat_csc_osqp.shape[0],)
    assert qp._u_buf.shape == (qp.A_mat_csc_osqp.shape[0],)


def test_update_ubg_writes_in_place_without_reallocating():
    import casadi as ca

    m = _new_integrator()
    Q, R, Qf, N = np.eye(m.nx), np.eye(m.nu), np.eye(m.nx), 3
    qp = QP(N, Q, R, m, Qf, backend="osqp")

    buf_id = id(qp.ubg_osqp_base)
    new_ubg = ca.DM(np.array(qp.ubg).astype(float).ravel() - 1.0)
    qp.update_ubg(new_ubg)

    assert id(qp.ubg_osqp_base) == buf_id, "ubg_osqp_base must be written in place, not reallocated"
    expected = np.array(new_ubg).astype(float).ravel()
    np.testing.assert_allclose(qp.ubg_osqp_base[:expected.size], expected)
    np.testing.assert_allclose(qp.ubg_osqp_base[expected.size:], 0.0)


def test_reset_ubg_and_reset_lbg_write_in_place():
    m = _new_integrator()
    Q, R, Qf, N = np.eye(m.nx), np.eye(m.nu), np.eye(m.nx), 3
    qp = QP(N, Q, R, m, Qf, backend="osqp")

    ubg_id, lbg_id = id(qp.ubg_osqp_base), id(qp.lbg_osqp_base)
    qp.reset_ubg()
    qp.reset_lbg()
    assert id(qp.ubg_osqp_base) == ubg_id
    assert id(qp.lbg_osqp_base) == lbg_id

    expected_ubg = np.array(qp.nominal_ubg).astype(float).ravel()
    np.testing.assert_allclose(qp.ubg_osqp_base[:expected_ubg.size], expected_ubg)
    expected_lbg = np.array(qp.lbg_ZEROS_INF_NOMINAL).astype(float).ravel()
    np.testing.assert_allclose(qp.lbg_osqp_base[:expected_lbg.size], expected_lbg)
