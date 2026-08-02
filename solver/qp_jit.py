import numpy as np
import casadi as ca
import scipy.sparse as sp
from scipy.io import savemat
from pathlib import Path
import os
import shutil
import stat
import time

from solver.ocp import OCP
from dyn.LTV import LTV
from dyn.LTI import LTI

try:
    import osqp  # optional: native OSQP path + codegen
    _HAS_OSQP = True
except Exception:
    _HAS_OSQP = False

EPSILON = 1e-10


def _clear_readonly_and_retry(func, path, exc_info):
    """shutil.rmtree onerror hook: git marks pack files read-only, which Windows'
    os.unlink refuses to delete (unlike POSIX). Clear the attribute and retry."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


class QP(OCP):
    """
    Minimal, fast QP front-end for fast-SLS with three interchangeable backends:
      • "casadi_osqp"   — CasADi conic using OSQP plugin (no codegen)
      • "osqp"          — Native OSQP Python API (fast updates)
      • "osqp_codegen"  — Compiled OSQP Python extension (generated with parameters='matrices')

    Contract with fast-SLS:
      - build once; then only q (linear term), l/u (bounds), and numeric A/P update per iteration
      - expose: solve(x0), update_dynamics(...), update_ubg(...), reset_ubg(),
                update_q_cost_lin(...), add_q_cost_lin(...), reset_q_cost_lin()

    backend="osqp_codegen" caveat: the compiled extension's module-level API
    (osqp_generated) is a process-wide singleton -- one shared C solver state, not one
    per QP instance. P is pushed once at construction (Q/R/Qf never change for the
    instance's life); A is pushed once at construction and again whenever
    update_dynamics() changes it. Only one such instance may be actively solved at a
    time without an intervening update_dynamics() call -- constructing or updating a
    second backend="osqp_codegen" instance in the same process silently makes the first
    one's next solve() use the second instance's A/P instead of its own.
    """

    # backend="osqp_codegen" is a process-wide singleton (see docstring above) -- track
    # which QP instance most recently pushed A/P to a given compiled module, so solve()
    # can detect and refuse to silently solve with another instance's stale data.
    _codegen_instance_counter = 0
    _codegen_singleton_owner: dict = {}

    def __init__(self, N, Q, R, m, Qf, *, backend="casadi_osqp", codegen_module=None, verbose=True,
                 export_standard_QP: bool = False, export_dir: str | Path | None = None):
        super().__init__(N, Q, R, m, Qf)
        assert backend in {"casadi_osqp", "osqp", "osqp_codegen"}
        self.backend = backend
        QP._codegen_instance_counter += 1
        self._instance_id = QP._codegen_instance_counter
        self.codegen_module = codegen_module
        self.verbose = verbose

        # export options
        self.export_standard_qp = bool(export_standard_QP)
        self.export_dir = Path(export_dir) if export_dir is not None else Path("build/quadprog_exports")
        self._export_counter = 0

        # structures
        self.A_mat_csc: sp.csc_matrix | None = None
        self.P_mat_csc: sp.csc_matrix | None = None
        self.A_mat_cas: ca.DM | None = None
        self.P_mat_cas: ca.DM | None = None

        # vectors
        self.lbg = None
        self.ubg = None
        self.nominal_ubg = None
        self.nominal_lbg = None
        self.q_cost_lin = None

        # casadi helpers
        self.lbx_fun = None
        self.ubx_fun = None

        # solver objects
        self.solver_qp = None     # CasADi conic
        self._osqp_prob = None    # osqp.OSQP() or codegen module/object

        # build once
        self.initialize_list_dynamics()
        self._assemble_structures()
        self._init_backend()

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------
    def _assemble_structures(self):
        nx, nu, ni, ni_f, N = self.m.nx, self.m.nu, self.m.ni, self.m.ni_f, self.N

        rows = []
        cols = []
        data = []

        def put(r0, c0, M):
            C = sp.coo_matrix(M)
            rows.extend(r0 + C.row)
            cols.extend(c0 + C.col)
            data.extend(C.data)

        def put_dense_block(r0, c0, M):
            """Insert a dense block M (including explicit zeros) to freeze sparsity pattern."""
            M = np.asarray(M)
            nr, nc = M.shape
            # generate all coordinates for the block
            rr = r0 + np.repeat(np.arange(nr), nc)
            cc = c0 + np.tile(np.arange(nc), nr)
            rows.extend(rr)
            cols.extend(cc)
            data.extend(M.reshape(-1))

        r = 0
        for k in range(N):
            Ak = self.A_list[k]
            Bk = self.B_list[k]
            # dynamics: [A B -I] on (x_k, u_k, x_{k+1})
            cx = k * (nx + nu)
            cu = cx + nx
            cxp = (k + 1) * (nx + nu)
            # Use dense placement for A and B so their sparsity never changes across updates
            put_dense_block(r, cx, Ak)
            put_dense_block(r, cu, Bk)
            put(r, cxp, -np.eye(nx))
            r += nx
            # ineq: G[x;u] <= g_k
            G = self.m.G
            put(r, cx, G[:, :nx])
            put(r, cu, G[:, nx:])
            r += ni
        # terminal ineq: Gf x_N <= g_f
        put(r, N * (nx + nu), self.m.Gf)
        r += ni_f

        A = sp.csc_matrix((data, (rows, cols)), shape=(r, (nx + nu) * N + nx))

        # Quadratic cost: blkdiag(Q,R,...,Qf)
        Qs = sp.csc_matrix(self.Q)
        Rs = sp.csc_matrix(self.R)
        Qf = sp.csc_matrix(self.Qf)
        QR = sp.bmat([[Qs, None], [None, Rs]], format='csc')
        P = sp.block_diag([QR for _ in range(N)] + [Qf], format='csc')

        self.A_mat_csc = A
        self.P_mat_csc = P
        self.A_mat_cas = ca.DM(A)
        self.P_mat_cas = ca.DM(P)

        # --- freeze CSC ordering for OSQP/codegen ---
        self.A_mat_csc.sort_indices()
        self.P_mat_csc.sort_indices()

        # Q/R/Qf never change after construction, so P never changes either -- freeze its
        # upper-triangular data once, here. Nothing downstream may reassign self.P_mat_csc,
        # self.P_mat_cas, or recompute self._Px_buf; see _reassemble_numeric_same_sparsity.
        P_ut0 = sp.triu(2 * self.P_mat_csc, format='csc')
        P_ut0.sort_indices()
        self._Px_buf = np.ascontiguousarray(P_ut0.data, dtype=np.float64)

        # bounds lbg/ubg
        if isinstance(self.m, LTI):
            ub = ca.kron(ca.DM.ones(N, 1), ca.vertcat(ca.DM.zeros(nx, 1), self.m.g))
            ub = ca.vertcat(ub, self.m.gf )
        elif isinstance(self.m, LTV):
            pieces = [ca.vertcat(ca.DM.zeros(nx, 1), g ) for g in self.m.g_list[:-1]]
            ub = ca.vertcat(*pieces)
            ub = ca.vertcat(ub, self.m.g_list[-1] )
        else:
            raise ValueError("Model must be LTI or LTV")
        self.nominal_ubg = ub
        self.ubg = ub

        lb = ca.kron(ca.DM.ones(N, 1), ca.vertcat(ca.DM.zeros(nx, 1), -ca.DM.inf(self.m.ni, 1)))
        self.lbg = ca.vertcat(lb, -ca.DM.inf(self.m.ni_f, 1))
        self.lbg_ZEROS_INF_NOMINAL = self.lbg

        # casadi-only variable box bounds for x0 (others free)
        x0 = ca.SX.sym('x0', nx)
        # Enforce x(0) + x0 = 0  => x(0) = -x0 (tight box around -x0)
        self.ubx_fun = ca.Function(
            'ubx_fun', [x0],
            [ca.vertcat(-x0 + EPSILON, ca.DM.inf(N * (nx + nu), 1))],
            {'jit': False, 'compiler': 'shell'}
        )
        self.lbx_fun = ca.Function(
            'lbx_fun', [x0],
            [ca.vertcat(-x0 - EPSILON, -ca.DM.inf(N * (nx + nu), 1))],
            {'jit': False, 'compiler': 'shell'}
        )

        self.reset_q_cost_lin()
        self._compute_A_update_indices()
        self._qv_buf = np.zeros((nx + nu) * N + nx, dtype=np.float64)

        # fingerprints to ensure sparsity unchanged when updating numerics
        self._A_sig = (tuple(self.A_mat_csc.indices), tuple(self.A_mat_csc.indptr))
        self._P_sig = (tuple(self.P_mat_csc.indices), tuple(self.P_mat_csc.indptr))

        # --- extra rows to pin x0 in OSQP ---
        nx, nu, N = self.m.nx, self.m.nu, self.N
        nv = (nx + nu) * N + nx

        I0 = sp.hstack([sp.eye(nx, format='csc'),
                        sp.csc_matrix((nx, nv - nx))], format='csc')  # acts on the first nx vars (x0)

        self.A_mat_csc_osqp = sp.vstack([self.A_mat_csc, I0], format='csc')
        self.A_mat_csc_osqp.sort_indices()

        # base l/u without x0 values yet (filled per-solve). Preallocated once here at the
        # fixed final size (A_mat_csc_osqp's row count). Every writer of these buffers
        # (_reassemble_numeric_same_sparsity below, and update_ubg/reset_ubg/reset_lbg)
        # writes in place via slice-assignment -- none of them reassign the buffer object,
        # so it stays the same persistent array for the object's whole lifetime.
        self.lbg_osqp_base = np.zeros(self.A_mat_csc_osqp.shape[0], dtype=np.float64)
        lbg_flat = np.array(self.lbg).astype(float).ravel()
        self.lbg_osqp_base[:lbg_flat.size] = lbg_flat
        self.ubg_osqp_base = np.zeros(self.A_mat_csc_osqp.shape[0], dtype=np.float64)
        ubg_flat = np.array(self.ubg).astype(float).ravel()
        self.ubg_osqp_base[:ubg_flat.size] = ubg_flat

        self._l_buf = np.zeros(self.A_mat_csc_osqp.shape[0], dtype=np.float64)
        self._u_buf = np.zeros(self.A_mat_csc_osqp.shape[0], dtype=np.float64)

    def _compute_A_update_indices(self):
        """One-time setup: precompute, for each SCP step k, exactly where Ak's and Bk's
        row-major-flattened values land in the frozen self.A_mat_csc.data array. Lets
        _reassemble_numeric_same_sparsity() write new dynamics values via plain numpy
        scatter-writes instead of rebuilding the whole sparse matrix from scratch.

        Standard "tagged permutation" trick: build the identical (row, col) structure with
        placeholder values 0..nnz-1 in the exact same insertion order _assemble_structures
        uses, sort it the same way scipy sorts the real matrix, and read off where each
        placeholder landed. Because scipy's CSC sort_indices() ordering depends only on
        (row, col) pairs -- never on the data values -- the real matrix (built from the same
        (row, col) pairs) is guaranteed to land in the identical relative order.
        """
        nx, nu, ni, ni_f, N = self.m.nx, self.m.nu, self.m.ni, self.m.ni_f, self.N

        rows, cols, tags = [], [], []
        a_ranges = []
        b_ranges = []

        def put(r0, c0, M):
            C = sp.coo_matrix(np.asarray(M))
            n = len(C.data)
            rows.extend(r0 + C.row)
            cols.extend(c0 + C.col)
            tags.extend(range(len(tags), len(tags) + n))

        def put_dense_block(r0, c0, shape):
            nr, nc = shape
            n = nr * nc
            rows.extend(r0 + np.repeat(np.arange(nr), nc))
            cols.extend(c0 + np.tile(np.arange(nc), nr))
            start = len(tags)
            tags.extend(range(start, start + n))
            return start, start + n

        r = 0
        for k in range(N):
            cx = k * (nx + nu)
            cu = cx + nx
            cxp = (k + 1) * (nx + nu)
            a_start, a_end = put_dense_block(r, cx, (nx, nx))
            b_start, b_end = put_dense_block(r, cu, (nx, nu))
            a_ranges.append((a_start, a_end))
            b_ranges.append((b_start, b_end))
            put(r, cxp, -np.eye(nx))
            r += nx
            G = self.m.G
            put(r, cx, G[:, :nx])
            put(r, cu, G[:, nx:])
            r += ni
        put(r, N * (nx + nu), self.m.Gf)
        r += ni_f

        nnz = len(tags)
        tagged = sp.csc_matrix((tags, (rows, cols)), shape=(r, (nx + nu) * N + nx))
        tagged.sort_indices()

        # inv_perm[original_flat_index] = position in the sorted CSC data array
        inv_perm = np.empty(nnz, dtype=np.int64)
        inv_perm[np.asarray(tagged.data, dtype=np.int64)] = np.arange(nnz)

        self._A_update_idx = [
            (inv_perm[a0:a1], inv_perm[b0:b1])
            for (a0, a1), (b0, b1) in zip(a_ranges, b_ranges)
        ]

    def _reassemble_numeric_same_sparsity(self):
        """Recompute numeric values of A given the current A_list/B_list. P is invariant and
        never rebuilt here -- see _assemble_structures/_init_backend. Uses the CSC
        scatter-index map computed once at setup (self._A_update_idx) to write new Ak/Bk
        values directly into self.A_mat_csc.data, instead of rebuilding the sparse matrix
        from scratch on every SCP iteration.
        """
        N = self.N
        for k in range(N):
            idx_A, idx_B = self._A_update_idx[k]
            self.A_mat_csc.data[idx_A] = np.asarray(self.A_list[k], dtype=float).ravel()
            self.A_mat_csc.data[idx_B] = np.asarray(self.B_list[k], dtype=float).ravel()

        if self.backend == "casadi_osqp":
            self.A_mat_cas = ca.DM(self.A_mat_csc)

        # OSQP variant (with extra x0 rows) -- cheap vstack of two already-built CSC
        # matrices (no Python-level rebuild of the much larger dynamics block).
        nx = self.m.nx
        nv = self.A_mat_csc.shape[1]
        I0 = sp.hstack([sp.eye(nx, format='csc'),
                        sp.csc_matrix((nx, nv - nx))], format='csc')
        self.A_mat_csc_osqp = sp.vstack([self.A_mat_csc, I0], format='csc')
        self.A_mat_csc_osqp.sort_indices()

        # refresh OSQP base bounds for extra rows
        lbg_flat = np.asarray(self.lbg, dtype=np.float64).ravel()
        self.lbg_osqp_base[:lbg_flat.size] = lbg_flat
        ubg_flat = np.asarray(self.ubg, dtype=np.float64).ravel()
        self.ubg_osqp_base[:ubg_flat.size] = ubg_flat

        # ubg: stage constraints from self.g_list; terminal from the model's per-stage g_list
        # (set by SCP/fast-SLS tightening) when present, else the plain-LTI terminal self.g_list[-1].
        pieces = [ca.vertcat(ca.DM.zeros(nx, 1), g) for g in self.g_list[:-1]]
        ub = ca.vertcat(*pieces)
        term = self.m.g_list[-1] if getattr(self.m, "g_list", None) is not None else self.g_list[-1]
        ub = ca.vertcat(ub, term)
        self.update_ubg(ub)
        self.reset_lbg()

    # ------------------------------------------------------------------
    # Backends
    # ------------------------------------------------------------------
    def _init_backend(self):
        if self.backend == "casadi_osqp":
            opts = {'osqp': {'verbose': False, 'polish': True}}
            self.solver_qp = ca.conic('solver', 'osqp',
                                      {'a': self.A_mat_cas.sparsity(), 'h': self.P_mat_cas.sparsity()},
                                      opts)
        elif self.backend == "osqp":
            if not _HAS_OSQP:
                raise RuntimeError("OSQP not installed. pip install osqp")
            prob = osqp.OSQP()
            prob.setup(
                P=2 * self.P_mat_csc,  # match CasADi’s h = 2*H
                q=np.zeros(self.P_mat_csc.shape[0]),
                A=self.A_mat_csc_osqp,  # includes x0 rows
                l=self.lbg_osqp_base,  # placeholders; updated per solve
                u=self.ubg_osqp_base,
                # --- Accuracy / robustness ---
                eps_abs=1e-9,  # tighten as needed (1e-9 if stable)
                eps_rel=1e-9,
                scaled_termination=False,  # check tolerances on ORIGINAL (unscaled) problem
                max_iter=50000,
                check_termination=1,  # check every iteration
                # --- Extras that help ---
                polish=True,
                polish_refine_iter=20,
                warm_start=False,  # reuse solution for the 2nd pass (below)
                adaptive_rho=True,  # keep OSQP’s automatic rho updates
                verbose=False
            )
            self._osqp_prob = prob
        elif self.backend == "osqp_codegen":
            if self.codegen_module is None:
                raise ValueError("backend='osqp_codegen' needs codegen_module")
            mod = __import__(self.codegen_module)
            self._osqp_mod = mod

            # Always compute and store the reference CSC patterns used at init
            P_ut = sp.triu(2 * self.P_mat_csc, format='csc')
            P_ut.sort_indices()
            A_osqp = self.A_mat_csc_osqp
            A_osqp.sort_indices()
            self._store_cg_signatures(P_ut, A_osqp)

            if hasattr(mod, 'OSQP'):
                self._osqp_prob = mod.OSQP()

                # Build matrices exactly as used at codegen time
                self._osqp_prob.setup(
                    P=P_ut,
                    q=np.zeros(P_ut.shape[0]),
                    A=A_osqp,
                    l=self.lbg_osqp_base,
                    u=self.ubg_osqp_base,
                    polishing=True, warm_starting=False, verbose=False
                )
            else:
                # Module-level API (functions like update_data_mat/update_data_vec/solve).
                # This is a process-wide singleton -- the SAME compiled C solver state is
                # shared by every QP instance in this process, not one per instance. Two
                # things must be pushed here, once, so a freshly constructed instance is
                # immediately self-consistent even if solve() is called before its first
                # update_dynamics() (which is exactly what several tests do, and which
                # otherwise leaves this instance solving with A/P left over from whatever
                # QP instance touched the singleton last):
                #  - P: solver/build_fast.py codegens this extension using a generic
                #    placeholder Q/Qf, not necessarily THIS instance's actual Q/R/Qf, so the
                #    codegen-time-baked-in P can be numerically wrong for this instance.
                #  - A: correct as of construction (self.A_mat_csc_osqp is freshly built in
                #    _assemble_structures, just before this runs), but never implicitly
                #    correct on the compiled side until pushed.
                # P never changes again after this (Q/R/Qf are fixed for the instance's
                # life), so this is the only place P is ever pushed. A gets pushed again
                # whenever update_dynamics() actually changes it.
                self._osqp_prob = mod
                Ax0 = np.ascontiguousarray(self.A_mat_csc_osqp.data, dtype=np.float64)
                pushed = mod.update_data_mat(P_x=self._Px_buf, A_x=Ax0)
                if pushed != 0:
                    raise RuntimeError(
                        f"Initial update_data_mat(P_x=..., A_x=...) failed with code {pushed}. "
                        "This means the compiled solver in build/osqp_fast was codegen'd for a "
                        "different horizon N (or different dynamics/constraints) than the one "
                        "you're running now. Rebuild it to match: "
                        "`rnmpc build-fast <system> --N <k>`."
                    )
                QP._codegen_singleton_owner[self.codegen_module] = self._instance_id

        else:
            raise ValueError(f"Unknown backend {self.backend}")

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    def solve(self, x0):
        nx, nu, ni, ni_f, N = self.m.nx, self.m.nu, self.m.ni, self.m.ni_f, self.N

        if self.backend == "casadi_osqp":
            try:
                sol = self.solver_qp(a=self.A_mat_cas,
                                     h=2 * self.P_mat_cas,  # CasADi/OSQP: 1/2 x' P x + q' x
                                     lba=self.lbg,
                                     uba=self.ubg,
                                     g=self.q_cost_lin,
                                     lbx=self.lbx_fun(x0),
                                     ubx=self.ubx_fun(x0))
            except Exception as e:
                if self.verbose:
                    print(e)
                    print('QP: infeasible (CasADi/OSQP).')
                return {'success': False}
            return self._pack_solution(sol, nx, nu, N, ni_f)

        if self.backend == "osqp":
            prob = self._osqp_prob

            # linear term
            qv = np.array(self.q_cost_lin).astype(float).ravel()

            # bounds with x0 box: last nx rows correspond to I*x0
            nx = self.m.nx
            l_curr = np.array(self.lbg, dtype=float).ravel()
            u_curr = np.array(self.ubg, dtype=float).ravel()
            # append placeholders for x0 equality rows
            l = np.concatenate([l_curr, np.zeros(nx)])
            u = np.concatenate([u_curr, np.zeros(nx)])

            x0v = np.array(x0, dtype=float).ravel()
            # Enforce x(0) + x0 = 0  => A*y = -x0
            l[-nx:] = -x0v - EPSILON
            u[-nx:] = -x0v + EPSILON

            # sanitize infinities to large magnitudes for OSQP
            BIG = 1e20
            l[np.isneginf(l)] = -BIG
            l[np.isposinf(l)] = BIG
            u[np.isposinf(u)] = BIG
            u[np.isneginf(u)] = -BIG

            # ensure latest dynamics values are used (pattern unchanged)
            A_osqp = self.A_mat_csc_osqp
            A_osqp.sort_indices()
            prob.update(Ax=A_osqp.data, q=qv, l=l, u=u)
            t0 = time.perf_counter()
            res = prob.solve()
            t1 = time.perf_counter()
            time_qp = (t1 - t0) * 1e3

            if res.info.status_val not in (1, 2):
                if self.verbose:
                    print(f"QP(OSQP): {res.info.status}")
                return {'success': False}
            sol = {'x': res.x, 'lam_a': res.y, 'cost': res.info.obj_val, 'success': True, 'time_ms': time_qp}
            return self._pack_solution(sol, nx, nu, N, ni_f)

        if self.backend == "osqp_codegen":
            cg = self._osqp_prob
            nx = self.m.nx

            # ---- vectors (with x0 box in last nx rows), written into persistent buffers ----
            self._qv_buf[:] = np.asarray(self.q_cost_lin, dtype=np.float64).ravel()

            l_curr = np.asarray(self.lbg, dtype=np.float64).ravel()
            u_curr = np.asarray(self.ubg, dtype=np.float64).ravel()
            self._l_buf[:l_curr.size] = l_curr
            self._u_buf[:u_curr.size] = u_curr

            x0v = np.asarray(x0, dtype=np.float64).ravel()
            # Enforce x(0) + x0 = 0  => A*y = -x0
            self._l_buf[-nx:] = -x0v - EPSILON
            self._u_buf[-nx:] = -x0v + EPSILON

            BIG = 1e20
            self._l_buf[np.isneginf(self._l_buf)] = -BIG
            self._l_buf[np.isposinf(self._l_buf)] = BIG
            self._u_buf[np.isposinf(self._u_buf)] = BIG
            self._u_buf[np.isneginf(self._u_buf)] = -BIG

            # ---- sanity sizes ----
            assert self._qv_buf.size == self.P_mat_csc.shape[0]
            assert self._l_buf.size == self.A_mat_csc_osqp.shape[0]
            assert self._u_buf.size == self.A_mat_csc_osqp.shape[0]

            # ---- push numerics: q/l/u only. A was already pushed in update_dynamics() (it
            # doesn't change between fast-SLS/RTI inner iterations); P was pushed once at
            # setup (_init_backend) since Q/R/Qf never change. ----
            self._cg_push_updates(cg, self._qv_buf, self._l_buf, self._u_buf)

            # ---- guard the process-wide singleton: refuse to solve with another QP
            # instance's stale A/P rather than silently producing wrong numbers ----
            owner = QP._codegen_singleton_owner.get(self.codegen_module)
            if owner != self._instance_id:
                raise RuntimeError(
                    f"backend='osqp_codegen' singleton conflict on module "
                    f"'{self.codegen_module}': this QP instance's A/P are no longer current "
                    "in the compiled solver -- another QP instance sharing the same compiled "
                    "module pushed to it more recently. Only one backend='osqp_codegen' "
                    "instance may be actively solved at a time without an intervening "
                    "update_dynamics() call on it (see the QP class docstring)."
                )

            # ---- solve & normalize result ----
            t0 = time.perf_counter()
            res = cg.solve()
            t1 = time.perf_counter()
            time_qp = (t1 - t0) * 1e3
            ok = False
            if isinstance(res, tuple):
                x = res[0] if len(res) >= 1 else None
                y = res[1] if len(res) >= 2 else None
                # osqp_generated returns: (x, y, status_code, iter, run_time)
                status_code = int(res[2]) if len(res) >= 3 else -1
                ok = (status_code == 0) and (x is not None) and (y is not None)
                status_val = status_code
            else:
                x = getattr(res, "x", None)
                y = getattr(res, "y", None)
                info = getattr(res, "info", None)
                status_val = int(getattr(info, "status_val", 0) if info is not None else 0)
                ok = (status_val in (1, 2)) and (x is not None) and (y is not None)

            if ok:
                cost = float(x @ (self.P_mat_csc @ x) + self._qv_buf @ x)
                sol = {'x': x, 'lam_a': y, 'cost': cost, 'success': True, 'time_ms': time_qp}
                ret = self._pack_solution(sol, self.m.nx, self.m.nu, self.N, self.m.ni_f)
                if self.export_standard_qp and ret.get('success', True):
                    try:
                        self._export_quadprog(x0, ret)
                    except Exception as ex:
                        if self.verbose:
                            print(f"Export (quadprog) failed: {ex}")
                return ret

            if self.verbose:
                print(f"QP(osqp_codegen): status={status_val} (failed)")
            return {'success': False}

    def _pack_solution(self, sol, nx, nu, N, ni_f):
        # ---- primal ----
        y = np.asarray(sol['x']).copy().reshape(-1)
        primal_x, primal_u = self.unpack_primal_nominal(y)

        # ---- duals ----
        lam = np.asarray(sol['lam_a']).copy().reshape(-1)

        # OSQP backend has extra last-nx duals for x0 equality rows -> remove them
        if self.backend in ("osqp", "osqp_codegen"):
            lam = lam[:-nx]

        dual_mu_f = lam[-ni_f:]  # terminal ineq duals
        dual_non_term = lam[:-ni_f].reshape(N, nx + self.m.ni)
        dual_mu_stage = dual_non_term[:, nx:].T  # shape (ni, N)

        return {
            'success': True,
            'primal_vec': y,
            'primal_x': primal_x,
            'primal_u': primal_u,
            'dual_vec': lam,
            'dual_mu': dual_mu_stage,
            'dual_mu_f': dual_mu_f,  # 1D; make .reshape(1,-1) if you prefer row
            'cost': float(sol['cost']),
            'time_ms': sol.get('time_ms', None),
        }

    # ------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------
    def update_dynamics(self, new_A_list, new_B_list, new_E_list=None, new_g_list=None):
        assert len(new_A_list) == self.N and len(new_B_list) == self.N
        self.A_list = new_A_list
        self.B_list = new_B_list
        self.m.A_list = new_A_list
        self.m.B_list = new_B_list
        if new_E_list is not None:
            assert len(new_E_list) == self.N + 1
            self.E_list = new_E_list
            self.m.E_list = new_E_list
        if new_g_list is not None:
            assert len(new_g_list) == self.N + 1
            self.g_list = new_g_list
            self.m.g_list = new_g_list

        # Always rebuild numerics to allow arbitrary (dense) A/B values
        self._reassemble_numeric_same_sparsity()

        # Backend handling
        if self.backend == "osqp":
            # Rebuild native OSQP problem to accept new sparsity structure
            prob = osqp.OSQP()
            prob.setup(
                P=2 * self.P_mat_csc,
                q=np.zeros(self.P_mat_csc.shape[0]),
                A=self.A_mat_csc_osqp,
                l=self.lbg_osqp_base,
                u=self.ubg_osqp_base,
                polishing=True, warm_starting=False, verbose=False
            )
            self._osqp_prob = prob
        elif self.backend == "osqp_codegen":
            # P never changes (pushed once at setup); only A changed here. The sparsity
            # pattern is fixed by construction (see _compute_A_update_indices) so there's
            # nothing to re-verify at runtime -- a genuine shape/pattern mismatch (e.g. a
            # stale compiled solver built for a different N) is still caught below via the
            # C API's own nonzero return code.
            cg = self._osqp_prob
            Ax = np.ascontiguousarray(self.A_mat_csc_osqp.data, dtype=np.float64)
            pushed = cg.update_data_mat(A_x=Ax)
            if pushed != 0:
                raise RuntimeError(
                    f"update_data_mat(A_x=...) failed with code {pushed}. This means the "
                    "compiled solver in build/osqp_fast was codegen'd for a different horizon "
                    "N (or different dynamics/constraints) than the one you're running now. "
                    "Rebuild it to match: `rnmpc build-fast <system> --N <k>` (note: "
                    "`--smoke-test` runs use N=3, not the default N=15 - pass `--smoke-test` to "
                    "build-fast too, or `--N 3` explicitly). Or run with `--stock-osqp` to "
                    "sidestep the compiled solver."
                )
            QP._codegen_singleton_owner[self.codegen_module] = self._instance_id
        else:
            # casadi_osqp uses CasADi sparsity; nothing to do here (handled at solve())
            pass

    def update_ubg(self, new_ubg):
        self.ubg = new_ubg
        ubg_flat = np.asarray(self.ubg, dtype=np.float64).ravel()
        self.ubg_osqp_base[:ubg_flat.size] = ubg_flat

    def reset_ubg(self):
        self.ubg = self.nominal_ubg
        ubg_flat = np.asarray(self.ubg, dtype=np.float64).ravel()
        self.ubg_osqp_base[:ubg_flat.size] = ubg_flat

    def reset_lbg(self):
        self.lbg = self.lbg_ZEROS_INF_NOMINAL
        lbg_flat = np.asarray(self.lbg, dtype=np.float64).ravel()
        self.lbg_osqp_base[:lbg_flat.size] = lbg_flat

    def offset_constraints(self, offset_equality_constraints):
        """
        Offset the equality constraints by a given amount, and keeps the inequality constraints unchanged.
        :param offset_equality_constraints:
        :return:
        """
        assert offset_equality_constraints.shape == (self.m.nx, self.N)
        off = ca.DM.zeros((0, 1))
        for k in range(self.N):
            off = ca.vertcat(off, ca.DM(offset_equality_constraints[:, k]), ca.DM.zeros((self.m.ni, 1)))
        off = ca.vertcat(off, ca.DM.zeros((self.m.ni_f, 1)))
        self.ubg = self.ubg - off + EPSILON
        self.lbg = self.lbg_ZEROS_INF_NOMINAL
        self.lbg = self.lbg - off - EPSILON

        return

    def update_q_cost_lin(self, q_cost_lin):
        expected = (self.m.nx + self.m.nu) * self.N + self.m.nx
        assert q_cost_lin.shape[0] == expected
        if not isinstance(q_cost_lin, ca.DM):
            q_cost_lin = ca.DM(q_cost_lin)
        self.q_cost_lin = q_cost_lin

    def add_q_cost_lin(self, q_cost_lin):
        expected = (self.m.nx + self.m.nu) * self.N + self.m.nx
        assert q_cost_lin.shape[0] == expected
        if not isinstance(q_cost_lin, ca.DM):
            q_cost_lin = ca.DM(q_cost_lin)
        self.q_cost_lin += q_cost_lin

    def reset_q_cost_lin(self):
        nx, nu, N = self.m.nx, self.m.nu, self.N
        self.q_cost_lin = ca.DM.zeros((nx + nu) * N + nx)

    def debug_codegen_pattern(self):
        """
        Return a dict with CSC pattern diagnostics comparing current P_ut and A_osqp
        against stored codegen signatures. Useful to pinpoint where the pattern changed.
        """
        info = {}
        if not hasattr(self, "_P_cg_sig") or not hasattr(self, "_A_cg_sig"):
            info["has_signatures"] = False
            return info
        info["has_signatures"] = True

        # current matrices
        P_ut = sp.triu(2 * self.P_mat_csc, format='csc'); P_ut.sort_indices()
        A_osqp = self.A_mat_csc_osqp; A_osqp.sort_indices()

        # summaries
        info["P_now_shape"] = P_ut.shape
        info["A_now_shape"] = A_osqp.shape
        info["P_now_nnz"] = int(P_ut.nnz)
        info["A_now_nnz"] = int(A_osqp.nnz)
        info["P_ref_nnz"] = len(self._P_cg_sig[0]) if self._P_cg_sig else None
        info["A_ref_nnz"] = len(self._A_cg_sig[0]) if self._A_cg_sig else None

        # matches?
        try:
            self._assert_same_csc_pattern(P_ut, self._P_cg_sig, "P_ut")
            info["P_match"] = True
            info["P_report"] = "OK"
        except Exception as e:
            info["P_match"] = False
            info["P_report"] = str(e)
        try:
            self._assert_same_csc_pattern(A_osqp, self._A_cg_sig, "A_osqp")
            info["A_match"] = True
            info["A_report"] = "OK"
        except Exception as e:
            info["A_match"] = False
            info["A_report"] = str(e)

        return info

    def _cg_push_updates(self, cg, qv, l, u):
        """Push q, l, u to the codegen module. A (data-only) is pushed separately in
        update_dynamics() only when it actually changes; P is pushed once at setup
        (_init_backend) since Q/R/Qf never change for the life of the instance.
        """
        pushed = cg.update_data_vec(qv, l, u)
        if pushed != 0:
            raise RuntimeError(f"update_data_vec failed with code {pushed}")

    def _store_cg_signatures(self, P_ut: sp.csc_matrix, A_osqp: sp.csc_matrix):
        # Both must be sorted before storing
        P_ut.sort_indices()
        A_osqp.sort_indices()
        self._P_cg_sig = (tuple(P_ut.indices), tuple(P_ut.indptr))
        self._A_cg_sig = (tuple(A_osqp.indices), tuple(A_osqp.indptr))

    def _assert_same_csc_pattern(self, M: sp.csc_matrix, sig, name: str):
        M.sort_indices()
        # If no reference signature, nothing to compare against
        if sig is None or sig[0] is None or sig[1] is None:
            return
        if (tuple(M.indices), tuple(M.indptr)) != sig:
            # Build a detailed diff to aid debugging
            raise RuntimeError(self._csc_diff_report(M, sig, name))

    def _csc_diff_report(self, M: sp.csc_matrix, sig, name: str) -> str:
        """
        Create a detailed human-readable report of how M's CSC pattern differs from the
        stored reference signature (indices, indptr). Highlights first differing column,
        per-column nnz and row-index mismatches, and overall stats.
        """
        idx_ref, ptr_ref = sig
        idx_now = tuple(M.indices)
        ptr_now = tuple(M.indptr)
        ncols = M.shape[1]
        cols_ref = len(ptr_ref) - 1
        cols_now = len(ptr_now) - 1
        nnz_ref = len(idx_ref)
        nnz_now = len(idx_now)

        lines = []
        lines.append(f"{name} CSC pattern/order changed vs. codegen")
        lines.append(f"  shape_now={M.shape}, ncols_now={ncols}")
        lines.append(f"  ref: cols={cols_ref}, nnz={nnz_ref}")
        lines.append(f"  now: cols={cols_now}, nnz={nnz_now}")

        # Column pointer differences
        max_cols = min(cols_ref, cols_now)
        first_col_ptr_diff = None
        for j in range(max_cols + 1):  # include last pointer
            pr = ptr_ref[j] if j < len(ptr_ref) else None
            pn = ptr_now[j] if j < len(ptr_now) else None
            if pr != pn:
                first_col_ptr_diff = j
                break
        if first_col_ptr_diff is not None:
            j = first_col_ptr_diff
            pr = ptr_ref[j] if j < len(ptr_ref) else None
            pn = ptr_now[j] if j < len(ptr_now) else None
            lines.append(f"  first indptr mismatch at j={j}: ref={pr}, now={pn}")
        else:
            lines.append("  indptr arrays identical")

        # If we can align columns, find first column with row-index diff
        first_col_idx_diff = None
        j_limit = min(cols_ref, cols_now)
        for j in range(j_limit):
            r0_ref, r1_ref = ptr_ref[j], ptr_ref[j + 1]
            r0_now, r1_now = ptr_now[j], ptr_now[j + 1]
            col_idx_ref = idx_ref[r0_ref:r1_ref]
            col_idx_now = idx_now[r0_now:r1_now]
            if tuple(col_idx_ref) != tuple(col_idx_now):
                first_col_idx_diff = j
                break
        if first_col_idx_diff is not None:
            j = first_col_idx_diff
            r0_ref, r1_ref = ptr_ref[j], ptr_ref[j + 1]
            r0_now, r1_now = ptr_now[j], ptr_now[j + 1]
            col_idx_ref = tuple(idx_ref[r0_ref:r1_ref])
            col_idx_now = tuple(idx_now[r0_now:r1_now])
            lines.append(f"  first column row-index mismatch at j={j}:")
            lines.append(f"    ref rows[{r0_ref}:{r1_ref}] len={len(col_idx_ref)} -> {list(col_idx_ref)[:20]}")
            lines.append(f"    now rows[{r0_now}:{r1_now}] len={len(col_idx_now)} -> {list(col_idx_now)[:20]}")
        else:
            lines.append("  per-column row indices identical for compared columns")

        # Also include a short checksum of arrays for quick compare
        import hashlib
        def _hs(x):
            h = hashlib.sha1(bytes(str(x), 'utf-8')).hexdigest()[:12]
            return h
        lines.append(f"  checksum idx_ref={_hs(idx_ref)}, idx_now={_hs(idx_now)}")
        lines.append(f"  checksum ptr_ref={_hs(ptr_ref)}, ptr_now={_hs(ptr_now)}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Codegen export helpers
    # ------------------------------------------------------------------


    def export_osqp_solver(self, out_dir, *, extension_name="osqp_generated"):
        out = Path(out_dir)
        # nuke existing folder to avoid copytree errors (simplest + robust)
        if out.exists():
            shutil.rmtree(out, onerror=_clear_readonly_and_retry)
        out.mkdir(parents=True, exist_ok=True)
        self.P_mat_csc.sort_indices()
        self.A_mat_csc_osqp.sort_indices()

        P = 2 * self.P_mat_csc
        A = self.A_mat_csc_osqp
        q = np.zeros(P.shape[0])
        l = self.lbg_osqp_base
        u = self.ubg_osqp_base
        self.P_mat_csc.sort_indices()
        self.A_mat_csc_osqp.sort_indices()

        P_ut = sp.triu(2 * self.P_mat_csc, format='csc')
        P_ut.sort_indices()

        prob = osqp.OSQP()
        # Tight tolerances are baked into the generated C at codegen time. Default OSQP eps is
        # ~1e-3; we tighten to 1e-6 (+ polish) so the compiled solver matches a general-purpose
        # solver (Clarabel) to ~1e-5, and matches the stock backend used elsewhere.
        prob.setup(P=P_ut, q=q, A=self.A_mat_csc_osqp, l=l, u=u,
                   eps_abs=1e-6, eps_rel=1e-6, polish=True, warm_start=True, verbose=False)
        # prob.setup(
        #     P=2 * self.P_mat_csc,  # match CasADi’s h = 2*H
        #     q=np.zeros(self.P_mat_csc.shape[0]),
        #     A=self.A_mat_csc_osqp,  # includes x0 rows
        #     l=self.lbg_osqp_base,  # placeholders; updated per solve
        #     u=self.ubg_osqp_base,
        #     polish=True, warm_start=False, verbose=False
        # )

        # import os, sys
        # os.environ["PYBIND11_FINDPYTHON"] = "NEW"
        # os.environ["CMAKE_ARGS"] = f"-DPython3_EXECUTABLE={sys.executable}"

        # OSQP ≥ 1.0 API
        prob.codegen(
            str(out),
            parameters="matrices",
            extension_name=extension_name,
            include_codegen_src=True,
            force_rewrite=True,  # <-- key
            compile=False,  # builds the Python module
            profiling_enable=False, # !!! profiling is not supported in codegen mode
        )
        if self.verbose:
            print(f"Built {extension_name} in {out}. You can now `import {extension_name}`.")

    # ------------------------------------------------------------------
    # Quadprog export helpers
    # ------------------------------------------------------------------
    def _convert_bounds_to_quadprog(self, A: sp.csc_matrix, l: np.ndarray, u: np.ndarray, tol_eq: float = 1e-8):
        """
        Convert l <= A x <= u into quadprog's A,b,Aeq,beq.
        Returns (A_ineq, b_ineq, A_eq, b_eq) as CSR matrices/vectors.
        """
        A = A.tocsr()
        m, n = A.shape
        idx = np.arange(m)

        is_l_fin = np.isfinite(l)
        is_u_fin = np.isfinite(u)
        is_eq = is_l_fin & is_u_fin & (np.abs(u - l) <= tol_eq)

        # Equality rows
        Aeq = A[is_eq]
        beq = 0.5 * (u[is_eq] + l[is_eq])

        # Inequalities
        # upper-only: A x <= u
        mask_up = (~is_l_fin) & is_u_fin
        A1 = A[mask_up]
        b1 = u[mask_up]

        # lower-only: -A x <= -l
        mask_lo = is_l_fin & (~is_u_fin)
        A2 = (-A[mask_lo])
        b2 = (-l[mask_lo])

        # double-bounded (not equal): both A x <= u and -A x <= -l
        mask_db = is_l_fin & is_u_fin & (~is_eq)
        A3a = A[mask_db]
        b3a = u[mask_db]
        A3b = (-A[mask_db])
        b3b = (-l[mask_db])

        # stack
        A_ineq = sp.vstack([A1, A2, A3a, A3b], format='csr') if (A1.shape[0] + A2.shape[0] + A3a.shape[0] + A3b.shape[0]) > 0 else sp.csr_matrix((0, n))
        b_ineq = np.concatenate([b1, b2, b3a, b3b]) if A_ineq.shape[0] > 0 else np.zeros((0,))

        return A_ineq, b_ineq, Aeq, beq

    def _export_quadprog(self, x0, solve_ret: dict):
        """
        Build quadprog inputs (H,f,A,b,Aeq,beq,lb,ub) from the current QP and save to .mat.
        Includes the solution vector and trajectories.
        """
        # Ensure export dir exists
        self.export_dir.mkdir(parents=True, exist_ok=True)

        # Matrices and vectors
        H = (2 * self.P_mat_csc).tocsc()  # quadprog uses 1/2 x' H x
        f = np.asarray(self.q_cost_lin, dtype=float).ravel()

        # Constraints in OSQP form (without x0 helper rows)
        A = self.A_mat_csc.tocsc()
        l = np.asarray(self.lbg, dtype=float).ravel()
        u = np.asarray(self.ubg, dtype=float).ravel()

        # Convert to quadprog form
        A_ineq, b_ineq, Aeq_base, beq_base = self._convert_bounds_to_quadprog(A, l, u)

        # Append x0 equality: I * x(0) + x0 = 0  => I*y = -x0 on first nx vars
        nx, nu, N = self.m.nx, self.m.nu, self.N
        nv = (nx + nu) * N + nx
        I0 = sp.hstack([sp.eye(nx, format='csr'), sp.csr_matrix((nx, nv - nx))], format='csr')
        Aeq = sp.vstack([Aeq_base, I0], format='csr') if Aeq_base.shape[0] > 0 else I0
        beq = np.concatenate([beq_base, -np.asarray(x0, dtype=float).ravel()]) if Aeq_base.shape[0] > 0 else (-np.asarray(x0, dtype=float).ravel())

        # Variable bounds: leave free (±inf) — x0 equality is already in Aeq
        lb = -np.inf * np.ones(nv)
        ub = +np.inf * np.ones(nv)

        # Solution info
        y = np.asarray(solve_ret.get('primal_vec')).ravel()
        x_traj = np.asarray(solve_ret.get('primal_x'))
        u_traj = np.asarray(solve_ret.get('primal_u'))
        cost = float(solve_ret.get('cost', np.nan))

        # Save
        k = self._export_counter
        self._export_counter += 1
        out_path = self.export_dir / f"qp_export_{k:06d}.mat"
        savemat(str(out_path), {
            'H': H,
            'f': f,
            'A': A_ineq,
            'b': b_ineq,
            'Aeq': Aeq,
            'beq': beq,
            'lb': lb,
            'ub': ub,
            'x0': np.asarray(x0, dtype=float).ravel(),
            'x_sol': y,
            'x_traj': x_traj,
            'u_traj': u_traj,
            'cost': cost,
            'backend': np.array(self.backend),
            'dimensions': np.array([nx, nu, N], dtype=np.int32),
        })
        if self.verbose:
            print(f"Saved quadprog export to {out_path}")
