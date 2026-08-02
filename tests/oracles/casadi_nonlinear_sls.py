import numpy as np
from scipy.linalg import block_diag

from solver.ocp import OCP
from util.SLS import SLS
from dyn.LTI import LTI

import casadi as ca

MINIMUM_BACKOFF = 1e-10

class CasadiNonlinearSLS(OCP):
    """
    casadi_nonlinear_sls class is a child of the OCP class. It is used to solve the SLS optimal control problem using casadi.
    """

    def __init__(self, N, Q, R, m, Qf, Q_reg=None, R_reg=None, Q_reg_f=None):
        super().__init__(N, Q, R, m, Qf, Q_reg, R_reg, Q_reg_f)
        """
        Constructor of the casadi_nonlinear_sls class.
        :param N: int: The horizon length.
        :param Q: numpy array: The state cost matrix.
        :param R: numpy array: The control cost matrix.
        :param m: object: The dynamical system.
        :param Qf: numpy array: The final state cost matrix.
        """

        self.solver_name = None
        # set verbose to False
        self.verbose = False
        self.solver_options = {
            'ipopt.print_level': 5 if self.verbose else 0,
            'print_time': self.verbose,
        }

        self.solver_params = {}
        self.solver = None

        self.initialize_solver()
        # solution list
        self.solution = {}
        self.EPSILON = 1e-5

    def solve(self, x0):
        y0 = self.solver_params['y0']
        lbg = self.solver_params['lbg']
        ubg = self.solver_params['ubg']
        try:
            sol = self.solver(x0=y0, p=x0, lbg=lbg, ubg=ubg)
        except Exception as e:
            print('Infeasible forward solution. Try with another initial condition.')
            print(e)
            self.solution['success'] = False

        self.solution['success'] = self.solver.stats()['success']
        if self.solution['success']:
            self.post_processing_solution(sol)
        return self.solution

    def post_processing_solution(self, sol):
        nx = self.m.nx
        nu = self.m.nu
        nw = self.m.nw
        N = self.N
        ni = self.m.ni
        ni_f = self.m.ni_f
        x = np.array(sol['x'])
        start_X = 0
        end_X = nx * (N + 1)
        start_U = end_X
        end_U = start_U + nu * N
        start_Phi_x = end_U
        end_Phi_x = start_Phi_x + nx * nx * (N + 1) * (N + 1)
        start_Phi_u = end_Phi_x
        end_Phi_u = start_Phi_u + nx * nu * (N + 1) * N
        start_beta = end_Phi_u
        end_beta = start_beta + N * N * ni
        start_beta_f = end_beta
        end_beta_f = start_beta_f + (N + 1) * ni_f

        self.solution['X'] = x[start_X:end_X].reshape(N + 1, nx).T
        self.solution['U'] = x[start_U:end_U].reshape(N, nu).T
        self.solution['Phi_x_vec'] = x[start_Phi_x:end_Phi_x]
        self.solution['Phi_x'] = x[start_Phi_x:end_Phi_x].reshape(N + 1, N + 1, nx, nx).transpose(0, 1, 3, 2)
        #SLS.convert_tensor_to_matrix(self.solution['Phi_x'])
        self.solution['Phi_u'] = x[start_Phi_u:end_Phi_u].reshape(N, N + 1, nu, nx)
        # todo: use nw instead of nx
        #self.solution['beta'] = x[start_beta:end_beta].reshape(N, N, ni)
        #self.solution['beta_f'] = x[start_beta_f:end_beta_f].reshape(N + 1, ni_f)
        # directly extracting the beta does not work, because of the inequality constraints

        sqrt_beta = np.full((N, N + 1, ni), np.nan)  # backoff
        sqrt_beta_f = np.full((N + 1, ni_f), np.nan)
        G = self.m.G
        Gf = self.m.Gf
        for kk in range(N):
            for jj in range(N+1):
                sqrt_beta[kk,jj,:] = np.linalg.norm(G @ np.vstack([self.solution['Phi_x'][kk, jj], self.solution['Phi_u'][kk, jj]]), axis=1)

        for jj in range(N + 1):
            sqrt_beta_f[jj, :] = np.linalg.norm(Gf @ self.solution['Phi_x'][N, jj], axis=1)

        self.solution['backoff'] = np.nansum(sqrt_beta, axis=1)
        self.solution['backoff_f'] = np.nansum(sqrt_beta_f, axis=0).T

        self.solution['backoff_x'] = self.solution['backoff'][:, :self.m.nx]
        # append the backoff_f to the solution
        self.solution['backoff_x'] = np.vstack((self.solution['backoff_x'], self.solution['backoff_f'][:self.m.nx]))
        self.solution['backoff_u'] = self.solution['backoff'][:, self.m.nx:self.m.nx+self.m.nu]

        self.solution['primal_x'] = self.solution['X']
        self.solution['primal_u'] = self.solution['U']
        self.solution['dual_vec'] = np.array(sol['lam_g'])
        self.solution['cost'] = sol['f']

    def initialize_solver(self):
        """
         Solve the SLS optimal control problem using casadi.
         :return: dict: The solution of the optimal control problem.
         """
        m = self.m
        nx = self.m.nx
        nu = self.m.nu
        nw = self.m.nw
        N = self.N
        ni = m.ni
        ni_f = m.ni_f
        G = m.G
        g = m.g
        Gf = m.Gf
        gf = m.gf

        # Define the decision variables
        X = ca.MX.sym('X', nx, N + 1)
        U = ca.MX.sym('U', nu, N)

        # Define the parameters
        initial_condition = ca.MX.sym('initial_condition', nx)

        # Define the Phi_x and Phi_u matrices
        Phi_x = [[ca.MX.sym(f'Phi_x_{k}_{j}', nx, nw) for j in range(N + 1)] for k in range(N + 1)]
        Phi_u = [[ca.MX.sym(f'Phi_u_{k}_{j}', nu, nw) for j in range(N + 1)] for k in range(N)]

        # Define the backoff
        beta = [[ca.MX.sym(f'beta_{k}_{j}', ni) for j in range(N + 1)] for k in range(N)]
        beta_f = [ca.MX.sym(f'beta_f_{i}', ni_f) for i in range(N + 1)]

        # Define the dynamics as jacobians of the dynamics
        x_fun = ca.SX.sym('x_fun', nx)
        u_fun = ca.SX.sym('u_fun', nu)

        A_fun = ca.Function('A_fun', [x_fun, u_fun], [ca.jacobian(self.m.ddyn(x_fun, u_fun), x_fun)])
        B_fun = ca.Function('B_fun', [x_fun, u_fun], [ca.jacobian(self.m.ddyn(x_fun, u_fun), u_fun)])

        # Initialize lists to store matrices
        A_block = []
        B_block = []
        E_block = []
        for i in range(N):
            current_A = A_fun(X[:, i], U[:, i])
            current_B = B_fun(X[:, i], U[:, i])
            A_block.append(current_A)
            B_block.append(current_B)
            E_block.append(self.m.E)
        E_block.append(self.m.E)

        # initialize cost and constraints
        cost = 0.
        eq_constraints = ca.vertcat()
        ineq_constraints = ca.vertcat()

        # Define the dynamics constraint
        for i in range(N):
            eq_constraints = ca.vertcat(eq_constraints, X[:, i + 1] - self.m.ddyn(X[:, i], U[:, i]))
        eq_constraints = ca.vertcat(eq_constraints, X[:, 0] - initial_condition)

        # Define the beta constraints
        for kk in range(N):
            for jj in range(N + 1):
                Phi = ca.vertcat(Phi_x[kk][jj], Phi_u[kk][jj])
                for ii in range(ni):
                    ineq_constraints = ca.vertcat(
                        ineq_constraints,
                        ca.sumsqr(G[ii,:] @ Phi) - beta[kk][jj][ii]
                    )

        for jj in range(N + 1):
            # terminal conditions
            for ii in range(ni_f):
                ineq_constraints = ca.vertcat(
                    ineq_constraints,
                    ca.sumsqr(Gf[ii,:] @ Phi_x[-1][jj]) - beta_f[jj][ii]
                )

        # Nominal constraints (could be removed without changing the solution)
        for i in range(N):
            ineq_constraints = ca.vertcat(ineq_constraints,
                G @ ca.vertcat(X[:, i], U[:, i]) - g
            )

        ineq_constraints = ca.vertcat(ineq_constraints,
            Gf @ X[:, -1] - gf
        )
        # Define the robust constraint satisfaction
        for k in range(N):
            Z = ca.vertcat(X[:, k], U[:, k])
            LHS = G@Z
            for j in range(N):
                LHS += ca.sqrt(ca.fmax(beta[k][j], 0.) + MINIMUM_BACKOFF)
            ineq_constraints = ca.vertcat(ineq_constraints,
                LHS - g
            )

        # Compute terminal backoff as sum of all contributions 
        LHS_f = Gf @ X[:, -1]
        for jj in range(N+1):
            LHS_f += ca.sqrt(ca.fmax(beta_f[jj], 0.) + MINIMUM_BACKOFF)
        
        ineq_constraints = ca.vertcat(ineq_constraints,
            LHS_f - gf
        )

        # System Level Parametrization
        for jj in range(N + 1):
            eq_constraints = ca.vertcat(
                eq_constraints,
                ca.reshape(Phi_x[jj][jj] - E_block[jj],-1,1)
            )

            for kk in range(jj, N):
                A = A_block[kk]
                B = B_block[kk]
                eq_constraints = ca.vertcat(eq_constraints,
                    ca.reshape(Phi_x[kk + 1][jj] - A @ Phi_x[kk][jj] - B @ Phi_u[kk][jj],-1,1)
                )

        # Add causality constraints
        for k in range(N):
            for j in range(N + 1):
                if j > k:
                    eq_constraints = ca.vertcat(eq_constraints,
                        ca.reshape(Phi_x[k][j],-1,1)
                    )
                    eq_constraints = ca.vertcat(eq_constraints,
                        ca.reshape(Phi_u[k][j],-1,1)
                    )

        # Define the cost function
        for i in range(N):
            cost += ca.mtimes([X[:, i].T, self.Q, X[:, i]]) + ca.mtimes([U[:, i].T, self.R, U[:, i]])
        cost += ca.mtimes([X[:, N].T, self.Qf, X[:, N]])

        # Cost on the regularization terms
        Q_reg = self.Q_reg
        R_reg = self.R_reg
        Q_reg_f = self.Q_reg_f

#        Q_reg_blk = block_diag(np.kron(np.eye(N), Q_reg), Q_reg_f)
#        R_reg_blk = np.kron(np.eye(N), R_reg)

        Phi_x_mat = ca.vertcat(*[ca.horzcat(*[Phi_x[i][j] for j in range(N + 1)]) for i in range(N + 1)])
        Phi_u_mat = ca.vertcat(*[ca.horzcat(*[Phi_u[i][j] for j in range(N + 1)]) for i in range(N)])
        Phi_mat = ca.vertcat(Phi_x_mat, Phi_u_mat)

        for jj in range(N):
            for kk in range(jj, N):
                cost += ca.sumsqr(Q_reg @  Phi_x[kk][jj]) + ca.sumsqr(R_reg @ Phi_u[kk][jj])

        for jj in range(N + 1):
            cost += ca.sumsqr(Q_reg_f @ Phi_x[N][jj])

        Phi_x_vec = ca.vertcat(*[ca.reshape(Phi_x[i][j], -1, 1) for i in range(N + 1) for j in range(N + 1)])
        Phi_u_vec = ca.vertcat(*[ca.reshape(Phi_u[i][j], -1, 1) for i in range(N) for j in range(N+1)])
        beta_vec = ca.vertcat(*[beta[k][j] for k in range(N) for j in range(N+1)])
        beta_f_vec = ca.vertcat(*[beta_f[j] for j in range(N + 1)])

        # Define the optimization problem
        y = ca.vertcat(
            # reshape vertically
            ca.reshape(X, -1, 1),
            ca.reshape(U, -1, 1),
            ca.reshape(Phi_x_vec, -1, 1),
            ca.reshape(Phi_u_vec, -1, 1),
            ca.reshape(beta_vec, -1, 1),
            ca.reshape(beta_f_vec, -1, 1)
        )
        g_eq = ca.vertcat(eq_constraints)
        g_ineq = ca.vertcat(ineq_constraints)
        nlp = {'f': cost, 'x': y, 'g': ca.vertcat(g_eq, g_ineq), 'p': initial_condition}
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, self.solver_options)
        self.solver_params['lbg'] = ca.vertcat(ca.DM.zeros(eq_constraints.shape[0]), [-ca.inf] * ineq_constraints.shape[0])
        self.solver_params['ubg'] = ca.DM.zeros(eq_constraints.shape[0]+ineq_constraints.shape[0])
        self.solver_params['y0'] = ca.DM.zeros(y.shape[0])

