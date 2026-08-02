import cvxpy as cp
import numpy as np
from scipy.linalg import block_diag

from solver.ocp import OCP
from util.SLS import SLS
from dyn.LTI import LTI


class cvxpy_SLS(OCP):
    """
    cvxpy_SLS class is a child of the OCP class. It is used to solve the SLS optimal control problem using cvxpy.
    """

    def __init__(self, N, Q, R, m, Qf, Q_reg=None, R_reg=None, Q_reg_f=None):
        super().__init__(N, Q, R, m, Qf, Q_reg, R_reg, Q_reg_f)
        """
        Constructor of the cvxpy_SLS class.
        :param N: int: The horizon length.
        :param Q: numpy array: The state cost matrix.
        :param R: numpy array: The control cost matrix.
        :param m: object: The dynamical system.
        :param Qf: numpy array: The final state cost matrix.
        """
        self.solver_name = None
        # set verbose to False
        self.solver_options = {'verbose': False}
        self.problem = None
        self.x0 = None
        self.Phi_x = None
        self.Phi_u = None
        self.x = None
        self.u = None
        self.constraints = None
        self.cost = None

        self.initialize_solver()
        # solution list
        self.solution = {}

        assert isinstance(m, LTI), "The dynamical system should be an LTI system."


    def initialize_solver(self):
        """
         Solve the SLS optimal control problem using cvxpy.
         :return: dict: The solution of the optimal control problem.
         """
        nx = self.m.nx
        nu = self.m.nu
        nw = self.m.nw
        N = self.N

        self.initialize_list_dynamics()
        A_list = self.A_list
        B_list = self.B_list
        E_list = self.E_list

        # Define the decision variables
        x = cp.Variable((nx, N + 1), name='primal_x')
        u = cp.Variable((nu, N), name='primal_u')

        # initial conditions parameter
        x0 = cp.Parameter(nx, name='x0')

        # System Response Matrix
        Phi_x = [[cp.Variable((nx, nw), name='Phi_x_{},{}'.format(i, j)) for j in range(N + 1)] for i in range(N + 1)]
        Phi_u = [[cp.Variable((nu, nw), name='Phi_u_{},{}'.format(i, j)) for j in range(N + 1)] for i in range(N + 1)]

        Phi_x_mat = cp.bmat([[Phi_x[i][j] for j in range(N + 1)] for i in range(N + 1)])
        Phi_u_mat = cp.bmat([[Phi_u[i][j] for j in range(N + 1)] for i in range(N + 1)])
        # last line of Phi_u_mat is not used ...

        Q = self.Q
        R = self.R
        Qf = self.Qf

        # Define the cost function
        cost = 0
        for k in range(N):
            cost += cp.quad_form(x[:, k], Q) + cp.quad_form(u[:, k], R)
        cost += cp.quad_form(x[:, N], Qf)

        # Add the system response matrix constraints in cost function

        Q_reg = self.Q_reg
        R_reg = self.R_reg
        Q_reg_f = self.Q_reg_f

        Q_reg_blk = block_diag(np.kron(np.eye(N), Q_reg), Q_reg_f)
        R_reg_blk = np.kron(np.eye(N + 1), R_reg)
        # concatenate the Phi matrices
        Phi_mat = cp.vstack([Phi_x_mat, Phi_u_mat])
        cost += cp.norm(block_diag(Q_reg_blk, R_reg_blk) @ Phi_mat, p='fro')**2
        # todo: this is not the correct way to add the regularization term, as it should be squared. Right now, the definition matches with fast_SLS only with unit regularization
        constraints = []
        # Define the constraints
        constraints += [x[:, 0] == x0]
        for k in range(N):
            constraints += [x[:, k + 1] == A_list[k] @ x[:, k] + B_list[k] @ u[:, k]]

        A_blk = SLS.convert_list_to_blk_matrix(A_list)
        B_blk = SLS.convert_list_to_blk_matrix(B_list)
        E_blk = SLS.convert_list_to_blk_matrix(E_list)

        # add a block at the end of A_blk
        A_blk = block_diag(A_blk, np.zeros((nx, nx)))
        B_blk = block_diag(B_blk, np.zeros((nx, nu)))

        Z_downshift = SLS.get_block_downshift_matrix(N + 1, nx)
        I_blk = np.eye(nx * (N + 1))

        # system response matrix constraints
        constraints += [(I_blk - Z_downshift @ A_blk) @ Phi_x_mat - Z_downshift @ B_blk @ Phi_u_mat == E_blk]

        # add causal structure constraints
        for k in range(N):
            for j in range(N + 1):
                if j > k:
                    constraints += [Phi_x[k][j] == np.zeros((nx, nw))]
                    constraints += [Phi_u[k][j] == np.zeros((nu, nw))]

        # Add state and input constraints
        G = self.m.G
        g = self.m.g
        for kk in range(N):
            z = cp.hstack([x[:, kk], u[:, kk]])
            LHS = G @ z # check dimensions
            for ii in range(kk + 1):
                Phi_ki = cp.vstack([Phi_x[kk][ii], Phi_u[kk][ii]])
                LHS = LHS + cp.norm(G @ Phi_ki, p=2, axis=1)
            constraints += [LHS <= g]

        # add terminal constraints
        Gf = self.m.Gf
        gf = self.m.gf
        LHS = Gf @ x[:, N]
        for kk in range(N + 1):
            Phi_ki = Phi_x[N][kk]
            LHS = LHS + cp.norm(Gf @ Phi_ki, p=2, axis=1)
        constraints += [LHS <= gf]

        # Define the problem
        prob = cp.Problem(cp.Minimize(cost), constraints)

        self.problem = prob
        self.constraints = constraints
        self.x0 = x0
        self.Phi_x = Phi_x
        self.Phi_u = Phi_u
        self.x = x
        self.u = u
        self.cost = cost

    def solve(self, x0):
        """
        Solve the SLS optimal control problem using cvxpy.
        :param x0:
        :return: dict: The solution of the optimal control problem. If the problem is infeasible, return False.
        """
        self.x0.value = x0
        try:
            # Clarabel (BSD, ships with cvxpy) handles the SOCP tube constraints; replaces MOSEK.
            self.problem.solve(solver=cp.CLARABEL)
        except cp.error.SolverError as e:
            self.solution['success'] = False
            # print the error message
            print("SolverError: ", str(e))
            print("Problem status before error:", self.problem.status)
            return self.solution
        if self.problem.status != cp.OPTIMAL:
            self.solution['success'] = False
            return self.solution

        self.solution['success'] = True
        # load solution into dictionary
        self.solution_postprocessing()

        return self.solution

    def solution_postprocessing(self):
        """
        Postprocess the solution of the optimal control problem.
        :return: dict: The postprocessed solution of the optimal control problem.
        """
        N = self.N
        nx = self.m.nx
        nu = self.m.nu
        self.solution['primal_x'] = self.x.value
        self.solution['primal_u'] = self.u.value
        self.solution['cost'] = self.cost.value

        # reconstruct the tensor Phi_x and Phi_u
        Phi_x_tensor = np.array([[
            self.Phi_x[i][j].value for j in range(len(self.Phi_x[0]))
        ] for i in range(len(self.Phi_x))])

        Phi_u_tensor = np.array([[
            self.Phi_u[i][j].value for j in range(len(self.Phi_u[0]))
        ] for i in range(len(self.Phi_u))])

        self.solution['Phi_x'] = Phi_x_tensor
        self.solution['Phi_u'] = Phi_u_tensor

        Phi_x_mat = SLS.convert_tensor_to_matrix(Phi_x_tensor)
        Phi_u_mat = SLS.convert_tensor_to_matrix(Phi_u_tensor)

        A_list = self.A_list
        B_list = self.B_list
        E_list = self.E_list

        A_blk = SLS.convert_list_to_blk_matrix(A_list)
        B_blk = SLS.convert_list_to_blk_matrix(B_list)
        E_blk = SLS.convert_list_to_blk_matrix(E_list)

        # add a block at the end of A_blk
        A_blk = block_diag(A_blk, np.zeros((nx, nx)))
        B_blk = block_diag(B_blk, np.zeros((nx, nu)))

        nx = self.m.nx
        nw = self.m.nw
        N = self.N

        Z_downshift = SLS.get_block_downshift_matrix(N + 1, nx)
        I_blk = np.eye(nx * (N + 1))

        # this constraint is satisfied: (I_blk - Z_downshift @ A_blk) @ Phi_x_mat - Z_downshift @ B_blk @ Phi_u_mat == E_blk

        # remove the last nu lines of Phi_u_mat
        Phi_u_mat = Phi_u_mat[:-nu, :]
        # calculate the controller K
        if nx == nw:
            K = Phi_u_mat @ np.linalg.inv(Phi_x_mat)
            self.solution['K_mat'] = K

        Phi_x = Phi_x_tensor
        Phi_u = Phi_u_tensor
        E = E_list
        A = A_list
        B = B_list

        ni = self.m.ni
        ni_f = self.m.ni_f
        # initialize the backoff
        beta = np.full((N, N, ni), np.nan)  # backoff
        beta_f = np.full((N + 1, ni_f), np.nan)

        # import constraints
        G = self.m.G
        gf = self.m.gf
        g = self.m.g
        Gf = self.m.Gf

        # calculate the backoff
        for jj in range(N + 1):
            for kk in range(jj, N):
                # evaluate the 2-norm of each component
                Phi = np.vstack([Phi_x[kk, jj], Phi_u[kk, jj]])
                beta[kk, jj] = np.linalg.norm(G @ Phi, axis=1)
            # terminal conditions
            beta_f[jj] = np.linalg.norm(Gf @ Phi_x[N, jj], axis=1)

        # evaluate new backoff as the sum of the contribution of each disturbance
        backoff = np.nansum(beta, axis=1)
        backoff_f = np.sum(beta_f, axis=0).T  # transpose to recover convention for time propagation

        self.solution['Phi_x_mat'] = Phi_x_mat
        self.solution['Phi_u_mat'] = Phi_u_mat
        self.solution['E_mat'] = E_blk
        self.solution['beta'] = beta
        self.solution['beta_f'] = beta_f
        self.solution['backoff'] = backoff
        self.solution['backoff_f'] = backoff_f
        self.solution['backoff_x'] = np.vstack((backoff[:, :nx], backoff_f[:nx]))
        self.solution['backoff_u'] = backoff[:, nx:nx + nu]

        return self.solution
