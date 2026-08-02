"""Double-integrator LTI model — TEST-ONLY fixture (not one of the published models).

Used to exercise the QP backends (osqp vs osqp_codegen) on a tiny, deterministic problem.
Model matrices/constraints are copied verbatim from the original monorepo model; the plotting
and tube helpers were dropped because they depended on the removed ``util.minkowski_*``
modules and are not needed by any test.
"""
import numpy as np

from dyn.LTI import LTI


class Integrator(LTI):
    def __init__(self):
        super().__init__()
        self.dt = 1
        self.E = 0.1 * np.eye(2)
        self.A = np.array([[1, 1], [0, 1]])
        self.B = np.array([[0.5], [1]])
        self.G = np.vstack((np.eye(3), -np.eye(3)))
        x_max = 5
        x_min = -5
        u_max = 3
        u_min = -3
        x_max_f = 5
        x_min_f = -5
        self.g = np.array([x_max, x_max, u_max, -x_min, -x_min, -u_min])
        # constraints are symmetric, stacked as [upper_bounds; -lower_bounds]
        self.Gf = np.vstack((np.eye(2), -np.eye(2)))
        self.gf = np.array([x_max_f, x_max_f, -x_min_f, -x_min_f])

        self.assign_dimensions()

    def replace_constraints(self, x_max, x_min, u_max, u_min, x_max_f, x_min_f):
        self.g = np.array([x_max, x_max, u_max, -x_min, -x_min, -u_min])
        self.gf = np.array([x_max_f, x_max_f, -x_min_f, -x_min_f])
