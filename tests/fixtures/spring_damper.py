"""Nonlinear spring-damper model — TEST-ONLY fixture (not one of the published models).

A small, cheap nonlinear system for exercising the NLP / SCP-SLS solvers in tests without the
cost of the full pendulum/quadrotor/rocket experiments. Dynamics/constraints are copied
verbatim from the original monorepo model; plotting/tube helpers were dropped (they depended
on the removed ``util.minkowski_*`` modules and are not needed by any test).
"""
import casadi as ca
import numpy as np

from dyn.model import Model


class SpringDamper(Model):
    def __init__(self):
        self.nx = 2  # [position, velocity]
        self.nu = 1  # control input u
        self.nw = 2  # disturbance w (one per state)
        self.dt = 0.25  # discretization time step
        self.discretization_method = 'euler'  # Euler instead of the default RK4

        x_max = np.array([10, 10])  # max position and velocity
        u_max = np.array([5])       # max control input

        # Inequality constraints: G @ [x; u] <= g
        self.G = ca.vertcat(np.eye(3), -np.eye(3))
        self.g = np.concatenate((x_max, u_max, x_max, u_max))
        self.ni = 6

        # Terminal constraints
        self.Gf = ca.vertcat(np.eye(2), -np.eye(2))
        self.gf = np.concatenate((x_max, x_max))
        self.ni_f = 4

        self.E = 0.1 * np.eye(2)

    def ode(self, x, u):
        """Continuous-time dynamics. State x = [position, velocity].
        Force(x) = 2*x1 + x2 + tanh(10*x2) + u (disturbance w=0 assumed); mass = 5.
        """
        x1, x2 = ca.vertsplit(x)
        w = 0
        force = 2 * x1 + x2 + ca.tanh(10 * x2) + w + u
        x1_dot = x2
        x2_dot = -force / 5
        return ca.vertcat(x1_dot, x2_dot)

    def replace_constraints(self, x_max, x_min, u_max, u_min, x_max_f, x_min_f):
        x_max = np.asarray(x_max)
        x_min = np.asarray(x_min)
        u_max = np.asarray(u_max)
        u_min = np.asarray(u_min)
        x_max_f = np.asarray(x_max_f)
        x_min_f = np.asarray(x_min_f)

        # G @ [x; u] <= g with [x; u] of size (nx + nu)
        self.G = ca.vertcat(np.eye(self.nx + self.nu), -np.eye(self.nx + self.nu))
        self.g = np.concatenate((x_max, u_max, -x_min, -u_min))
        self.ni = 2 * (self.nx + self.nu)

        self.Gf = ca.vertcat(np.eye(self.nx), -np.eye(self.nx))
        self.gf = np.concatenate((x_max_f, -x_min_f))
        self.ni_f = 2 * self.nx
