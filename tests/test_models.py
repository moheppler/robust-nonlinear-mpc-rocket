import numpy as np
import pytest


def test_integrator_dimensions():
    from tests.fixtures.integrator import Integrator
    m = Integrator()
    assert (m.nx, m.nu, m.nw, m.ni, m.ni_f) == (2, 1, 2, 6, 4)


def test_live_model_dimensions():
    from dyn.pendulum import Pendulum
    from dyn.quadrotor import Quadrotor
    from dyn.rocket import Rocket
    assert (Pendulum().nx, Pendulum().nu) == (4, 1)
    assert (Quadrotor().nx, Quadrotor().nu) == (13, 4)
    assert (Rocket().nx, Rocket().nu) == (17, 4)


def test_spring_damper_dynamics():
    import casadi as ca
    from tests.fixtures.spring_damper import SpringDamper
    m = SpringDamper()
    xdot = m.ode(ca.DM([1.0, 0.5]), ca.DM([0.2]))
    expected_force = 2 * 1 + 0.5 + float(ca.tanh(10 * 0.5)) + 0.2
    assert abs(float(xdot[0]) - 0.5) < 1e-10
    assert abs(float(xdot[1]) - (-expected_force / 5)) < 1e-10


def test_spring_damper_nlp_integration():
    from tests.fixtures.spring_damper import SpringDamper
    from solver.nlp import NLP
    m = SpringDamper()
    N = 10
    sol = NLP(N, np.eye(m.nx), np.eye(m.nu), m, 10 * np.eye(m.nx)).solve(np.array([2.0, 0.0]))
    assert sol['success'] is True
    assert sol['primal_x'].shape == (m.nx, N + 1)
    assert sol['primal_u'].shape == (m.nu, N)
    np.testing.assert_allclose(sol['primal_x'][:, 0], np.array([2.0, 0.0]), atol=1e-6)
