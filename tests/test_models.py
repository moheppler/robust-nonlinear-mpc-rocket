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


def test_ddyn_uses_model_dt_and_is_side_effect_free():
    """`ddyn(x, u)` must integrate over the model's own `self.dt` (not a
    hard-coded 0.05) and must not mutate `self.dt` (previously the call
    re-wrote model.dt to the default, silently pinning every solver pipeline
    to 50 ms)."""
    from dyn.rocket_direct import RocketDirect

    m = RocketDirect()  # native dt = 1/25 s (not 0.05)
    x = np.zeros(m.nx)
    x[3] = 0.1  # perturbed velocity: ẋ(x=0,u=0) = 0 would hide the step size
    u = np.zeros(m.nu)

    assert m.dt == 1.0 / 25.0
    one_step = m.ddyn(x, u)
    assert m.dt == 1.0 / 25.0, "ddyn must not overwrite model.dt"

    # explicit h is equivalent to self.dt when equal, and scales linearly otherwise
    np.testing.assert_allclose(m.ddyn(x, u, h=1.0 / 25.0), one_step, atol=1e-14)
    # RK4 order: a quarter-length step is ~16x more accurate O(h^4); the map
    # itself must differ when h differs (guards against a silently pinned h)
    assert not np.allclose(m.ddyn(x, u, h=0.1), one_step, atol=1e-9)


def test_ddyn_explicit_h_beats_model_dt():
    from dyn.rocket_direct import RocketDirect

    m = RocketDirect()
    x = np.zeros(m.nx)
    u = np.zeros(m.nu)
    a = np.asarray(m.ddyn(x, u, h=1.0 / 25.0))
    assert np.allclose(a, np.asarray(m.ddyn(x, u)))  # h defaults to self.dt
    assert m.dt == 1.0 / 25.0
