import numpy as np


def test_quadrotor_smoke_runs_on_stock_osqp():
    import solver.osqp_backend as ob
    ob.set_force_stock(True)
    try:
        from expe import main_quadrotor_robust_closed_loop as quad
        out = quad.generate(N=3, smoke_test=True)
    finally:
        ob.set_force_stock(False)

    data = np.load(out)
    nx, nu = int(data["nx"]), int(data["nu"])
    T = int(data["simulation_time_steps"])
    assert data["state_trajectory"].shape == (nx, T)
    assert np.all(np.isfinite(data["state_trajectory"]))


def test_quadrotor_is_deterministic():
    import solver.osqp_backend as ob
    ob.set_force_stock(True)
    try:
        from expe import main_quadrotor_robust_closed_loop as quad
        a = np.load(quad.generate(N=3, smoke_test=True))["state_trajectory"]
        b = np.load(quad.generate(N=3, smoke_test=True))["state_trajectory"]
    finally:
        ob.set_force_stock(False)
    np.testing.assert_allclose(a, b)
