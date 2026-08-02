import numpy as np


def test_rocket_smoke_runs_on_stock_osqp():
    import solver.osqp_backend as ob
    ob.set_force_stock(True)
    try:
        from expe import main_rocket_robust_closed_loop as rocket
        out = rocket.generate(N=3, smoke_test=True)
    finally:
        ob.set_force_stock(False)

    data = np.load(out)
    nx = int(data["nx"])
    T = int(data["simulation_time_steps"])
    assert data["state_trajectory"].shape == (nx, T)
    assert np.all(np.isfinite(data["state_trajectory"]))
