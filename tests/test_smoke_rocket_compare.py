import numpy as np


def test_rocket_compare_smoke_runs():
    import solver.osqp_backend as ob
    ob.set_force_stock(True)
    try:
        from expe import main_rocket_compare_closed_loop as cmp
        out = cmp.generate(N=3, smoke_test=True)
    finally:
        ob.set_force_stock(False)

    data = np.load(out, allow_pickle=True)
    # both controllers produced a closed-loop trajectory
    assert "r_state_trajectory" in data.files
    assert "s_state_trajectory" in data.files
    nx = int(data["nx"])
    T = int(data["simulation_time_steps"])
    assert data["r_state_trajectory"].shape == (nx, T)
    assert data["s_state_trajectory"].shape == (nx, T)
    assert np.all(np.isfinite(data["r_state_trajectory"]))
    assert np.all(np.isfinite(data["s_state_trajectory"]))
