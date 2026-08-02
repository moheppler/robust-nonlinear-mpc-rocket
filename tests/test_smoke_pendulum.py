import numpy as np

from util.paths import results_dir


def test_pendulum_smoke_runs_on_stock_osqp():
    import solver.osqp_backend as ob
    ob.set_force_stock(True)
    try:
        from expe import main_pendulum_robust_closed_loop as pend
        out = pend.generate(N=3, smoke_test=True)
    finally:
        ob.set_force_stock(False)

    data = np.load(out)
    nx, nu = int(data["nx"]), int(data["nu"])
    T = int(data["simulation_time_steps"])
    assert data["state_trajectory"].shape == (nx, T)
    assert data["input_trajectory"].shape == (nu, T - 1)
    assert np.all(np.isfinite(data["state_trajectory"]))


def test_results_dir_is_repo_anchored(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    d = results_dir("unit_probe")
    assert d.is_dir()
    assert d.name == "unit_probe"
    assert d.parent.name == "results"
