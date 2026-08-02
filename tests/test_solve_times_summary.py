import os
import subprocess
import sys
from datetime import datetime

import numpy as np
import pytest

import expe.main_solve_times_summary as sts


def _write_fake_npz(folder, t_jac, t_qp, t_riccati):
    folder.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    path = folder / f"fake_{timestamp}.npz"
    np.savez(
        path,
        t_jac=np.asarray(t_jac).reshape(-1, 1),
        t_qp=np.asarray(t_qp).reshape(-1, 1),
        t_riccati=np.asarray(t_riccati).reshape(-1, 1),
    )
    return path


def test_latest_npz_raises_clear_error_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(sts, "results_dir", lambda name: tmp_path / name)
    with pytest.raises(FileNotFoundError) as exc:
        sts._latest_npz("pendulum_robust_closed_loop", "pendulum")
    msg = str(exc.value)
    assert "pendulum" in msg
    assert "rnmpc pendulum --run" in msg


def test_latest_npz_picks_the_most_recently_created_file(tmp_path, monkeypatch):
    monkeypatch.setattr(sts, "results_dir", lambda name: tmp_path / name)
    folder = tmp_path / "pendulum_robust_closed_loop"
    _write_fake_npz(folder, [1.0], [2.0], [3.0])
    import time
    time.sleep(0.01)
    newest = _write_fake_npz(folder, [9.0], [9.0], [9.0])

    data = sts._latest_npz("pendulum_robust_closed_loop", "pendulum")
    assert float(np.asarray(data["t_jac"]).ravel()[0]) == 9.0


def test_load_timings_returns_flat_arrays(tmp_path, monkeypatch):
    monkeypatch.setattr(sts, "results_dir", lambda name: tmp_path / name)
    folder = tmp_path / "quadrotor_robust_closed_loop"
    _write_fake_npz(folder, [0.1, 0.2], [0.3, 0.4], [0.5, 0.6])

    t_jac, t_qp, t_riccati = sts._load_timings("quadrotor_robust_closed_loop", "quadrotor")
    assert t_jac.shape == (2,)
    assert t_qp.shape == (2,)
    assert t_riccati.shape == (2,)
    np.testing.assert_allclose(t_jac, [0.1, 0.2])
    np.testing.assert_allclose(t_qp, [0.3, 0.4])
    np.testing.assert_allclose(t_riccati, [0.5, 0.6])


def test_run_all_builds_and_runs_each_system_in_order(monkeypatch):
    calls = []

    def fake_check_call(cmd, cwd=None):
        calls.append(cmd)

    monkeypatch.setattr(subprocess, "check_call", fake_check_call)

    sts.run_all(N=3, smoke_test=True)

    expected = []
    for system in sts.SYSTEMS:
        expected.append(
            [sys.executable, "-m", "rnmpc_cli", "build-fast", system, "--N", "3", "--smoke-test"]
        )
        expected.append(
            [sys.executable, "-m", "rnmpc_cli", system, "--run", "--N", "3", "--smoke-test"]
        )
    assert calls == expected


def test_summarize_produces_one_png_covering_all_three_systems(tmp_path, monkeypatch):
    def fake_results_dir(name):
        d = tmp_path / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    monkeypatch.setattr(sts, "results_dir", fake_results_dir)
    for name in sts.SYSTEM_RESULT_NAMES.values():
        _write_fake_npz(fake_results_dir(name), [1.0, 2.0], [0.5, 0.6], [3.0, 4.0])

    monkeypatch.setattr(sts.plt, "show", lambda: None)

    out_png = sts.summarize()

    assert os.path.exists(out_png)
    assert out_png.endswith(".png")
    sts.plt.close("all")
