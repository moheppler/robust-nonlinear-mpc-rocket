import pytest

import rnmpc_cli


def test_systems_registry_complete():
    assert set(rnmpc_cli.SYSTEMS) == {"pendulum", "quadrotor", "rocket", "rocket-compare", "rocket-direct"}


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        rnmpc_cli.main(["--help"])
    assert exc.value.code == 0
    assert "rnmpc" in capsys.readouterr().out


def test_unknown_system_errors():
    with pytest.raises(SystemExit) as exc:
        rnmpc_cli.main(["nope"])
    assert exc.value.code != 0


def test_stock_flag_forces_stock(monkeypatch):
    calls = {}
    monkeypatch.setattr("solver.osqp_backend.set_force_stock", lambda f: calls.setdefault("v", f))
    # plot path avoids running a solve; we only assert the flag was propagated
    monkeypatch.setattr(rnmpc_cli, "_dispatch_experiment", lambda *a, **k: 0)
    rnmpc_cli.main(["pendulum", "--stock-osqp"])
    assert calls["v"] is True


def test_fast_solver_required_error_is_reported_cleanly(monkeypatch, capsys):
    import solver.osqp_backend as ob

    def _boom(*a, **k):
        raise ob.FastSolverRequiredError("build it or pass --stock-osqp")

    monkeypatch.setattr(rnmpc_cli, "_dispatch_experiment", _boom)
    rc = rnmpc_cli.main(["pendulum", "--run"])
    assert rc == 1
    assert "--stock-osqp" in capsys.readouterr().err


def test_build_fast_smoke_test_flag_propagates(monkeypatch):
    calls = {}
    monkeypatch.setattr(rnmpc_cli, "_dispatch_build", lambda system, N, smoke_test: calls.update(
        system=system, N=N, smoke_test=smoke_test))
    rnmpc_cli.main(["build-fast", "pendulum", "--smoke-test"])
    assert calls == {"system": "pendulum", "N": None, "smoke_test": True}


def test_build_error_is_reported_cleanly(monkeypatch, capsys):
    import solver.build_fast as bf

    def _boom(*a, **k):
        raise bf.FastSolverBuildError("install a C compiler + CMake")

    monkeypatch.setattr(bf, "build", _boom)
    rc = rnmpc_cli.main(["build-fast", "pendulum"])
    assert rc == 1
    assert "C compiler" in capsys.readouterr().err
