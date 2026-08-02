import pytest

import solver.osqp_backend as ob


def test_force_stock_selects_stock():
    ob.set_force_stock(True)
    try:
        assert ob.fast_solver_available() is False
        assert ob.select_backend() == "osqp"
    finally:
        ob.set_force_stock(False)


def test_select_backend_refuses_without_consent(monkeypatch):
    """No compiled extension + no --stock-osqp/set_force_stock opt-in => hard error, not a
    silent downgrade to the slow backend."""
    monkeypatch.setattr(ob, "_force_stock", False)
    monkeypatch.setattr(ob, "_available_cache", False)
    with pytest.raises(ob.FastSolverRequiredError):
        ob.select_backend()


def test_select_backend_honors_explicit_stock_opt_in_even_if_unavailable(monkeypatch):
    monkeypatch.setattr(ob, "_force_stock", True)
    monkeypatch.setattr(ob, "_available_cache", False)
    assert ob.select_backend() == "osqp"


def test_announce_is_once(capsys):
    ob._announced = False  # reset module state for the test
    ob.announce_backend_once("osqp")
    ob.announce_backend_once("osqp")
    out = capsys.readouterr().out
    assert out.count("pure-Python") == 1


def test_announce_silent_for_fast(capsys):
    ob._announced = False
    ob.announce_backend_once("osqp_codegen")
    assert capsys.readouterr().out == ""
