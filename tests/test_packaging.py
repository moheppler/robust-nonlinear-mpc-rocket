import importlib

import pytest


@pytest.mark.parametrize("mod", [
    "dyn.pendulum", "dyn.quadrotor", "dyn.rocket",
    "dyn.model", "dyn.LTI", "dyn.LTV",
    "solver.qp_jit", "solver.ocp", "solver.nlp",
    "util.footnote",
    "tests.fixtures.integrator", "tests.fixtures.spring_damper",
])
def test_core_modules_import(mod):
    assert importlib.import_module(mod) is not None


def test_pendulum_model_dimensions():
    from dyn.pendulum import Pendulum
    m = Pendulum()
    assert m.nx == 4
    assert m.nu == 1
