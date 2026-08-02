import numpy as np

from dyn.rocket import Rocket
from solver.nlp import NLP


def test_rocket_nlp_nominal_solves():
    m = Rocket()
    m.dt = 0.05
    N = 3
    x0 = np.array(m.neutral_state, dtype=float)
    x0[0] = 0.5
    sol = NLP(N, 10 * np.eye(m.nx), np.eye(m.nu), m, 10 * np.eye(m.nx)).solve(x0)
    assert sol['success']
    assert sol['primal_x'].shape == (m.nx, N + 1)
