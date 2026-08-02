"""Render the latest pendulum closed-loop run (results/pendulum_robust_closed_loop/*.npz) as an
animated GIF: the physical cart-pole alongside the state trajectory against its constraints.

Run `rnmpc pendulum --run` first to produce a result, then:
    python -m expe.gif_pendulum_rollout
"""
import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from dyn.pendulum import Pendulum
from util.paths import results_dir

_RESULT_NAME = "pendulum_robust_closed_loop"
POLE_LENGTH = 0.5  # matches the cart-pole length `l` in dyn/pendulum.py::ode


def _latest_npz():
    folder = str(results_dir(_RESULT_NAME))
    files = [f for f in os.listdir(folder) if f.endswith(".npz")]
    if not files:
        raise FileNotFoundError(f"No results in {folder}; run `rnmpc pendulum --run` first.")
    latest = max(files, key=lambda f: os.path.getctime(os.path.join(folder, f)))
    return os.path.join(folder, latest), folder


def make_gif(out_path: str | None = None, fps: int = 15) -> str:
    npz_path, folder = _latest_npz()
    data = np.load(npz_path)
    X = data["state_trajectory"]  # (4, T): cart x, x_dot, theta, theta_dot
    dt = float(data["dt"])
    T = X.shape[1]
    time = np.arange(T) * dt
    x_max = float(Pendulum().g[0])

    cart_x = X[0]
    theta = X[2]
    bob_x = cart_x + POLE_LENGTH * np.sin(theta)
    bob_y = POLE_LENGTH * np.cos(theta)

    fig, (ax_cart, ax_state) = plt.subplots(2, 1, figsize=(7, 8), height_ratios=[1.3, 1])

    # --- physical cart-pole panel --- (auto-scaled to the actual rollout: the tube keeps this
    # system tightly regulated, well inside the +/-x_max constraints, so fixing the view to the
    # constraint scale would make the motion invisible)
    cart_pad = max(0.3, 0.25 * (cart_x.max() - cart_x.min()))
    ax_cart.set_xlim(cart_x.min() - cart_pad, cart_x.max() + cart_pad)
    ax_cart.set_ylim(-0.15, POLE_LENGTH * 1.3)
    ax_cart.set_aspect("equal")
    ax_cart.axhline(0, color="k", lw=1)
    ax_cart.set_title("Cart-pole closed-loop rollout (SLS tube MPC)")
    ax_cart.set_xticks([])
    ax_cart.set_yticks([])
    cart_w, cart_h = 0.3, 0.15
    cart_patch = plt.Rectangle((cart_x[0] - cart_w / 2, -cart_h / 2), cart_w, cart_h,
                                fc="#2b6cb0", ec="k", zorder=3)
    ax_cart.add_patch(cart_patch)
    pole_line, = ax_cart.plot([], [], "-", color="#333333", lw=3, zorder=2)
    bob_dot, = ax_cart.plot([], [], "o", color="#e53e3e", ms=12, zorder=4)
    trace_line, = ax_cart.plot([], [], "-", color="#999999", lw=1, alpha=0.6, zorder=1)

    # --- state-vs-constraints panel --- (auto-scaled to the data; the tube keeps this run
    # far inside +/-x_max, so the constraint lines only get drawn when they're in view)
    labels = ["x", "x_dot", "theta", "theta_dot"]
    colors = plt.cm.viridis(np.linspace(0, 1, 4))
    for i in range(4):
        ax_state.plot(time, X[i], color=colors[i], label=labels[i])
    data_lo, data_hi = X.min(), X.max()
    state_pad = max(0.05, 0.25 * (data_hi - data_lo))
    ylim = (data_lo - state_pad, data_hi + state_pad)
    ax_state.set_ylim(*ylim)
    if ylim[0] <= x_max <= ylim[1] or ylim[0] <= -x_max <= ylim[1]:
        ax_state.axhline(x_max, color="k", lw=0.8, ls="--")
        ax_state.axhline(-x_max, color="k", lw=0.8, ls="--")
        constraint_note = "dashed = constraints"
    else:
        constraint_note = f"constraints at +/-{x_max:g} (off-scale - tube keeps this run tight)"
    ax_state.set_xlabel("time [s]")
    ax_state.set_title(f"State trajectory ({constraint_note})")
    ax_state.legend(loc="upper right", ncol=4, fontsize=8)
    ax_state.grid(alpha=0.3)
    cursor = ax_state.axvline(time[0], color="k", lw=1)

    fig.tight_layout()

    def update(frame):
        cx = cart_x[frame]
        cart_patch.set_xy((cx - cart_w / 2, -cart_h / 2))
        pole_line.set_data([cx, bob_x[frame]], [0, bob_y[frame]])
        bob_dot.set_data([bob_x[frame]], [bob_y[frame]])
        trace_line.set_data(bob_x[: frame + 1], bob_y[: frame + 1])
        cursor.set_xdata([time[frame], time[frame]])
        return cart_patch, pole_line, bob_dot, trace_line, cursor

    ani = animation.FuncAnimation(fig, update, frames=T, blit=False, interval=1000 / fps)
    out_path = out_path or os.path.join(folder, "rollout.gif")
    ani.save(out_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Saved rollout GIF to {out_path}")
    return out_path


if __name__ == "__main__":
    make_gif()
