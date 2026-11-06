
import casadi as ca
import numpy as np
from dyn.model import Model

MAX_QUAT_BOUND = 1.5


class RocketDirect(Model):
    """
    Direct-input rocket model without attitude-rate dynamics.

    Differences from ``dyn.rocket.Rocket``:

    * The actuator states (thrust-magnitude lag, torque lag, two servo-angle
      lags) and the full angular-velocity state are gone.  The body angular
      rates ``[omega_x, omega_y, omega_z]`` are *inputs*, consumed directly by
      the attitude kinematics -- there is no second-order attitude dynamics
      (no ``J * omega_dot = tau - omega x (J omega)`` equation).
    * ``u_F`` is the commanded thrust magnitude.  As in the original model,
      the zero input commands hover; the actual (absolute) thrust is
      ``u_F + m*g``, with hover thrust kept equal to the original
      11.3796 N convention so that linearisation about the neutral point
      (x = 0, u = 0) still describes a hover equilibrium.

    State x (nx=10):
      [0:3]   position (x, y, z) in world [m]
      [3:6]   linear velocity (v_x, v_y, v_z) in world [m/s]
      [6:10]  unit quaternion (q_x, q_y, q_z, q_w) body->world (Hamilton, w last)

    Input u (nu=4):
      [0]  u_F: thrust magnitude commanded along body +Z [N, deviation from hover]
      [1:4] omega = (omega_x, omega_y, omega_z): body angular rates [rad/s]
    """

    def __init__(self):
        self.params = {
            'mass': 1.16,
            'gravity_constant': 9.81,
        }

        # Hover thrust (used to convert the deviation input to absolute thrust)
        self.hover_thrust = self.params['mass'] * self.params['gravity_constant']  # 11.3796 N

        self.state_names = [
            'x', 'y', 'z',
            'v_x', 'v_y', 'v_z',
            'quat_x', 'quat_y', 'quat_z', 'quat_w',
        ]
        self.control_names = ['u_F', 'omega_x', 'omega_y', 'omega_z']

        self.state = [ca.SX.sym(name) for name in self.state_names]
        self.control = [ca.SX.sym(name) for name in self.control_names]

        self.state_index = {name: i for i, name in enumerate(self.state_names)}
        self.control_index = {name: i for i, name in enumerate(self.control_names)}
        self.get_state_index = lambda n: self.state_index[n] if isinstance(n, str) else [self.state_index[k] for k in n]
        self.get_control_index = lambda n: self.control_index[n] if isinstance(n, str) else [self.control_index[k] for k in n]

        self.state_groups = {
            'pos': slice(0, 3),
            'vel': slice(3, 6),
            'quat': slice(6, 10),
        }

        # --- Neutral state (hover at the origin, level attitude) ---
        self.neutral_state = ca.vertcat(
            0, 0, 0,   # position
            0, 0, 0,   # velocity
            0, 0, 0, 1,  # quaternion [q_x, q_y, q_z, q_w] (w last, matches state_names)
        )
        self.neutral_input = ca.vertcat(
            0,   # u_F = 0  =>  thrust = m*g (hover)
            0, 0, 0,   # omega = 0 (level attitude, body rates zeroed)
        )

        self.nx = len(self.state)
        self.nu = len(self.control)
        self.dt = 1.0 / 25.0  # 30 Hz controller rate

        # --- Bounds (monitor-style, matching the original Rocket layout) ---
        # quaternion uses the same +/-1.5 bound as the original; body rates use
        # the original angular-velocity bound of +/- 2 rad/s.
        state_bounds = {
            'x': (-10.0, 10.0),
            'y': (-10.0, 10.0),
            'z': (-10.0, 10.0),
            'v_x': (-1.0, 1.0),
            'v_y': (-1.0, 1.0),
            'v_z': (-1.0, 1.0),
            'quat_x': (-MAX_QUAT_BOUND, MAX_QUAT_BOUND),
            'quat_y': (-MAX_QUAT_BOUND, MAX_QUAT_BOUND),
            'quat_z': (-MAX_QUAT_BOUND, MAX_QUAT_BOUND),
            'quat_w': (-MAX_QUAT_BOUND, MAX_QUAT_BOUND),
        }
        control_bounds = {
            'u_F': (-50.0, 50.0),              # deviation from hover; |u_F + m*g| <= 61.38 N
            'omega_x': (-2.0, 2.0),
            'omega_y': (-2.0, 2.0),
            'omega_z': (-2.0, 2.0),
        }

        x_lb, x_ub = [], []
        for s in self.state:
            lb, ub = state_bounds[s.name()]
            x_lb.append(lb)
            x_ub.append(ub)
        u_lb, u_ub = [], []
        for ui in self.control:
            lb, ub = control_bounds[ui.name()]
            u_lb.append(lb)
            u_ub.append(ub)

        x_lb = np.array(x_lb)
        x_ub = np.array(x_ub)
        u_lb = np.array(u_lb)
        u_ub = np.array(u_ub)

        self.G = ca.vertcat(np.eye(self.nx + self.nu), -np.eye(self.nx + self.nu))
        self.g = np.concatenate((np.concatenate((x_ub, u_ub)), -np.concatenate((x_lb, u_lb))))
        self.ni = 2 * (self.nx + self.nu)
        self.Gf = ca.vertcat(np.eye(self.nx), -np.eye(self.nx))
        self.gf = np.concatenate((x_ub, -x_lb))
        self.ni_f = 2 * self.nx

        # Disturbance scaling (diagonal) over the 10 states; no per-rate or
        # per-actuator entries since those states no longer exist.
        self.E = np.diag([
            0.03, 0.03, 0.03,     # pos [m]
            0.08, 0.08, 0.08,     # vel [m/s]
            0.0175, 0.0175, 0.0175,  # quat vector parts
            0.00175,              # quat w
        ])
        self.E_crs = np.diag([
            4.2, 3.5, 3.5,        # pos (compare-experiment scale)
            1.8, 1.6, 1.6,        # vel
            20.0, 20.0, 20.0,     # quat vector parts
            20.0,                 # quat w
        ])
        self.nw = self.nx

    def ode(self, X, u):
        p = self.params
        m = p['mass']
        g = p['gravity_constant']

        # unpack state
        v = ca.vertcat(X[3], X[4], X[5])
        q = X[6:10]  # [qx, qy, qz, qw]
        qw = q[3]
        qx = q[0]
        qy = q[1]
        qz = q[2]

        # unpack inputs
        u_F = u[0]                      # thrust deviation from hover; thrust = u_F + m*g
        wx, wy, wz = u[1], u[2], u[3]   # body angular rates [rad/s]

        # rotation matrix body -> world (same formulation as dyn.rocket.Rocket)
        r00 = 1 - 2 * qy ** 2 - 2 * qz ** 2
        r01 = 2 * qx * qy - 2 * qz * qw
        r02 = 2 * qx * qz + 2 * qy * qw
        r10 = 2 * qx * qy + 2 * qz * qw
        r11 = 1 - 2 * qx ** 2 - 2 * qz ** 2
        r12 = 2 * qy * qz - 2 * qx * qw
        r20 = 2 * qx * qz - 2 * qy * qw
        r21 = 2 * qy * qz + 2 * qx * qw
        r22 = 1 - 2 * qx ** 2 - 2 * qy ** 2
        R = ca.vertcat(
            ca.hcat([r00, r01, r02]),
            ca.hcat([r10, r11, r12]),
            ca.hcat([r20, r21, r22])
        )

        # translational dynamics: thrust along body +Z, gravity in world -Z
        Fb = ca.vertcat(0, 0, u_F + m * g)
        acc = (1 / m) * ca.mtimes(R, Fb)
        acc = ca.vertcat(acc[0], acc[1], acc[2] - g)

        # attitude kinematics only, in unambiguous vector form
        # (q = [v; s], robot-local rates):  s_dot = -0.5*(w . v),  v_dot = 0.5*(s*w + cross(w, v))
        # No attitude-rate dynamics: omega enters as a direct input.
        w_vec = ca.vertcat(wx, wy, wz)
        v_vec = ca.vertcat(qx, qy, qz)
        s = qw

        s_dot = -0.5 * ca.dot(w_vec, v_vec)
        v_dot = 0.5 * (s * w_vec + ca.cross(w_vec, v_vec))

        return ca.vertcat(v, acc, ca.vertcat(v_dot, s_dot))
