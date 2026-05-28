"""Common utilities — geometry, time, msg <-> dataclass conversions."""
from __future__ import annotations

import math
import time
from typing import List, Optional, Tuple

import numpy as np

from .types import PieceWisePol, RobotState


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def angle_wrap(a: float) -> float:
    """Wrap to (-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def angle_diff(target: float, current: float) -> float:
    """Smallest signed angular distance from current to target."""
    return angle_wrap(target - current)


def yaw_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
    """ZYX Tait-Bryan yaw."""
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)


def quat_from_yaw(yaw: float) -> Tuple[float, float, float, float]:
    half = 0.5 * yaw
    return (0.0, 0.0, math.sin(half), math.cos(half))


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


def saturate(v: np.ndarray, vmax: float) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n > vmax and n > 1e-9:
        return v * (vmax / n)
    return v


def project_point_to_sphere(P1: np.ndarray, P2: np.ndarray, radius: float) -> np.ndarray:
    """Project P2 onto a sphere of `radius` centered at P1.  Returns P2 unchanged
    if already inside the sphere. Mirrors sando_utils::projectPointToSphere.
    """
    diff = P2 - P1
    n = float(np.linalg.norm(diff))
    if n <= radius:
        return np.array(P2, dtype=float)
    return P1 + diff * (radius / n)


def project_point_to_box(P1: np.ndarray, P2: np.ndarray,
                         wdx: float, wdy: float, wdz: float) -> np.ndarray:
    """Project P2 onto the surface of an AABB of half-widths (wdx/2, wdy/2, wdz/2)
    centered at P1. Returns P2 unchanged if already inside the box.
    """
    x_max = P1[0] + wdx / 2.0
    x_min = P1[0] - wdx / 2.0
    y_max = P1[1] + wdy / 2.0
    y_min = P1[1] - wdy / 2.0
    z_max = P1[2] + wdz / 2.0
    z_min = P1[2] - wdz / 2.0
    if (x_min < P2[0] < x_max and y_min < P2[1] < y_max and z_min < P2[2] < z_max):
        return np.array(P2, dtype=float)
    # Direction from P1 to P2; parametric search for the closest box face intersection
    v = P2 - P1
    if np.linalg.norm(v) < 1e-12:
        return np.array(P1, dtype=float)
    best_t = float("inf")
    best_pt = np.array(P2, dtype=float)
    planes = [
        (np.array([1.0, 0.0, 0.0]), -x_max),
        (np.array([-1.0, 0.0, 0.0]), x_min),
        (np.array([0.0, 1.0, 0.0]), -y_max),
        (np.array([0.0, -1.0, 0.0]), y_min),
        (np.array([0.0, 0.0, 1.0]), -z_max),
        (np.array([0.0, 0.0, -1.0]), z_min),
    ]
    for n_pl, d_pl in planes:
        denom = float(np.dot(n_pl, v))
        if abs(denom) < 1e-12:
            continue
        t = -(float(np.dot(n_pl, P1)) + d_pl) / denom
        if 0.0 < t < best_t:
            best_t = t
            best_pt = P1 + v * t
    return best_pt


def min_time_double_integrator_1d(p0: float, v0: float, pf: float, vf: float,
                                  v_max: float, a_max: float) -> float:
    """Time-optimal point-to-point transfer time for a 1D double integrator
    with bounded velocity and acceleration. Mirrors getMinTimeDoubleIntegrator1D.
    """
    x1, x2 = v0, p0
    x1r, x2r = vf, pf
    k1 = a_max
    k2 = 1.0
    x1_bar = v_max
    sign = 1.0 if (-x1 + x1r) > 0 else (-1.0 if (-x1 + x1r) < 0 else 0.0)
    B = (k2 / (2 * k1)) * sign * (x1 ** 2 - x1r ** 2) + x2r
    C = (k2 / (2 * k1)) * (x1 ** 2 + x1r ** 2) - (k2 / k1) * x1_bar ** 2 + x2r
    D = (-k2 / (2 * k1)) * (x1 ** 2 + x1r ** 2) + (k2 / k1) * x1_bar ** 2 + x2r
    if x2 <= B and x2 >= C:
        inside = (k2 ** 2) * (x1 ** 2) - k1 * k2 * ((k2 / (2 * k1)) * (x1 ** 2 - x1r ** 2) + x2 - x2r)
        inside = max(0.0, inside)
        t = (-k2 * (x1 + x1r) + 2 * math.sqrt(inside)) / (k1 * k2)
    elif x2 <= B and x2 < C:
        t = (x1_bar - x1 - x1r) / k1 + (x1 ** 2 + x1r ** 2) / (2 * k1 * x1_bar) + (x2r - x2) / (k2 * x1_bar)
    elif x2 > B and x2 <= D:
        inside = (k2 ** 2) * (x1 ** 2) + k1 * k2 * ((k2 / (2 * k1)) * (-x1 ** 2 + x1r ** 2) + x2 - x2r)
        inside = max(0.0, inside)
        t = (k2 * (x1 + x1r) + 2 * math.sqrt(inside)) / (k1 * k2)
    else:
        t = (x1_bar + x1 + x1r) / k1 + (x1 ** 2 + x1r ** 2) / (2 * k1 * x1_bar) + (-x2r + x2) / (k2 * x1_bar)
    return float(t)


def min_time_double_integrator_3d(p0: np.ndarray, v0: np.ndarray,
                                  pf: np.ndarray, vf: np.ndarray,
                                  v_max: np.ndarray, a_max: np.ndarray) -> float:
    """Maximum over axes of the per-axis 1D minimum-time transfer."""
    tx = min_time_double_integrator_1d(p0[0], v0[0], pf[0], vf[0], v_max[0], a_max[0])
    ty = min_time_double_integrator_1d(p0[1], v0[1], pf[1], vf[1], v_max[1], a_max[1])
    tz = min_time_double_integrator_1d(p0[2], v0[2], pf[2], vf[2], v_max[2], a_max[2])
    return max(tx, ty, tz)


def create_more_vertexes(path: List[np.ndarray], d: float) -> List[np.ndarray]:
    """Subdivide each segment of `path` so that no two consecutive vertices are
    farther apart than `d`. Mirrors createMoreVertexes.
    """
    if len(path) < 2 or d <= 0.0:
        return [p.copy() for p in path]
    out: List[np.ndarray] = [path[0].copy()]
    for a, b in zip(path[:-1], path[1:]):
        seg = b - a
        L = float(np.linalg.norm(seg))
        if L <= 1e-9:
            continue
        n = int(np.floor(L / d))
        u = seg / L
        for k in range(1, n + 1):
            out.append(a + u * (k * d))
        if np.linalg.norm(out[-1] - b) > 1e-6:
            out.append(b.copy())
    return out


def transform_stamped_to_matrix(transform_stamped) -> np.ndarray:
    """Convert a geometry_msgs/TransformStamped to a 4x4 numpy matrix."""
    t = transform_stamped.transform.translation
    q = transform_stamped.transform.rotation
    qx, qy, qz, qw = float(q.x), float(q.y), float(q.z), float(q.w)
    R = np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])
    M = np.eye(4)
    M[:3, :3] = R
    M[0, 3] = float(t.x); M[1, 3] = float(t.y); M[2, 3] = float(t.z)
    return M


def transform_inverse_se3(M: np.ndarray) -> np.ndarray:
    """Analytic SE(3) inverse of a 4x4 transform: [R | t; 0 0 0 1]^{-1} = [R^T | -R^T t; 0 0 0 1]."""
    R = M[:3, :3]
    t = M[:3, 3]
    RT = R.T
    M_inv = np.eye(4)
    M_inv[:3, :3] = RT
    M_inv[:3, 3] = -RT @ t
    return M_inv


# ---------------------------------------------------------------------------
# ROS time helpers
# ---------------------------------------------------------------------------

def ros_time_seconds(node) -> float:
    """Return the node's current ROS clock in seconds (float)."""
    t = node.get_clock().now()
    return float(t.nanoseconds) * 1e-9


def builtin_to_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def seconds_to_builtin(seconds: float):
    from builtin_interfaces.msg import Time
    msg = Time()
    msg.sec = int(seconds)
    msg.nanosec = int((seconds - int(seconds)) * 1e9)
    return msg


# ---------------------------------------------------------------------------
# Wall-clock timing (for profiling — independent of ROS clock)
# ---------------------------------------------------------------------------

class Timer:
    def __init__(self):
        self._t0 = time.perf_counter()

    def reset(self) -> None:
        self._t0 = time.perf_counter()

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000.0

    def elapsed_s(self) -> float:
        return time.perf_counter() - self._t0


# ---------------------------------------------------------------------------
# Message <-> dataclass conversion (dynus_interfaces)
# ---------------------------------------------------------------------------

def state_to_msg(s: RobotState):
    """Convert RobotState -> dynus_interfaces.msg.State (best effort)."""
    try:
        from dynus_interfaces.msg import State as StateMsg
    except ImportError:
        return None
    from geometry_msgs.msg import Vector3, Quaternion

    m = StateMsg()
    m.pos = Vector3(x=float(s.pos[0]), y=float(s.pos[1]), z=float(s.pos[2]))
    m.vel = Vector3(x=float(s.vel[0]), y=float(s.vel[1]), z=float(s.vel[2]))
    qx, qy, qz, qw = quat_from_yaw(s.yaw)
    m.quat = Quaternion(x=qx, y=qy, z=qz, w=qw)
    return m


def msg_to_state(msg) -> RobotState:
    s = RobotState()
    if hasattr(msg, "pos"):
        s.pos = np.array([msg.pos.x, msg.pos.y, msg.pos.z])
    if hasattr(msg, "vel"):
        s.vel = np.array([msg.vel.x, msg.vel.y, msg.vel.z])
    if hasattr(msg, "quat"):
        s.yaw = yaw_from_quat(msg.quat.x, msg.quat.y, msg.quat.z, msg.quat.w)
    if hasattr(msg, "header") and hasattr(msg.header, "stamp"):
        s.t = builtin_to_seconds(msg.header.stamp)
    return s


def state_to_goal_msg(s: RobotState):
    """Convert RobotState -> dynus_interfaces.msg.Goal."""
    try:
        from dynus_interfaces.msg import Goal as GoalMsg
    except ImportError:
        return None
    from geometry_msgs.msg import Vector3

    g = GoalMsg()
    g.p = Vector3(x=float(s.pos[0]), y=float(s.pos[1]), z=float(s.pos[2]))
    g.v = Vector3(x=float(s.vel[0]), y=float(s.vel[1]), z=float(s.vel[2]))
    g.a = Vector3(x=float(s.accel[0]), y=float(s.accel[1]), z=float(s.accel[2]))
    g.j = Vector3(x=float(s.jerk[0]), y=float(s.jerk[1]), z=float(s.jerk[2]))
    g.yaw = float(s.yaw)
    g.dyaw = float(s.dyaw)
    g.power = True
    g.mode_xy = 0
    g.mode_z = 0
    return g


def pwp_to_msg(pwp: PieceWisePol):
    try:
        from dynus_interfaces.msg import PWPTraj, CoeffPoly3
    except ImportError:
        return None

    m = PWPTraj()
    m.times = list(map(float, pwp.times))
    for cx, cy, cz in zip(pwp.coeff_x, pwp.coeff_y, pwp.coeff_z):
        m.coeff_x.append(CoeffPoly3(a=float(cx[0]), b=float(cx[1]), c=float(cx[2]), d=float(cx[3])))
        m.coeff_y.append(CoeffPoly3(a=float(cy[0]), b=float(cy[1]), c=float(cy[2]), d=float(cy[3])))
        m.coeff_z.append(CoeffPoly3(a=float(cz[0]), b=float(cz[1]), c=float(cz[2]), d=float(cz[3])))
    return m


def msg_to_pwp(msg) -> PieceWisePol:
    pwp = PieceWisePol()
    pwp.times = list(msg.times)
    for cx in msg.coeff_x:
        pwp.coeff_x.append(np.array([cx.a, cx.b, cx.c, cx.d]))
    for cy in msg.coeff_y:
        pwp.coeff_y.append(np.array([cy.a, cy.b, cy.c, cy.d]))
    for cz in msg.coeff_z:
        pwp.coeff_z.append(np.array([cz.a, cz.b, cz.c, cz.d]))
    return pwp


# ---------------------------------------------------------------------------
# Path simplification helpers (collinear / corner removal)
# ---------------------------------------------------------------------------

def remove_collinear(points: List[np.ndarray], tol: float = 1e-3) -> List[np.ndarray]:
    if len(points) <= 2:
        return [p.copy() for p in points]
    out = [points[0].copy()]
    for i in range(1, len(points) - 1):
        a = out[-1]
        b = points[i]
        c = points[i + 1]
        ab = b - a
        bc = c - b
        n_ab = np.linalg.norm(ab)
        n_bc = np.linalg.norm(bc)
        if n_ab < 1e-9 or n_bc < 1e-9:
            continue
        cross = np.linalg.norm(np.cross(ab, bc))
        if cross / (n_ab * n_bc) > tol:
            out.append(b.copy())
    out.append(points[-1].copy())
    return out


def collapse_short_edges(points: List[np.ndarray], min_len: float,
                         is_blocked=None) -> List[np.ndarray]:
    if not points:
        return []
    out = [points[0].copy()]
    n = len(points)
    for i in range(1, n):
        p = points[i]
        if np.linalg.norm(p - out[-1]) >= min_len:
            out.append(p.copy())
        elif is_blocked is not None and i + 1 < n and is_blocked(out[-1], points[i + 1]):
            # Dropping p would merge across an obstacle -> keep it (safety over brevity).
            out.append(p.copy())
    if len(out) >= 2 and np.linalg.norm(out[-1] - points[-1]) > 1e-9:
        out[-1] = points[-1].copy()
    elif len(out) == 1:
        out.append(points[-1].copy())
    return out


def angle_spacing_filter(points: List[np.ndarray], min_turn_deg: float,
                         min_seg_len: float, is_blocked=None) -> List[np.ndarray]:
    if len(points) <= 2 or min_turn_deg <= 0:
        return [p.copy() for p in points]
    cos_thr = math.cos(math.radians(min_turn_deg))
    out = [points[0].copy()]
    for i in range(1, len(points) - 1):
        a = out[-1]
        b = points[i]
        c = points[i + 1]
        ab = b - a
        bc = c - b
        nab = np.linalg.norm(ab)
        nbc = np.linalg.norm(bc)
        want_skip = False
        if nab < min_seg_len or nbc < min_seg_len:
            want_skip = True
        else:
            cosang = float(np.dot(ab, bc) / (nab * nbc))
            if cosang > cos_thr:  # big cosang (close to 1) means tiny turn
                want_skip = True
        # Never drop b if the merged segment a->c would clip an obstacle.
        if want_skip and (is_blocked is None or not is_blocked(a, c)):
            continue
        out.append(b.copy())
    out.append(points[-1].copy())
    return out
