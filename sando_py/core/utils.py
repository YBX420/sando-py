"""Algorithmic utilities ported from include/sando/utils.hpp + src/sando/utils.cpp.

Faithful translation of functions in namespace ``sando_utils``. ROS-message-only
helpers (convertPwpMsg2Pwp, convertPwp2PwpMsg, convertPwp2ColoredMarkerArray,
convertEigen2Point, convertCovMsg2Cov, convertCov2CovMsg, getColor as RGBA,
identityGeometryMsgsPose, convertMarkerArray2Vec_Vec_Vecf3) are intentionally
omitted — they have no ROS analog in a pure-Python build. The color *constants*
are preserved as ints for cross-reference.

Eigen::Vector3d -> np.ndarray shape (3,).
vec_Vecf<3>     -> List[np.ndarray].
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as _Rot

from .piecewise_poly import PieceWisePol
from .types import RobotState


# --- color id constants (utils.hpp:28-40) -----------------------------------
red_normal        = 1
red_trans         = 2
red_trans_trans   = 3
green_normal      = 4
blue_normal       = 5
blue_trans        = 6
blue_trans_trans  = 7
blue_light        = 8
yellow_normal     = 9
orange_trans      = 10
black_trans       = 11
teal_normal       = 12
green_trans_trans = 13


# --- sgn   (utils.hpp:258) ---------------------------------------------------
def sgn(val) -> int:
    # Python int needed because numpy bool subtraction is unsupported on np.float64.
    return int(0 < val) - int(val < 0)


# --- saturate   (utils.cpp:421) ----------------------------------------------
def saturate(var: float, lo: float, hi: float) -> float:
    """Functional form; C++ version mutates in place — Python returns clamped value."""
    if var < lo:
        return lo
    if var > hi:
        return hi
    return var


# --- angle_wrap   (utils.cpp:429) --------------------------------------------
def angle_wrap(diff: float) -> float:
    """Wrap angle to [-pi, pi]. C++ mutates in place; Python returns wrapped value."""
    diff = math.fmod(diff + math.pi, 2.0 * math.pi)
    if diff < 0:
        diff += 2.0 * math.pi
    return diff - math.pi


# --- euclideanDistance   (utils.cpp:558) -------------------------------------
def euclideanDistance(p1: np.ndarray, p2: np.ndarray) -> float:
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    return float(np.linalg.norm(p1 - p2))


# --- getIntersectionWithPlane   (hgp/utils.cpp:529) --------------------------
def getIntersectionWithPlane(P1: np.ndarray, P2: np.ndarray, coeff: np.ndarray):
    """Plane (Ax+By+Cz+D=0) vs segment P1-P2.

    Returns (hit, intersection). `hit` is True iff intersection parameter t ∈ [0,1].
    """
    A, B, C, D = float(coeff[0]), float(coeff[1]), float(coeff[2]), float(coeff[3])
    x1, y1, z1 = float(P1[0]), float(P1[1]), float(P1[2])
    a = float(P2[0]) - x1
    b = float(P2[1]) - y1
    c = float(P2[2]) - z1
    denom = A * a + B * b + C * c
    if denom == 0.0:
        return False, np.array([x1, y1, z1])
    t = -(A * x1 + B * y1 + C * z1 + D) / denom
    inter = np.array([x1 + a * t, y1 + b * t, z1 + c * t])
    return (0.0 <= t <= 1.0), inter


# --- projectPointToBox   (utils.cpp:437) -------------------------------------
def projectPointToBox(P1: np.ndarray, P2: np.ndarray,
                      wdx: float, wdy: float, wdz: float) -> np.ndarray:
    """Project P2 onto the surface of an AABB centered at P1 with full widths wdx/wdy/wdz."""
    P1 = np.asarray(P1, dtype=float)
    P2 = np.asarray(P2, dtype=float)

    x_max = P1[0] + wdx / 2; x_min = P1[0] - wdx / 2
    y_max = P1[1] + wdy / 2; y_min = P1[1] - wdy / 2
    z_max = P1[2] + wdz / 2; z_min = P1[2] - wdz / 2

    # goal already inside the box
    if (x_min < P2[0] < x_max) and (y_min < P2[1] < y_max) and (z_min < P2[2] < z_max):
        return P2

    planes = [
        np.array([ 1.0, 0.0, 0.0, -x_max]),
        np.array([-1.0, 0.0, 0.0,  x_min]),
        np.array([ 0.0, 1.0, 0.0, -y_max]),
        np.array([ 0.0,-1.0, 0.0,  y_min]),
        np.array([ 0.0, 0.0, 1.0, -z_max]),
        np.array([ 0.0, 0.0,-1.0,  z_min]),
    ]

    intersections: List[np.ndarray] = []
    for pl in planes:
        hit, inter = getIntersectionWithPlane(P1, P2, pl)
        if hit:
            intersections.append(inter)

    if len(intersections) == 0:
        # Mirrors the RCLCPP_ERROR in C++ — should not happen if P2 is outside the box.
        raise RuntimeError("projectPointToBox: no intersection found (should be impossible)")

    dists = [np.linalg.norm(p - P1) for p in intersections]
    inter = intersections[int(np.argmin(dists))]

    # utils.cpp:491 — pull back 1.0 along (inter - P1) to keep inside the box
    v = (inter - P1)
    n = np.linalg.norm(v)
    if n > 0:
        inter = inter - (v / n) * 1.0
    return inter


# --- projectPointToSphere   (utils.cpp:498) ----------------------------------
def projectPointToSphere(P1: np.ndarray, P2: np.ndarray, radius: float) -> np.ndarray:
    P1 = np.asarray(P1, dtype=float)
    P2 = np.asarray(P2, dtype=float)
    diff = P2 - P1
    n = np.linalg.norm(diff)
    if n <= radius:
        return P2
    return P1 + (diff / n) * radius


# --- createMoreVertexes   (utils.cpp:540) ------------------------------------
def createMoreVertexes(path: List[np.ndarray], d: float) -> None:
    """Subdivide any segment longer than d.  Mutates `path` in place."""
    if len(path) < 2:
        return
    j = 0
    while j < len(path) - 1:
        dist = float(np.linalg.norm(path[j + 1] - path[j]))
        if dist > d:
            v = (path[j + 1] - path[j]) / dist
            n_add = int(math.floor(dist / d))
            for _ in range(n_add):
                path.insert(j + 1, path[j] + v * d)
                j += 1
        j += 1


# --- quaternion2Euler   (utils.cpp:407-419) ---------------------------------
def quaternion2Euler(q) -> Tuple[float, float, float]:
    """Quaternion -> (roll, pitch, yaw). Accepts (x,y,z,w) tuple/array."""
    qx, qy, qz, qw = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    # scipy expects (x,y,z,w)
    r = _Rot.from_quat([qx, qy, qz, qw])
    roll, pitch, yaw = r.as_euler("xyz", degrees=False)
    return float(roll), float(pitch), float(yaw)


# --- transformStampedToMatrix   (utils.cpp:564) ------------------------------
def transformStampedToMatrix(translation, quaternion) -> np.ndarray:
    """Build a 4x4 homogeneous transform from translation (tx,ty,tz) and quaternion (qx,qy,qz,qw)."""
    tx, ty, tz = float(translation[0]), float(translation[1]), float(translation[2])
    qx, qy, qz, qw = (
        float(quaternion[0]), float(quaternion[1]),
        float(quaternion[2]), float(quaternion[3]),
    )
    R = _Rot.from_quat([qx, qy, qz, qw]).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[0, 3], T[1, 3], T[2, 3] = tx, ty, tz
    return T


# --- convertCoefficients2ControlPoints   (utils.cpp:311) --------------------
def convertCoefficients2ControlPoints(
    pwp: PieceWisePol, A_rest_pos_basis_inverse: np.ndarray
) -> List[np.ndarray]:
    """P_i = [coeff_x ; coeff_y ; coeff_z],   V_i = P_i * A^-1."""
    out: List[np.ndarray] = []
    for i in range(len(pwp.coeff_x)):
        P = np.vstack([pwp.coeff_x[i], pwp.coeff_y[i], pwp.coeff_z[i]])  # 3x4
        V = P @ A_rest_pos_basis_inverse                                   # 3x4
        out.append(V)
    return out


# --- getMinTimeDoubleIntegrator1D   (utils.cpp:333) -------------------------
def getMinTimeDoubleIntegrator1D(p0: float, v0: float, pf: float, vf: float,
                                 v_max: float, a_max: float) -> float:
    """Time-optimal double-integrator (1D)  — IOP 2017 paper notation."""
    x1, x2 = v0, p0
    x1r, x2r = vf, pf
    k1 = a_max
    k2 = 1.0
    x1_bar = v_max

    B = (k2 / (2 * k1)) * sgn(-x1 + x1r) * (x1 ** 2 - x1r ** 2) + x2r
    C = (k2 / (2 * k1)) * (x1 ** 2 + x1r ** 2) - (k2 / k1) * x1_bar ** 2 + x2r
    D = (-k2 / (2 * k1)) * (x1 ** 2 + x1r ** 2) + (k2 / k1) * x1_bar ** 2 + x2r

    if (x2 <= B) and (x2 >= C):
        return (
            -k2 * (x1 + x1r)
            + 2 * math.sqrt(
                k2 ** 2 * x1 ** 2
                - k1 * k2 * ((k2 / (2 * k1)) * (x1 ** 2 - x1r ** 2) + x2 - x2r)
            )
        ) / (k1 * k2)
    elif (x2 <= B) and (x2 < C):
        return (
            (x1_bar - x1 - x1r) / k1
            + (x1 ** 2 + x1r ** 2) / (2 * k1 * x1_bar)
            + (x2r - x2) / (k2 * x1_bar)
        )
    elif (x2 > B) and (x2 <= D):
        return (
            k2 * (x1 + x1r)
            + 2 * math.sqrt(
                k2 ** 2 * x1 ** 2
                + k1 * k2 * ((k2 / (2 * k1)) * (-x1 ** 2 + x1r ** 2) + x2 - x2r)
            )
        ) / (k1 * k2)
    else:  # (x2 > B) and (x2 > D)
        return (
            (x1_bar + x1 + x1r) / k1
            + (x1 ** 2 + x1r ** 2) / (2 * k1 * x1_bar)
            + (-x2r + x2) / (k2 * x1_bar)
        )


# --- getMinTimeDoubleIntegrator3D   (utils.cpp:391) -------------------------
def getMinTimeDoubleIntegrator3D(p0, v0, pf, vf, v_max, a_max) -> float:
    p0 = np.asarray(p0, dtype=float); v0 = np.asarray(v0, dtype=float)
    pf = np.asarray(pf, dtype=float); vf = np.asarray(vf, dtype=float)
    v_max = np.asarray(v_max, dtype=float); a_max = np.asarray(a_max, dtype=float)
    tx = getMinTimeDoubleIntegrator1D(p0[0], v0[0], pf[0], vf[0], v_max[0], a_max[0])
    ty = getMinTimeDoubleIntegrator1D(p0[1], v0[1], pf[1], vf[1], v_max[1], a_max[1])
    tz = getMinTimeDoubleIntegrator1D(p0[2], v0[2], pf[2], vf[2], v_max[2], a_max[2])
    return max(tx, ty, tz)


# --- findVelocitiesInPath   (utils.cpp:591) ---------------------------------
def findVelocitiesInPath(path: List[np.ndarray],
                         A: RobotState,
                         v_max_3d: np.ndarray,
                         verbose: bool = False,
                         *,
                         threshold_distance: float = 5.0) -> List[np.ndarray]:
    """Heuristic velocity at every waypoint along ``path``.

    ``threshold_distance`` (metres) controls how aggressively velocity
    ramps from zero up to ``v_max_3d``: distances larger than this get
    full velocity, smaller distances scale linearly. The default 5 m
    matches the C++ ``utils.cpp:601`` hardcoded value; pass
    ``Parameters.threshold_distance_velocity`` to override.
    """
    velocities: List[np.ndarray] = []
    velocities.append(np.asarray(A.vel, dtype=float).copy())

    for i in range(len(path) - 2):
        p0, p1, p2 = path[i], path[i + 1], path[i + 2]
        d01 = p1 - p0
        d12 = p2 - p1
        d02 = p2 - p0

        v_next = np.zeros(3)
        for j in range(3):
            # Case 1: both non-negative, at least one positive
            if (d01[j] > 0 and d12[j] > 0) or (d01[j] == 0 and d12[j] > 0) or (d01[j] > 0 and d12[j] == 0):
                dist02 = abs(d02[j])
                if dist02 > threshold_distance:
                    v_next[j] = v_max_3d[j]
                else:
                    v_next[j] = v_max_3d[j] * dist02 / threshold_distance
                continue
            # Case 2: positive then negative -> 0
            if d01[j] > 0 and d12[j] < 0:
                v_next[j] = 0.0
                continue
            # Case 3: negative then positive -> 0
            if d01[j] < 0 and d12[j] > 0:
                v_next[j] = 0.0
                continue
            # Case 4: both non-positive, at least one negative
            if (d01[j] < 0 and d12[j] < 0) or (d01[j] == 0 and d12[j] < 0) or (d01[j] < 0 and d12[j] == 0):
                dist02 = abs(d02[j])
                if dist02 > threshold_distance:
                    v_next[j] = -v_max_3d[j]
                else:
                    v_next[j] = -v_max_3d[j] * dist02 / threshold_distance
                continue
            # Case 5: zero & zero
            v_next[j] = 0.0
        velocities.append(v_next)

    velocities.append(np.zeros(3))
    assert len(path) == len(velocities), "path.size() != velocities.size()"
    return velocities


# --- getTravelTimes   (utils.cpp:697) ---------------------------------------
def getTravelTimes(path: List[np.ndarray],
                   A: RobotState,
                   v_max_3d: np.ndarray,
                   a_max_3d: np.ndarray,
                   debug_verbose: bool = False,
                   *,
                   threshold_distance: float = 5.0) -> List[float]:
    if len(path) == 1:
        return []
    if len(path) == 2:
        return [
            getMinTimeDoubleIntegrator3D(path[0], A.vel, path[1], np.zeros(3), v_max_3d, a_max_3d)
        ]
    velocities = findVelocitiesInPath(
        path, A, np.asarray(v_max_3d, dtype=float), debug_verbose,
        threshold_distance=threshold_distance,
    )

    travel_times: List[float] = []
    for i in range(len(path) - 1):
        t = getMinTimeDoubleIntegrator3D(
            path[i], velocities[i], path[i + 1], velocities[i + 1], v_max_3d, a_max_3d
        )
        if debug_verbose:
            print(f"i: {i} of {len(path) - 1}")
            print(f"Start: {path[i]}")
            print(f"start velocity: {velocities[i]}")
            print(f"End: {path[i + 1]}")
            print(f"end velocity: {velocities[i + 1]}")
            print(f"travel time: {t}")
        travel_times.append(t)
    return travel_times
