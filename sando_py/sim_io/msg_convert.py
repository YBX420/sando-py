"""Bi-directional conversion between ROS 2 messages and ``sando_py.core``
types. Mirrors the C++ helpers in ``utils.cpp`` and the conversion code
inline in ``sando_node.cpp`` (``stateCallback``, ``trajCallback``,
``terminalGoalCallback``, ``publishGoal``).

Only loadable inside a sourced ROS 2 environment — ``dynus_interfaces.msg``
and ``geometry_msgs.msg`` are imported at module top. The pure-Python
algorithm subpackages (``core``, ``hgp``, ``decomp``, ``solver``) do NOT
import this module.

C++ -> Python mapping table
---------------------------
* ``stateCallback``                       -> ``state_msg_to_robot_state``
* ``terminalGoalCallback``                -> ``pose_stamped_to_goal``
* ``publishGoal``                         -> ``robot_state_to_goal_msg``
* ``sando_utils::convertPwpMsg2Pwp``      -> ``pwp_msg_to_pwp``
* ``sando_utils::convertPwp2PwpMsg``      -> ``pwp_to_pwp_msg``
* ``SANDO_NODE::convertDynTrajMsg2DynTraj`` -> ``dyn_traj_msg_to_dyn_traj``
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

# ROS messages — these imports require a sourced ROS 2 environment.
from dynus_interfaces.msg import (  # noqa: E402
    CoeffPoly3 as CoeffPoly3Msg,
    DynTraj as DynTrajMsg,
    DynTrajArray as DynTrajArrayMsg,
    Goal as GoalMsg,
    PWPTraj as PWPTrajMsg,
    State as StateMsg,
)
from geometry_msgs.msg import (  # noqa: E402
    PoseStamped,
    Quaternion,
    Vector3,
)
from sensor_msgs.msg import PointCloud2  # noqa: E402
from sensor_msgs_py import point_cloud2 as _pcl2  # noqa: E402
from std_msgs.msg import Header  # noqa: E402, F401  (re-exported convenience)

from sando_py.core import (
    DynTraj,
    Parameters,
    PieceWisePol,
    RobotState,
    TrajMode,
)
from sando_py.core.utils import quaternion2Euler

__all__ = [
    # low-level helpers
    "vec3_to_array",
    "array_to_vec3",
    "quat_to_yaw",
    # State <-> RobotState
    "state_msg_to_robot_state",
    # PoseStamped -> goal point
    "pose_stamped_to_goal",
    # RobotState -> Goal
    "robot_state_to_goal_msg",
    # PWPTraj <-> PieceWisePol
    "pwp_msg_to_pwp",
    "pwp_to_pwp_msg",
    # Own trajectory broadcast (multi-agent /trajs)
    "pwp_to_own_dyn_traj_msg",
    # DynTraj msg -> core.DynTraj
    "dyn_traj_msg_to_dyn_traj",
    "dyn_traj_array_msg_to_dyn_trajs",
    # PointCloud2 -> (N, 3) numpy
    "cloud_msg_to_points",
    "transform_points_to_frame",
]


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def vec3_to_array(v: Vector3) -> np.ndarray:
    """``geometry_msgs/Vector3`` -> ``np.ndarray`` shape (3,)."""
    return np.array([v.x, v.y, v.z], dtype=float)


def array_to_vec3(a: Iterable[float]) -> Vector3:
    """``np.ndarray`` shape (3,) -> ``geometry_msgs/Vector3``."""
    arr = np.asarray(a, dtype=float).reshape(3)
    out = Vector3()
    out.x, out.y, out.z = float(arr[0]), float(arr[1]), float(arr[2])
    return out


def quat_to_yaw(q: Quaternion) -> float:
    """Extract the yaw component (rotation about z) from a quaternion.

    Delegates to ``core.utils.quaternion2Euler`` which uses scipy's rXYZ
    convention — same as the C++ ``quaternion2Euler`` in utils.cpp.
    """
    _, _, yaw = quaternion2Euler([q.x, q.y, q.z, q.w])
    return float(yaw)


# ---------------------------------------------------------------------------
# State <-> RobotState
# ---------------------------------------------------------------------------

def state_msg_to_robot_state(
    msg: StateMsg, t_now: Optional[float] = None
) -> RobotState:
    """``dynus_interfaces/State`` -> ``core.RobotState``.

    Mirrors ``SANDO_NODE::stateCallback`` (sando_node.cpp:807-841): pos / vel
    copied through, accel zeroed (State carries no accel), yaw extracted from
    the quaternion.
    """
    s = RobotState()
    if t_now is not None:
        s.t = float(t_now)
    s.setPos(msg.pos.x, msg.pos.y, msg.pos.z)
    s.setVel(msg.vel.x, msg.vel.y, msg.vel.z)
    s.setAccel(0.0, 0.0, 0.0)
    s.setYaw(quat_to_yaw(msg.quat))
    return s


# ---------------------------------------------------------------------------
# PoseStamped -> terminal goal
# ---------------------------------------------------------------------------

def pose_stamped_to_goal(msg: PoseStamped, params: Parameters) -> np.ndarray:
    """``geometry_msgs/PoseStamped`` -> 3D goal point as ``np.ndarray`` (3,).

    Honours ``params.force_goal_z`` / ``params.default_goal_z`` and validates
    against ``[z_min, z_max]``. Mirrors ``SANDO_NODE::terminalGoalCallback``
    (sando_node.cpp:951-979).

    Raises:
        ValueError if the resolved z is outside ``[z_min, z_max]``.
    """
    if params.force_goal_z:
        z = float(params.default_goal_z)
    else:
        z = float(msg.pose.position.z)
    if z < params.z_min or z > params.z_max:
        raise ValueError(
            f"Goal z={z} is out of bounds [{params.z_min}, {params.z_max}]"
        )
    return np.array(
        [float(msg.pose.position.x), float(msg.pose.position.y), z], dtype=float
    )


# ---------------------------------------------------------------------------
# RobotState -> Goal setpoint
# ---------------------------------------------------------------------------

def robot_state_to_goal_msg(
    state: RobotState,
    frame_id: str = "map",
    stamp=None,
) -> GoalMsg:
    """Build a ``dynus_interfaces/Goal`` from a setpoint ``RobotState``.

    Mirrors ``SANDO_NODE::publishGoal`` (sando_node.cpp:1534-1566). Mode is
    set to position control on both XY and Z by default — matches the C++
    behaviour (only the explicit p/v/a/j/yaw/dyaw fields are filled in C++;
    mode_xy and mode_z default to zero = MODE_POSITION_CONTROL).
    """
    g = GoalMsg()
    g.header.frame_id = frame_id
    if stamp is not None:
        g.header.stamp = stamp
    g.p = array_to_vec3(state.pos)
    g.v = array_to_vec3(state.vel)
    g.a = array_to_vec3(state.accel)
    g.j = array_to_vec3(state.jerk)
    g.yaw = float(state.yaw)
    g.dyaw = float(state.dyaw)
    g.power = True
    g.mode_xy = GoalMsg.MODE_POSITION_CONTROL
    g.mode_z = GoalMsg.MODE_POSITION_CONTROL
    return g


# ---------------------------------------------------------------------------
# PWPTraj <-> PieceWisePol
# ---------------------------------------------------------------------------

def pwp_msg_to_pwp(msg: PWPTrajMsg) -> PieceWisePol:
    """``dynus_interfaces/PWPTraj`` -> ``core.PieceWisePol``.

    Mirrors ``sando_utils::convertPwpMsg2Pwp`` (utils.cpp:53-80).
    """
    n = len(msg.coeff_x)
    if n != len(msg.coeff_y) or n != len(msg.coeff_z):
        raise ValueError(
            "pwp_msg_to_pwp: coeff_x, coeff_y, coeff_z must have the same length "
            f"(got {n}, {len(msg.coeff_y)}, {len(msg.coeff_z)})"
        )

    out = PieceWisePol()
    out.times = [float(t) for t in msg.times]
    for cx, cy, cz in zip(msg.coeff_x, msg.coeff_y, msg.coeff_z):
        out.coeff_x.append(np.array([cx.a, cx.b, cx.c, cx.d], dtype=float))
        out.coeff_y.append(np.array([cy.a, cy.b, cy.c, cy.d], dtype=float))
        out.coeff_z.append(np.array([cz.a, cz.b, cz.c, cz.d], dtype=float))
    return out


def pwp_to_own_dyn_traj_msg(
    pwp: PieceWisePol,
    current_state,
    params: Parameters,
    agent_id: int,
    *,
    frame_id: str,
    stamp,
    terminal_goal=None,
) -> DynTrajMsg:
    """Build the ``/trajs`` ``DynTrajMsg`` an agent publishes about its
    own committed trajectory. Mirrors ``SANDO_NODE::publishOwnTraj``
    (sando_node.cpp:1413-1445)."""
    msg = DynTrajMsg()
    msg.header.frame_id = frame_id
    if stamp is not None:
        msg.header.stamp = stamp
    bbox = list(params.drone_bbox) if params.drone_bbox else [0.4, 0.4, 0.4]
    msg.bbox = [float(b) for b in bbox[:3]]
    msg.id = int(agent_id)
    msg.mode = "pwp"
    msg.pwp = pwp_to_pwp_msg(pwp)
    msg.is_agent = True
    if current_state is not None:
        msg.pos = array_to_vec3(np.asarray(current_state.pos, dtype=float))
    if terminal_goal is not None:
        g = np.asarray(terminal_goal, dtype=float).reshape(3)
        msg.goal = [float(g[0]), float(g[1]), float(g[2])]
    return msg


def pwp_to_pwp_msg(pwp: PieceWisePol) -> PWPTrajMsg:
    """``core.PieceWisePol`` -> ``dynus_interfaces/PWPTraj``.

    Mirrors ``sando_utils::convertPwp2PwpMsg`` (utils.cpp:13-51).
    """
    msg = PWPTrajMsg()
    msg.times = [float(t) for t in pwp.times]

    def _pack(coeffs):
        m = CoeffPoly3Msg()
        c = np.asarray(coeffs, dtype=float).reshape(4)
        m.a, m.b, m.c, m.d = float(c[0]), float(c[1]), float(c[2]), float(c[3])
        return m

    msg.coeff_x = [_pack(c) for c in pwp.coeff_x]
    msg.coeff_y = [_pack(c) for c in pwp.coeff_y]
    msg.coeff_z = [_pack(c) for c in pwp.coeff_z]
    return msg


# ---------------------------------------------------------------------------
# DynTraj message -> core.DynTraj
# ---------------------------------------------------------------------------

def dyn_traj_msg_to_dyn_traj(
    msg: DynTrajMsg, params: Parameters, t_now: float
) -> DynTraj:
    """``dynus_interfaces/DynTraj`` -> ``core.DynTraj``.

    Mirrors ``SANDO_NODE::convertDynTrajMsg2DynTraj`` (sando_node.cpp:1050-1150).
    Key behaviours preserved:
      * ``bbox`` is the obstacle half-bbox + ego drone half-bbox.
      * If ``params.use_only_curr_pos_for_dynamic_obst`` and the trajectory
        is not an agent, the future trajectory is replaced by a stationary
        analytic expression at ``msg.pos``.
      * Otherwise, the PWP is parsed unconditionally and the analytic
        expressions (``msg.function`` / ``msg.velocity``) override the mode
        if they compile successfully.
      * Covariances are only copied for non-agent trajs.
      * Communication delay is only computed for agent trajs.
    """
    traj = DynTraj()

    # bbox inflation: obstacle half-extent + ego drone half-extent
    obs_bbox = list(msg.bbox)
    drone_bbox = params.drone_bbox or [0.0, 0.0, 0.0]
    traj.bbox = np.array(
        [
            obs_bbox[0] / 2.0 + drone_bbox[0] / 2.0,
            obs_bbox[1] / 2.0 + drone_bbox[1] / 2.0,
            obs_bbox[2] / 2.0 + drone_bbox[2] / 2.0,
        ],
        dtype=float,
    )
    traj.id = int(msg.id)

    skip_future = (
        bool(params.use_only_curr_pos_for_dynamic_obst) and not bool(msg.is_agent)
    )

    if not skip_future:
        traj.pwp = pwp_msg_to_pwp(msg.pwp)
        traj.mode = TrajMode.Piecewise  # may be overridden to Analytic below
    else:
        # Stationary analytic obstacle pinned to its current reported pos
        traj.traj_x = str(float(msg.pos.x))
        traj.traj_y = str(float(msg.pos.y))
        traj.traj_z = str(float(msg.pos.z))
        traj.traj_vx = "0.0"
        traj.traj_vy = "0.0"
        traj.traj_vz = "0.0"
        if traj.compileAnalytic():
            traj.mode = TrajMode.Analytic

    # Covariances — only for non-agent trajs, and only when future is kept
    if not bool(msg.is_agent) and not skip_future:
        if msg.ekf_cov_p:
            traj.ekf_cov_p = np.array(msg.ekf_cov_p, dtype=float)
        if msg.ekf_cov_q:
            traj.ekf_cov_q = np.array(msg.ekf_cov_q, dtype=float)
        if msg.poly_cov:
            traj.poly_cov = np.array(msg.poly_cov, dtype=float)

    # Analytic expressions override the PWP mode when they compile
    if not skip_future:
        if len(msg.function) == 3:
            traj.traj_x, traj.traj_y, traj.traj_z = (
                msg.function[0],
                msg.function[1],
                msg.function[2],
            )
        if len(msg.velocity) == 3:
            traj.traj_vx, traj.traj_vy, traj.traj_vz = (
                msg.velocity[0],
                msg.velocity[1],
                msg.velocity[2],
            )
        if len(msg.function) == 3 and len(msg.velocity) == 3:
            if traj.compileAnalytic():
                traj.mode = TrajMode.Analytic
            # else: keep whatever mode we set (Piecewise from above)

    traj.time_received = float(t_now)
    traj.current_pos = vec3_to_array(msg.pos)
    traj.is_agent = bool(msg.is_agent)

    if traj.is_agent and len(msg.goal) == 3:
        traj.goal = np.array(msg.goal, dtype=float)

    # Communication delay — only for agent trajs, when enabled
    if traj.is_agent and params.use_comm_delay_inflation:
        stamp_sec = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        delay = float(t_now) - stamp_sec
        if delay < 0:
            delay = 0.0
        traj.communication_delay = delay

    return traj


def dyn_traj_array_msg_to_dyn_trajs(
    msg: DynTrajArrayMsg, params: Parameters, t_now: float
) -> list:
    """Convenience: ``DynTrajArray`` -> ``List[core.DynTraj]``.

    The sando node subscribes to per-obstacle ``DynTraj`` messages, not
    arrays, but ``dynamic_forest_node`` publishes an array on some topics —
    this saves callers from looping by hand.
    """
    return [dyn_traj_msg_to_dyn_traj(m, params, t_now) for m in msg.trajs]


# ---------------------------------------------------------------------------
# PointCloud2 -> (N, 3) numpy array
# ---------------------------------------------------------------------------

def cloud_msg_to_points(msg: PointCloud2) -> np.ndarray:
    """``sensor_msgs/PointCloud2`` -> ``np.ndarray`` of shape ``(N, 3)``,
    dtype ``float32``.

    NaN-bearing rows are skipped (typical for depth-cam clouds that
    include unmeasured pixels). Returns an empty ``(0, 3)`` array when
    the cloud is empty.

    Mirrors ``pcl::fromROSMsg`` + ``.points`` access in
    ``SANDO_NODE::occupancyMapCallback`` (sando_node.cpp:2078) — just
    the (x, y, z) channels, no intensity / color preserved. The Python
    port uses ``sensor_msgs_py.point_cloud2.read_points`` which returns
    a numpy structured array; we restack to a plain (N, 3).

    The cloud is assumed to already be in the map / world frame.
    Sensor-frame clouds need a separate TF transform before this
    conversion (the upstream demo publishes the global cloud in the
    map frame directly via ``map_generator/global_cloud``).
    """
    arr = _pcl2.read_points_numpy(
        msg, field_names=("x", "y", "z"), skip_nans=True,
    )
    if arr.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    # read_points_numpy returns (N, len(field_names)); already the
    # right shape, just enforce dtype.
    return arr.astype(np.float32, copy=False)


def transform_points_to_frame(
    points: np.ndarray,
    source_frame: str,
    stamp,
    tf_buffer,
    target_frame: str,
    *,
    timeout_s: float = 0.05,
) -> Optional[np.ndarray]:
    """Transform ``(N, 3)`` ``points`` from ``source_frame`` into
    ``target_frame`` using a tf2 ``Buffer``. Returns ``None`` when the
    transform isn't available (transient — try again next tick).

    No-op (returns the input array) when frames match or the source
    frame is empty (some publishers don't populate ``header.frame_id``).

    Mirrors what the C++ does implicitly via ``tf2_ros::Buffer`` +
    ``tf2::doTransform`` in callbacks that touch sensor data.
    """
    import rclpy.time
    import rclpy.duration
    from sando_py.core.utils import transformStampedToMatrix

    if points is None or points.size == 0:
        return points
    if not source_frame or source_frame == target_frame:
        return points

    try:
        tf = tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            rclpy.time.Time.from_msg(stamp),
            rclpy.duration.Duration(seconds=float(timeout_s)),
        )
    except Exception:  # noqa: BLE001  — tf2 raises several distinct types
        return None

    t = tf.transform.translation
    q = tf.transform.rotation
    T = transformStampedToMatrix(
        [t.x, t.y, t.z], [q.x, q.y, q.z, q.w],
    )
    R = T[:3, :3]
    tr = T[:3, 3]
    # Vectorised: out = points @ R.T + tr
    pts = np.asarray(points, dtype=np.float64)
    out = pts @ R.T + tr
    return out.astype(points.dtype, copy=False)
