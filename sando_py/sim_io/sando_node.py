"""``rclpy.Node`` that wraps the pure-Python algorithm code so it can drop
into the upstream ROS 2 simulation world (RViz, ``dynamic_forest_node``,
``fake_sim``, ``goal_sender``). Mirrors the subscriber / publisher / timer
topology of the C++ ``sando_node.cpp``.

State machine
-------------
``DroneStatus``: YAWING -> TRAVELING -> GOAL_SEEN -> GOAL_REACHED. Mirrors
the C++ ``enum class DroneStatus`` in ``sando.hpp``.

Algorithm coupling
------------------
``_replan_cb`` runs the v1 pipeline: ``hgp.plan -> decomp.sfc ->
solver.solve``. On success the new ``PieceWisePol`` replaces
``committed_traj_``; on failure the previous trajectory is kept (matches
the C++ Gurobi-failure branch). ``_publish_goal_cb`` evaluates
``committed_traj_`` at ``t_now`` for each goal-publish tick, falling
back to hovering at the current state until the first plan commits.

Launching
---------
    python -m sando_py.sim_io
    python -m sando_py.sim_io --ros-args -r __ns:=/NX01 \\
        -p config_file:=/path/to/sando.yaml -p agent_id:=0

The conftest opt-out flag plus ``scripts/with_ros_env.sh`` makes import
work without needing to ``colcon build`` anything inside sando-py.
"""

from __future__ import annotations

import math
import time
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from tf2_ros import Buffer, TransformListener

from decomp_ros_msgs.msg import PolyhedronArray
from dynus_interfaces.msg import (
    DynTraj as DynTrajMsg,
    Goal as GoalMsg,
    State as StateMsg,
)
from geometry_msgs.msg import PoseStamped, TransformStamped
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import MarkerArray

from sando_py.core import (
    DynTraj,
    Parameters,
    PieceWisePol,
    RobotState,
    load_parameters_from_yaml,
    warn_if_extras,
)
from sando_py.core.utils import angle_wrap, projectPointToSphere
from sando_py.decomp import sfc, sfc_time_layered
from sando_py.decomp.sfc import _voxel_map_occupied_points as _voxel_map_occupied_points_cached
from sando_py.hgp import HGPPlanner
from sando_py.core.hardware import InitPoseTransform, init_pose_from_transform_stamped
from sando_py.core.step_timing import StepTimingRow, StepTimingWriter
from sando_py.sim_io.msg_convert import (
    cloud_msg_to_points,
    dyn_traj_msg_to_dyn_traj,
    pose_stamped_to_goal,
    pwp_to_own_dyn_traj_msg,
    robot_state_to_goal_msg,
    state_msg_to_robot_state,
    transform_points_to_frame,
)
from sando_py.sim_io.viz import (
    color_rgba,
    path_to_marker_array,
    polytopes_to_polyhedron_array,
    positions_to_marker_array,
    pwp_to_colored_marker_array,
)
from sando_py.solver import GurobiSolver, estimate_initial_dt


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class DroneStatus(Enum):
    """Mirror of the C++ ``enum class DroneStatus`` in ``sando.hpp``."""
    YAWING = 0
    TRAVELING = 1
    GOAL_SEEN = 2
    GOAL_REACHED = 3
    HOVER_AVOIDING = 4


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[2] / "config" / "sando.yaml"
)

# How often to push a new sample onto ``actual_traj_hist_``. The C++ uses
# 50 ms throttle on the publish; we throttle the *sample* too so the
# history doesn't grow at full state-callback rate.
_ACTUAL_TRAJ_SAMPLE_DT = 0.05  # 20 Hz
# How often to publish the MarkerArray. RViz drops messages at high rates
# when the array grows over time, so we publish less frequently than we
# sample.
_ACTUAL_TRAJ_PUB_DT = 0.10  # 10 Hz


def _xyz_to_pointcloud2_msg(points: np.ndarray, *, frame_id: str, stamp) -> PointCloud2:
    """Pack an ``(N, 3)`` float32 array into a minimal PointCloud2
    message. Used to re-publish the rasterised occupancy map for RViz
    overlays — same wire format the C++ ``pub_sando_occupied_cloud_``
    emits."""
    msg = PointCloud2()
    msg.header.frame_id = frame_id
    if stamp is not None:
        msg.header.stamp = stamp
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    msg.height = 1
    msg.width = int(pts.shape[0])
    msg.is_bigendian = False
    msg.is_dense = True
    msg.point_step = 12
    msg.row_step = msg.point_step * msg.width
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.data = pts.tobytes()
    return msg


def _critical_qos() -> QoSProfile:
    """Matches the C++ ``critical_qos`` (depth=10, RELIABLE)."""
    return QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)


def _sensor_qos() -> QoSProfile:
    """Matches the ROS 2 ``rmw_qos_profile_sensor_data`` defaults: depth
    10, BEST_EFFORT. Used for point-cloud and similar high-rate topics
    where dropping a frame is fine."""
    from rclpy.qos import ReliabilityPolicy as _R
    return QoSProfile(depth=10, reliability=_R.BEST_EFFORT)


# ---------------------------------------------------------------------------
# SandoNode
# ---------------------------------------------------------------------------

class SandoNode(Node):
    """ROS 2 wrapper around the SANDO planner pipeline.

    Subscribers, publishers, and timers mirror ``sando_node.cpp:133-180``.
    The replan callback body is a stub until the algorithm subpackages
    land — at that point the body becomes ``hgp.plan -> decomp.sfc ->
    solver.solve``.
    """

    def __init__(self) -> None:
        super().__init__("sando_node")

        # -------- Parameters --------
        self.declare_parameter("config_file", str(_DEFAULT_CONFIG))
        self.declare_parameter("agent_id", 0)
        self.declare_parameter("frame_id", "map")

        config_file = self.get_parameter("config_file").value
        self.id_: int = int(self.get_parameter("agent_id").value)
        self.frame_id_: str = self.get_parameter("frame_id").value

        self.get_logger().info(f"Loading config from {config_file}")
        self.params: Parameters = self._load_params(config_file)

        # -------- Planner state --------
        self.current_state_: Optional[RobotState] = None
        self.term_goal_: Optional[np.ndarray] = None
        # Mirrors C++ ``terminal_goal_initialized_`` (sando.hpp). Used by
        # ``_term_goal_cb`` to gate the duplicate-goal short-circuit and the
        # mid-flight smooth-update branch (sando.cpp:1896, 1909, 1978).
        self.terminal_goal_initialized_: bool = False
        self.trajs_: Dict[int, DynTraj] = {}      # id -> DynTraj
        self.committed_traj_: Optional[PieceWisePol] = None
        self.status_: DroneStatus = DroneStatus.GOAL_REACHED  # C++ initial state
        self.state_initialized_: bool = False

        # -------- Hover-avoidance state --------
        # ``p_hover_`` is the position the agent wants to return to once
        # the obstacle clears. Mirrors C++ ``p_hover_`` (sando.hpp:570) —
        # snapshotted at terminal-goal arrival and refreshed on entry to
        # HOVER_AVOIDING. ``term_goal_`` is repurposed during avoidance
        # to point at the evasion target (matches C++ ``setGterm`` calls
        # inside ``checkHoverAvoidance``).
        self.p_hover_: np.ndarray = np.zeros(3)

        # -------- findA k_value adaptation --------
        # Port of C++ ``startAdaptKValue`` (sando.cpp:101-111) +
        # ``findAandAtime`` (sando.cpp:185-235). Warm-up phase
        # accumulates wall-time samples from completed replans; once
        # ``num_replanning_before_adapt`` are in hand, ``startAdaptKValue``
        # seeds the EMA and adaptation kicks in. ``_last_replan_comp_time``
        # carries the previous replan's measured wall time across ticks,
        # same role as ``replanning_computation_time_`` in C++
        # ``sando_node.cpp::replanCallback``.
        self._use_adapt_k_value: bool = False
        self._got_enough_replanning: bool = False
        self._num_replanning: int = 0
        self._store_computation_times: List[float] = []
        self._est_comp_time: float = 0.0
        self._last_replan_comp_time: float = 0.0

        # -------- Pipeline --------
        # ``solver_`` is the C++-aligned Gurobi QP. On failure we hold
        # the previous committed trajectory, matching the C++ planner.
        self.hgp_ = HGPPlanner(self.params)
        self.solver_ = GurobiSolver(self.params)

        # -------- Yaw planning state --------
        # ``previous_yaw_`` is the *commanded* yaw reference — the one
        # we last published. The yaw step computes ``diff = desired -
        # previous_yaw_`` so we get smooth marching toward the target
        # without overshoot. ``yaw_start_pos_`` / ``yaw_start_time_``
        # snapshot the agent's state when YAWING begins so the desired
        # yaw (atan2 toward goal) stays stable even if the drone drifts
        # slightly during in-place rotation. Mirrors C++
        # ``previous_yaw_`` / ``yaw_start_pos_`` / ``yaw_start_time_``.
        self.previous_yaw_: float = 0.0
        self.yaw_start_pos_: Optional[np.ndarray] = None
        self.yaw_start_time_: float = 0.0

        # -------- Hardware / multi-agent SE(3) transform --------
        # Mirrors sando.cpp:2207-2308 — once the init_pose service /
        # topic delivers a TransformStamped, we cache the SE(3) forward
        # (local→global) and inverse (global→local) transforms plus the
        # yaw offset. When ``use_hardware`` + ``provide_goal_in_global_frame``
        # are on, state callbacks lift the incoming pose into the
        # planner's global frame and goal-publish lowers it back into
        # the local frame for MAVROS. Same transforms drive
        # applyInitiPoseTransform / Inverse on committed trajectories.
        self.init_pose_xform_: InitPoseTransform = InitPoseTransform()

        # -------- Replanning failure counter --------
        # Mirrors C++ ``replanning_failure_count_`` (sando.cpp:678/747/828,
        # reset on YAWING transition / initial plan setup). Once it
        # exceeds ``yaw_spinning_threshold``, the goal-publish callback
        # commands a constant ``yaw_spinning_dyaw`` rotation so the agent
        # spins in place and waits for the world (dynamic obstacles,
        # sensor refresh) to change. Re-zeroed in ``_set_status`` on
        # YAWING entry to match sando.cpp:1880.
        self.replanning_failure_count_: int = 0

        # -------- Sensor state --------
        # Latest cloud rasterised into the voxel map on every replan
        # tick. Stored as an ``(N, 3)`` numpy array **in the map
        # frame** (TF-transformed on the way in if the publisher used a
        # different frame). ``None`` until the first PointCloud2
        # arrives — in ``rviz_only`` modes this stays ``None`` forever
        # and the planner just uses the trajectory snapshot.
        self.latest_cloud_: Optional[np.ndarray] = None
        self._last_cloud_t: float = -1.0
        # TF buffer for transforming sensor-frame clouds (and any
        # future TF-aware messages) into ``frame_id_``. Listener
        # starts spinning a thread as soon as it's constructed.
        self.tf_buffer_ = Buffer()
        self.tf_listener_ = TransformListener(self.tf_buffer_, self)

        # -------- Viz state --------
        # Running history of where the agent actually went, for the
        # ``actual_traj`` MarkerArray. Throttled by ``_ACTUAL_TRAJ_DT``
        # in ``_state_cb`` so we don't bloat the message at full state
        # callback rate.
        self.actual_traj_hist_: list = []  # list[np.ndarray]
        self._last_actual_traj_hist_t: float = -1.0
        # Last successful viz publish time for ``actual_traj`` — RViz can't
        # keep up if we resend the growing MarkerArray every 10 ms.
        self._last_actual_traj_pub_t: float = -1.0

        # -------- ROS I/O --------
        qos = _critical_qos()

        # Subscribers
        if not self.params.ignore_other_trajs:
            self.sub_traj_ = self.create_subscription(
                DynTrajMsg, "/trajs", self._traj_cb, qos
            )
        self.sub_predicted_traj_ = self.create_subscription(
            DynTrajMsg, "predicted_trajs", self._traj_cb, qos
        )
        self.sub_state_ = self.create_subscription(
            StateMsg, "state", self._state_cb, qos
        )
        self.sub_term_goal_ = self.create_subscription(
            PoseStamped, "term_goal", self._term_goal_cb, qos
        )
        # init_pose: one-shot SE(3) transform that aligns the agent's
        # local frame with the planner-global frame (Vicon, multi-agent
        # mesh, etc.). C++ exposes setInitialPose as a service in
        # sando_node.cpp:1035 — we accept it as a TransformStamped topic
        # which is simpler and equivalent. ``init_pose_set_`` flips on
        # first message; subsequent messages overwrite the cached
        # transform.
        self.sub_init_pose_ = self.create_subscription(
            TransformStamped, "init_pose", self._init_pose_cb, qos
        )
        # Point-cloud occupancy data. The C++ subscribes to one of two
        # topics depending on ``use_global_pc``: per-tick sensor scans
        # (``sensor_point_cloud``, ``BEST_EFFORT``) or a one-shot global
        # ground-truth cloud (``/map_generator/global_cloud``). We
        # subscribe to both — whichever is being published wins, and
        # the latest cloud is rasterised into the voxel map at replan
        # time. In ``rviz_only`` modes nothing is published on either,
        # which is fine — the planner just sees the trajectory snapshot.
        self.sub_sensor_cloud_ = self.create_subscription(
            PointCloud2, "sensor_point_cloud", self._cloud_cb, _sensor_qos(),
        )
        self.sub_global_cloud_ = self.create_subscription(
            PointCloud2, "/map_generator/global_cloud",
            self._cloud_cb, _sensor_qos(),
        )

        # Publishers — control output
        self.pub_goal_ = self.create_publisher(GoalMsg, "goal", qos)
        # Publishers — RViz visualization. Best-effort QoS would be more
        # typical for viz, but mirroring the C++ which uses depth=10 only.
        viz_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.pub_hgp_path_marker_ = self.create_publisher(
            MarkerArray, "hgp_path_marker", viz_qos,
        )
        self.pub_poly_safe_ = self.create_publisher(
            PolyhedronArray, "poly_safe", viz_qos,
        )
        self.pub_traj_committed_colored_ = self.create_publisher(
            MarkerArray, "traj_committed_colored", viz_qos,
        )
        self.pub_actual_traj_ = self.create_publisher(
            MarkerArray, "actual_traj", viz_qos,
        )
        # ---- Extra viz / multi-agent publishers (sando_node.cpp:79-132) ----
        # ``original_hgp_path_marker``: the global path BEFORE
        # findSafeSubGoal trims it. Useful for debugging when the
        # planner is replanning around unreachable goals.
        self.pub_original_hgp_path_marker_ = self.create_publisher(
            MarkerArray, "original_hgp_path_marker", viz_qos,
        )
        # ``free_hgp_path_marker``: the HGP path through free-only
        # cells. In the Python port we only run one HGP variant, so
        # this republishes ``hgp_path_marker``; the topic exists so the
        # upstream RViz config doesn't complain about a missing
        # subscriber. C++ sando_node.cpp:81 + 1695.
        self.pub_free_hgp_path_marker_ = self.create_publisher(
            MarkerArray, "free_hgp_path_marker", viz_qos,
        )
        # ``poly_whole``: full polyhedron set (vs ``poly_safe`` which
        # the solver actually consumed). Same in single-thread Python.
        self.pub_poly_whole_ = self.create_publisher(
            PolyhedronArray, "poly_whole", viz_qos,
        )
        # ``static_push_points``: empty in C++ unless a feature stores
        # into ``static_push_points_`` — published so the RViz config
        # gets its subscriber.
        self.pub_static_push_points_ = self.create_publisher(
            MarkerArray, "static_push_points", viz_qos,
        )
        # ``sando_occupied_cloud``: occupancy as PointCloud2 (cheaper
        # than MarkerArray for RViz). Sent every replan tick.
        self.pub_sando_occupied_cloud_ = self.create_publisher(
            PointCloud2, "sando_occupied_cloud", _sensor_qos(),
        )
        # ``free_map_marker``: free cells as a MarkerArray (only useful
        # for small maps; we throttle by ``debug_verbose`` to avoid
        # flooding RViz on default config).
        self.pub_free_map_marker_ = self.create_publisher(
            MarkerArray, "free_map_marker", viz_qos,
        )
        # ``/trajs``: own trajectory broadcast to other agents
        # (sando_node.cpp:132 + publishOwnTraj at sando_node.cpp:1413).
        # Always created so multi-agent setups can hear us even when
        # ``ignore_other_trajs`` is on (that flag only filters incoming
        # /trajs; it does not silence the outgoing publisher).
        self.pub_own_traj_ = self.create_publisher(
            DynTrajMsg, "/trajs", qos,
        )

        # Timers — created started, then cancelled to mimic the C++ pattern of
        # waiting for first state / first term_goal before doing work.
        dc = self.params.dc if self.params.dc > 0 else 0.01
        self.timer_replan_ = self.create_timer(0.010, self._replan_cb)
        self.timer_replan_.cancel()
        self.timer_goal_pub_ = self.create_timer(dc, self._publish_goal_cb)
        self.timer_goal_pub_.cancel()
        self.timer_cleanup_ = self.create_timer(0.5, self._cleanup_old_trajs_cb)

        self.get_logger().info(
            f"SandoNode initialized (id={self.id_}, frame={self.frame_id_}, "
            f"goal_radius={self.params.goal_radius}, "
            f"goal_seen_radius={self.params.goal_seen_radius})"
        )

    # ------------------------------------------------------------------
    # Parameter loading
    # ------------------------------------------------------------------

    def _load_params(self, config_file: str) -> Parameters:
        params, extras = load_parameters_from_yaml(config_file)
        warn_if_extras(extras)
        return params

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------

    def _t_now(self) -> float:
        """Current ROS clock as floating-point seconds."""
        return self.get_clock().now().nanoseconds * 1e-9

    def _traj_cb(self, msg: DynTrajMsg) -> None:
        """Mirrors ``trajCallback`` (sando_node.cpp:790-803). Filters own
        traj, converts the msg, stores by id."""
        if int(msg.id) == self.id_:
            return
        try:
            traj = dyn_traj_msg_to_dyn_traj(msg, self.params, self._t_now())
        except Exception as e:  # noqa: BLE001
            self.get_logger().warning(f"trajCallback: dropping bad msg id={msg.id}: {e}")
            return
        self.trajs_[traj.id] = traj

    def _init_pose_cb(self, msg: TransformStamped) -> None:
        """Cache the local→global SE(3) transform. Mirrors C++
        ``setInitialPose`` (sando.cpp:2207-2259). Idempotent — re-publishing
        the init_pose just refreshes the cache."""
        new_xform = init_pose_from_transform_stamped(msg)
        if new_xform is None:
            return
        self.init_pose_xform_ = new_xform
        if self.params.debug_verbose:
            self.get_logger().info(
                f"init_pose received: yaw_offset={new_xform.yaw_offset:.3f} rad"
            )

    def _state_cb(self, msg: StateMsg) -> None:
        """Mirrors ``stateCallback`` (sando_node.cpp:807-853)."""
        t = self._t_now()
        state = state_msg_to_robot_state(msg, t_now=t)

        # Hardware mode: lift incoming local-frame state into the
        # planner's global frame. Mirrors sando.cpp:1525-1547.
        if (
            self.params.use_hardware
            and self.params.provide_goal_in_global_frame
            and not self.params.state_already_in_global_frame
            and self.init_pose_xform_.set_
        ):
            pos, vel, accel, jerk, yaw = self.init_pose_xform_.apply_state_local_to_global(
                state.pos, state.vel, state.accel, state.jerk, float(state.yaw),
            )
            state.pos = pos
            state.vel = vel
            state.accel = accel
            state.jerk = jerk
            state.yaw = yaw

        if self.params.use_state_update:
            self.current_state_ = state
        elif self.current_state_ is None:
            # First state callback when use_state_update is False:
            # still initialize once so downstream code has something to read.
            self.current_state_ = state

        if not self.state_initialized_:
            self.state_initialized_ = True
            self.timer_goal_pub_.reset()
            # Seed previous_yaw_ to the actual drone yaw so the first
            # yaw step doesn't make a huge jump.
            self.previous_yaw_ = float(state.yaw)
            self.get_logger().info("State initialized")

        # Throttled actual-trajectory history sample (for ``actual_traj``
        # viz publisher).
        if t - self._last_actual_traj_hist_t >= _ACTUAL_TRAJ_SAMPLE_DT:
            self.actual_traj_hist_.append(state.pos.copy())
            self._last_actual_traj_hist_t = t

    def _cloud_cb(self, msg: PointCloud2) -> None:
        """Mirrors ``occupancyMapCallback`` (sando_node.cpp:2078). The
        cloud is parsed, transformed into ``frame_id_`` if it arrives
        in a different frame, and stored verbatim — its contents are
        rasterised into the voxel map on the next replan tick rather
        than maintained as an explicit grid here, so we don't carry
        stale cells around between replans.

        Transient TF failures (the listener hasn't accumulated the
        sensor → map transform yet) cause the message to be dropped
        without raising — the next tick's cloud will be tried again.
        """
        try:
            pts = cloud_msg_to_points(msg)
        except Exception as e:  # noqa: BLE001
            self.get_logger().warning(f"cloud_cb: failed to parse: {e}")
            return

        # TF transform sensor-frame -> map frame
        source_frame = msg.header.frame_id
        transformed = transform_points_to_frame(
            pts, source_frame, msg.header.stamp,
            self.tf_buffer_, self.frame_id_,
        )
        if transformed is None:
            # Transient TF failure (transform not yet available); skip
            # this cloud and let the next one try again. Log only when
            # verbose so we don't spam during the first second of
            # operation while TF accumulates.
            if self.params.debug_verbose:
                self.get_logger().warning(
                    f"cloud_cb: no TF {source_frame} -> {self.frame_id_}; "
                    "dropping cloud"
                )
            return

        self.latest_cloud_ = transformed
        self._last_cloud_t = self._t_now()
        if self.params.debug_verbose:
            self.get_logger().info(
                f"cloud_cb: stored {transformed.shape[0]} pts from "
                f"{source_frame} (-> {self.frame_id_})"
            )

    def _term_goal_cb(self, msg: PoseStamped) -> None:
        """Mirrors ``terminalGoalCallback`` (sando_node.cpp:951-979) +
        ``setTerminalGoal`` (sando.cpp:1893-1979)."""
        try:
            goal = pose_stamped_to_goal(msg, self.params)
        except ValueError as e:
            self.get_logger().error(str(e))
            return

        # Ignore duplicate goals — the goal_sender re-publishes for
        # reliability, but re-triggering YAWING clears the plan and
        # stops the drone mid-flight. Mirrors sando.cpp:1894-1900.
        if self.terminal_goal_initialized_ and self.term_goal_ is not None:
            if float(np.linalg.norm(self.term_goal_ - goal)) < 0.1:
                return

        # If the drone is already in TRAVELING or GOAL_SEEN state (i.e.
        # mid-flight), smoothly update the terminal goal without
        # stopping. The replanning timer will pick up the new goal on
        # the next cycle and replan toward it. Mirrors sando.cpp:1909-1925.
        if self.terminal_goal_initialized_ and self.status_ in (
            DroneStatus.TRAVELING, DroneStatus.GOAL_SEEN,
        ):
            self.term_goal_ = goal
            self.p_hover_ = goal.copy()
            # NB: C++ also updates ``G_.pos`` to the sphere-projected
            # goal here (sando.cpp:1915-1917). Python recomputes the
            # projection inline inside ``_replan_cb`` each tick, so
            # there is no persistent ``G_`` field to update — the next
            # replan tick picks up the new ``term_goal_`` automatically.
            self.get_logger().info(
                f"SMOOTH_UPDATE (mid-flight): Terminal goal set to {goal.tolist()}"
            )
            # Go back to TRAVELING if we were in GOAL_SEEN (since we
            # have a new goal now). Mirrors sando.cpp:1922.
            if self.status_ == DroneStatus.GOAL_SEEN:
                self._set_status(DroneStatus.TRAVELING)
            return

        # First goal or drone is in YAWING / GOAL_REACHED / HOVER_AVOIDING:
        # full initialization. Mirrors sando.cpp:1927-1978.
        self.term_goal_ = goal
        # Snapshot the user-requested goal for hover-avoidance return.
        # Mirrors C++ ``setGoal`` (sando.cpp:1836, 1873): ``p_hover_`` is
        # set whenever a fresh terminal goal arrives so the agent has a
        # valid hover position even if it never trips the "first time
        # reaching goal" branch in needReplan.
        self.p_hover_ = goal.copy()
        self.timer_replan_.reset()  # start replanning on first terminal goal

        # Snapshot the current state so YAWING's desired yaw stays
        # stable (atan2 toward goal from a fixed start point) even if
        # the drone drifts during in-place rotation. Mirrors
        # ``yaw_start_pos_`` / ``yaw_start_time_`` in C++ ``sando.cpp``.
        if self.current_state_ is not None:
            self.yaw_start_pos_ = self.current_state_.pos.copy()
            # Seed previous_yaw_ to the actual drone yaw — the first
            # YAWING tick measures progress from here.
            self.previous_yaw_ = float(self.current_state_.yaw)
        self.yaw_start_time_ = self._t_now()

        # Reset replanning failure count so YAWING uses getDesiredYaw
        # (not spinning mode). Mirrors sando.cpp:1956. Note _set_status
        # also resets it on YAWING entry, but the skip_initial_yawing
        # path goes straight to TRAVELING and would otherwise miss it.
        self.replanning_failure_count_ = 0

        # Mirrors the C++ state-machine entry on first term_goal:
        # GOAL_REACHED -> YAWING (or TRAVELING if skip_initial_yawing).
        if self.params.skip_initial_yawing:
            self._set_status(DroneStatus.TRAVELING)
            self.get_logger().info(
                f"FULL_INIT (TRAVELING, skip_yaw): Terminal goal set to {goal.tolist()}"
            )
        else:
            self._set_status(DroneStatus.YAWING)
            self.get_logger().info(
                f"FULL_INIT (YAWING): Terminal goal set to {goal.tolist()}"
            )

        self.terminal_goal_initialized_ = True

    def _replan_cb(self) -> None:
        """One replanning tick: step the state machine, then run
        ``hgp.plan → decomp.sfc → solver.solve``.

        Mirrors ``sando.cpp::replanCB``:
          * The replan start ``A`` is picked via :meth:`_find_A` — a
            point on the previously committed trajectory, looked ahead
            by ``params.lookahead_replan_time``. This anticipates the
            agent will have moved by the time the new plan goes live,
            so successive plans hand off smoothly. On the first tick
            (no committed trajectory yet) we fall back to the agent's
            current state.
          * Hover-avoidance: when ``hover_avoidance_enabled`` is on and
            we're at the goal, ``_check_hover_avoidance`` overrides
            ``term_goal_`` with an evasion point if a dynamic obstacle
            comes within ``hover_avoidance_d_trigger``. Replanning then
            proceeds toward that evasion point. When the obstacle
            clears, the override flips back to ``p_hover_``. Mirrors
            C++ ``needReplan`` (sando.cpp:144-160) +
            ``replanCB`` hover branch (sando.cpp:597-600).
          * On any pipeline failure we hold the previous trajectory
            (matches the C++ Gurobi-failure branch).
        """
        if self.current_state_ is None or self.term_goal_ is None:
            return

        # Skip replanning while YAWING. The drone rotates in place;
        # ``_publish_goal_cb`` drives the yaw step and transitions out
        # of YAWING once the heading converges. Mirrors the C++ early
        # ``return`` in ``replanCallback`` for ``YAWING`` /
        # ``GOAL_REACHED`` states.
        if self.status_ == DroneStatus.YAWING:
            return

        t_now = self._t_now()
        dist = float(np.linalg.norm(self.current_state_.pos - self.term_goal_))
        vel_mag = float(np.linalg.norm(self.current_state_.vel))
        max_goal_vel = 0.1  # m/s — C++ sando.cpp:142 `max_goal_velocity`

        hover_on = bool(self.params.hover_avoidance_enabled)
        hover_state = self.status_ in (
            DroneStatus.GOAL_REACHED, DroneStatus.HOVER_AVOIDING,
        )

        # Distance-based status transitions. Skipped when we're already
        # in a hover state with hover-avoidance enabled — in that mode
        # the dist check is performed against an evasion goal that's
        # constantly being overwritten, so we'd flip between GOAL_REACHED
        # and HOVER_AVOIDING spuriously. Mirrors the early return at
        # sando.cpp:147-151.
        if not (hover_on and hover_state):
            if dist <= self.params.goal_radius and vel_mag < max_goal_vel:
                if hover_on:
                    # Enter HOVER_AVOIDING — keep replanning (so the
                    # avoidance check runs every tick) instead of
                    # cancelling the timer.
                    self.p_hover_ = self.term_goal_.copy()
                    if self.status_ != DroneStatus.HOVER_AVOIDING:
                        self._set_status(DroneStatus.HOVER_AVOIDING)
                else:
                    if self.status_ != DroneStatus.GOAL_REACHED:
                        self._set_status(DroneStatus.GOAL_REACHED)
                        self.timer_replan_.cancel()
                    return
            elif dist <= self.params.goal_seen_radius:
                if self.status_ == DroneStatus.TRAVELING:
                    self._set_status(DroneStatus.GOAL_SEEN)

        # Hover-avoidance pre-pipeline gate: may flip term_goal_ to an
        # evasion point, return to p_hover_, or report "no avoidance
        # needed" (return False -> skip the pipeline this tick).
        if hover_on and self.status_ in (
            DroneStatus.GOAL_REACHED, DroneStatus.HOVER_AVOIDING,
        ):
            if not self._check_hover_avoidance(t_now):
                return

        self._run_pipeline()

    # ------------------------------------------------------------------
    # Hover avoidance
    # ------------------------------------------------------------------
    # Lookahead window for ``_is_point_threatened`` — picks up periodic
    # orbits whose closest approach is in the near future. Matches the
    # ``lookahead_window`` / ``lookahead_step`` C++ constants at
    # sando.cpp:2336-2337.
    _HOVER_LOOKAHEAD_WINDOW: float = 15.0
    _HOVER_LOOKAHEAD_STEP:   float = 0.5
    # Rotation candidates tried when the straight repulsion direction
    # lands in a threat / occupied cell. Two sets (with / without π)
    # mirror the C++ two passes at sando.cpp:2406, 2430.
    _HOVER_THREAT_ANGLES = (math.pi / 6, -math.pi / 6, math.pi / 3, -math.pi / 3,
                            math.pi / 2, -math.pi / 2, math.pi)
    _HOVER_OCCUPIED_ANGLES = (math.pi / 6, -math.pi / 6, math.pi / 3, -math.pi / 3,
                              math.pi / 2, -math.pi / 2)

    def _is_point_threatened(
        self,
        pt: np.ndarray,
        t_now: float,
        lookahead: bool = True,
    ) -> bool:
        """Return True if ``pt`` is within ``hover_avoidance_d_trigger``
        of any obstacle's current position, or — when ``lookahead`` is
        True — of any predicted position in the next 15 s.

        Direct port of the C++ ``isPointThreatened`` lambda
        (sando.cpp:2338-2360). For Piecewise (agent) trajectories the
        lookahead window is clamped to ``pwp.times.back()`` so we don't
        treat stale extrapolated points as future predictions; analytic
        (obstacle) trajectories use the full window.
        """
        d_trig = float(self.params.hover_avoidance_d_trigger)
        pt = np.asarray(pt, dtype=float)
        for traj in self.trajs_.values():
            # Current reported position — always checked.
            if np.linalg.norm(pt - traj.current_pos) < d_trig:
                return True
            if not lookahead:
                continue
            t_end = t_now + self._HOVER_LOOKAHEAD_WINDOW
            try:
                from sando_py.core.dyn_traj import TrajMode
                if traj.mode == TrajMode.Piecewise and traj.pwp.times:
                    t_end = min(t_end, float(traj.pwp.times[-1]))
            except Exception:  # noqa: BLE001
                pass
            t = t_now
            while t <= t_end:
                p_obs = traj.eval(t)
                if np.linalg.norm(pt - p_obs) < d_trig:
                    return True
                t += self._HOVER_LOOKAHEAD_STEP
        return False

    def _is_point_occupied(self, pt: np.ndarray) -> bool:
        """Return True if ``pt`` lies in an occupied cell of the most
        recent voxel map. Falls back to False when no voxel map has
        been built yet (start-up, before the first HGP plan) — same
        best-effort behaviour as the C++ planner during the bootstrap
        tick. The map is the one HGP built on the previous replan
        tick; it lags by one cycle but covers the relevant area
        around the agent."""
        vm = self.hgp_.last_voxel_map
        if vm is None:
            return False
        idx = vm.world_to_idx(np.asarray(pt, dtype=float))
        if not vm.in_bounds(idx):
            return False
        return bool(vm.is_occupied(idx))

    def _check_hover_avoidance(self, t_now: float) -> bool:
        """Port of C++ ``SANDO::checkHoverAvoidance`` (sando.cpp:2322).

        Computes a 1/d² weighted repulsion vector from each obstacle
        within ``hover_avoidance_d_trigger`` of the agent's current
        position. If the resulting vector exceeds
        ``hover_avoidance_min_repulsion_norm``:

          * On first entry, snapshot ``p_hover_`` and transition to
            ``HOVER_AVOIDING``.
          * Pick an evasion point ``h`` metres along the repulsion
            direction (with the vertical component zeroed when
            ``hover_avoidance_2d`` is set). If that point is either
            threatened or occupied, try rotation candidates and bail
            if none are safe.
          * Overwrite ``term_goal_`` (= C++ ``G_term_``) with the
            evasion point so the next HGP / decomp / solve pass plans
            toward it.

        When no repulsion is felt and we're already in ``HOVER_AVOIDING``:
          * If ``p_hover_`` itself is threatened, stay where we are
            (return False) — the obstacle will eventually move away.
          * Otherwise set ``term_goal_`` back to ``p_hover_`` so the
            agent drifts back to where it was hovering.

        Returns True iff the caller should proceed with the regular
        planning pipeline; False to hold the previous trajectory.
        """
        local_state = self.current_state_
        assert local_state is not None  # gated by caller

        d_trig = float(self.params.hover_avoidance_d_trigger)
        n_total = np.zeros(3)
        for traj in self.trajs_.values():
            r_i = local_state.pos - traj.current_pos
            dist_i = float(np.linalg.norm(r_i))
            if 1e-6 < dist_i < d_trig:
                n_total += (1.0 / (dist_i * dist_i)) * (r_i / dist_i)

        if float(np.linalg.norm(n_total)) > float(self.params.hover_avoidance_min_repulsion_norm):
            # Snapshot p_hover_ on the *first* entry — once we're
            # already HOVER_AVOIDING we leave it alone (matches
            # sando.cpp:2378-2384: only refresh on the transition).
            if self.status_ != DroneStatus.HOVER_AVOIDING:
                self.p_hover_ = self.term_goal_.copy()
                self._set_status(DroneStatus.HOVER_AVOIDING)

            direction = n_total / float(np.linalg.norm(n_total))
            if self.params.hover_avoidance_2d:
                direction[2] = 0.0
            n = float(np.linalg.norm(direction))
            if n < 1e-6:
                direction = np.array([1.0, 0.0, 0.0], dtype=float)
            else:
                direction = direction / n

            p_evasion = self._build_evasion_point(local_state.pos, direction)

            if self._is_point_threatened(p_evasion, t_now, lookahead=True):
                p_evasion = self._try_rotated_evasion(
                    local_state.pos, direction, t_now,
                    angles=self._HOVER_THREAT_ANGLES,
                    avoid_threat=True, avoid_occupied=True,
                )
                if p_evasion is None:
                    return False

            if self._is_point_occupied(p_evasion):
                p_evasion = self._try_rotated_evasion(
                    local_state.pos, direction, t_now,
                    angles=self._HOVER_OCCUPIED_ANGLES,
                    avoid_threat=True, avoid_occupied=True,
                )
                if p_evasion is None:
                    return False

            # Set evasion as the new (transient) terminal goal so HGP +
            # solver target it. Mirrors C++ ``setGterm(evasion_goal)`` +
            # ``G_.pos = projectPointToSphere(...)``; in our pipeline
            # the sphere-projection is handled implicitly by HGP using
            # ``horizon`` as the search budget.
            self.term_goal_ = p_evasion
            return True

        # Repulsion vector too small to act on.
        if self.status_ == DroneStatus.HOVER_AVOIDING:
            # If p_hover_ is still threatened, wait it out. Don't
            # overwrite p_hover_ — the obstacle will eventually move.
            if self._is_point_threatened(self.p_hover_, t_now, lookahead=False):
                return False
            # Safe to return to the original hover position.
            self.term_goal_ = self.p_hover_.copy()
            return True

        # GOAL_REACHED with no nearby obstacle: nothing to do.
        return False

    def _build_evasion_point(self, agent_pos: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """Place an evasion candidate ``hover_avoidance_h`` metres along
        ``direction`` from ``agent_pos``. In 2D mode the candidate keeps
        the agent's current altitude; otherwise it's clamped to a 0.5 m
        margin inside ``[z_min, z_max]``. Mirrors C++ sando.cpp:2398-2403.
        """
        h = float(self.params.hover_avoidance_h)
        p = agent_pos + h * direction
        if self.params.hover_avoidance_2d:
            p[2] = float(agent_pos[2])
        else:
            p[2] = max(
                float(self.params.z_min) + 0.5,
                min(float(p[2]), float(self.params.z_max) - 0.5),
            )
        return p

    def _try_rotated_evasion(
        self,
        agent_pos: np.ndarray,
        direction: np.ndarray,
        t_now: float,
        angles,
        avoid_threat: bool,
        avoid_occupied: bool,
    ) -> Optional[np.ndarray]:
        """Walk through rotation candidates around the repulsion
        direction and return the first one that satisfies the requested
        safety checks. Returns ``None`` if no candidate passes.
        """
        for angle in angles:
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            rotated = np.array([
                direction[0] * cos_a - direction[1] * sin_a,
                direction[0] * sin_a + direction[1] * cos_a,
                direction[2],
            ], dtype=float)
            n = float(np.linalg.norm(rotated))
            if n < 1e-9:
                continue
            rotated = rotated / n
            candidate = self._build_evasion_point(agent_pos, rotated)
            if avoid_threat and self._is_point_threatened(candidate, t_now, lookahead=True):
                continue
            if avoid_occupied and self._is_point_occupied(candidate):
                continue
            return candidate
        return None

    # ------------------------------------------------------------------
    # findSafeSubGoal — port of sando.cpp:267-413
    # ------------------------------------------------------------------

    _SAFE_SUBGOAL_SAMPLE_DIST: float = 0.1   # m, sample step along segments
    _SAFE_SUBGOAL_BACKTRACK_DIST: float = 0.1  # m, backward walk step

    def _find_safe_sub_goal(self, path, vm) -> list:
        """Trim ``path`` to the first reachable sub-goal when the tail
        crosses inflated-unknown space.

        Port of ``SANDO::findSafeSubGoal`` (sando.cpp:267-413). Python
        equivalent: use the voxel map's occupied cells as the danger
        set (the HGP-inflated boundary the global path was allowed to
        skirt). Walk each segment at 0.1 m steps; if any sample lies
        within ``drone_radius`` of an occupied cell, backtrack along the
        segment until outside ``drone_radius +
        obst_max_vel * horizon`` and use that point as the new terminal
        waypoint.

        Returns a new path list (possibly identical to the input). The
        original ``term_goal_`` is left untouched — only the planning
        path is rewritten, mirroring C++ behaviour where ``G_term``
        stays fixed.
        """
        path = list(path)
        if vm is None or vm.occupied.size == 0 or len(path) < 2:
            return path
        # Build the occupied-cell world coords once per call (small set
        # in typical local maps).
        occ_pts = _voxel_map_occupied_points_cached(vm)
        if occ_pts.size == 0:
            return path

        from scipy.spatial import cKDTree  # type: ignore

        tree = cKDTree(occ_pts)

        thr_orig = max(1e-6, float(self.params.drone_radius))
        horizon = max(0.0, float(self.params.horizon))
        thr_inflated = thr_orig + float(self.params.obst_max_vel) * horizon

        sample = float(self._SAFE_SUBGOAL_SAMPLE_DIST)
        back_step = float(self._SAFE_SUBGOAL_BACKTRACK_DIST)

        new_path = [np.asarray(path[0], dtype=float).reshape(3)]
        for i in range(len(path) - 1):
            p0 = np.asarray(path[i], dtype=float).reshape(3)
            p1 = np.asarray(path[i + 1], dtype=float).reshape(3)
            seg = p1 - p0
            seg_len = float(np.linalg.norm(seg))
            if seg_len < 1e-9:
                new_path.append(p1)
                continue
            dir_ = seg / seg_len
            n_samples = max(1, int(np.ceil(seg_len / sample)))
            hit_arc = -1.0
            for k in range(1, n_samples + 1):
                arc = min(k * sample, seg_len)
                pt = p0 + dir_ * arc
                d, _ = tree.query(pt, k=1)
                if d < thr_orig:
                    hit_arc = arc
                    break
            if hit_arc < 0:
                new_path.append(p1)
                continue

            # Backtrack until outside the inflated-unknown threshold.
            safe_arc = hit_arc
            while safe_arc > 0.0:
                safe_arc = max(0.0, safe_arc - back_step)
                pt = p0 + dir_ * safe_arc
                d, _ = tree.query(pt, k=1)
                if d >= thr_inflated:
                    break
            safe_pt = p0 + dir_ * safe_arc
            new_path.append(safe_pt)
            # Truncate — anything beyond the safe sub-goal is unreachable.
            return new_path

        return new_path

    def _find_A(self, t_now: float) -> Optional[RobotState]:
        """Pick the replanning start state ``A`` for this tick.

        Two modes, controlled by ``params.use_adaptive_lookahead``:

        * **Adaptive (C++-faithful)** — port of ``findAandAtime``
          (sando.cpp:185-235). Two sub-phases:

          - *Warm-up* (first ``num_replanning_before_adapt`` ticks):
            use ``default_k_value`` as a fixed lookahead in discrete
            ``dc`` steps, and collect wall-time samples on each
            replan. The very-first sample is skipped to mirror C++
            ``if (num_replanning_ != 1)``.
          - *Adapt*: low-pass-filter the previous replan's
            wall time (``α * t_last + (1-α) * t_est``) and choose
            ``n_steps = int(k_value_factor * t_est / dc)``. This
            anticipates the agent will have moved by however long
            the planner is currently taking — handing off smoothly
            across replans.

        * **Fixed lookahead** — anchors ``A`` at
          ``t_now + lookahead_replan_time`` regardless of past
          compute time. Useful for unit tests and benchmarks that
          need a deterministic A.

        Returns the current state verbatim on the first tick (no
        committed trajectory yet) or ``None`` when no state has been
        received at all.
        """
        if self.current_state_ is None:
            return None
        cur = self.current_state_

        # use_state_update=False benchmark mode (sando.cpp:233-238) — anchor
        # A at the current state with A_time = current_time, regardless of
        # the committed trajectory. Used for global-planner benchmarking
        # where the agent is frozen in place.
        if not self.params.use_state_update:
            A = RobotState()
            A.setTimeStamp(t_now)
            A.setPos(cur.pos)
            A.setVel(cur.vel)
            A.setAccel(cur.accel)
            A.setYaw(cur.yaw)
            return A

        traj = self.committed_traj_
        if traj is None or not traj.times:
            return cur

        dc = max(1e-6, float(self.params.dc))

        if not self.params.use_adaptive_lookahead:
            lookahead = max(0.0, float(self.params.lookahead_replan_time))
        else:
            lookahead = self._adaptive_lookahead(traj.times[-1] - traj.times[0], dc)

        t_eval = t_now + lookahead
        # Clamp to the trajectory's domain so the polynomial is valid
        # (PieceWisePol.eval clamps to the boundary automatically, but
        # we want the explicit t for the new plan's start time).
        t_eval = min(max(t_eval, traj.times[0]), traj.times[-1])

        A = RobotState()
        A.setTimeStamp(t_eval)
        A.setPos(traj.eval(t_eval))
        A.setVel(traj.velocity(t_eval))
        A.setAccel(traj.acceleration(t_eval))
        # Yaw is held — the planner doesn't optimise yaw in v2.
        A.setYaw(cur.yaw)
        return A

    def _adaptive_lookahead(self, traj_horizon_s: float, dc: float) -> float:
        """Compute the A-anchor lookahead (in seconds) for the C++
        k_value adaptation. Warm-up bookkeeping (sample collection
        + adapt-flip) is handled in :meth:`_record_warmup_sample`
        and :meth:`_advance_warmup_state` so the pre-pipeline STORE
        still fires on the very first replan tick (when
        ``committed_traj_`` is still ``None`` and ``_find_A`` would
        short-circuit before reaching this code path).
        """
        if not self._use_adapt_k_value:
            # Warm-up: fixed lookahead = (default_k_value - 1) * dc
            # (matches C++ ``plan_size - 1 - k_value_`` when
            # ``k_value_ = max(plan_size - default_k_value, 0)`` and
            # ``plan_size > default_k_value``).
            n_steps = int(self.params.default_k_value)
        else:
            # Adaptive: EMA of measured replan wall time.
            alpha = float(self.params.alpha_k_value_filtering)
            self._est_comp_time = (
                alpha * self._last_replan_comp_time
                + (1.0 - alpha) * self._est_comp_time
            )
            n_steps = int(float(self.params.k_value_factor)
                          * self._est_comp_time / dc)

        # Trajectory horizon in dc-discretised steps — the C++ has
        # the equivalent clamp via ``plan_size - 1`` because ``plan_``
        # is bounded.
        horizon_steps = max(1, int(math.floor(traj_horizon_s / dc)))
        # C++ clamps invalid bounds to ``k_value_ = plan_size - 1`` →
        # effective index 0 → A = current state, i.e. n_steps = 1.
        if n_steps < 1 or n_steps > horizon_steps:
            n_steps = 1
        return (n_steps - 1) * dc

    def _run_pipeline(self) -> None:
        """The hgp → decomp → solver chain. Updates ``committed_traj_``
        on success; silently holds the previous trajectory otherwise.

        Wall time of the full chain is captured into
        ``_last_replan_comp_time`` and feeds into the next tick's
        :meth:`_find_A` adaptive lookahead. Mirrors the C++ measurement
        in ``sando_node.cpp::replanCallback`` (runs after ``replan``
        returns) plus the warm-up state machine in
        ``planLocalTrajectory`` (sando.cpp:1366-1376) that flips from
        fixed-``default_k_value`` lookahead to the EMA-filtered estimate
        once enough samples are in hand.
        """
        t_start = time.perf_counter()
        success = self._run_pipeline_inner()
        elapsed = time.perf_counter() - t_start

        # Update the carry-forward comp time only on success — matches
        # C++ ``if (replanning_result) replanning_computation_time_ = ...``
        # (sando_node.cpp:869-871). Failed ticks shouldn't pollute the
        # EMA with their truncated timings.
        if success:
            self._last_replan_comp_time = elapsed
            self._advance_warmup_state()

    def _record_warmup_sample(self) -> None:
        """Store the previous replan's wall time into the warm-up
        ring as long as we're still in the warm-up phase. Port of the
        C++ STORE call inside ``findAandAtime`` (sando.cpp:204-208).

        Gated by ``use_adaptive_lookahead`` because the legacy fixed
        lookahead mode doesn't need samples at all. The
        ``num_replanning_ != 1`` skip is a faithful port of the C++
        off-by-one quirk: the second-tick sample is dropped (the
        first stored value is the all-zero seed from before any
        pipeline ran)."""
        if not self.params.use_adaptive_lookahead:
            return
        if self._use_adapt_k_value:
            return
        if self._num_replanning != 1:
            self._store_computation_times.append(self._last_replan_comp_time)

    def _advance_warmup_state(self) -> None:
        """Increment the warm-up counter and, once enough samples
        are in hand, flip into the EMA-adapted mode. Port of the
        ``planLocalTrajectory`` warm-up tail (sando.cpp:1366-1376)
        + ``startAdaptKValue`` (sando.cpp:101-111)."""
        if self._got_enough_replanning:
            return
        if len(self._store_computation_times) < int(
            self.params.num_replanning_before_adapt
        ):
            self._num_replanning += 1
            return
        # startAdaptKValue — seed the EMA from the warm-up samples.
        if self._store_computation_times:
            self._est_comp_time = (
                sum(self._store_computation_times)
                / len(self._store_computation_times)
            )
        self._use_adapt_k_value = True
        self._got_enough_replanning = True

    def _run_pipeline_inner(self) -> bool:
        # Stage timers — Python-end mirror of C++ ``MyTimer`` blocks in
        # ``SANDO::replan`` (sando.cpp:622-732). The breakdown is dumped
        # to a CSV at every "attempted replan" so a Python run can be
        # diffed against the C++ run with the upstream plotter.
        row = StepTimingRow()
        total_t0 = time.perf_counter()

        # --- Housekeeping (warm-up STORE + start-state pick) ---
        t_hk = time.perf_counter()
        # Warm-up STORE happens BEFORE _find_A so the first-tick
        # zero-seed sample lands even when committed_traj_ is None
        # and _find_A short-circuits to current_state. Matches the
        # ordering in C++ ``findAandAtime`` where the STORE happens
        # at the top of the warm-up branch (sando.cpp:204-208) before
        # ``A = plan_[plan_size - 1 - k_value_]``.
        self._record_warmup_sample()

        t_now = self._t_now()
        A = self._find_A(t_now)
        if A is None:
            # findA failed — mirrors sando.cpp:678 (++failure_count then return).
            self.replanning_failure_count_ += 1
            row.housekeeping_ms = (time.perf_counter() - t_hk) * 1000.0
            row.total_replan_ms = (time.perf_counter() - total_t0) * 1000.0
            self._write_step_timing(row)
            return False

        # Project the terminal goal onto a sphere of radius ``horizon``
        # around A so the local planner only ever sees a near-future
        # target. Mirrors C++ ``computeG`` (sando.cpp:115-120) +
        # ``projectPointToSphere``. Without this, HGP runs end-to-end
        # over the full 105 m terminal goal and the LOS simplifier
        # collapses the path to a 2-point segment — which then makes
        # EllipsoidDecomp produce huge corridors.
        horizon = float(self.params.horizon or 0.0)
        if horizon > 0:
            local_goal = projectPointToSphere(A.pos, self.term_goal_, horizon)
        else:
            local_goal = self.term_goal_.copy()

        # Ground robot z-clamp (sando.cpp:711-714, 843-846). Pin start
        # and terminal goal to z=1.0 for planar vehicles so the
        # solver doesn't waste DOFs on z.
        term_goal_planner = local_goal
        if self.params.vehicle_type and self.params.vehicle_type != "uav":
            A.pos[2] = 1.0
            term_goal_planner = local_goal.copy()
            term_goal_planner[2] = 1.0

        obstacles = list(self.trajs_.values())
        row.housekeeping_ms = (time.perf_counter() - t_hk) * 1000.0

        # --- Global planning (HGP + safe-sub-goal trim) ---
        t_gp = time.perf_counter()
        path = self.hgp_.plan(
            A, term_goal_planner, obstacles, t_now,
            cloud_points=self.latest_cloud_,
        )

        # Cap the global path at ``num_P + 1`` waypoints (sando.cpp:788-791).
        if path is not None and self.params.num_P > 0:
            max_pts = int(self.params.num_P) + 1
            if len(path) > max_pts:
                path = list(path[:max_pts])
        if path is None or len(path) < 2:
            # HGP failed / degenerate path — mirrors sando.cpp:747+828.
            self.replanning_failure_count_ += 1
            row.global_planning_ms = (time.perf_counter() - t_gp) * 1000.0
            row.total_replan_ms = (time.perf_counter() - total_t0) * 1000.0
            self._write_step_timing(row)
            if self.params.debug_verbose:
                self.get_logger().warning("HGP failed; holding previous trajectory")
            return False

        # If the global path is exactly 2 points, subdivide the segment
        # by inserting midpoints until size >= 3. Mirrors sando.cpp:820-824.
        if len(path) == 2:
            p0 = np.asarray(path[0], dtype=float)
            p1 = np.asarray(path[1], dtype=float)
            mid = 0.5 * (p0 + p1)
            path = [p0, mid, p1]

        vm = self.hgp_.last_voxel_map
        if vm is None:
            self.replanning_failure_count_ += 1
            row.global_planning_ms = (time.perf_counter() - t_gp) * 1000.0
            row.total_replan_ms = (time.perf_counter() - total_t0) * 1000.0
            self._write_step_timing(row)
            return False  # defensive — HGP shouldn't return a path without a map

        # findSafeSubGoal: if the trimmed global path can't reach the
        # terminal goal because the tail crosses inflated unknown space,
        # truncate to a reachable sub-goal. C++ sando.cpp:267-413 +
        # call site at sando.cpp:795.
        original_path = list(path)
        path = self._find_safe_sub_goal(path, vm)
        row.global_planning_ms = (time.perf_counter() - t_gp) * 1000.0

        # --- Local trajectory optimization (cvx decomp + solver) ---
        # Time-layered decomp when ``environment_assumption == "dynamic"``
        # — generate [N_time × P_spatial] polytopes so the solver MIQP
        # can pick a different corridor per time layer (each layer's
        # dynamic obstacles are inflated by a different reachability
        # radius). Mirrors C++ ``cvxEllipsoidDecompTimeLayered`` driven
        # from ``generateLocalTrajectory`` (sando.cpp:1334-1376).
        # Other modes (static / dynamic_worst_case / aabb / empty) keep
        # the single-layer ``sfc()``.
        env_mode = (self.params.environment_assumption or "").lower()
        t_cvx = time.perf_counter()
        if env_mode == "dynamic" and self.params.num_N > 0:
            num_N = max(1, int(self.params.num_N))
            factor_hint = float(
                getattr(self.solver_.time_alloc, "current_mean",
                        self.params.dynamic_factor_initial_mean or 1.0)
            )
            initial_dt = estimate_initial_dt(
                A, np.asarray(path[-1], dtype=float), self.params, num_N,
            )
            dt_layer = max(1e-3, initial_dt * factor_hint)
            polys = sfc_time_layered(
                path, vm, obstacles, self.params,
                num_time_layers=num_N,
                dt_layer=dt_layer,
                t_now=t_now,
            )
        else:
            polys = sfc(path, vm, obstacles, self.params, t_now=t_now)
        row.cvx_decomp_ms = (time.perf_counter() - t_cvx) * 1000.0

        t_lto = time.perf_counter()
        traj_new, info = self.solver_.solve(A, path, polys)
        row.local_traj_opt_ms = (time.perf_counter() - t_lto) * 1000.0

        # Copy the winning solver's per-stage breakdown into the CSV row.
        tb = getattr(info, "timing", None)
        if tb is not None:
            row.gurobi_findDT_ms = tb.findDT_ms
            row.gurobi_setX_ms = tb.setX_ms
            row.gurobi_polytopes_ms = tb.polytopes_ms
            row.gurobi_dynamic_ms = tb.dynamic_ms
            row.gurobi_objective_setup_ms = tb.objective_ms
            row.gurobi_mapsize_ms = tb.mapsize_ms
            row.gurobi_callOptimizer_ms = tb.callOptimizer_ms
            row.gurobi_postsolve_ms = tb.postsolve_ms

        if info.success:
            row.successful_factor = float(getattr(info, "successful_factor", 1.0))
            row.gurobi_objective_value = float(info.cost)
        else:
            # Match C++ "nan on failure" convention.
            row.gurobi_objective_value = float("nan")

        if info.success:
            # --- Append (Python: direct commit; ~0 ms) ---
            t_app = time.perf_counter()
            self.committed_traj_ = traj_new
            row.append_ms = (time.perf_counter() - t_app) * 1000.0

            self._publish_plan_viz(path, polys, traj_new,
                                   original_path=original_path, vm=vm)
            self._publish_own_traj(traj_new)

            # --- Final housekeeping (reset failure counter) ---
            t_fh = time.perf_counter()
            self.replanning_failure_count_ = 0
            row.final_housekeeping_ms = (time.perf_counter() - t_fh) * 1000.0
            row.success = True
            row.total_replan_ms = (time.perf_counter() - total_t0) * 1000.0
            self._write_step_timing(row)
            return True

        if self.params.debug_verbose:
            self.get_logger().warning(
                f"Solver failed (cost={info.cost:.2f}); holding previous trajectory"
            )
        # Solver failure — C++ tolerates this without bumping the counter.
        row.total_replan_ms = (time.perf_counter() - total_t0) * 1000.0
        self._write_step_timing(row)
        return False

    def _write_step_timing(self, row: StepTimingRow) -> None:
        """Lazy-init the step-timing writer and append ``row``. Path
        respects the ``SANDO_PY_STEP_TIMING_CSV`` env var; unset → default
        ``/tmp/sando_py_step_timing.csv``; set-to-empty → disabled."""
        writer = StepTimingWriter.get()
        if writer is None:
            return
        writer.write(row)

    def _publish_goal_cb(self) -> None:
        """Evaluate the committed trajectory at ``t_now`` and publish the
        resulting setpoint. Falls back to hovering at the current state
        when no trajectory has been committed yet (start-up, or after a
        failed replan with no previous trajectory).

        Also publishes the throttled ``actual_traj`` MarkerArray off this
        timer — it ticks at ``params.dc`` (~100 Hz) which is the fastest
        cadence in the node, so it's the natural place to gate viz on.
        """
        if self.current_state_ is None:
            return

        setpoint = RobotState()
        t_now = self._t_now()
        if self.committed_traj_ is not None and self.committed_traj_.times:
            setpoint.setPos(self.committed_traj_.eval(t_now))
            setpoint.setVel(self.committed_traj_.velocity(t_now))
            setpoint.setAccel(self.committed_traj_.acceleration(t_now))
        else:
            setpoint.setPos(self.current_state_.pos)

        # Yaw planning — pos comes from the trajectory above; the yaw
        # step is per-state-machine-mode (see ``_compute_yaw_setpoint``).
        yaw, dyaw = self._compute_yaw_setpoint(setpoint)
        setpoint.setYaw(yaw)
        setpoint.setDYaw(dyaw)
        self.previous_yaw_ = yaw

        # Hardware mode: lower the planner-global setpoint into the
        # local (MAVROS) frame. Mirrors sando.cpp:1648-1667.
        if (
            self.params.use_hardware
            and self.params.provide_goal_in_global_frame
            and self.init_pose_xform_.set_
        ):
            pos, vel, accel, jerk, yaw_local = self.init_pose_xform_.apply_state_global_to_local(
                setpoint.pos, setpoint.vel, setpoint.accel, setpoint.jerk,
                float(setpoint.yaw),
            )
            setpoint.pos = pos
            setpoint.vel = vel
            setpoint.accel = accel
            setpoint.jerk = jerk
            setpoint.yaw = yaw_local

        stamp = self.get_clock().now().to_msg()
        msg = robot_state_to_goal_msg(setpoint, frame_id=self.frame_id_, stamp=stamp)
        self.pub_goal_.publish(msg)

        # Throttled ``actual_traj`` viz publish
        if (
            len(self.actual_traj_hist_) >= 2
            and t_now - self._last_actual_traj_pub_t >= _ACTUAL_TRAJ_PUB_DT
        ):
            ma = positions_to_marker_array(
                self.actual_traj_hist_,
                frame_id=self.frame_id_,
                stamp=stamp,
                color=color_rgba(0.0, 1.0, 0.0),
            )
            self.pub_actual_traj_.publish(ma)
            self._last_actual_traj_pub_t = t_now

    def _compute_yaw_setpoint(self, setpoint: RobotState) -> tuple:
        """Compute (yaw, dyaw) for the current state-machine status,
        mirroring C++ ``SANDO::getDesiredYaw`` + ``yaw``.

        State-machine modes:

        * **YAWING**: closed-loop yaw from the agent's actual heading
          toward ``atan2(goal - yaw_start_pos)``. Step capped at
          ``w_max_yawing * dc``. Transition to TRAVELING when
          ``|diff| < 0.3 rad`` (~17°) or after a 10s timeout with
          ``|diff| < 1.0 rad`` (~57°).
        * **TRAVELING / GOAL_SEEN**: open-loop yaw follows the
          commanded velocity direction. Filtered via
          ``(1 - alpha_filter_dyaw) * diff``, clamped to
          ``w_max * dc``. Holds previous yaw when speed is < 0.01 m/s
          (atan2 unstable at zero velocity).
        * **GOAL_REACHED**: yaw held, dyaw = 0.
        * **yaw_spinning fallback**: when consecutive replan failures
          exceed ``yaw_spinning_threshold`` (and we're not already
          HOVER_AVOIDING), short-circuit any other yaw mode and command
          ``previous_yaw_ + yaw_spinning_dyaw * dc`` so the agent rotates
          in place while sensors / dynamic obstacles refresh. Mirrors
          C++ sando.cpp:1623-1626.

        Returns ``(yaw, dyaw)``. ``self.previous_yaw_`` is **not**
        updated here — the caller writes it after publishing so the
        commanded reference advances by exactly one step per goal tick.
        """
        dc = max(1e-6, float(self.params.dc))

        # yaw_spinning fallback — highest-priority override.
        if (
            int(self.params.yaw_spinning_threshold) > 0
            and self.replanning_failure_count_ > int(self.params.yaw_spinning_threshold)
            and self.status_ != DroneStatus.HOVER_AVOIDING
        ):
            dyaw = float(self.params.yaw_spinning_dyaw)
            new_yaw = self.previous_yaw_ + dyaw * dc
            return new_yaw, dyaw

        # GOAL_REACHED: hold previous yaw.
        if self.status_ == DroneStatus.GOAL_REACHED:
            return self.previous_yaw_, 0.0

        # YAWING: closed-loop toward the goal direction
        if self.status_ == DroneStatus.YAWING:
            if self.yaw_start_pos_ is None or self.term_goal_ is None:
                return self.previous_yaw_, 0.0
            dx = float(self.term_goal_[0] - self.yaw_start_pos_[0])
            dy = float(self.term_goal_[1] - self.yaw_start_pos_[1])
            desired_yaw = math.atan2(dy, dx)

            # Closed-loop convergence check uses the *actual* drone yaw.
            actual_yaw = float(self.current_state_.yaw)
            diff_actual = angle_wrap(desired_yaw - actual_yaw)

            elapsed = self._t_now() - self.yaw_start_time_
            if abs(diff_actual) < 0.3 or (elapsed > 10.0 and abs(diff_actual) < 1.0):
                # Converged or timeout — transition out of YAWING.
                self._set_status(DroneStatus.TRAVELING)
                # On the same tick, still publish the (close-to-target)
                # yaw so the setpoint matches the new mode's behaviour.

            # Step from previous_yaw_ (the commanded reference) toward
            # desired_yaw. Using the *commanded* reference (not the
            # actual drone yaw) ensures the quaternion target marches
            # steadily forward — PX4 tracks the commanded value, not
            # the dyaw feedforward.
            diff_cmd = angle_wrap(desired_yaw - self.previous_yaw_)
            max_step = float(self.params.w_max_yawing) * dc
            step = max(-max_step, min(max_step, diff_cmd))
            new_yaw = self.previous_yaw_ + step
            dyaw = step / dc
            return new_yaw, dyaw

        # TRAVELING / GOAL_SEEN: yaw follows commanded velocity.
        if self.status_ in (DroneStatus.TRAVELING, DroneStatus.GOAL_SEEN):
            vx, vy = float(setpoint.vel[0]), float(setpoint.vel[1])
            speed_xy = math.hypot(vx, vy)
            if speed_xy < 0.01:
                # Velocity too small — atan2 unstable, hold yaw.
                return self.previous_yaw_, 0.0
            desired_yaw = math.atan2(vy, vx)
            diff = angle_wrap(desired_yaw - self.previous_yaw_)
            # Exponential convergence by filtering the angle (not the
            # rate): each step covers a fraction of the remaining
            # error.
            alpha = float(self.params.alpha_filter_dyaw or 0.0)
            step = (1.0 - alpha) * diff
            max_step = float(self.params.w_max) * dc
            step = max(-max_step, min(max_step, step))
            new_yaw = self.previous_yaw_ + step
            dyaw = step / dc
            return new_yaw, dyaw

        # HOVER_AVOIDING: face the stored hover position. Open-loop yaw
        # step from previous_yaw_ (commanded reference), same pattern as
        # YAWING but driven by ``atan2(p_hover_ - setpoint.pos)``.
        # Mirrors C++ ``getDesiredYaw`` HOVER_AVOIDING branch
        # (sando.cpp:1690-1693, 1752-1758).
        if self.status_ == DroneStatus.HOVER_AVOIDING:
            dx = float(self.p_hover_[0] - setpoint.pos[0])
            dy = float(self.p_hover_[1] - setpoint.pos[1])
            dist_xy = math.hypot(dx, dy)
            if dist_xy < 0.3:
                # Too close — atan2 is unstable, hold yaw.
                return self.previous_yaw_, 0.0
            desired_yaw = math.atan2(dy, dx)
            diff_cmd = angle_wrap(desired_yaw - self.previous_yaw_)
            max_step = float(self.params.w_max_yawing) * dc
            step = max(-max_step, min(max_step, diff_cmd))
            new_yaw = self.previous_yaw_ + step
            dyaw = step / dc
            return new_yaw, dyaw

        # Any unhandled status — hold previous yaw.
        return self.previous_yaw_, 0.0

    def _publish_plan_viz(self, path, polys, traj, *,
                          original_path=None, vm=None) -> None:
        """Push the fresh global path, safety corridors, and committed
        trajectory to RViz. Called from ``_run_pipeline`` after a
        successful solve so the markers track the planner's actual
        decisions (rather than a stale snapshot)."""
        stamp = self.get_clock().now().to_msg()

        # hgp_path_marker — polyline + dots
        ma_path = path_to_marker_array(
            path,
            frame_id=self.frame_id_,
            stamp=stamp,
            color=color_rgba(1.0, 0.0, 0.0),
        )
        self.pub_hgp_path_marker_.publish(ma_path)

        # original_hgp_path_marker — pre-findSafeSubGoal path (orange).
        if original_path is not None and len(original_path) >= 2:
            ma_orig = path_to_marker_array(
                original_path,
                frame_id=self.frame_id_,
                stamp=stamp,
                color=color_rgba(1.0, 0.5, 0.0),
            )
            self.pub_original_hgp_path_marker_.publish(ma_orig)
        # free_hgp_path_marker — same path in green (single-variant port).
        ma_free = path_to_marker_array(
            path,
            frame_id=self.frame_id_,
            stamp=stamp,
            color=color_rgba(0.0, 1.0, 0.0),
        )
        self.pub_free_hgp_path_marker_.publish(ma_free)

        # poly_safe — corridors
        # ``polys`` may be ``List[List[Polytope]]`` (time-layered) or
        # ``List[Polytope]`` (single-layer). Flatten time layers for the
        # RViz ``poly_safe`` / ``poly_whole`` arrays — RViz doesn't
        # distinguish time layers, so we just stack them all.
        if polys and isinstance(polys[0], (list, tuple)):
            polys_flat = [p for layer in polys for p in layer]
        else:
            polys_flat = list(polys)
        pa = polytopes_to_polyhedron_array(
            polys_flat, frame_id=self.frame_id_, stamp=stamp,
        )
        self.pub_poly_safe_.publish(pa)
        # poly_whole — full polytope set (= all time layers when
        # time-layered decomp is active, single layer otherwise).
        self.pub_poly_whole_.publish(pa)

        # traj_committed_colored — sampled committed trajectory, JET by |v|
        ma_traj = pwp_to_colored_marker_array(
            traj,
            frame_id=self.frame_id_,
            stamp=stamp,
            v_max=max(0.1, float(self.params.v_max)),
        )
        self.pub_traj_committed_colored_.publish(ma_traj)

        # static_push_points — published empty (not populated by the
        # Python port; matches the dormant C++ branch).
        self.pub_static_push_points_.publish(MarkerArray())

        # sando_occupied_cloud — PointCloud2 of occupied voxel centres.
        if vm is not None:
            try:
                from sando_py.decomp.sfc import _voxel_map_occupied_points
                occ_pts = _voxel_map_occupied_points(vm)
                cloud_msg = _xyz_to_pointcloud2_msg(
                    occ_pts, frame_id=self.frame_id_, stamp=stamp,
                )
                self.pub_sando_occupied_cloud_.publish(cloud_msg)
            except Exception:  # noqa: BLE001
                pass

    def _publish_own_traj(self, traj) -> None:
        """Broadcast the committed PieceWisePol on ``/trajs`` so other
        agents can pick it up. Mirrors C++ ``publishOwnTraj``."""
        if traj is None or not traj.times:
            return
        if self.current_state_ is None:
            return
        stamp = self.get_clock().now().to_msg()
        msg = pwp_to_own_dyn_traj_msg(
            traj, self.current_state_, self.params, self.id_,
            frame_id=self.frame_id_, stamp=stamp,
            terminal_goal=self.term_goal_,
        )
        self.pub_own_traj_.publish(msg)

    def _cleanup_old_trajs_cb(self) -> None:
        """Mirrors ``cleanUpOldTrajsCallback`` (sando_node.cpp ~500ms timer):
        drop obstacle trajectories not refreshed within ``params.traj_lifetime``."""
        if self.params.traj_lifetime <= 0:
            return
        cutoff = self._t_now() - self.params.traj_lifetime
        stale = [tid for tid, tr in self.trajs_.items() if tr.time_received < cutoff]
        for tid in stale:
            del self.trajs_[tid]
        if stale and self.params.debug_verbose:
            self.get_logger().info(f"Dropped {len(stale)} stale obstacle trajs")

    # ------------------------------------------------------------------
    # State-machine helper
    # ------------------------------------------------------------------

    def _set_status(self, new_status: DroneStatus) -> None:
        if new_status == self.status_:
            return
        # Same exact phrasing as the C++ log message so it diffs cleanly
        # against the upstream demo.
        self.get_logger().info(
            f"Changing DroneStatus from status_={self.status_.name} "
            f"to status_={new_status.name}"
        )
        # YAWING transition zeros the failure counter (sando.cpp:1880).
        if new_status == DroneStatus.YAWING:
            self.replanning_failure_count_ = 0
        self.status_ = new_status


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None) -> None:
    rclpy.init(args=args)
    node = SandoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
