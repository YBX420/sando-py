"""SANDO ROS 2 node — Python port of src/sando/sando_node.cpp.

Subscribers:
  state            -- dynus_interfaces/State    (robot odometry / sim state)
  term_goal        -- geometry_msgs/PoseStamped (terminal goal from rviz / sender)
  /trajs           -- dynus_interfaces/DynTraj  (other agents' shared trajectories)
  predicted_trajs  -- dynus_interfaces/DynTraj  (dynamic obstacle predictions)
  occupancy_grid + unknown_grid -- sensor_msgs/PointCloud2 (gazebo / hardware)
  sensor_point_cloud OR /map_generator/global_cloud -- sensor_msgs/PointCloud2 (fake_sim)

Publishers (matching C++):
  goal             -- dynus_interfaces/Goal     (per-tick setpoint at 1/dc Hz)
  /trajs           -- dynus_interfaces/DynTraj  (own trajectory for other agents)
  goal_reached     -- std_msgs/Empty
  computation_times-- dynus_interfaces/ComputationTimes
  + various visualization markers / point clouds.

Timers:
  10 ms replanCallback  (mirrors C++ 100 Hz replan loop)
  par.dc publishGoal     (typically 1 kHz / 100 Hz, drives MAVROS)
  500 ms cleanUpOldTrajs
  100 ms goal_reached_check if use_benchmark
  100 ms get_initial_pose_hw if use_hardware
"""
from __future__ import annotations

import math
import time
from threading import Lock
from typing import List, Optional

import numpy as np

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from geometry_msgs.msg import PointStamped, PoseStamped, Vector3
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import ColorRGBA, Empty
from visualization_msgs.msg import Marker, MarkerArray

from ..planner import SANDO
from ..types import DroneStatus, DynTraj, Parameters, PieceWisePol, RobotState
from ..utils import (
    angle_wrap,
    builtin_to_seconds,
    msg_to_pwp,
    pwp_to_msg,
    state_to_goal_msg,
    yaw_from_quat,
)


def _critical_qos() -> QoSProfile:
    return QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST, depth=10,
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.VOLATILE,
    )


def _sensor_qos() -> QoSProfile:
    return QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST, depth=5,
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        durability=QoSDurabilityPolicy.VOLATILE,
    )


def _pointcloud2_to_numpy(msg: PointCloud2) -> np.ndarray:
    """Convert PointCloud2 → (N, 3) numpy array of XYZ.

    Uses sensor_msgs_py.point_cloud2.read_points, the rclpy-side replacement
    for PCL conversions used by the C++ node.
    """
    gen = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
    return np.array([(float(p[0]), float(p[1]), float(p[2])) for p in gen],
                    dtype=np.float64) if gen is not None else np.zeros((0, 3))


class SANDONode(Node):
    """ROS 2 node wrapper. Holds the SANDO planner and bridges topics ↔ planner state."""

    def __init__(self):
        super().__init__("sando_node")

        # Namespace / id (matches C++ ns -> NX## convention)
        ns = self.get_namespace() or "/"
        ns = ns.rstrip("/").split("/")[-1] if ns.rstrip("/") else "NX01"
        self.ns_ = ns
        self.id_str = ns[-2:] if len(ns) >= 2 and ns[-2:].isdigit() else "01"
        try:
            self.id_ = int(self.id_str)
        except ValueError:
            self.id_ = 1

        # Parameters
        self.par = Parameters.from_ros_node(self)
        # Visualization frame: "world" for hw+global-frame, "map" otherwise (C++ parity)
        self.viz_frame = ("world" if (self.par.use_hardware and self.par.state_already_in_global_frame)
                          else "map")

        # Planner
        self.sando = SANDO(self.par)

        # Callback groups
        self.cb_re = ReentrantCallbackGroup()
        self.cb_replan = MutuallyExclusiveCallbackGroup()
        self.cb_goal = ReentrantCallbackGroup()
        self.cb_map = ReentrantCallbackGroup()

        critical = _critical_qos()
        sensor_q = _sensor_qos()

        # Publishers — only the essentials and a small viz set (visual_level≥1).
        # The C++ node publishes ~25 topics; here we keep parity for the
        # control-essential ones and the dominant visualization streams.
        self.pub_goal = self.create_publisher(_get_goal_msg(), "goal", critical)
        self.pub_own_traj = self.create_publisher(_get_dyntraj_msg(), "/trajs", critical)
        self.pub_goal_reached = self.create_publisher(Empty, "goal_reached", critical)
        self.pub_computation_times = self.create_publisher(
            _get_computation_times_msg(), "computation_times", 10
        ) if _get_computation_times_msg() else None

        self.pub_point_A = self.create_publisher(PointStamped, "point_A", 10)
        self.pub_point_G = self.create_publisher(PointStamped, "point_G", 10)
        self.pub_point_E = self.create_publisher(PointStamped, "point_E", 10)
        self.pub_point_G_term = self.create_publisher(PointStamped, "point_G_term", 10)
        self.pub_current_state = self.create_publisher(PointStamped, "point_current_state", 10)
        self.pub_setpoint = self.create_publisher(PointStamped, "setpoint_vis", 10)
        self.pub_vel_text = self.create_publisher(Marker, "vel_text", 10)
        self.pub_hgp_path_marker = self.create_publisher(MarkerArray, "hgp_path_marker", 10)
        self.pub_original_hgp_path_marker = self.create_publisher(
            MarkerArray, "original_hgp_path_marker", 10)
        self.pub_traj_committed = self.create_publisher(MarkerArray, "traj_committed_colored", 10)
        self.pub_fov = self.create_publisher(Marker, "fov", 10)

        # Subscribers
        state_msg = _get_state_msg()
        if state_msg is not None:
            self.sub_state = self.create_subscription(
                state_msg, "state", self.state_callback, critical,
                callback_group=self.cb_re,
            )
        self.sub_term_goal = self.create_subscription(
            PoseStamped, "term_goal", self.terminal_goal_callback, critical,
        )

        dyntraj_msg = _get_dyntraj_msg()
        if dyntraj_msg is not None:
            if not self.par.ignore_other_trajs:
                self.sub_traj = self.create_subscription(
                    dyntraj_msg, "/trajs", self.traj_callback, critical,
                    callback_group=self.cb_re,
                )
            self.sub_predicted_traj = self.create_subscription(
                dyntraj_msg, "predicted_trajs", self.traj_callback, critical,
                callback_group=self.cb_re,
            )

        # Point cloud subscribers — fake_sim/rviz_only vs gazebo+hardware
        if self.par.sim_env == "fake_sim":
            topic = "/map_generator/global_cloud" if self.par.use_global_pc else "sensor_point_cloud"
            self.sub_occ = self.create_subscription(
                PointCloud2, topic, self.occupancy_map_callback, sensor_q,
                callback_group=self.cb_map,
            )
        elif self.par.sim_env == "rviz_only":
            # Seed an empty map so the planner can pass check_ready_to_replan
            self.sando.update_occupancy_map_ptr(np.zeros((0, 3)))
            self.get_logger().info("[rviz_only] Initialized map with empty point cloud")
        else:
            # gazebo / hardware — two synchronized clouds
            self._latest_occ: Optional[np.ndarray] = None
            self._latest_unk: Optional[np.ndarray] = None
            self._cloud_lock = Lock()
            self.sub_occ = self.create_subscription(
                PointCloud2, "occupancy_grid", self._occ_callback, sensor_q,
                callback_group=self.cb_map,
            )
            self.sub_unk = self.create_subscription(
                PointCloud2, "unknown_grid", self._unk_callback, sensor_q,
                callback_group=self.cb_map,
            )

        # Timers — initially cancelled, started after first state msg
        self._timer_replan = self.create_timer(0.01, self.replan_callback,
                                                callback_group=self.cb_replan)
        self._timer_goal = self.create_timer(max(1e-3, self.par.dc), self.publish_goal,
                                              callback_group=self.cb_goal)
        self._timer_cleanup = self.create_timer(0.5, self.cleanup_old_trajs_callback)

        self._timer_replan.cancel()
        self._timer_goal.cancel()
        self.state_initialized = False

        # Replan diagnostics
        self.replanning_computation_time = 0.0
        self.last_replan_viz_publish_t = 0.0
        self.last_actual_traj_viz_publish_t = 0.0
        self.pwp_to_share = PieceWisePol()

        self.get_logger().info(f"SANDO Python node ready (ns={self.ns_}, id={self.id_})")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def state_callback(self, msg) -> None:
        st = RobotState()
        st.pos = np.array([msg.pos.x, msg.pos.y, msg.pos.z], dtype=float)
        st.vel = np.array([msg.vel.x, msg.vel.y, msg.vel.z], dtype=float)
        st.accel = np.zeros(3)
        st.yaw = yaw_from_quat(msg.quat.x, msg.quat.y, msg.quat.z, msg.quat.w)
        if self.par.use_state_update:
            self.sando.update_state(st)
            self._publish_point(st.pos, self.pub_current_state)
            if self.par.visual_level >= 1:
                self._publish_velocity_text(st.pos, float(np.linalg.norm(st.vel)))
        if not self.state_initialized:
            if not self.par.use_state_update:
                st.t = self._ros_now()
                self.sando.update_state(st)
            self.get_logger().info("State initialized")
            self.state_initialized = True
            self._timer_goal.reset()

    def terminal_goal_callback(self, msg: PoseStamped) -> None:
        goal_z = self.par.default_goal_z if self.par.force_goal_z else float(msg.pose.position.z)
        if goal_z < self.par.z_min or goal_z > self.par.z_max:
            self.get_logger().error(f"Goal z is out of bounds: {goal_z}")
            return
        G_term = RobotState()
        G_term.set_pos(float(msg.pose.position.x), float(msg.pose.position.y), goal_z)
        self.sando.set_terminal_goal(G_term)
        self._publish_point(G_term.pos, self.pub_point_G_term)
        # Start replanning loop on first valid goal
        self._timer_replan.reset()

    def traj_callback(self, msg) -> None:
        if msg.id == self.id_:
            return
        current_time = self._ros_now()
        traj = self._convert_dyntraj_msg(msg, current_time)
        self.sando.add_traj(traj, current_time)

    def occupancy_map_callback(self, msg: PointCloud2) -> None:
        pc = _pointcloud2_to_numpy(msg)
        self.sando.update_occupancy_map_ptr(pc)
        # In global-pc mode the cloud is static — drop subscription after first read
        if self.par.use_global_pc:
            self.destroy_subscription(self.sub_occ)

    def _occ_callback(self, msg: PointCloud2) -> None:
        with self._cloud_lock:
            self._latest_occ = _pointcloud2_to_numpy(msg)
            self._maybe_push_map_pair()

    def _unk_callback(self, msg: PointCloud2) -> None:
        with self._cloud_lock:
            self._latest_unk = _pointcloud2_to_numpy(msg)
            self._maybe_push_map_pair()

    def _maybe_push_map_pair(self) -> None:
        if self._latest_occ is not None and self._latest_unk is not None:
            self.sando.update_map_ptr(self._latest_occ, self._latest_unk)
            # Keep the latest pair; matches "approximate time sync" semantics

    def cleanup_old_trajs_callback(self) -> None:
        self.sando.clean_up_old_trajs(self._ros_now())

    # ------------------------------------------------------------------
    # Main replan + goal loops
    # ------------------------------------------------------------------
    def replan_callback(self) -> None:
        current_time = self._ros_now()
        t0 = time.perf_counter()
        replan_ok, hgp_ok = self.sando.replan(self.replanning_computation_time, current_time)
        if replan_ok:
            self.replanning_computation_time = time.perf_counter() - t0
            self._publish_own_traj()

        do_viz = False
        if self.par.visual_level >= 1:
            t_now = self._ros_now()
            if t_now - self.last_replan_viz_publish_t >= 0.05:
                do_viz = True
                self.last_replan_viz_publish_t = t_now

        if do_viz and hgp_ok:
            self._publish_global_path()
        if do_viz and replan_ok:
            self._publish_traj_committed()
            self._publish_point(self.sando.get_A().pos, self.pub_point_A)
            self._publish_point(self.sando.get_G().pos, self.pub_point_G)
            self._publish_point(self.sando.get_E().pos, self.pub_point_E)

        # Always publish computation times (matches C++ behavior)
        if self.pub_computation_times is not None:
            self._publish_computation_times(replan_ok)

    def publish_goal(self) -> None:
        if self.par.use_hardware and self.sando.get_drone_status() == int(DroneStatus.GOAL_REACHED):
            return
        ok, next_goal = self.sando.get_next_goal()
        if not (ok and self.par.use_state_update):
            return

        GoalMsg = _get_goal_msg()
        if GoalMsg is None:
            return
        g = GoalMsg()
        g.header.stamp = self.get_clock().now().to_msg()
        g.header.frame_id = self.viz_frame
        g.p = Vector3(x=float(next_goal.pos[0]), y=float(next_goal.pos[1]), z=float(next_goal.pos[2]))
        g.v = Vector3(x=float(next_goal.vel[0]), y=float(next_goal.vel[1]), z=float(next_goal.vel[2]))
        g.a = Vector3(x=float(next_goal.accel[0]), y=float(next_goal.accel[1]), z=float(next_goal.accel[2]))
        g.j = Vector3(x=float(next_goal.jerk[0]), y=float(next_goal.jerk[1]), z=float(next_goal.jerk[2]))
        g.yaw = float(next_goal.yaw)
        g.dyaw = float(next_goal.dyaw)
        g.power = True
        g.mode_xy = 0
        g.mode_z = 0
        self.pub_goal.publish(g)
        if self.par.visual_level >= 1:
            self._publish_point(next_goal.pos, self.pub_setpoint)

    # ------------------------------------------------------------------
    # DynTraj message conversion
    # ------------------------------------------------------------------
    def _convert_dyntraj_msg(self, msg, current_time: float) -> DynTraj:
        traj = DynTraj()
        bbox_in = list(msg.bbox) if hasattr(msg, "bbox") and msg.bbox else [0.0, 0.0, 0.0]
        traj.bbox = np.array([
            bbox_in[0] / 2.0 + self.par.drone_bbox[0] / 2.0,
            bbox_in[1] / 2.0 + self.par.drone_bbox[1] / 2.0,
            bbox_in[2] / 2.0 + self.par.drone_bbox[2] / 2.0,
        ])
        traj.id = int(msg.id)

        skip_future = self.par.use_only_curr_pos_for_dynamic_obst and not bool(msg.is_agent)
        if not skip_future:
            if hasattr(msg, "pwp") and msg.pwp.times:
                traj.set_piecewise(msg_to_pwp(msg.pwp))
            if hasattr(msg, "function") and len(msg.function) == 3:
                traj.traj_x = msg.function[0]
                traj.traj_y = msg.function[1]
                traj.traj_z = msg.function[2]
            if hasattr(msg, "velocity") and len(msg.velocity) == 3:
                traj.traj_vx = msg.velocity[0]
                traj.traj_vy = msg.velocity[1]
                traj.traj_vz = msg.velocity[2]
            if traj.traj_x and traj.traj_y and traj.traj_z:
                if traj.compile_analytic():
                    traj.mode = "Analytic"
        else:
            traj.traj_x = str(float(msg.pos.x))
            traj.traj_y = str(float(msg.pos.y))
            traj.traj_z = str(float(msg.pos.z))
            traj.traj_vx = "0.0"
            traj.traj_vy = "0.0"
            traj.traj_vz = "0.0"
            traj.compile_analytic()
            traj.mode = "Analytic"

        traj.time_received = current_time
        traj.current_pos = np.array([msg.pos.x, msg.pos.y, msg.pos.z], dtype=float)
        traj.is_agent = bool(msg.is_agent)
        if traj.is_agent and hasattr(msg, "goal") and len(msg.goal) >= 3:
            traj.goal = np.array([msg.goal[0], msg.goal[1], msg.goal[2]], dtype=float)
        if traj.is_agent and self.par.use_comm_delay_inflation:
            stamp = msg.header.stamp
            t_msg = float(stamp.sec) + float(stamp.nanosec) * 1e-9
            delay = self._ros_now() - t_msg
            traj.communication_delay = max(0.0, delay)
        return traj

    # ------------------------------------------------------------------
    # Visualization helpers (minimal — matches the C++ critical-path topics)
    # ------------------------------------------------------------------
    def _publish_point(self, pos: np.ndarray, publisher) -> None:
        p = PointStamped()
        p.header.frame_id = self.viz_frame
        p.header.stamp = self.get_clock().now().to_msg()
        p.point.x = float(pos[0]); p.point.y = float(pos[1]); p.point.z = float(pos[2])
        publisher.publish(p)

    def _publish_velocity_text(self, pos: np.ndarray, velocity: float) -> None:
        m = Marker()
        m.header.frame_id = self.viz_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.action = Marker.ADD
        m.ns = "velocity"; m.id = 0
        m.type = Marker.TEXT_VIEW_FACING
        m.scale.z = 1.0
        m.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        m.text = f"{velocity:.2f}m/s"
        m.pose.position.x = float(pos[0])
        m.pose.position.y = float(pos[1])
        m.pose.position.z = float(pos[2]) + 5.0
        m.pose.orientation.w = 1.0
        self.pub_vel_text.publish(m)

    def _publish_global_path(self) -> None:
        gp = self.sando.get_global_path()
        if not gp:
            return
        ma = MarkerArray()
        ma.markers.append(self._line_marker(gp, ns="global_path", id_=0,
                                            color=ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),
                                            width=0.03))
        self.pub_hgp_path_marker.publish(ma)
        og = self.sando.get_original_global_path()
        if og:
            ma2 = MarkerArray()
            ma2.markers.append(self._line_marker(og, ns="orig_global", id_=0,
                                                  color=ColorRGBA(r=1.0, g=0.6, b=0.0, a=1.0),
                                                  width=0.03))
            self.pub_original_hgp_path_marker.publish(ma2)

    def _publish_traj_committed(self) -> None:
        setpoints = self.sando.retrieve_goal_setpoints()
        if not setpoints:
            return
        pts = [s.pos for s in setpoints]
        ma = MarkerArray()
        ma.markers.append(self._line_marker(pts, ns="committed", id_=1,
                                             color=ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0),
                                             width=0.10))
        self.pub_traj_committed.publish(ma)

    def _line_marker(self, points: List[np.ndarray], ns: str, id_: int,
                     color: ColorRGBA, width: float) -> Marker:
        m = Marker()
        m.header.frame_id = self.viz_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.action = Marker.ADD
        m.type = Marker.LINE_STRIP
        m.ns = ns; m.id = id_
        m.scale.x = float(width)
        m.color = color
        m.pose.orientation.w = 1.0
        from geometry_msgs.msg import Point as _Point
        for p in points:
            pt = _Point(); pt.x = float(p[0]); pt.y = float(p[1]); pt.z = float(p[2])
            m.points.append(pt)
        return m

    def _publish_own_traj(self) -> None:
        DynTrajMsg = _get_dyntraj_msg()
        if DynTrajMsg is None:
            return
        self.pwp_to_share = self.sando.get_pwp()
        msg = DynTrajMsg()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.viz_frame
        msg.bbox = [float(self.par.drone_bbox[0]),
                    float(self.par.drone_bbox[1]),
                    float(self.par.drone_bbox[2])]
        msg.id = self.id_
        msg.mode = "pwp"
        pwp_msg = pwp_to_msg(self.pwp_to_share)
        if pwp_msg is not None:
            msg.pwp = pwp_msg
        msg.is_agent = True
        cur = self.sando.get_state()
        msg.pos.x = float(cur.pos[0]); msg.pos.y = float(cur.pos[1]); msg.pos.z = float(cur.pos[2])
        Gp = self.sando.get_G().pos
        msg.goal = [float(Gp[0]), float(Gp[1]), float(Gp[2])]
        self.pub_own_traj.publish(msg)

    def _publish_computation_times(self, result: bool) -> None:
        CT = _get_computation_times_msg()
        if CT is None or self.pub_computation_times is None:
            return
        d = self.sando.retrieve_data()
        msg = CT()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.result = bool(result)
        msg.successful_factor = float(d["successful_factor"])
        msg.total_replanning_ms = float(self.replanning_computation_time * 1000.0)
        msg.global_planning_ms = float(d["global_planning_time"])
        msg.hgp_static_jps_ms = float(d["hgp_static_jps_time"])
        msg.hgp_check_path_ms = float(d["hgp_check_path_time"])
        msg.hgp_dynamic_astar_ms = float(d["hgp_dynamic_astar_time"])
        msg.hgp_recover_path_ms = float(d["hgp_recover_path_time"])
        msg.cvx_decomp_ms = float(d["cvx_decomp_time"])
        msg.local_traj_ms = float(d["local_traj_computation_time"])
        msg.safe_paths_ms = float(d["safe_paths_time"])
        msg.safety_check_ms = float(d["safety_check_time"])
        msg.yaw_sequence_ms = float(d["yaw_sequence_time"])
        msg.yaw_fitting_ms = float(d["yaw_fitting_time"])
        self.pub_computation_times.publish(msg)

    # ------------------------------------------------------------------
    def _ros_now(self) -> float:
        t = self.get_clock().now()
        return float(t.nanoseconds) * 1e-9


def _get_dyntraj_msg():
    try:
        from dynus_interfaces.msg import DynTraj as DT
        return DT
    except ImportError:
        return None


def _get_state_msg():
    try:
        from dynus_interfaces.msg import State
        return State
    except ImportError:
        return None


def _get_goal_msg():
    try:
        from dynus_interfaces.msg import Goal
        return Goal
    except ImportError:
        return None


def _get_computation_times_msg():
    try:
        from dynus_interfaces.msg import ComputationTimes
        return ComputationTimes
    except ImportError:
        return None


def main(args=None):
    rclpy.init(args=args)
    node = SANDONode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
