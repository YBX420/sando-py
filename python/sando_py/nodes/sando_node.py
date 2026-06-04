"""SANDO ROS 2 node — Python port of src/sando/sando_node.cpp.

中文说明
========
这是整个规划器的「对外接线盒」(ROS 2 节点)。它本身不做规划算法,而是负责:
  1. 从外面收消息(机器人自身状态、终点目标、别的无人机轨迹、动态障碍预测、点云地图);
  2. 把这些信息喂给真正干活的规划器对象 self.sando(SANDO 类,在 ../planner 里);
  3. 按固定节奏(定时器)反复调用 self.sando.replan() 重新规划,
     再把算出来的轨迹/下一步控制点(goal)发出去给飞控,顺便发一堆 RViz 可视化。
这是 C++ 版 sando_node.cpp 的 Python 移植,topic 名和发布内容尽量和 C++ 对齐。

关键输入:state(里程计/状态)、term_goal(终点)、predicted_trajs(动障预测)、点云(地图)。
关键输出:goal(每一拍发给飞控的位置/速度/加速度等设定点)、/trajs(自己的轨迹给别的飞机看)、
         以及各种 marker 可视化。

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


# QoS = ROS 2 里收发消息的「服务质量」配置。
# critical 用于控制相关、不能丢的消息:RELIABLE(保证送达,丢了会重发)。
def _critical_qos() -> QoSProfile:
    return QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST, depth=10,
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.VOLATILE,
    )


# sensor 用于点云这种又大又频繁的传感器数据:BEST_EFFORT(尽力发,允许丢,换低延迟)。
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

    中文:把 ROS 的点云消息(PointCloud2)转成一个 N 行 3 列的 numpy 数组,每行是一个点的 xyz。
    C++ 那边用 PCL 库做这件事,Python 这边改用 sensor_msgs_py。
    """
    gen = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
    return np.array([(float(p[0]), float(p[1]), float(p[2])) for p in gen],
                    dtype=np.float64) if gen is not None else np.zeros((0, 3))


class SANDONode(Node):
    """ROS 2 node wrapper. Holds the SANDO planner and bridges topics ↔ planner state.

    中文:ROS 2 节点的外壳。内部持有一个 SANDO 规划器(self.sando),
    负责把 ROS topic 上来回的消息和规划器的内部状态对接起来。
    """

    def __init__(self):
        super().__init__("sando_node")

        # 从命名空间里解析出本机编号。多机时每架飞机跑在不同 ns(如 /NX01),
        # 取末尾两位数字当 id_,用来区分「这是不是我自己发的轨迹」。
        # Namespace / id (matches C++ ns -> NX## convention)
        ns = self.get_namespace() or "/"
        ns = ns.rstrip("/").split("/")[-1] if ns.rstrip("/") else "NX01"
        self.ns_ = ns
        self.id_str = ns[-2:] if len(ns) >= 2 and ns[-2:].isdigit() else "01"
        try:
            self.id_ = int(self.id_str)
        except ValueError:
            self.id_ = 1

        # 一次性从 ROS 参数服务器把所有规划参数读进来(v_max、bbox、各种开关等)。
        # Parameters
        self.par = Parameters.from_ros_node(self)
        # 可视化用哪个坐标系名:硬件且已在全局系时用 "world",否则用 "map"。
        # Visualization frame: "world" for hw+global-frame, "map" otherwise (C++ parity)
        self.viz_frame = ("world" if (self.par.use_hardware and self.par.state_already_in_global_frame)
                          else "map")

        # 真正干活的规划器对象。本节点只是它的「传话筒」。
        # Planner
        self.sando = SANDO(self.par)

        # 回调组:控制哪些回调能并行跑、哪些必须串行。
        #  - ReentrantCallbackGroup:允许并发(同一组里多个回调可同时执行)。
        #  - MutuallyExclusiveCallbackGroup:互斥,组内一次只跑一个(replan 用它,防止重入)。
        # Callback groups
        self.cb_re = ReentrantCallbackGroup()
        self.cb_replan = MutuallyExclusiveCallbackGroup()
        self.cb_goal = ReentrantCallbackGroup()
        self.cb_map = ReentrantCallbackGroup()

        critical = _critical_qos()
        sensor_q = _sensor_qos()

        # 发布者。C++ 那边发约 25 个 topic,这里只保留控制必需的 + 主要的可视化。
        # pub_goal 最关键:每一拍发给飞控的设定点(位置/速度/...)。
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

        # 订阅者。state=自身状态,term_goal=终点,/trajs=别人的轨迹,predicted_trajs=动障预测。
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

        # 别人的轨迹(/trajs)和动障预测(predicted_trajs)走同一个回调 traj_callback,
        # 进去后靠 is_agent 字段区分是「飞机」还是「障碍」。
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

        # 点云(地图)的来源随仿真环境不同而不同:
        #  - fake_sim:订一路点云(全局静态点云 或 实时传感器点云)。
        #  - rviz_only:没有真点云,塞一个空地图进去,好让规划器通过「准备就绪」检查。
        #  - gazebo/硬件:订两路点云(已知占据 + 未知区域),凑齐一对再喂给规划器。
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

        # 三个定时器(节点的「心跳」):
        #  - replan:每 10ms 重规划一次(对应 C++ 的 100Hz 主循环)。
        #  - goal:每 par.dc 秒发一次设定点给飞控(频率高,通常 100Hz~1kHz)。
        #  - cleanup:每 0.5s 清理过期的别家轨迹。
        # 注意:replan 和 goal 先 cancel 掉,等条件满足才 reset 启动(见下)——
        # 没收到状态就发 goal 会发出垃圾值,没设终点就 replan 没意义。
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
        # 收到自身状态(位置/速度/姿态)。把消息转成内部 RobotState 喂给规划器。
        # 第一次收到时:把 goal 定时器启动起来(state_initialized 这道闸只放一次)。
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
        # 收到终点(一般来自 RViz 点击或外部发送)。设进规划器,并启动 replan 主循环。
        # force_goal_z:忽略点来的高度,强行用配置里的默认高度(很多 demo 在固定高度飞)。
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
        # 收到别家轨迹或动障预测。先把「自己发的」过滤掉(id 相同),剩下的转成内部 DynTraj
        # 加进规划器,后续避障会用到。
        if msg.id == self.id_:
            return
        current_time = self._ros_now()
        traj = self._convert_dyntraj_msg(msg, current_time)
        self.sando.add_traj(traj, current_time)

    def occupancy_map_callback(self, msg: PointCloud2) -> None:
        # fake_sim 用的地图回调:点云转 numpy 喂进规划器。
        pc = _pointcloud2_to_numpy(msg)
        self.sando.update_occupancy_map_ptr(pc)
        # 全局点云模式下地图是静态的,读一次就够了——读完就退订,省得反复处理。
        # In global-pc mode the cloud is static — drop subscription after first read
        if self.par.use_global_pc:
            self.destroy_subscription(self.sub_occ)

    # gazebo/硬件:占据点云和未知点云分两路来,各自存最新一帧,再尝试凑成一对。
    def _occ_callback(self, msg: PointCloud2) -> None:
        with self._cloud_lock:
            self._latest_occ = _pointcloud2_to_numpy(msg)
            self._maybe_push_map_pair()

    def _unk_callback(self, msg: PointCloud2) -> None:
        with self._cloud_lock:
            self._latest_unk = _pointcloud2_to_numpy(msg)
            self._maybe_push_map_pair()

    def _maybe_push_map_pair(self) -> None:
        # 两路点云都到齐了才一起喂给规划器(近似时间同步:用各自最新一帧凑对)。
        if self._latest_occ is not None and self._latest_unk is not None:
            self.sando.update_map_ptr(self._latest_occ, self._latest_unk)
            # Keep the latest pair; matches "approximate time sync" semantics

    def cleanup_old_trajs_callback(self) -> None:
        # 定时清掉太久没更新的别家轨迹,免得拿陈旧数据当障碍躲。
        self.sando.clean_up_old_trajs(self._ros_now())

    # ------------------------------------------------------------------
    # Main replan + goal loops
    # ------------------------------------------------------------------
    def replan_callback(self) -> None:
        # 规划主循环(10ms 一次)。核心就一句 self.sando.replan():
        # 它内部跑 heat-A* 全局向导 + MINCO 局部优化 + 避障,算出一条新轨迹。
        # 返回两个标志:replan_ok=局部轨迹成功,hgp_ok=全局路径成功。
        # 算成功了就把自己的轨迹发出去给别的飞机,顺便(限频)发可视化。
        current_time = self._ros_now()
        t0 = time.perf_counter()
        replan_ok, hgp_ok = self.sando.replan(self.replanning_computation_time, current_time)
        if replan_ok:
            # 记录这次重规划耗时,下次 replan 会用它做时间补偿(轨迹起点要往前挪一点)。
            self.replanning_computation_time = time.perf_counter() - t0
            self._publish_own_traj()

        # 可视化限频:最多 20Hz(每 0.05s 一次),避免刷 marker 把 RViz 拖垮。
        do_viz = False
        if self.par.visual_level >= 1:
            t_now = self._ros_now()
            if t_now - self.last_replan_viz_publish_t >= 0.05:
                do_viz = True
                self.last_replan_viz_publish_t = t_now

        if do_viz and hgp_ok:
            self._publish_global_path()
        if do_viz and replan_ok:
            # A=本次轨迹起点,G=局部子目标,E=轨迹终点。发出来在 RViz 里看规划状态。
            self._publish_traj_committed()
            self._publish_point(self.sando.get_A().pos, self.pub_point_A)
            self._publish_point(self.sando.get_G().pos, self.pub_point_G)
            self._publish_point(self.sando.get_E().pos, self.pub_point_E)

        # Always publish computation times (matches C++ behavior)
        if self.pub_computation_times is not None:
            self._publish_computation_times(replan_ok)

    def publish_goal(self) -> None:
        # 高频发设定点给飞控(频率由 par.dc 定)。从已提交轨迹上取「下一个该到的点」,
        # 填好位置/速度/加速度/jerk/yaw,发到 goal topic。这是真正驱动飞机动起来的那根线。
        # 硬件且已到点时不再发,避免到点后继续推设定点。
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
        # 把收到的 DynTraj 消息转成内部 DynTraj 对象。这里有两个要点:
        #  1) bbox 做了「闵可夫斯基膨胀」:障碍半尺寸 + 自身无人机半尺寸,
        #     这样后面可以把无人机当质点来碰撞判定。
        #  2) 障碍的未来轨迹可能以两种形式给:pwp(分段多项式)或 function(解析表达式字符串)。
        traj = DynTraj()
        bbox_in = list(msg.bbox) if hasattr(msg, "bbox") and msg.bbox else [0.0, 0.0, 0.0]
        traj.bbox = np.array([
            bbox_in[0] / 2.0 + self.par.drone_bbox[0] / 2.0,
            bbox_in[1] / 2.0 + self.par.drone_bbox[1] / 2.0,
            bbox_in[2] / 2.0 + self.par.drone_bbox[2] / 2.0,
        ])
        traj.id = int(msg.id)

        # skip_future:对动态障碍只用「当前位置」、不信它的未来预测(配置开关)。
        # 此时把轨迹退化成「停在原地」的常数表达式;别的飞机(is_agent)不受影响。
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
        # 通信延迟补偿:多机时别家消息有传输延迟,记下来好把对方轨迹「多躲一点」。
        if traj.is_agent and self.par.use_comm_delay_inflation:
            stamp = msg.header.stamp
            t_msg = float(stamp.sec) + float(stamp.nanosec) * 1e-9
            delay = self._ros_now() - t_msg
            traj.communication_delay = max(0.0, delay)
        return traj

    # ------------------------------------------------------------------
    # Visualization helpers (minimal — matches the C++ critical-path topics)
    # ------------------------------------------------------------------
    # 下面几个 _publish_* 都是纯可视化:把点/线/文字打包成 RViz marker 发出去,不影响规划。
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
        # 画 heat-A* 全局向导路径(红线)和它优化前的原始路径(橙线),方便对比。
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
        # 画「已提交轨迹」(绿线)——就是飞机接下来真正要走、会逐点发给飞控的那条。
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
        # 工具函数:把一串点连成一条 RViz 折线(LINE_STRIP)marker。
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
        # 把自己刚算好的轨迹(以 pwp 分段多项式形式)发到 /trajs,让别的飞机能预测并躲开我。
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
        # 发各阶段耗时统计(全局规划/A*/局部轨迹/安全检查/yaw 等),给 benchmark 用来分析性能。
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
        # 当前 ROS 时间,转成「秒」的浮点数(规划器内部都用秒算)。
        t = self.get_clock().now()
        return float(t.nanoseconds) * 1e-9


# 下面这几个 _get_*_msg 是「软依赖」:dynus_interfaces 消息包没装时返回 None,
# 节点照样能起来(对应的发布/订阅自动跳过),不会一 import 就崩。
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
    # 节点入口。用多线程 executor,这样高频的 goal 发布、replan、点云回调能分头并行跑,
    # 不会互相卡住(配合前面的回调组设置)。
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
