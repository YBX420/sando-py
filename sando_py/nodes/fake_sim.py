"""Lightweight kinematic simulator — Python port of src/sando/fake_sim.cpp.

Publishes the drone's State (and optionally Odometry) by integrating from the
last Goal received. Uses Hopf-fibration quaternion for attitude (matches C++).
Skips the Gazebo set_entity_state service path — that's only relevant when
running with Gazebo, in which case the C++ node is preferred anyway. The TF
broadcast and visualization-marker pipeline is kept intact.

中文说明:
这是一个「假飞机」运动学仿真节点(没有真实物理引擎,只做简单运动外推)。
它在 RViz-only 的演示模式里替代真飞机:订阅规划器发来的控制指令 Goal
(目标位置/速度/加速度/偏航),把它当成飞机的真实状态,再发布出去给规划器闭环。

它的作用 / 在整套规划器里扮演的角色:
  - 输入:话题 `goal`(dynus_interfaces/Goal,规划器算出来的下一拍指令)。
  - 输出:话题 `state`(飞机当前状态,规划器下一拍要读它当起点)、
          TF 变换(map -> <namespace>/base_link,给 RViz 显示用)、
          可选的 `odom` 里程计、可选的飞机三维模型 marker。
  - 姿态用 Hopf 纤维化(Hopf fibration)从「加速度+重力」反推出来,
    跟 C++ 版本算法一致,保证仿真行为对得上基线。
跳过了 Gazebo 的 set_entity_state 服务,因为那条路只有真用 Gazebo 时才需要,
而那种场景直接用 C++ 节点更合适。
"""
from __future__ import annotations

import math
from threading import Lock

import numpy as np

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from geometry_msgs.msg import TransformStamped, Vector3, Quaternion
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker


# 延迟导入 State 消息类型:dynus_interfaces 可能没装,没装就返回 None(不直接崩)
def _state_msg():
    try:
        from dynus_interfaces.msg import State
        return State
    except ImportError:
        return None


# 同上,延迟导入 Goal 消息类型(规划器发给飞机的控制指令)
def _goal_msg():
    try:
        from dynus_interfaces.msg import Goal
        return Goal
    except ImportError:
        return None


# 一个「可靠传输」的 QoS 配置:保证消息不丢,留最近 10 条
def _qos_reliable() -> QoSProfile:
    return QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=10,
                      reliability=QoSReliabilityPolicy.RELIABLE,
                      durability=QoSDurabilityPolicy.VOLATILE)


def _hopf_quat_xyzw(thrust: np.ndarray, yaw: float) -> np.ndarray:
    """Hopf fibration: build q from (acceleration + gravity) and yaw.
    Returns xyzw quaternion matching the C++ FakeSim.goalCallback.

    中文:用 Hopf 纤维化把「期望加速度 + 重力」和偏航角 yaw 反算出飞机姿态四元数。
    直觉:飞机要产生这个加速度,机身 z 轴(推力方向)必须指向「加速度+重力」的方向,
    这就定下了俯仰/横滚;再叠加一个绕 z 的偏航旋转 yaw。返回 xyzw 顺序的四元数,
    算法和 C++ 版 FakeSim.goalCallback 完全一致。
    入参 thrust 实为期望加速度向量,yaw 为偏航角(弧度)。
    """
    # 推力方向 = 加速度 + 重力补偿(9.81 抵消重力,飞机才能悬停或加速)
    g = np.array([thrust[0], thrust[1], thrust[2] + 9.81])
    n = float(np.linalg.norm(g))
    # 数值兜底:推力几乎为零时方向没意义,只保留偏航旋转
    if n < 1e-9:
        return np.array([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])
    a, b, c = g / n
    tmp = 1.0 / math.sqrt(2.0 * (1.0 + c))
    # qabc:让机身 z 轴对准推力方向的姿态; qpsi:绕 z 轴的偏航旋转
    qabc = np.array([-b * tmp, a * tmp, 0.0, tmp * (1.0 + c)])
    qpsi = np.array([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])
    return _quat_mul(qabc, qpsi)


# Hamilton 四元数乘法(xyzw 顺序),用来把两个旋转拼起来
def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


class FakeSim(Node):
    """假飞机仿真节点:订阅控制指令 goal,外推出飞机状态并对外发布。

    构造时声明一堆参数(起始位姿、是否发 odom、可视化等级等),建好发布/订阅,
    再开一个 100Hz 的定时器 _tick 持续往外发状态和 TF。
    """
    def __init__(self):
        super().__init__("fake_sim")
        self.get_logger().info("Initializing FakeSim...")

        # ---- 以下都是可在 launch 里覆盖的参数 ----
        start_pos = self.declare_parameter("start_pos", [0.0, 0.0, 4.0]).value
        start_yaw = float(self.declare_parameter("start_yaw", -1.57).value)
        self.send_state_to_gazebo = bool(
            self.declare_parameter("send_state_to_gazebo", False).value)
        self.default_goal_z = float(self.declare_parameter("default_goal_z", 0.3).value)
        self.visual_level = int(self.declare_parameter("visual_level", 0).value)
        self.publish_odom = bool(self.declare_parameter("publish_odom", False).value)
        self.odom_topic = str(self.declare_parameter("odom_topic", "odom").value)
        self.odom_frame_id = str(self.declare_parameter("odom_frame_id", "map").value)
        self.base_frame_id = str(self.declare_parameter("base_frame_id", "").value)

        # 命名空间决定 TF 的子坐标系名(多机时每架飞机一个独立 frame)
        ns = self.get_namespace() or "/"
        self.ns_ = ns.lstrip("/")
        self.target_frame = f"{self.ns_}/base_link"
        if not self.base_frame_id:
            self.base_frame_id = self.target_frame

        StateMsg = _state_msg()
        GoalMsg = _goal_msg()
        # State 是必须的,没有就没法工作,直接报错
        if StateMsg is None:
            raise RuntimeError("dynus_interfaces.msg.State not available")

        # 初始状态:停在起点,姿态由起始偏航算出
        self.state_pos = np.array([float(start_pos[0]), float(start_pos[1]), float(start_pos[2])])
        self.state_vel = np.zeros(3)
        self.state_quat = _hopf_quat_xyzw(np.zeros(3), start_yaw)

        self.pub_state = self.create_publisher(StateMsg, "state", _qos_reliable())
        self.pub_marker = self.create_publisher(Marker, "drone_marker", _qos_reliable())
        self.pub_odom = self.create_publisher(Odometry, self.odom_topic, _qos_reliable()) \
            if self.publish_odom else None

        if GoalMsg is not None:
            self.sub_goal = self.create_subscription(GoalMsg, "goal", self._goal_cb, 10)
        else:
            self.get_logger().warn("dynus_interfaces.msg.Goal not available — goal subscription disabled")

        self.br = TransformBroadcaster(self, qos=_qos_reliable())
        # 100Hz 定时器:每 0.01s 把当前状态/TF/marker 发一遍
        self.timer = self.create_timer(0.01, self._tick)

        # goal 回调和定时器可能并发跑,用锁保护共享的目标/状态变量
        self._lock = Lock()
        self.goal_received = False
        self.goal_stamp = self.get_clock().now()
        self.goal_pos = np.zeros(3)
        self.goal_vel = np.zeros(3)
        self.goal_acc = np.zeros(3)
        self.goal_quat = np.array([0.0, 0.0, 0.0, 1.0])

        self.get_logger().info("FakeSim initialized")

    def _goal_cb(self, msg) -> None:
        """收到规划器的控制指令:把飞机状态直接「贴」到指令上(瞬时跟随,无动力学误差)。"""
        # msg.a 是期望加速度,既用来外推位置,也用来反算姿态
        thrust = np.array([msg.a.x, msg.a.y, msg.a.z])
        q = _hopf_quat_xyzw(thrust, float(msg.yaw))
        with self._lock:
            self.goal_stamp = self.get_clock().now()
            self.goal_pos = np.array([msg.p.x, msg.p.y, msg.p.z])
            self.goal_vel = np.array([msg.v.x, msg.v.y, msg.v.z])
            self.goal_acc = thrust
            self.goal_quat = q
            self.goal_received = True
            # 假飞机理想跟随:状态瞬间等于目标,不模拟跟踪误差
            self.state_pos = self.goal_pos.copy()
            self.state_vel = self.goal_vel.copy()
            self.state_quat = q

    def _tick(self) -> None:
        """100Hz 主循环:从最近一次 goal 做匀加速外推,发布 TF / State / Odom / marker。"""
        now = self.get_clock().now()
        with self._lock:
            if self.goal_received:
                # dt = 距上次收到 goal 的时间,夹在 [0, 0.1] 防止指令断流时位置飞出去
                dt = max(0.0, min(0.1, (now - self.goal_stamp).nanoseconds * 1e-9))
                # 匀加速外推: p = p0 + v0*dt + 0.5*a*dt^2
                p_interp = self.goal_pos + self.goal_vel * dt + 0.5 * self.goal_acc * dt * dt
                quat = self.goal_quat
            else:
                # 还没收到任何指令:就停在初始位置
                p_interp = self.state_pos.copy()
                quat = self.state_quat.copy()

        # TF broadcast —— 广播 map -> base_link,RViz 据此把飞机画在正确位置
        # TF broadcast
        tfm = TransformStamped()
        tfm.header.stamp = now.to_msg()
        tfm.header.frame_id = "map"
        tfm.child_frame_id = self.target_frame
        tfm.transform.translation.x = float(p_interp[0])
        tfm.transform.translation.y = float(p_interp[1])
        tfm.transform.translation.z = float(p_interp[2])
        tfm.transform.rotation.x = float(quat[0])
        tfm.transform.rotation.y = float(quat[1])
        tfm.transform.rotation.z = float(quat[2])
        tfm.transform.rotation.w = float(quat[3])
        self.br.sendTransform(tfm)

        # State publish —— 发布飞机当前状态,这是规划器下一拍闭环要读的核心话题
        # State publish
        StateMsg = _state_msg()
        if StateMsg is not None:
            s = StateMsg()
            s.header.stamp = now.to_msg()
            s.header.frame_id = "map"
            s.pos = Vector3(x=float(p_interp[0]), y=float(p_interp[1]), z=float(p_interp[2]))
            s.vel = Vector3(x=float(self.state_vel[0]), y=float(self.state_vel[1]),
                            z=float(self.state_vel[2]))
            s.quat = Quaternion(x=float(quat[0]), y=float(quat[1]),
                                z=float(quat[2]), w=float(quat[3]))
            self.pub_state.publish(s)

        # Odom —— 可选:有些下游(如建图)只认 nav_msgs/Odometry,按需发一份
        # Odom
        if self.pub_odom is not None:
            odom = Odometry()
            odom.header.stamp = now.to_msg()
            odom.header.frame_id = self.odom_frame_id
            odom.child_frame_id = self.base_frame_id
            odom.pose.pose.position.x = float(p_interp[0])
            odom.pose.pose.position.y = float(p_interp[1])
            odom.pose.pose.position.z = float(p_interp[2])
            odom.pose.pose.orientation.x = float(quat[0])
            odom.pose.pose.orientation.y = float(quat[1])
            odom.pose.pose.orientation.z = float(quat[2])
            odom.pose.pose.orientation.w = float(quat[3])
            odom.twist.twist.linear.x = float(self.state_vel[0])
            odom.twist.twist.linear.y = float(self.state_vel[1])
            odom.twist.twist.linear.z = float(self.state_vel[2])
            self.pub_odom.publish(odom)

        # Drone mesh marker (optional) —— 可选:在 RViz 里画一架三维飞机模型,纯展示
        # Drone mesh marker (optional)
        if self.visual_level > 0:
            m = Marker()
            m.id = 1
            m.ns = f"mesh_{self.ns_}"
            m.header.frame_id = "map"
            m.header.stamp = now.to_msg()
            m.type = Marker.MESH_RESOURCE
            m.action = Marker.ADD
            m.pose.position.x = float(p_interp[0])
            m.pose.position.y = float(p_interp[1])
            m.pose.position.z = float(p_interp[2])
            m.pose.orientation.x = float(quat[0])
            m.pose.orientation.y = float(quat[1])
            m.pose.orientation.z = float(quat[2])
            m.pose.orientation.w = float(quat[3])
            m.mesh_use_embedded_materials = True
            m.mesh_resource = "package://sando/meshes/quadrotor/quadrotor.dae"
            m.scale.x = m.scale.y = m.scale.z = 0.75
            self.pub_marker.publish(m)


def main(args=None):
    # 用多线程 executor:回调(goal)和定时器(tick)能并发,响应更及时
    rclpy.init(args=args)
    node = FakeSim()
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
