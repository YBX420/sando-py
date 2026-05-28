"""Lightweight kinematic simulator — Python port of src/sando/fake_sim.cpp.

Publishes the drone's State (and optionally Odometry) by integrating from the
last Goal received. Uses Hopf-fibration quaternion for attitude (matches C++).
Skips the Gazebo set_entity_state service path — that's only relevant when
running with Gazebo, in which case the C++ node is preferred anyway. The TF
broadcast and visualization-marker pipeline is kept intact.
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


def _state_msg():
    try:
        from dynus_interfaces.msg import State
        return State
    except ImportError:
        return None


def _goal_msg():
    try:
        from dynus_interfaces.msg import Goal
        return Goal
    except ImportError:
        return None


def _qos_reliable() -> QoSProfile:
    return QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=10,
                      reliability=QoSReliabilityPolicy.RELIABLE,
                      durability=QoSDurabilityPolicy.VOLATILE)


def _hopf_quat_xyzw(thrust: np.ndarray, yaw: float) -> np.ndarray:
    """Hopf fibration: build q from (acceleration + gravity) and yaw.
    Returns xyzw quaternion matching the C++ FakeSim.goalCallback.
    """
    g = np.array([thrust[0], thrust[1], thrust[2] + 9.81])
    n = float(np.linalg.norm(g))
    if n < 1e-9:
        return np.array([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])
    a, b, c = g / n
    tmp = 1.0 / math.sqrt(2.0 * (1.0 + c))
    qabc = np.array([-b * tmp, a * tmp, 0.0, tmp * (1.0 + c)])
    qpsi = np.array([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])
    return _quat_mul(qabc, qpsi)


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
    def __init__(self):
        super().__init__("fake_sim")
        self.get_logger().info("Initializing FakeSim...")

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

        ns = self.get_namespace() or "/"
        self.ns_ = ns.lstrip("/")
        self.target_frame = f"{self.ns_}/base_link"
        if not self.base_frame_id:
            self.base_frame_id = self.target_frame

        StateMsg = _state_msg()
        GoalMsg = _goal_msg()
        if StateMsg is None:
            raise RuntimeError("dynus_interfaces.msg.State not available")

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
        self.timer = self.create_timer(0.01, self._tick)

        self._lock = Lock()
        self.goal_received = False
        self.goal_stamp = self.get_clock().now()
        self.goal_pos = np.zeros(3)
        self.goal_vel = np.zeros(3)
        self.goal_acc = np.zeros(3)
        self.goal_quat = np.array([0.0, 0.0, 0.0, 1.0])

        self.get_logger().info("FakeSim initialized")

    def _goal_cb(self, msg) -> None:
        thrust = np.array([msg.a.x, msg.a.y, msg.a.z])
        q = _hopf_quat_xyzw(thrust, float(msg.yaw))
        with self._lock:
            self.goal_stamp = self.get_clock().now()
            self.goal_pos = np.array([msg.p.x, msg.p.y, msg.p.z])
            self.goal_vel = np.array([msg.v.x, msg.v.y, msg.v.z])
            self.goal_acc = thrust
            self.goal_quat = q
            self.goal_received = True
            self.state_pos = self.goal_pos.copy()
            self.state_vel = self.goal_vel.copy()
            self.state_quat = q

    def _tick(self) -> None:
        now = self.get_clock().now()
        with self._lock:
            if self.goal_received:
                dt = max(0.0, min(0.1, (now - self.goal_stamp).nanoseconds * 1e-9))
                p_interp = self.goal_pos + self.goal_vel * dt + 0.5 * self.goal_acc * dt * dt
                quat = self.goal_quat
            else:
                p_interp = self.state_pos.copy()
                quat = self.state_quat.copy()

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
