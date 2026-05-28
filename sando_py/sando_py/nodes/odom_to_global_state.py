"""Odometry (local) -> global-frame State + PoseStamped — port of odom_to_global_state.cpp.

Looks up TF: world -> <veh_name>/init_pose, caches it, then transforms every
incoming Odometry message into the global "world" frame.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped

import tf2_ros
from tf2_ros import Buffer, TransformListener


def _state_msg():
    try:
        from dynus_interfaces.msg import State
        return State
    except ImportError:
        return None


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton quaternion product (xyzw)."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = q
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


class OdomToGlobalState(Node):
    def __init__(self):
        super().__init__("odom_to_global_state")

        ns = self.get_namespace() or "/"
        self.veh_name = ns.rstrip("/").split("/")[-1] or "PX04"

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        StateMsg = _state_msg()
        if StateMsg is None:
            raise RuntimeError("dynus_interfaces not available")
        self.pub_state = self.create_publisher(StateMsg, "state", 10)
        self.pub_pose = self.create_publisher(PoseStamped, "global_pose", 10)
        self.sub_odom = self.create_subscription(Odometry, "odom", self._odom_cb, 10)
        self.tf_timer = self.create_timer(0.1, self._poll_tf)

        self.tf_acquired = False
        self.t_init = np.zeros(3)
        self.q_init = np.array([0.0, 0.0, 0.0, 1.0])  # xyzw
        self.R_init = np.eye(3)
        self.get_logger().info(f"Waiting for TF: world -> {self.veh_name}/init_pose")

    def _poll_tf(self) -> None:
        target = "world"
        source = f"{self.veh_name}/init_pose"
        if not self.tf_buffer.can_transform(target, source, Time()):
            return
        try:
            tf = self.tf_buffer.lookup_transform(target, source, Time())
        except tf2_ros.TransformException as ex:
            self.get_logger().warn(f"TF lookup failed: {ex}", throttle_duration_sec=2.0)
            return
        self.t_init = np.array([tf.transform.translation.x,
                                tf.transform.translation.y,
                                tf.transform.translation.z])
        self.q_init = np.array([tf.transform.rotation.x,
                                tf.transform.rotation.y,
                                tf.transform.rotation.z,
                                tf.transform.rotation.w])
        self.R_init = _quat_to_rot(self.q_init)
        self.tf_acquired = True
        self.tf_timer.cancel()
        yaw = math.atan2(self.R_init[1, 0], self.R_init[0, 0])
        self.get_logger().info(
            f"TF acquired: t=[{self.t_init[0]:.3f}, {self.t_init[1]:.3f}, "
            f"{self.t_init[2]:.3f}], yaw={yaw:.3f} rad"
        )

    def _odom_cb(self, msg: Odometry) -> None:
        if not self.tf_acquired:
            return
        local_pos = np.array([msg.pose.pose.position.x,
                              msg.pose.pose.position.y,
                              msg.pose.pose.position.z])
        local_vel = np.array([msg.twist.twist.linear.x,
                              msg.twist.twist.linear.y,
                              msg.twist.twist.linear.z])
        local_quat = np.array([msg.pose.pose.orientation.x,
                               msg.pose.pose.orientation.y,
                               msg.pose.pose.orientation.z,
                               msg.pose.pose.orientation.w])
        global_pos = self.R_init @ local_pos + self.t_init
        global_vel = self.R_init @ local_vel
        global_quat = _quat_mul(self.q_init, local_quat)
        n = float(np.linalg.norm(global_quat))
        if n > 1e-9:
            global_quat = global_quat / n

        StateMsg = _state_msg()
        if StateMsg is None:
            return
        s = StateMsg()
        s.header.stamp = msg.header.stamp
        s.header.frame_id = "world"
        s.pos.x, s.pos.y, s.pos.z = float(global_pos[0]), float(global_pos[1]), float(global_pos[2])
        s.vel.x, s.vel.y, s.vel.z = float(global_vel[0]), float(global_vel[1]), float(global_vel[2])
        s.quat.x = float(global_quat[0]); s.quat.y = float(global_quat[1])
        s.quat.z = float(global_quat[2]); s.quat.w = float(global_quat[3])
        self.pub_state.publish(s)

        p = PoseStamped()
        p.header.stamp = msg.header.stamp
        p.header.frame_id = "world"
        p.pose.position.x, p.pose.position.y, p.pose.position.z = (
            float(global_pos[0]), float(global_pos[1]), float(global_pos[2]),
        )
        p.pose.orientation.x = float(global_quat[0])
        p.pose.orientation.y = float(global_quat[1])
        p.pose.orientation.z = float(global_quat[2])
        p.pose.orientation.w = float(global_quat[3])
        self.pub_pose.publish(p)


def main(args=None):
    rclpy.init(args=args)
    node = OdomToGlobalState()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
