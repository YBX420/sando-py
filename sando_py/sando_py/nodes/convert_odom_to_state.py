"""Odometry -> dynus_interfaces/State — port of convert_odom_to_state.cpp."""
from __future__ import annotations

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry


def _get_state_msg():
    try:
        from dynus_interfaces.msg import State
        return State
    except ImportError:
        return None


class OdometryToStateNode(Node):
    def __init__(self):
        super().__init__("odometry_to_state_node")
        StateMsg = _get_state_msg()
        if StateMsg is None:
            raise RuntimeError("dynus_interfaces not available")
        self.pub_state = self.create_publisher(StateMsg, "state", 10)
        self.sub_odom = self.create_subscription(Odometry, "odom", self._cb, 10)

    def _cb(self, msg: Odometry) -> None:
        StateMsg = _get_state_msg()
        if StateMsg is None:
            return
        s = StateMsg()
        s.header.stamp = msg.header.stamp
        s.header.frame_id = msg.header.frame_id
        s.pos.x = msg.pose.pose.position.x
        s.pos.y = msg.pose.pose.position.y
        s.pos.z = msg.pose.pose.position.z
        s.vel.x = msg.twist.twist.linear.x
        s.vel.y = msg.twist.twist.linear.y
        s.vel.z = msg.twist.twist.linear.z
        s.quat.x = msg.pose.pose.orientation.x
        s.quat.y = msg.pose.pose.orientation.y
        s.quat.z = msg.pose.pose.orientation.z
        s.quat.w = msg.pose.pose.orientation.w
        self.pub_state.publish(s)


def main(args=None):
    rclpy.init(args=args)
    node = OdometryToStateNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
