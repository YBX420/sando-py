"""Vicon Pose+Twist -> dynus_interfaces/State — port of convert_vicon_to_state.cpp.

The C++ node uses message_filters::ApproximateTime. We replicate that with a
simple monotonic-time match (closest twist within a short window) since
message_filters is C++-only.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from geometry_msgs.msg import PoseStamped, TwistStamped


def _state_msg():
    try:
        from dynus_interfaces.msg import State
        return State
    except ImportError:
        return None


def _stamp_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class PoseTwistToStateNode(Node):
    def __init__(self):
        super().__init__("pose_twist_to_state_node")
        StateMsg = _state_msg()
        if StateMsg is None:
            raise RuntimeError("dynus_interfaces not available")
        qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1,
                         reliability=QoSReliabilityPolicy.BEST_EFFORT,
                         durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.sub_pose = self.create_subscription(PoseStamped, "world", self._pose_cb, qos)
        self.sub_twist = self.create_subscription(TwistStamped, "twist", self._twist_cb, qos)
        self.pub_state = self.create_publisher(StateMsg, "state", 10)
        self._poses: Deque[PoseStamped] = deque(maxlen=20)
        self._twists: Deque[TwistStamped] = deque(maxlen=20)
        # 30 ms sync slop — matches ApproximateTime defaults
        self._slop = 0.03

    def _pose_cb(self, msg: PoseStamped) -> None:
        self._poses.append(msg)
        self._try_pair()

    def _twist_cb(self, msg: TwistStamped) -> None:
        self._twists.append(msg)
        self._try_pair()

    def _try_pair(self) -> None:
        if not self._poses or not self._twists:
            return
        StateMsg = _state_msg()
        if StateMsg is None:
            return
        # Match latest pose with nearest twist (within slop)
        latest_pose = self._poses[-1]
        t_pose = _stamp_seconds(latest_pose.header.stamp)
        best_dt = float("inf")
        best_twist = None
        for tw in self._twists:
            dt = abs(_stamp_seconds(tw.header.stamp) - t_pose)
            if dt < best_dt:
                best_dt = dt; best_twist = tw
        if best_twist is None or best_dt > self._slop:
            return
        s = StateMsg()
        s.header.stamp = self.get_clock().now().to_msg()
        s.header.frame_id = latest_pose.header.frame_id
        s.pos.x = latest_pose.pose.position.x
        s.pos.y = latest_pose.pose.position.y
        s.pos.z = latest_pose.pose.position.z
        s.vel.x = best_twist.twist.linear.x
        s.vel.y = best_twist.twist.linear.y
        s.vel.z = best_twist.twist.linear.z
        s.quat.x = latest_pose.pose.orientation.x
        s.quat.y = latest_pose.pose.orientation.y
        s.quat.z = latest_pose.pose.orientation.z
        s.quat.w = latest_pose.pose.orientation.w
        self.pub_state.publish(s)


def main(args=None):
    rclpy.init(args=args)
    node = PoseTwistToStateNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
