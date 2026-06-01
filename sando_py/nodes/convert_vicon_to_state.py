"""Vicon Pose+Twist -> dynus_interfaces/State — port of convert_vicon_to_state.cpp.

The C++ node uses message_filters::ApproximateTime. We replicate that with a
simple monotonic-time match (closest twist within a short window) since
message_filters is C++-only.

中文说明:
桥接节点,用于真机的 Vicon 动捕系统。Vicon 把「位置/姿态」(PoseStamped,话题 world)
和「速度」(TwistStamped,话题 twist)分成两条话题发,这里要把同一时刻的这两条配对、
合并成规划器要的 dynus_interfaces/State,再发到话题 `state`。

难点是两条话题时间戳不会完全对齐。C++ 版用 message_filters 的 ApproximateTime 做近似
时间同步;Python 没这个库,所以这里自己实现:拿最新的一条 pose,去 twist 缓存里找
时间戳最接近的一条,只要时间差在容差(slop=30ms)内就算配上、合并发布。
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from geometry_msgs.msg import PoseStamped, TwistStamped


# 延迟导入 State 消息类型(没装就返回 None)
def _state_msg():
    try:
        from dynus_interfaces.msg import State
        return State
    except ImportError:
        return None


# 把 ROS 时间戳(sec + nanosec)换算成浮点秒,方便比较时间差
def _stamp_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class PoseTwistToStateNode(Node):
    """订阅 Vicon 的 pose 和 twist 两条话题,按时间近似配对合并成 State 后发布。"""
    def __init__(self):
        super().__init__("pose_twist_to_state_node")
        StateMsg = _state_msg()
        if StateMsg is None:
            raise RuntimeError("dynus_interfaces not available")
        # BEST_EFFORT + 只留最新一条:动捕高频实时数据,丢几帧无所谓,要的是最新
        qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1,
                         reliability=QoSReliabilityPolicy.BEST_EFFORT,
                         durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.sub_pose = self.create_subscription(PoseStamped, "world", self._pose_cb, qos)
        self.sub_twist = self.create_subscription(TwistStamped, "twist", self._twist_cb, qos)
        self.pub_state = self.create_publisher(StateMsg, "state", 10)
        # 各留最近 20 条做缓冲,供配对时回查
        self._poses: Deque[PoseStamped] = deque(maxlen=20)
        self._twists: Deque[TwistStamped] = deque(maxlen=20)
        # 30 ms sync slop — matches ApproximateTime defaults
        # 时间同步容差 30ms:两条消息时间差小于它才算「同一时刻」,对齐 C++ 默认值
        self._slop = 0.03

    # 收到 pose 就入缓存并尝试配对
    def _pose_cb(self, msg: PoseStamped) -> None:
        self._poses.append(msg)
        self._try_pair()

    # 收到 twist 就入缓存并尝试配对
    def _twist_cb(self, msg: TwistStamped) -> None:
        self._twists.append(msg)
        self._try_pair()

    # 核心:拿最新 pose 找时间最近的 twist,差值在容差内就合并成 State 发出去
    def _try_pair(self) -> None:
        if not self._poses or not self._twists:
            return
        StateMsg = _state_msg()
        if StateMsg is None:
            return
        # Match latest pose with nearest twist (within slop)
        latest_pose = self._poses[-1]
        t_pose = _stamp_seconds(latest_pose.header.stamp)
        # 遍历 twist 缓存,挑时间戳和 pose 最接近的那条
        best_dt = float("inf")
        best_twist = None
        for tw in self._twists:
            dt = abs(_stamp_seconds(tw.header.stamp) - t_pose)
            if dt < best_dt:
                best_dt = dt; best_twist = tw
        # 最接近的也超出容差,就放弃这次配对(等更接近的 twist 到来)
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
