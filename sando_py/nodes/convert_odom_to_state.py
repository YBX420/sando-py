"""Odometry -> dynus_interfaces/State — port of convert_odom_to_state.cpp.

中文说明:
一个格式转换的「桥接」节点。订阅标准的 nav_msgs/Odometry(里程计:位置+速度+姿态),
原样搬进规划器自己用的 dynus_interfaces/State 消息再发出去。
作用:让规划器能吃各种数据源(仿真/真机)给的里程计,统一成它认的 State 类型。
输入话题 `odom`,输出话题 `state`。纯字段拷贝,不做任何坐标变换。
"""
from __future__ import annotations

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry


# 延迟导入 State 消息类型(没装 dynus_interfaces 就返回 None,不直接崩)
def _get_state_msg():
    try:
        from dynus_interfaces.msg import State
        return State
    except ImportError:
        return None


class OdometryToStateNode(Node):
    """订阅 odom、转成 State 再发布的桥接节点。"""
    def __init__(self):
        super().__init__("odometry_to_state_node")
        StateMsg = _get_state_msg()
        if StateMsg is None:
            raise RuntimeError("dynus_interfaces not available")
        self.pub_state = self.create_publisher(StateMsg, "state", 10)
        self.sub_odom = self.create_subscription(Odometry, "odom", self._cb, 10)

    # 每来一条 odom 就拷贝字段(位置/速度/姿态)到 State 并发布
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
