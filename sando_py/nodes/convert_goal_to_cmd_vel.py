"""Goal -> cmd_vel for ground robots — Python port of convert_goal_to_cmd_vel.cpp.

Implements the same Lyapunov-based controller as the C++ node:
  v_command   = v_desired * cos(eyaw) - kx * ex
  yawd_command = yawd_desired
                  - v_desired * (ky*ey + sin(eyaw)) / sqrt(ey^2 + eps^2)
                  - kyaw * eyaw
where (ex, ey) is the position error rotated into the desired yaw frame.
"""
from __future__ import annotations

import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

from ..utils import yaw_from_quat


def _get_goal_msg():
    try:
        from dynus_interfaces.msg import Goal
        return Goal
    except ImportError:
        return None


def _get_state_msg():
    try:
        from dynus_interfaces.msg import State
        return State
    except ImportError:
        return None


class GoalToCmdVel(Node):
    def __init__(self):
        super().__init__("goal_to_cmd_vel")
        self.current_yaw = 0.0
        self.state_initialized = False
        self.goal_initialized = False

        x = self.declare_parameter("x", 0.0).value
        y = self.declare_parameter("y", 0.0).value
        z = self.declare_parameter("z", 0.0).value
        yaw = self.declare_parameter("yaw", 0.0).value
        cmd_vel_topic = self.declare_parameter("cmd_vel_topic_name", "cmd_vel").value
        self.kx = float(self.declare_parameter("ground_robot_kx", 1.0).value)
        self.ky = float(self.declare_parameter("ground_robot_ky", 1.0).value)
        self.kyaw = float(self.declare_parameter("ground_robot_kyaw", 1.0).value)
        self.eps = float(self.declare_parameter("ground_robot_eps", 1e-2).value)

        self.state_pos = (float(x), float(y), float(z))
        self.state_yaw = float(yaw)
        self.goal_p = (0.0, 0.0, 0.0)
        self.goal_v = (0.0, 0.0, 0.0)
        self.goal_yaw = 0.0
        self.goal_dyaw = 0.0

        self.get_logger().info(
            f"kx={self.kx}, ky={self.ky}, kyaw={self.kyaw}, eps={self.eps}"
        )

        self.pub_cmd_vel = self.create_publisher(Twist, cmd_vel_topic, 10)
        GoalMsg = _get_goal_msg()
        StateMsg = _get_state_msg()
        if GoalMsg is not None:
            self.sub_goal = self.create_subscription(GoalMsg, "goal", self._goal_cb, 10)
        if StateMsg is not None:
            self.sub_state = self.create_subscription(StateMsg, "state", self._state_cb, 10)
        self.timer = self.create_timer(0.01, self._tick)

    def _state_cb(self, msg) -> None:
        self.state_pos = (msg.pos.x, msg.pos.y, msg.pos.z)
        self.state_yaw = yaw_from_quat(msg.quat.x, msg.quat.y, msg.quat.z, msg.quat.w)
        self.state_initialized = True

    def _goal_cb(self, msg) -> None:
        self.goal_p = (msg.p.x, msg.p.y, msg.p.z)
        self.goal_v = (msg.v.x, msg.v.y, msg.v.z)
        self.goal_yaw = float(msg.yaw)
        self.goal_dyaw = float(msg.dyaw)
        self.goal_initialized = True

    def _tick(self) -> None:
        if not self.state_initialized or not self.goal_initialized:
            return
        x_d, y_d, _ = self.goal_p
        xd_d, yd_d, _ = self.goal_v
        yaw_d, dyaw_d = self.goal_yaw, self.goal_dyaw
        v_d = math.hypot(xd_d, yd_d)
        sx, sy, _ = self.state_pos
        ex = math.cos(yaw_d) * (sx - x_d) + math.sin(yaw_d) * (sy - y_d)
        ey = -math.sin(yaw_d) * (sx - x_d) + math.cos(yaw_d) * (sy - y_d)
        eyaw = self.state_yaw - yaw_d
        v_command = v_d * math.cos(eyaw) - self.kx * ex
        denom = math.sqrt(ey * ey + self.eps * self.eps)
        yawd_command = dyaw_d - v_d * (self.ky * ey + math.sin(eyaw)) / denom - self.kyaw * eyaw
        twist = Twist()
        twist.linear.x = float(v_command)
        twist.angular.z = float(yawd_command)
        self.pub_cmd_vel.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = GoalToCmdVel()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
