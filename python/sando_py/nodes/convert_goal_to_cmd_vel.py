"""Goal -> cmd_vel for ground robots — Python port of convert_goal_to_cmd_vel.cpp.

Implements the same Lyapunov-based controller as the C++ node:
  v_command   = v_desired * cos(eyaw) - kx * ex
  yawd_command = yawd_desired
                  - v_desired * (ky*ey + sin(eyaw)) / sqrt(ey^2 + eps^2)
                  - kyaw * eyaw
where (ex, ey) is the position error rotated into the desired yaw frame.

中文说明:
给「地面机器人」用的桥接 + 控制节点。规划器输出的是飞机式的 Goal(期望位置/速度/偏航),
但差速底盘只能接收「前进线速度 + 转向角速度」(geometry_msgs/Twist,话题 cmd_vel)。
这个节点把 Goal 转成底盘能执行的 cmd_vel。

它用的是一个基于李雅普诺夫(Lyapunov)思想设计的轨迹跟踪控制器——大白话就是:
比较「当前位姿」和「期望位姿」的差,根据这个误差算出该给多大的前进速度和转向速度,
让机器人逐渐贴回期望轨迹。这是经典的非完整移动机器人(只能前进+转向)跟踪控制律。

输入:话题 `goal`(期望状态) + 话题 `state`(机器人当前状态)。
输出:话题 `cmd_vel`(底盘的速度指令)。控制律和 C++ 版完全一致。
"""
from __future__ import annotations

import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

from ..utils import yaw_from_quat


# 延迟导入 Goal 消息类型(没装就返回 None)
def _get_goal_msg():
    try:
        from dynus_interfaces.msg import Goal
        return Goal
    except ImportError:
        return None


# 延迟导入 State 消息类型(没装就返回 None)
def _get_state_msg():
    try:
        from dynus_interfaces.msg import State
        return State
    except ImportError:
        return None


class GoalToCmdVel(Node):
    """把规划器的 Goal 转成差速底盘的 cmd_vel(线速度+角速度)的跟踪控制节点。"""
    def __init__(self):
        super().__init__("goal_to_cmd_vel")
        self.current_yaw = 0.0
        # 两个就绪标志:state 和 goal 都收到过才开始算控制量
        self.state_initialized = False
        self.goal_initialized = False

        x = self.declare_parameter("x", 0.0).value
        y = self.declare_parameter("y", 0.0).value
        z = self.declare_parameter("z", 0.0).value
        yaw = self.declare_parameter("yaw", 0.0).value
        cmd_vel_topic = self.declare_parameter("cmd_vel_topic_name", "cmd_vel").value
        # kx/ky/kyaw 是控制增益(误差放大系数):越大纠偏越猛但越容易抖
        self.kx = float(self.declare_parameter("ground_robot_kx", 1.0).value)
        self.ky = float(self.declare_parameter("ground_robot_ky", 1.0).value)
        self.kyaw = float(self.declare_parameter("ground_robot_kyaw", 1.0).value)
        # eps:防止后面公式分母 sqrt(ey^2+eps^2) 为零的小正数
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
        # 100Hz 控制循环
        self.timer = self.create_timer(0.01, self._tick)

    # 收到当前状态:记下位置,并把四元数转成平面偏航角 yaw
    def _state_cb(self, msg) -> None:
        self.state_pos = (msg.pos.x, msg.pos.y, msg.pos.z)
        self.state_yaw = yaw_from_quat(msg.quat.x, msg.quat.y, msg.quat.z, msg.quat.w)
        self.state_initialized = True

    # 收到期望状态:记下期望位置/速度/偏航/偏航角速度
    def _goal_cb(self, msg) -> None:
        self.goal_p = (msg.p.x, msg.p.y, msg.p.z)
        self.goal_v = (msg.v.x, msg.v.y, msg.v.z)
        self.goal_yaw = float(msg.yaw)
        self.goal_dyaw = float(msg.dyaw)
        self.goal_initialized = True

    # 100Hz:根据当前与期望的差算出 cmd_vel(控制律见文件头公式)
    def _tick(self) -> None:
        if not self.state_initialized or not self.goal_initialized:
            return
        x_d, y_d, _ = self.goal_p
        xd_d, yd_d, _ = self.goal_v
        yaw_d, dyaw_d = self.goal_yaw, self.goal_dyaw
        # v_d:期望前进速率(水平速度的模长)
        v_d = math.hypot(xd_d, yd_d)
        sx, sy, _ = self.state_pos
        # 把位置误差旋转到「期望朝向」坐标系:ex=纵向(前后)误差,ey=横向(左右)误差
        ex = math.cos(yaw_d) * (sx - x_d) + math.sin(yaw_d) * (sy - y_d)
        ey = -math.sin(yaw_d) * (sx - x_d) + math.cos(yaw_d) * (sy - y_d)
        # eyaw:朝向误差
        eyaw = self.state_yaw - yaw_d
        # 线速度指令:前馈期望速度,减去纵向误差的纠偏项
        v_command = v_d * math.cos(eyaw) - self.kx * ex
        # 分母加 eps 防止除零(eps 在构造里声明)
        denom = math.sqrt(ey * ey + self.eps * self.eps)
        # 角速度指令:前馈期望角速度,再叠加横向误差和朝向误差的纠偏项
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
