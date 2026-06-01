"""Odometry (local) -> global-frame State + PoseStamped — port of odom_to_global_state.cpp.

Looks up TF: world -> <veh_name>/init_pose, caches it, then transforms every
incoming Odometry message into the global "world" frame.

中文说明:
桥接 + 坐标变换节点。真机的里程计(odom)通常是相对「飞机自己出发点」的局部坐标,
而规划器要的是统一的全局 world 坐标。这个节点负责把局部 odom 搬到 world 坐标系下。

做法:先去 TF 树里查一次「world -> <飞机名>/init_pose」这个变换(即飞机出发点在
world 里的位姿),查到后缓存下来(平移 t_init + 旋转 R_init),之后每来一条 odom 就用
它做刚体变换:global = R_init * local + t_init,速度只转方向,姿态用四元数相乘叠加。

输入:话题 `odom`(局部里程计)。
输出:话题 `state`(world 坐标系下的规划器 State) + 话题 `global_pose`(同位姿的 PoseStamped)。
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


# 延迟导入 State 消息类型(没装就返回 None)
def _state_msg():
    try:
        from dynus_interfaces.msg import State
        return State
    except ImportError:
        return None


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton quaternion product (xyzw).

    中文:四元数乘法(xyzw 顺序),用来把两个旋转叠加(这里叠加出全局姿态)。
    """
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


# 四元数(xyzw)转 3x3 旋转矩阵,后面用它把局部坐标旋到全局
def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = q
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


class OdomToGlobalState(Node):
    """查一次 world->init_pose 的 TF 并缓存,之后把每条 odom 变换到 world 坐标系发布。"""
    def __init__(self):
        super().__init__("odom_to_global_state")

        # 从命名空间取飞机名(多机时区分),取不到就用默认 PX04
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
        # 定时轮询 TF,直到查到 init_pose 变换为止(查到后自己取消)
        self.tf_timer = self.create_timer(0.1, self._poll_tf)

        # tf_acquired:是否已拿到并缓存出发点变换;没拿到前 odom 回调直接跳过
        self.tf_acquired = False
        self.t_init = np.zeros(3)          # 出发点在 world 里的平移
        self.q_init = np.array([0.0, 0.0, 0.0, 1.0])  # xyzw  出发点姿态(四元数)
        self.R_init = np.eye(3)            # 出发点姿态对应的旋转矩阵
        self.get_logger().info(f"Waiting for TF: world -> {self.veh_name}/init_pose")

    # 轮询并缓存「world -> 飞机出发点」的变换;成功后停掉这个定时器
    def _poll_tf(self) -> None:
        target = "world"
        source = f"{self.veh_name}/init_pose"
        # 还查不到就返回,下次定时器再试
        if not self.tf_buffer.can_transform(target, source, Time()):
            return
        try:
            tf = self.tf_buffer.lookup_transform(target, source, Time())
        except tf2_ros.TransformException as ex:
            # 查 TF 偶尔会抛异常;限流打印警告,别刷屏
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
        # 变换只查一次(出发点固定不变),拿到就停掉轮询
        self.tf_timer.cancel()
        yaw = math.atan2(self.R_init[1, 0], self.R_init[0, 0])
        self.get_logger().info(
            f"TF acquired: t=[{self.t_init[0]:.3f}, {self.t_init[1]:.3f}, "
            f"{self.t_init[2]:.3f}], yaw={yaw:.3f} rad"
        )

    # 每条 odom:用缓存的出发点变换,把局部位姿/速度/姿态搬到 world 坐标系并发布
    def _odom_cb(self, msg: Odometry) -> None:
        # 变换还没拿到就不处理(否则坐标是错的)
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
        # 位置:旋转后再加平移(完整刚体变换);速度:只旋转方向,不加平移
        global_pos = self.R_init @ local_pos + self.t_init
        global_vel = self.R_init @ local_vel
        # 姿态:出发点姿态 叠加 局部姿态
        global_quat = _quat_mul(self.q_init, local_quat)
        # 归一化,消掉数值误差让它保持单位四元数
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
