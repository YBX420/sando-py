#!/usr/bin/env python3
"""
sando_py_bridge — drives the user's headless sando-py planner (LBFGSpp, NO Gurobi)
inside the Gazebo+D435i ROS2 sim, replacing sando_ws's stock sando_node 1:1.

Loop:  /NX01/occupancy_grid (D435->map points)  +  /NX01/term_goal  +  /NX01/state
         -> sando-py CAPI (update_state / update_occupancy / set_terminal_goal / replan / get_next_goal)
         -> /NX01/goal (dynus_interfaces/Goal)  ->  fake_sim moves drone & republishes /NX01/state.

Run (workspace + ROS sourced):
  python3 ~/code/sando_ws/sando_py_bridge.py --ros-args -r __ns:=/NX01
"""
import sys
import math
import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from geometry_msgs.msg import PoseStamped
from dynus_interfaces.msg import State, Goal

# the headless planner's ctypes wrapper (sando-py/python). This file is committed to
# sando-py/ros2_bridge/, so the wrapper is at ../python; fall back to a known abs path.
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, '..', 'python'), os.path.expanduser('~/code/sando-py/python')):
    if os.path.isdir(_p):
        sys.path.insert(0, _p)
        break
import sando_cpp_bridge as B   # noqa: E402


def quat_to_yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class SandoPyBridge(Node):
    def __init__(self):
        super().__init__('sando_py_bridge')

        # ---- params (sane defaults; override via --ros-args -p) ----
        self.declare_parameter('v_max', 3.0)
        self.declare_parameter('a_max', 10.0)
        self.declare_parameter('j_max', 30.0)
        self.declare_parameter('res', 0.3)
        self.declare_parameter('horizon', 15.0)
        self.declare_parameter('replan_dt', 0.1)
        self.declare_parameter('control_dt', 0.02)
        self.declare_parameter('default_goal_z', 2.0)
        self.declare_parameter('force_goal_z', True)
        g = lambda n: self.get_parameter(n).value

        par = B.Parameters()
        for k, v in [('v_max', float(g('v_max'))), ('a_max', float(g('a_max'))),
                     ('j_max', float(g('j_max'))), ('res', float(g('res'))),
                     ('horizon', float(g('horizon'))), ('replan_dt', float(g('replan_dt'))),
                     ('dc', float(g('control_dt')))]:
            try:
                setattr(par, k, v)
            except Exception:
                pass
        # go straight to TRAVELING (the YAWING state needs a yaw-align loop we don't drive yet;
        # yaw-to-look-ahead is part of the M3 RGBD-safety layer).
        try:
            par.skip_initial_yawing = True
        except Exception:
            pass
        self.sando = B.SANDO(par)

        self.default_goal_z = float(g('default_goal_z'))
        self.force_goal_z = bool(g('force_goal_z'))

        # ---- runtime state ----
        self.cur_pos = None
        self.cur_vel = np.zeros(3)
        self.goal_pos = None
        self.cloud = np.zeros((0, 3))
        self.map_inited = False
        self.last_goal = None
        self.last_yaw = 0.0
        self._t0 = self.get_clock().now()
        self._last_rt = 0.0
        self._nrep = 0

        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST, depth=5,
                                durability=DurabilityPolicy.VOLATILE)
        rel_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             history=HistoryPolicy.KEEP_LAST, depth=10,
                             durability=DurabilityPolicy.VOLATILE)

        self.create_subscription(PointCloud2, 'occupancy_grid', self.cb_cloud, sensor_qos)
        self.create_subscription(PoseStamped, 'term_goal', self.cb_goal, rel_qos)
        self.create_subscription(State, 'state', self.cb_state, rel_qos)
        self.pub_goal = self.create_publisher(Goal, 'goal', rel_qos)

        self.create_timer(float(g('replan_dt')), self.replan_tick)
        self.create_timer(float(g('control_dt')), self.control_tick)
        self.get_logger().info("sando_py_bridge up: occupancy_grid+term_goal+state -> [sando-py] -> goal")

    # ---- inputs ----
    def cb_cloud(self, msg):
        try:
            p = point_cloud2.read_points_numpy(msg, field_names=['x', 'y', 'z'], skip_nans=True)
            self.cloud = np.asarray(p, dtype=np.float64).reshape(-1, 3)
        except Exception as e:
            self.get_logger().warn(f"cloud read failed: {e}", throttle_duration_sec=2.0)

    def cb_goal(self, msg):
        z = self.default_goal_z if self.force_goal_z else msg.pose.position.z
        self.goal_pos = np.array([msg.pose.position.x, msg.pose.position.y, z])
        G = B.RobotState()
        G.pos = self.goal_pos
        self.sando.set_terminal_goal(G)
        self.get_logger().info(f"terminal goal set: {np.round(self.goal_pos, 2)}")

    def cb_state(self, msg):
        self.cur_pos = np.array([msg.pos.x, msg.pos.y, msg.pos.z])
        self.cur_vel = np.array([msg.vel.x, msg.vel.y, msg.vel.z])

    def _now_s(self):
        return (self.get_clock().now() - self._t0).nanoseconds * 1e-9

    # ---- planning (replan_dt) ----
    def replan_tick(self):
        if self.cur_pos is None or self.goal_pos is None:
            return
        st = B.RobotState()
        st.pos = self.cur_pos
        st.vel = self.cur_vel
        st.accel = np.zeros(3)
        self.sando.update_state(st)
        self.sando.update_occupancy_map_ptr(self.cloud if self.cloud.size else np.zeros((0, 3)))
        self.map_inited = True
        t = self._now_s()
        t0 = time.perf_counter()
        try:
            ok, att = self.sando.replan(self._last_rt, t)
        except Exception as e:
            self.get_logger().warn(f"replan exception: {e}", throttle_duration_sec=2.0)
            return
        self._last_rt = time.perf_counter() - t0
        self._nrep += 1
        if self._nrep % 10 == 0:
            self.get_logger().info(
                f"replan ok={ok} pts={self.cloud.shape[0]} pos={np.round(self.cur_pos,2)} "
                f"goal={np.round(self.goal_pos,2)} status={self.sando.get_drone_status()} "
                f"rt={self._last_rt*1e3:.1f}ms")

    # ---- control streaming (control_dt) ----
    def control_tick(self):
        if not self.map_inited:
            return
        try:
            okg, ng = self.sando.get_next_goal()
        except Exception:
            okg = False
        if okg:
            pos = np.asarray(ng.pos, dtype=float)
            vel = np.asarray(ng.vel, dtype=float)
            acc = np.asarray(ng.accel, dtype=float)
            sp = float(np.linalg.norm(vel[:2]))
            yaw = math.atan2(vel[1], vel[0]) if sp > 0.15 else self.last_yaw
            self.last_yaw = yaw
            g = Goal()
            g.header.stamp = self.get_clock().now().to_msg()
            g.header.frame_id = 'map'
            g.p.x, g.p.y, g.p.z = float(pos[0]), float(pos[1]), float(pos[2])
            g.v.x, g.v.y, g.v.z = float(vel[0]), float(vel[1]), float(vel[2])
            g.a.x, g.a.y, g.a.z = float(acc[0]), float(acc[1]), float(acc[2])
            g.yaw = float(yaw)
            self.last_goal = g
            self.pub_goal.publish(g)
        elif self.last_goal is not None:
            self.last_goal.header.stamp = self.get_clock().now().to_msg()
            self.pub_goal.publish(self.last_goal)


def main():
    rclpy.init()
    node = SandoPyBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
