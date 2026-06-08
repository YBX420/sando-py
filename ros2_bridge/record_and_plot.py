#!/usr/bin/env python3
"""Record the closed-loop flight (odom + a D435 occupancy snapshot) and render a PNG."""
import sys, time
import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

BOXES = [(6.0, -0.6), (10.0, 0.8)]   # (x,y) of the 1x1 obstacle boxes
GOAL = (16.0, 0.0)

rclpy.init()
node = rclpy.create_node('recorder')
traj = []
t0 = [None]
occ = [None]


def cb_odom(m):
    tt = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
    if t0[0] is None:
        t0[0] = tt
    p = m.pose.pose.position
    traj.append((tt - t0[0], p.x, p.y, p.z))


def cb_occ(m):
    pts = np.asarray(point_cloud2.read_points_numpy(m, field_names=['x', 'y', 'z'], skip_nans=True)).reshape(-1, 3)
    if occ[0] is None or len(pts) > len(occ[0]):   # keep densest view (boxes most visible)
        occ[0] = pts


rel = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)
se = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=5,
                durability=DurabilityPolicy.VOLATILE)
node.create_subscription(Odometry, '/NX01/visual_slam/odom', cb_odom, rel)
node.create_subscription(PointCloud2, '/NX01/occupancy_grid', cb_occ, se)

T = float(sys.argv[1]) if len(sys.argv) > 1 else 14.0
t_end = time.time() + T
while time.time() < t_end:
    rclpy.spin_once(node, timeout_sec=0.02)
node.destroy_node()
rclpy.shutdown()

tr = np.array(traj) if traj else np.zeros((0, 4))
o = occ[0] if occ[0] is not None else np.zeros((0, 3))
print(f"recorded {len(tr)} odom samples, occupancy snapshot {len(o)} pts", flush=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# --- top-down x-y: occupancy + boxes + flight path (shows avoidance) ---
if len(o):
    ax1.scatter(o[:, 0], o[:, 1], s=3, c='0.55', label='D435i occupancy')
for (bx, by) in BOXES:
    ax1.add_patch(Rectangle((bx - 0.5, by - 0.5), 1.0, 1.0, color='firebrick', alpha=0.45))
ax1.add_patch(Rectangle((-100, -100), 0, 0, color='firebrick', alpha=0.45, label='obstacle box'))
if len(tr):
    ax1.plot(tr[:, 1], tr[:, 2], 'b-', lw=2.2, label='drone path')
    ax1.plot(tr[0, 1], tr[0, 2], 'go', ms=11, label='start')
ax1.plot(GOAL[0], GOAL[1], 'r*', ms=18, label='goal')
ax1.set_xlabel('x [m]'); ax1.set_ylabel('y [m]'); ax1.set_xlim(-1, 18); ax1.set_ylim(-3, 3)
ax1.set_aspect('equal'); ax1.grid(True, alpha=0.3); ax1.legend(loc='upper left', fontsize=8)
ax1.set_title('Top-down: D435i occupancy + flight (obstacle avoidance)')

# --- forward progress x(t) ---
if len(tr):
    ax2.plot(tr[:, 0], tr[:, 1], 'b-', lw=2.2, label='x(t)')
ax2.axhline(GOAL[0], color='r', ls='--', label='goal x=16')
ax2.set_xlabel('t [s]'); ax2.set_ylabel('x [m]'); ax2.grid(True, alpha=0.3); ax2.legend(fontsize=8)
ax2.set_title('Forward progress x(t)')

plt.suptitle('SANDO-py closed loop in Gazebo:  D435i depth -> occupancy -> planner -> motion', fontsize=12)
plt.tight_layout()
plt.savefig('/tmp/sando_flight.png', dpi=115)
print("saved /tmp/sando_flight.png", flush=True)
