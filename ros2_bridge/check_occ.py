#!/usr/bin/env python3
"""One-shot verifier: subscribe /NX01/occupancy_grid, print point count + centroid + bbox.
Expected (box at map 2.5,0,1): centroid near x~2 y~0 z~1."""
import time
import numpy as np
import rclpy
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

rclpy.init()
node = rclpy.create_node('check_occ')
got = {}


def cb(m):
    p = point_cloud2.read_points_numpy(m, field_names=['x', 'y', 'z'], skip_nans=True)
    p = np.asarray(p, dtype=np.float64).reshape(-1, 3)
    got['n'] = int(p.shape[0])
    if p.shape[0]:
        got['centroid'] = np.round(p.mean(axis=0), 3).tolist()
        got['min'] = np.round(p.min(axis=0), 3).tolist()
        got['max'] = np.round(p.max(axis=0), 3).tolist()


qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                 history=HistoryPolicy.KEEP_LAST, depth=5,
                 durability=DurabilityPolicy.VOLATILE)
node.create_subscription(PointCloud2, '/NX01/occupancy_grid', cb, qos)

t0 = time.time()
while time.time() - t0 < 10 and 'centroid' not in got:
    rclpy.spin_once(node, timeout_sec=0.2)

if 'centroid' in got:
    print(f"OCC OK: points={got['n']} centroid={got['centroid']} min={got['min']} max={got['max']}")
else:
    print(f"OCC EMPTY/NONE: got={got}")
rclpy.shutdown()
