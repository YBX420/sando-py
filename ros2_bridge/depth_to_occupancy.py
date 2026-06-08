#!/usr/bin/env python3
"""
depth_to_occupancy — fills the missing acl-mapping seam for SANDO's gazebo path.

Subscribes the D435 depth point cloud (optical frame), transforms it into the
planner 'map' frame via TF, voxel-downsamples the occupied returns, and publishes
the synchronized (occupancy_grid, unknown_grid) PointCloud2 pair that
SANDO_NODE::mapCallback -> SANDO::updateMapPtr consumes (sim_env == "gazebo").

Milestone 1: unknown_grid is published EMPTY (same stamp) so the ApproximateTime
sync fires; occupancy comes straight from what the D435 actually sees. Milestone 2
will replace the empty unknown with a real frustum / free-carve unknown shell so the
planner gets genuine partial-observability (the RGBD-safety win).
"""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
import tf2_ros


def quat_to_mat(tf):
    x, y, z, w = tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        R = np.eye(3)
    else:
        s = 2.0 / n
        R = np.array([
            [1 - s * (y * y + z * z), s * (x * y - z * w),     s * (x * z + y * w)],
            [s * (x * y + z * w),     1 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w),     s * (y * z + x * w),     1 - s * (x * x + y * y)],
        ])
    T = np.eye(4)
    T[:3, :3] = R
    T[0, 3] = tf.translation.x
    T[1, 3] = tf.translation.y
    T[2, 3] = tf.translation.z
    return T


class DepthToOccupancy(Node):
    def __init__(self):
        super().__init__('depth_to_occupancy')
        self.declare_parameter('input_topic', '/NX01/d435/depth/color/points')
        self.declare_parameter('occ_topic', '/NX01/occupancy_grid')
        self.declare_parameter('unk_topic', '/NX01/unknown_grid')
        self.declare_parameter('target_frame', 'map')
        self.declare_parameter('voxel', 0.3)
        self.declare_parameter('range_min', 0.3)
        self.declare_parameter('range_max', 10.0)
        self.declare_parameter('min_period', 0.1)

        g = lambda n: self.get_parameter(n).value
        self.target_frame = g('target_frame')
        self.voxel = float(g('voxel'))
        self.range_min = float(g('range_min'))
        self.range_max = float(g('range_max'))
        self.min_period = float(g('min_period'))

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=5,
                         durability=DurabilityPolicy.VOLATILE)
        self.pub_occ = self.create_publisher(PointCloud2, g('occ_topic'), qos)
        self.pub_unk = self.create_publisher(PointCloud2, g('unk_topic'), qos)
        self.sub = self.create_subscription(PointCloud2, g('input_topic'), self.cb, qos)
        self._last_ns = None
        self._count = 0
        self.get_logger().info(
            f"depth_to_occupancy: {g('input_topic')} -> {g('occ_topic')} (+{g('unk_topic')}), "
            f"frame={self.target_frame}, voxel={self.voxel}, range=[{self.range_min},{self.range_max}]")

    def cb(self, msg):
        now_ns = self.get_clock().now().nanoseconds
        if self._last_ns is not None and (now_ns - self._last_ns) < self.min_period * 1e9:
            return
        self._last_ns = now_ns

        try:
            pts = point_cloud2.read_points_numpy(msg, field_names=['x', 'y', 'z'], skip_nans=True)
        except Exception as e:
            self.get_logger().warn(f"read_points failed: {e}")
            return
        if pts is None or pts.size == 0:
            self._publish(np.empty((0, 3)), msg.header.stamp)
            return
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)

        # range filter in camera/optical frame (Euclidean norm)
        d = np.linalg.norm(pts, axis=1)
        pts = pts[(d >= self.range_min) & (d <= self.range_max)]
        if pts.shape[0] == 0:
            self._publish(np.empty((0, 3)), msg.header.stamp)
            return

        src = msg.header.frame_id
        try:
            tr = self.tf_buffer.lookup_transform(self.target_frame, src, rclpy.time.Time())
        except Exception as e:
            self.get_logger().warn(f"TF {self.target_frame}<-{src} unavailable: {e}",
                                   throttle_duration_sec=2.0)
            return

        T = quat_to_mat(tr.transform)
        world = (T @ np.hstack([pts, np.ones((pts.shape[0], 1))]).T).T[:, :3]

        # voxel downsample -> voxel centers
        keys = np.floor(world / self.voxel).astype(np.int64)
        _, idx = np.unique(keys, axis=0, return_index=True)
        occ = (keys[idx].astype(np.float64) + 0.5) * self.voxel
        self._publish(occ, msg.header.stamp)

        self._count += 1
        if self._count % 30 == 0:
            self.get_logger().info(f"occupancy_grid: {occ.shape[0]} voxels (frame={self.target_frame})")

    def _publish(self, occ_pts, stamp):
        h = Header()
        h.stamp = stamp
        h.frame_id = self.target_frame
        occ_msg = point_cloud2.create_cloud_xyz32(h, occ_pts.tolist())
        unk_msg = point_cloud2.create_cloud_xyz32(h, [])  # milestone 1: empty unknown
        self.pub_occ.publish(occ_msg)
        self.pub_unk.publish(unk_msg)


def main():
    rclpy.init()
    node = DepthToOccupancy()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
