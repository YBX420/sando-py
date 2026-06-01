"""ObstacleTracker — Python port of src/sando/obstacle_tracker_node.cpp.

Subscribes to a sensor point cloud, runs Euclidean clustering (replacing PCL's
EuclideanClusterExtraction with scipy's DBSCAN — same parameters: cluster_tolerance
maps to DBSCAN's eps, min_cluster_size maps to min_samples). Each cluster is
associated with an adaptive Kalman filter (constant-acceleration model in 9D state)
and a future trajectory is fit with a cubic + quintic polynomial. The resulting
DynTraj message is published on `predicted_trajs` for the planner to consume.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.duration import Duration

from geometry_msgs.msg import Point, Vector3
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from ..types import PieceWisePol
from ..utils import pwp_to_msg


@dataclass
class EKFState:
    """9-D state [x, y, z, vx, vy, vz, ax, ay, az] with constant-acceleration model."""
    x: np.ndarray = field(default_factory=lambda: np.zeros(9))
    P: np.ndarray = field(default_factory=lambda: np.eye(9))
    Q: np.ndarray = field(default_factory=lambda: np.eye(9) * 0.01)
    R: np.ndarray = field(default_factory=lambda: np.eye(3) * 0.01)
    time_updated: float = 0.0
    bbox: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5, 0.5]))
    id: int = 0
    color: Tuple[float, float, float, float] = (0.5, 0.5, 0.5, 0.4)


def _make_color() -> Tuple[float, float, float, float]:
    rng = np.random.default_rng()
    return (float(rng.random()), float(rng.random()), float(rng.random()), 0.4)


def _ekf_predict(s: EKFState, dt: float) -> None:
    F = np.eye(9)
    F[0, 3] = dt; F[1, 4] = dt; F[2, 5] = dt
    F[0, 6] = 0.5 * dt * dt; F[1, 7] = 0.5 * dt * dt; F[2, 8] = 0.5 * dt * dt
    F[3, 6] = dt; F[4, 7] = dt; F[5, 8] = dt
    s.x = F @ s.x
    s.P = F @ s.P @ F.T + s.Q


def _aekf_update(s: EKFState, z: np.ndarray, alpha: float, time_updated: float,
                 bbox: np.ndarray, use_adaptive: bool) -> None:
    H = np.zeros((3, 9))
    H[0, 0] = 1.0; H[1, 1] = 1.0; H[2, 2] = 1.0
    d = z - H @ s.x
    S = H @ s.P @ H.T + s.R
    K = s.P @ H.T @ np.linalg.inv(S)
    s.x = s.x + K @ d
    epsilon = z - H @ s.x
    if use_adaptive:
        s.R = alpha * s.R + (1 - alpha) * (
            np.outer(epsilon, epsilon) + H @ s.P @ H.T
        )
        s.Q = alpha * s.Q + (1 - alpha) * (K @ np.outer(d, d) @ K.T)
    else:
        s.R = np.eye(3) * 0.01
        s.Q = np.eye(9) * 0.01
    s.P = (np.eye(9) - K @ H) @ s.P
    s.time_updated = time_updated
    s.bbox = 0.5 * s.bbox + 0.5 * bbox


def _associate_cluster(centroid: np.ndarray, ekf_states: List[EKFState],
                       tol: float) -> int:
    min_d = tol
    idx = -1
    for i, s in enumerate(ekf_states):
        d = float(np.linalg.norm(centroid - s.x[:3]))
        if d < min_d:
            min_d = d
            idx = i
    return idx


def _polyfit_descending(t: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    """numpy.polyfit returns coefficients in descending order (highest first).
    The C++ code uses ax^3 + bx^2 + cx + d order so this matches directly."""
    return np.polyfit(t, y, degree)


def _calculate_variance(t: np.ndarray, y: np.ndarray, beta: np.ndarray,
                         degree: int) -> float:
    fitted = np.polyval(beta, t)
    residuals = y - fitted
    n = len(t)
    if n - degree - 1 <= 0:
        return 0.0
    return float(np.sum(residuals * residuals) / (n - degree - 1))


def _euclidean_cluster(points: np.ndarray, eps: float, min_samples: int,
                       max_samples: int) -> List[np.ndarray]:
    """DBSCAN clustering — equivalent to PCL's EuclideanClusterExtraction.
    Returns a list of (N_i, 3) arrays, one per cluster."""
    if points.shape[0] == 0:
        return []
    try:
        from sklearn.cluster import DBSCAN
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)
    except ImportError:
        # Fallback: union-find with KDTree (no sklearn dep)
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        parent = list(range(points.shape[0]))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i in range(points.shape[0]):
            for j in tree.query_ball_point(points[i], r=eps):
                if j != i:
                    union(i, j)
        labels = np.array([find(i) for i in range(points.shape[0])])

    clusters: List[np.ndarray] = []
    unique = np.unique(labels)
    for u in unique:
        if u == -1:
            continue
        sel = labels == u
        n = int(np.sum(sel))
        if n < min_samples or n > max_samples:
            continue
        clusters.append(points[sel])
    return clusters


def _pointcloud2_to_xyz(msg: PointCloud2) -> np.ndarray:
    gen = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
    if gen is None:
        return np.zeros((0, 3))
    return np.array([(float(p[0]), float(p[1]), float(p[2])) for p in gen],
                    dtype=np.float64)


def _voxel_downsample(points: np.ndarray, leaf: float) -> np.ndarray:
    if points.shape[0] == 0 or leaf <= 0:
        return points
    keys = np.floor(points / leaf).astype(np.int64)
    # Use unique with axis=0 — keep first occurrence
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(idx)]


class ObstacleTrackerNode(Node):
    def __init__(self):
        super().__init__("obstacle_tracker_node")
        self.get_logger().info("Obstacle Tracker Node Started")

        # Parameters
        self.visual_level = self.declare_parameter("visual_level", 1).value
        self.use_adaptive_kf = self.declare_parameter("use_adaptive_kf", True).value
        self.adaptive_kf_alpha = float(self.declare_parameter("adaptive_kf_alpha", 0.98).value)
        self.adaptive_kf_dt = float(self.declare_parameter("adaptive_kf_dt", 0.1).value)
        self.cluster_tolerance = float(self.declare_parameter("cluster_tolerance", 2.0).value)
        self.min_cluster_size = int(self.declare_parameter("min_cluster_size", 10).value)
        self.max_cluster_size = int(self.declare_parameter("max_cluster_size", 2000).value)
        self.prediction_horizon = float(self.declare_parameter("prediction_horizon", 2.0).value)
        self.prediction_dt = float(self.declare_parameter("prediction_dt", 0.1).value)
        self.time_to_delete_old_obstacles = float(self.declare_parameter("time_to_delete_old_obstacles", 10.0).value)
        self.cluster_bbox_cutoff_size = float(self.declare_parameter("cluster_bbox_cutoff_size", 5.0).value)
        self.velocity_threshold = float(self.declare_parameter("velocity_threshold", 0.1).value)
        self.acceleration_threshold = float(self.declare_parameter("acceleration_threshold", 0.1).value)
        self.use_hardware = bool(self.declare_parameter("use_hardware", False).value)
        self.frame_id = "map"
        self.degree_pwp = 3
        self.degree_poly = 5

        # State
        self.ekf_states: List[EKFState] = []
        self.ekf_state_id = 0

        # Subscriber / publishers
        self.sub_pointcloud = self.create_subscription(
            PointCloud2, "point_cloud", self.pointcloud_callback, 10
        )
        self.pub_markers = self.create_publisher(MarkerArray, "tracked_obstacles", 10)
        self.pub_bboxes = self.create_publisher(MarkerArray, "cluster_bounding_boxes", 10)
        self.pub_unc_sphere = self.create_publisher(MarkerArray, "uncertainty_spheres", 10)

        DynTrajMsg = self._get_dyntraj_msg()
        if DynTrajMsg is not None:
            self.pub_predicted_traj = self.create_publisher(DynTrajMsg, "predicted_trajs", 10)
        else:
            self.pub_predicted_traj = None
            self.get_logger().warn("dynus_interfaces not available — predicted_trajs disabled")

        self.get_logger().info("Obstacle Tracker Node Initialized")

    @staticmethod
    def _get_dyntraj_msg():
        try:
            from dynus_interfaces.msg import DynTraj
            return DynTraj
        except ImportError:
            return None

    def pointcloud_callback(self, msg: PointCloud2) -> None:
        points = _pointcloud2_to_xyz(msg)
        if points.shape[0] == 0:
            self.get_logger().warn("Received empty point cloud!")
            return

        # Voxel downsample (matches PCL VoxelGrid leaf 0.2)
        points = _voxel_downsample(points, leaf=0.2)

        # TF transform — skip; assume cloud already in map frame for the Python port.
        # If running with a non-map source, configure the upstream node to publish in `map`.

        # Z passthrough filter (0.5 → 6.0)
        mask = (points[:, 2] >= 0.5) & (points[:, 2] <= 6.0)
        points = points[mask]

        # Cluster
        clusters_pts = _euclidean_cluster(points, eps=self.cluster_tolerance,
                                          min_samples=self.min_cluster_size,
                                          max_samples=self.max_cluster_size)
        centroids: List[np.ndarray] = []
        bboxes: List[np.ndarray] = []
        for c in clusters_pts:
            pmin = c.min(axis=0); pmax = c.max(axis=0)
            centroids.append((pmin + pmax) * 0.5)
            bboxes.append(pmax - pmin)

        now = self._ros_now()
        self._delete_old_ekf_states(now)

        cluster_records: List[Tuple[EKFState, np.ndarray]] = []
        for centroid, bbox in zip(centroids, bboxes):
            if float(np.linalg.norm(bbox)) > self.cluster_bbox_cutoff_size:
                continue
            idx = _associate_cluster(centroid, self.ekf_states, self.cluster_tolerance)
            if idx >= 0:
                _ekf_predict(self.ekf_states[idx], self.adaptive_kf_dt)
                _aekf_update(self.ekf_states[idx], centroid, self.adaptive_kf_alpha,
                              now, bbox, self.use_adaptive_kf)
                cluster_records.append((self.ekf_states[idx], centroid))
            else:
                Q_avg, R_avg = self._average_q_r()
                s = EKFState(
                    x=np.zeros(9), P=np.eye(9),
                    Q=Q_avg, R=R_avg,
                    time_updated=now, bbox=bbox, id=self.ekf_state_id,
                    color=_make_color(),
                )
                s.x[:3] = centroid
                self.ekf_state_id += 1
                self.ekf_states.append(s)
                cluster_records.append((s, centroid))

        if self.visual_level >= 0:
            self._publish_boxes(cluster_records)
        self._publish_predictions(cluster_records)

    def _delete_old_ekf_states(self, now: float) -> None:
        self.ekf_states = [s for s in self.ekf_states
                            if now - s.time_updated <= self.time_to_delete_old_obstacles]

    def _average_q_r(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.ekf_states:
            return np.eye(9) * 0.01, np.eye(3) * 0.01
        Q = np.mean(np.stack([s.Q for s in self.ekf_states]), axis=0)
        R = np.mean(np.stack([s.R for s in self.ekf_states]), axis=0)
        return Q, R

    def _publish_boxes(self, records: List[Tuple[EKFState, np.ndarray]]) -> None:
        markers = MarkerArray()
        unc = MarkerArray()
        lifetime = Duration(seconds=0.5).to_msg()
        # DELETEALL
        m = Marker(); m.header.frame_id = self.frame_id; m.header.stamp = self.get_clock().now().to_msg()
        m.action = Marker.DELETEALL; m.ns = "cluster_bounding_box"
        markers.markers.append(m)
        m2 = Marker(); m2.header.frame_id = self.frame_id; m2.header.stamp = self.get_clock().now().to_msg()
        m2.action = Marker.DELETEALL; m2.ns = "uncertainty_sphere"
        unc.markers.append(m2)

        max_scale = 2.5
        for i, (state, centroid) in enumerate(records):
            color = ColorRGBA(r=state.color[0], g=state.color[1], b=state.color[2], a=1.0)
            mk = Marker()
            mk.header.frame_id = self.frame_id; mk.header.stamp = self.get_clock().now().to_msg()
            mk.ns = "cluster_bounding_box"; mk.id = i
            mk.type = Marker.CUBE; mk.action = Marker.ADD
            mk.lifetime = lifetime
            mk.pose.position.x = float(centroid[0])
            mk.pose.position.y = float(centroid[1])
            mk.pose.position.z = float(centroid[2])
            mk.pose.orientation.w = 1.0
            mk.scale.x = float(state.bbox[0])
            mk.scale.y = float(state.bbox[1])
            mk.scale.z = float(state.bbox[2])
            mk.color = color
            markers.markers.append(mk)

            sp = Marker()
            sp.header.frame_id = self.frame_id; sp.header.stamp = self.get_clock().now().to_msg()
            sp.ns = "uncertainty_sphere"; sp.id = i
            sp.type = Marker.SPHERE; sp.action = Marker.ADD
            sp.lifetime = lifetime
            sp.pose.position.x = float(centroid[0])
            sp.pose.position.y = float(centroid[1])
            sp.pose.position.z = float(centroid[2])
            sp.pose.orientation.w = 1.0
            sp.scale.x = min(state.P[0, 0] * 2e3, max_scale)
            sp.scale.y = min(state.P[1, 1] * 2e3, max_scale)
            sp.scale.z = min(state.P[2, 2] * 2e3, max_scale)
            sp.color = ColorRGBA(r=state.color[0], g=state.color[1], b=state.color[2], a=0.6)
            unc.markers.append(sp)

        self.pub_bboxes.publish(markers)
        self.pub_unc_sphere.publish(unc)

    def _publish_predictions(self, records: List[Tuple[EKFState, np.ndarray]]) -> None:
        DynTrajMsg = self._get_dyntraj_msg()
        if DynTrajMsg is None or self.pub_predicted_traj is None:
            return
        ma = MarkerArray()
        num_steps = int(self.prediction_horizon / self.prediction_dt)
        marker_id = 0
        clear = Marker()
        clear.action = Marker.DELETEALL
        clear.header.frame_id = self.frame_id
        clear.header.stamp = self.get_clock().now().to_msg()
        ma.markers.append(clear)

        for state, centroid in records:
            pos = state.x[:3].copy()
            vel = state.x[3:6].copy()
            acc = np.clip(state.x[6:9].copy(), -2.0, 2.0)

            t_vals: List[float] = [0.0]
            x_vals: List[float] = [pos[0]]
            y_vals: List[float] = [pos[1]]
            z_vals: List[float] = [pos[2]]

            for step in range(num_steps):
                vel = np.clip(vel, -0.5, 0.5)
                acc = np.clip(acc, -0.8, 0.8)
                t = step * self.prediction_dt
                future_pos = pos + vel * t + 0.5 * acc * t * t
                t_vals.append(t)
                x_vals.append(future_pos[0])
                y_vals.append(future_pos[1])
                z_vals.append(future_pos[2])

                mk = Marker()
                mk.header.frame_id = self.frame_id
                mk.header.stamp = self.get_clock().now().to_msg()
                mk.lifetime = Duration(seconds=0.5).to_msg()
                mk.id = marker_id; marker_id += 1
                mk.type = Marker.ARROW; mk.action = Marker.ADD
                pt0 = Point(); pt0.x = float(pos[0]); pt0.y = float(pos[1]); pt0.z = float(pos[2])
                pt1 = Point(); pt1.x = float(future_pos[0]); pt1.y = float(future_pos[1]); pt1.z = float(future_pos[2])
                mk.points = [pt0, pt1]
                mk.scale.x = 0.1; mk.scale.y = 0.2
                mk.color = ColorRGBA(r=state.color[0], g=state.color[1],
                                      b=state.color[2], a=state.color[3])
                ma.markers.append(mk)

                pos = future_pos
                vel = vel + acc * self.prediction_dt

            cutoff = 0.1
            if (abs(x_vals[0] - x_vals[-1]) < cutoff
                    and abs(y_vals[0] - y_vals[-1]) < cutoff
                    and abs(z_vals[0] - z_vals[-1]) < cutoff):
                continue

            t_arr = np.array(t_vals)
            x_arr = np.array(x_vals)
            y_arr = np.array(y_vals)
            z_arr = np.array(z_vals)
            beta_x = _polyfit_descending(t_arr, x_arr, self.degree_pwp)
            beta_y = _polyfit_descending(t_arr, y_arr, self.degree_pwp)
            beta_z = _polyfit_descending(t_arr, z_arr, self.degree_pwp)
            var_x = _calculate_variance(t_arr, x_arr, beta_x, self.degree_pwp)
            var_y = _calculate_variance(t_arr, y_arr, beta_y, self.degree_pwp)
            var_z = _calculate_variance(t_arr, z_arr, beta_z, self.degree_pwp)

            pwp = PieceWisePol()
            t0 = self._ros_now()
            pwp.times = [t0, t0 + self.prediction_horizon]
            pwp.coeff_x = [np.array([beta_x[0], beta_x[1], beta_x[2], beta_x[3]])]
            pwp.coeff_y = [np.array([beta_y[0], beta_y[1], beta_y[2], beta_y[3]])]
            pwp.coeff_z = [np.array([beta_z[0], beta_z[1], beta_z[2], beta_z[3]])]

            beta_x_quintic = _polyfit_descending(t_arr, x_arr, self.degree_poly)
            beta_y_quintic = _polyfit_descending(t_arr, y_arr, self.degree_poly)
            beta_z_quintic = _polyfit_descending(t_arr, z_arr, self.degree_poly)

            msg = DynTrajMsg()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.frame_id
            msg.id = state.id
            msg.bbox = [float(state.bbox[0]), float(state.bbox[1]), float(state.bbox[2])]
            pwp_msg = pwp_to_msg(pwp)
            if pwp_msg is not None:
                msg.pwp = pwp_msg
            msg.ekf_cov_p = [float(state.P[0, 0]), float(state.P[1, 1]), float(state.P[2, 2])]
            msg.ekf_cov_q = [float(state.Q[0, 0]), float(state.Q[1, 1]), float(state.Q[2, 2])]
            if hasattr(msg, "ekf_cov_r"):
                msg.ekf_cov_r = [float(state.R[0, 0]), float(state.R[1, 1]), float(state.R[2, 2])]
            msg.poly_cov = [float(var_x), float(var_y), float(var_z)]
            msg.poly_coeffs_x = [float(c) for c in beta_x_quintic]
            msg.poly_coeffs_y = [float(c) for c in beta_y_quintic]
            msg.poly_coeffs_z = [float(c) for c in beta_z_quintic]
            msg.poly_start_time = float(t0)
            msg.poly_end_time = float(t0 + self.prediction_horizon)
            msg.pos.x = float(state.x[0]); msg.pos.y = float(state.x[1]); msg.pos.z = float(state.x[2])
            msg.is_agent = False
            self.pub_predicted_traj.publish(msg)

        if self.visual_level >= 0:
            self.pub_markers.publish(ma)

    def _ros_now(self) -> float:
        return float(self.get_clock().now().nanoseconds) * 1e-9


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleTrackerNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
