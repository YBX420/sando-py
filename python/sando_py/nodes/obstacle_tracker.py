"""ObstacleTracker — Python port of src/sando/obstacle_tracker_node.cpp.

中文说明
========
这个节点是「动态障碍感知器」。它跟规划器(sando_node)分开跑,职责是:
  原始点云 → 聚类成一团一团的物体 → 每团配一个卡尔曼滤波器估计它的位置/速度/加速度
  → 外推出它未来一小段的轨迹(拟合成多项式)→ 发到 predicted_trajs 让规划器去躲。
说白了:把「看到一堆点」变成「这有个东西,正往这个方向以这个速度运动,接下来大概会到哪」。

流程关键词:
  - 聚类:把空间上挨得近的点归成一个障碍(这里用 DBSCAN 代替 C++ 的 PCL 欧式聚类,参数等价)。
  - 卡尔曼滤波(KF):用「匀加速」模型平滑估计每个障碍的状态;自适应版还会在线调噪声协方差。
  - 数据关联:新一帧的某个团,和上一帧哪个已知障碍是同一个(按质心距离最近匹配)。
  - 轨迹外推 + 多项式拟合:把未来若干步的位置拟合成 3 次(给 pwp)和 5 次多项式(给 poly_coeffs)。

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
    """9-D state [x, y, z, vx, vy, vz, ax, ay, az] with constant-acceleration model.

    中文:一个被跟踪障碍的卡尔曼滤波状态。
    x = 9 维状态[位置 3 + 速度 3 + 加速度 3];P=状态协方差(不确定度);
    Q=过程噪声、R=观测噪声(自适应模式下会在线更新);
    bbox=包围盒尺寸;id=该障碍编号;color=可视化随机色。
    """
    x: np.ndarray = field(default_factory=lambda: np.zeros(9))
    P: np.ndarray = field(default_factory=lambda: np.eye(9))
    Q: np.ndarray = field(default_factory=lambda: np.eye(9) * 0.01)
    R: np.ndarray = field(default_factory=lambda: np.eye(3) * 0.01)
    time_updated: float = 0.0
    bbox: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5, 0.5]))
    id: int = 0
    color: Tuple[float, float, float, float] = (0.5, 0.5, 0.5, 0.4)


def _make_color() -> Tuple[float, float, float, float]:
    # 给每个新障碍随机分配一个颜色,纯为了在 RViz 里区分不同障碍。
    rng = np.random.default_rng()
    return (float(rng.random()), float(rng.random()), float(rng.random()), 0.4)


def _ekf_predict(s: EKFState, dt: float) -> None:
    # 卡尔曼「预测」步:按匀加速运动模型把状态往前推 dt 秒。
    # F 是状态转移矩阵:位置 += 速度*dt + 0.5*加速度*dt^2,速度 += 加速度*dt。
    # P 同步膨胀(加过程噪声 Q)表示「往前猜会更不确定」。
    F = np.eye(9)
    F[0, 3] = dt; F[1, 4] = dt; F[2, 5] = dt
    F[0, 6] = 0.5 * dt * dt; F[1, 7] = 0.5 * dt * dt; F[2, 8] = 0.5 * dt * dt
    F[3, 6] = dt; F[4, 7] = dt; F[5, 8] = dt
    s.x = F @ s.x
    s.P = F @ s.P @ F.T + s.Q


def _aekf_update(s: EKFState, z: np.ndarray, alpha: float, time_updated: float,
                 bbox: np.ndarray, use_adaptive: bool) -> None:
    # 卡尔曼「更新」步:用这一帧实测到的质心 z(只观测得到位置,所以 H 只取 xyz)修正状态。
    # d=观测和预测的差(新息),K=卡尔曼增益(信观测多还是信预测多)。
    # 自适应版(AEKF):用残差大小在线调 R、Q,observation 噪声大就少信它——这是相对普通 KF 的关键改动。
    # alpha 是遗忘因子(越接近 1 越平滑、变化越慢)。
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
    # bbox 用一半旧一半新做平滑,避免尺寸一帧一跳。
    s.bbox = 0.5 * s.bbox + 0.5 * bbox


def _associate_cluster(centroid: np.ndarray, ekf_states: List[EKFState],
                       tol: float) -> int:
    # 数据关联:这个新质心离哪个已知障碍最近(且在 tol 阈值内)就认作同一个,返回其下标;
    # 都不够近就返回 -1,表示「这是个新障碍」。最近邻 + 阈值的朴素关联。
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
    The C++ code uses ax^3 + bx^2 + cx + d order so this matches directly.

    中文:对 (t, y) 拟合一条 degree 次多项式,返回系数(高次在前)。
    系数顺序刻意和 C++ 的 ax^3+bx^2+cx+d 对齐,所以两边能直接对照。
    """
    return np.polyfit(t, y, degree)


def _calculate_variance(t: np.ndarray, y: np.ndarray, beta: np.ndarray,
                         degree: int) -> float:
    # 算拟合残差的方差(衡量这条多项式拟合得好不好)。下游用它当预测不确定度。
    # 分母 n-degree-1 是无偏估计的自由度;样本不够时直接返回 0。
    fitted = np.polyval(beta, t)
    residuals = y - fitted
    n = len(t)
    if n - degree - 1 <= 0:
        return 0.0
    return float(np.sum(residuals * residuals) / (n - degree - 1))


def _euclidean_cluster(points: np.ndarray, eps: float, min_samples: int,
                       max_samples: int) -> List[np.ndarray]:
    """DBSCAN clustering — equivalent to PCL's EuclideanClusterExtraction.
    Returns a list of (N_i, 3) arrays, one per cluster.

    中文:把点云聚成一团团(每团就是一个候选障碍),返回若干个 (N_i,3) 数组。
    eps=多近算一伙(等价 PCL 的 cluster_tolerance),min/max_samples=一团点数的上下限。
    """
    if points.shape[0] == 0:
        return []
    try:
        from sklearn.cluster import DBSCAN
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)
    except ImportError:
        # 没装 sklearn 时的兜底:用 KDTree 找邻居 + 并查集自己实现等价聚类(把互为邻居的点并成一组)。
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

    # 按标签收集每一团点;label==-1 是 DBSCAN 标的「噪声点」,丢掉;点数不在 [min,max] 的也丢。
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
    # 体素降采样:把空间切成边长 leaf 的小立方体,每个立方体里只留一个点,
    # 用来稀释稠密点云、减少后续聚类的计算量(等价 PCL 的 VoxelGrid)。
    if points.shape[0] == 0 or leaf <= 0:
        return points
    keys = np.floor(points / leaf).astype(np.int64)
    # 每个体素(相同整数坐标)只保留第一次出现的那个点。
    # Use unique with axis=0 — keep first occurrence
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(idx)]


class ObstacleTrackerNode(Node):
    """动态障碍跟踪节点:订点云、聚类、卡尔曼跟踪、外推未来轨迹并发布。"""

    def __init__(self):
        super().__init__("obstacle_tracker_node")
        self.get_logger().info("Obstacle Tracker Node Started")

        # 一堆可调参数(都从 ROS 参数读,可在 launch/yaml 里改):
        #  cluster_tolerance/min/max_cluster_size 控聚类;adaptive_kf_* 控卡尔曼;
        #  prediction_horizon/dt 控外推多远多密;各种 threshold/cutoff 控过滤。
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

        # 当前正在跟踪的所有障碍(每个一个 EKFState);ekf_state_id 是自增的障碍编号发号器。
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
        # 软依赖:消息包没装就返回 None,节点照常起,只是不发预测轨迹。
        try:
            from dynus_interfaces.msg import DynTraj
            return DynTraj
        except ImportError:
            return None

    def pointcloud_callback(self, msg: PointCloud2) -> None:
        # 主流程(每来一帧点云跑一次):降采样 → 高度过滤 → 聚类 → 关联/卡尔曼更新 → 发预测。
        points = _pointcloud2_to_xyz(msg)
        if points.shape[0] == 0:
            self.get_logger().warn("Received empty point cloud!")
            return

        # Voxel downsample (matches PCL VoxelGrid leaf 0.2)
        points = _voxel_downsample(points, leaf=0.2)

        # TF transform — skip; assume cloud already in map frame for the Python port.
        # If running with a non-map source, configure the upstream node to publish in `map`.

        # 只保留高度 0.5~6.0m 的点:把地面和过高的杂点滤掉。
        # Z passthrough filter (0.5 → 6.0)
        mask = (points[:, 2] >= 0.5) & (points[:, 2] <= 6.0)
        points = points[mask]

        # 聚类成一团团,再对每团算质心(当作障碍中心)和包围盒尺寸(max-min)。
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
        # 先清掉太久没被观测到的旧障碍(可能已经走出视野)。
        self._delete_old_ekf_states(now)

        # 逐团处理:太大的团(整面墙之类)直接跳过——只跟踪动态小障碍。
        # 能关联上已知障碍 → 预测+更新它;关联不上 → 新建一个障碍并用质心初始化位置。
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
                # 新障碍的噪声协方差用现有障碍的均值初始化,起步更稳。
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
        # 删掉超过 time_to_delete_old_obstacles 秒没更新过的障碍。
        self.ekf_states = [s for s in self.ekf_states
                            if now - s.time_updated <= self.time_to_delete_old_obstacles]

    def _average_q_r(self) -> Tuple[np.ndarray, np.ndarray]:
        # 取当前所有障碍 Q、R 的平均,给新障碍当初值;没有障碍时退回默认小值。
        if not self.ekf_states:
            return np.eye(9) * 0.01, np.eye(3) * 0.01
        Q = np.mean(np.stack([s.Q for s in self.ekf_states]), axis=0)
        R = np.mean(np.stack([s.R for s in self.ekf_states]), axis=0)
        return Q, R

    def _publish_boxes(self, records: List[Tuple[EKFState, np.ndarray]]) -> None:
        # 纯可视化:画每个障碍的包围盒(立方体)和不确定度球(P 越大球越大)。
        markers = MarkerArray()
        unc = MarkerArray()
        lifetime = Duration(seconds=0.5).to_msg()
        # 先发一个 DELETEALL 把上一帧的 marker 清掉,避免旧框残留。
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
        # 核心输出:对每个障碍,用它的状态外推未来 prediction_horizon 秒的轨迹,
        # 拟合成多项式,打包成 DynTraj 发到 predicted_trajs 给规划器躲;同时画外推箭头。
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
            # 取当前估计的位置/速度/加速度作为外推起点;加速度先夹到 ±2,防滤波器抖出离谱大值。
            pos = state.x[:3].copy()
            vel = state.x[3:6].copy()
            acc = np.clip(state.x[6:9].copy(), -2.0, 2.0)

            t_vals: List[float] = [0.0]
            x_vals: List[float] = [pos[0]]
            y_vals: List[float] = [pos[1]]
            z_vals: List[float] = [pos[2]]

            # 一步步往未来外推。速度/加速度都再夹一道(±0.5、±0.8),让预测保守、不外推到天上去。
            # 注意:每步用「当前 pos 为基点 + t 从 0 算起」推出 future_pos,循环末尾再把 pos 推进一步。
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

            # 几乎没动的障碍(整段位移都小于 cutoff)当作静态,不发预测——动态障碍才需要规划器特别躲。
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

            # 把 3 次拟合结果塞进 pwp(单段分段多项式,带绝对起止时间)——这是给规划器用的主形式。
            pwp = PieceWisePol()
            t0 = self._ros_now()
            # 外推样本只到 t=(num_steps-1)*dt(= t_arr[-1]),比 prediction_horizon 短一个 dt;
            # 宣告窗口必须缩到数据实际末点,否则多项式在 (t_arr[-1], horizon] 区间外插超出拟合数据。
            t_end_rel = float(t_arr[-1])
            pwp.times = [t0, t0 + t_end_rel]
            pwp.coeff_x = [np.array([beta_x[0], beta_x[1], beta_x[2], beta_x[3]])]
            pwp.coeff_y = [np.array([beta_y[0], beta_y[1], beta_y[2], beta_y[3]])]
            pwp.coeff_z = [np.array([beta_z[0], beta_z[1], beta_z[2], beta_z[3]])]

            # 另外再拟一份 5 次多项式,单独走 poly_coeffs_* 字段(消息里两种表示都带上)。
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
            # 把协方差/拟合方差也一起带上,规划器可据此把不确定的障碍多留点安全裕度。
            msg.poly_cov = [float(var_x), float(var_y), float(var_z)]
            msg.poly_coeffs_x = [float(c) for c in beta_x_quintic]
            msg.poly_coeffs_y = [float(c) for c in beta_y_quintic]
            msg.poly_coeffs_z = [float(c) for c in beta_z_quintic]
            msg.poly_start_time = float(t0)
            msg.poly_end_time = float(t0 + t_end_rel)   # 同 pwp.times:窗口缩到拟合数据末点,不外插
            # is_agent=False:明确告诉规划器「这是障碍,不是另一架飞机」。
            msg.pos.x = float(state.x[0]); msg.pos.y = float(state.x[1]); msg.pos.z = float(state.x[2])
            msg.is_agent = False
            self.pub_predicted_traj.publish(msg)

        if self.visual_level >= 0:
            self.pub_markers.publish(ma)

    def _ros_now(self) -> float:
        # 当前 ROS 时间,转成秒。
        return float(self.get_clock().now().nanoseconds) * 1e-9


def main(args=None):
    # 节点入口,多线程 executor 跑起来。
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
