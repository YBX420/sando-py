// obstacle_tracker_kernels.hpp — golden-verified C++ port of the deterministic math in
// sando_py/nodes/obstacle_tracker.py. Pure Eigen, no ROS.
//
// Reproduced verbatim in behaviour:
//   voxel_downsample      (PCL VoxelGrid stand-in: first point per leaf, original order)
//   connected_components  (the union-find/KDTree FALLBACK path the Python takes when sklearn
//                          is absent — connected components under the dist<=eps graph;
//                          NOTE: differs from a sklearn-DBSCAN run, which the Python only uses
//                          when scikit-learn is installed. See cpp/README.md.)
//   ekf_predict / aekf_update   (9-D constant-accel adaptive Kalman filter)
//   polyfit               (replicates numpy.polyfit: column-scaled Vandermonde + SVD lstsq)
//   polyval / variance    (numpy.polyval Horner + residual variance)
//   predict_samples       (the clipped constant-accel extrapolation in _publish_predictions)
#pragma once
#include <Eigen/Dense>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <functional>
#include <limits>
#include <cmath>
#include <cstdint>

namespace sando {
namespace obstacle_tracker {

// --- voxel downsample: keep the first point that falls in each leaf cube, original order ---
inline std::vector<Eigen::Vector3d> voxel_downsample(const std::vector<Eigen::Vector3d>& pts, double leaf) {
  if (pts.empty() || leaf <= 0) return pts;
  std::vector<Eigen::Vector3d> out;
  struct Key { int64_t a, b, c; bool operator==(const Key& o) const { return a == o.a && b == o.b && c == o.c; } };
  struct KeyHash { size_t operator()(const Key& k) const {
    return std::hash<int64_t>()(k.a) ^ (std::hash<int64_t>()(k.b) * 1000003ULL) ^ (std::hash<int64_t>()(k.c) * 1000000007ULL); } };
  std::unordered_map<Key, char, KeyHash> seen;
  for (const auto& p : pts) {
    Key k{static_cast<int64_t>(std::floor(p[0] / leaf)),
          static_cast<int64_t>(std::floor(p[1] / leaf)),
          static_cast<int64_t>(std::floor(p[2] / leaf))};
    if (seen.find(k) == seen.end()) { seen.emplace(k, 1); out.push_back(p); }
  }
  return out;
}

// --- connected components under dist<=eps; clusters kept only if size in [min_samples, max_samples] ---
// Returns each cluster as the list of original point indices (ascending), clusters ordered by
// ascending union-find root (matching np.unique(labels) over root labels).
inline std::vector<std::vector<int>> connected_components(
    const std::vector<Eigen::Vector3d>& pts, double eps, int min_samples, int max_samples) {
  int n = static_cast<int>(pts.size());
  std::vector<std::vector<int>> result;
  if (n == 0) return result;
  std::vector<int> parent(n);
  for (int i = 0; i < n; ++i) parent[i] = i;
  std::function<int(int)> find = [&](int i) { while (parent[i] != i) { parent[i] = parent[parent[i]]; i = parent[i]; } return i; };
  auto uni = [&](int a, int b) { int ra = find(a), rb = find(b); if (ra != rb) parent[ra] = rb; };
  double eps2 = eps * eps;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      if (j != i && (pts[i] - pts[j]).squaredNorm() <= eps2) uni(i, j);
  // group by root; iterate roots in ascending order (np.unique sorts labels)
  std::vector<int> root(n);
  for (int i = 0; i < n; ++i) root[i] = find(i);
  std::vector<int> roots_sorted = root;
  std::sort(roots_sorted.begin(), roots_sorted.end());
  roots_sorted.erase(std::unique(roots_sorted.begin(), roots_sorted.end()), roots_sorted.end());
  for (int r : roots_sorted) {
    std::vector<int> members;
    for (int i = 0; i < n; ++i) if (root[i] == r) members.push_back(i);
    int sz = static_cast<int>(members.size());
    if (sz < min_samples || sz > max_samples) continue;
    result.push_back(members);
  }
  return result;
}

// --- 9-D EKF state ---
struct EKFState {
  Eigen::Matrix<double, 9, 1> x = Eigen::Matrix<double, 9, 1>::Zero();
  Eigen::Matrix<double, 9, 9> P = Eigen::Matrix<double, 9, 9>::Identity();
  Eigen::Matrix<double, 9, 9> Q = Eigen::Matrix<double, 9, 9>::Identity() * 0.01;
  Eigen::Matrix<double, 3, 3> R = Eigen::Matrix<double, 3, 3>::Identity() * 0.01;
  double time_updated = 0.0;
  Eigen::Vector3d bbox = Eigen::Vector3d(0.5, 0.5, 0.5);
  int id = 0;
};

inline Eigen::Matrix<double, 9, 9> ekf_F(double dt) {
  Eigen::Matrix<double, 9, 9> F = Eigen::Matrix<double, 9, 9>::Identity();
  F(0, 3) = dt; F(1, 4) = dt; F(2, 5) = dt;
  F(0, 6) = 0.5 * dt * dt; F(1, 7) = 0.5 * dt * dt; F(2, 8) = 0.5 * dt * dt;
  F(3, 6) = dt; F(4, 7) = dt; F(5, 8) = dt;
  return F;
}
inline void ekf_predict(EKFState& s, double dt) {
  Eigen::Matrix<double, 9, 9> F = ekf_F(dt);
  s.x = F * s.x;
  s.P = F * s.P * F.transpose() + s.Q;
}
inline Eigen::Matrix<double, 3, 9> ekf_H() {
  Eigen::Matrix<double, 3, 9> H = Eigen::Matrix<double, 3, 9>::Zero();
  H(0, 0) = 1.0; H(1, 1) = 1.0; H(2, 2) = 1.0;
  return H;
}
inline void aekf_update(EKFState& s, const Eigen::Vector3d& z, double alpha, double time_updated,
                        const Eigen::Vector3d& bbox, bool use_adaptive) {
  Eigen::Matrix<double, 3, 9> H = ekf_H();
  Eigen::Vector3d d = z - H * s.x;
  Eigen::Matrix3d S = H * s.P * H.transpose() + s.R;
  Eigen::Matrix<double, 9, 3> K = s.P * H.transpose() * S.inverse();
  s.x = s.x + K * d;
  Eigen::Vector3d epsilon = z - H * s.x;
  if (use_adaptive) {
    s.R = alpha * s.R + (1 - alpha) * (epsilon * epsilon.transpose() + H * s.P * H.transpose());
    s.Q = alpha * s.Q + (1 - alpha) * (K * (d * d.transpose()) * K.transpose());
  } else {
    s.R = Eigen::Matrix3d::Identity() * 0.01;
    s.Q = Eigen::Matrix<double, 9, 9>::Identity() * 0.01;
  }
  s.P = (Eigen::Matrix<double, 9, 9>::Identity() - K * H) * s.P;
  s.time_updated = time_updated;
  s.bbox = 0.5 * s.bbox + 0.5 * bbox;
}

// data association: nearest existing track within tol, else -1
inline int associate_cluster(const Eigen::Vector3d& centroid, const std::vector<EKFState>& states, double tol) {
  double min_d = tol; int idx = -1;
  for (int i = 0; i < static_cast<int>(states.size()); ++i) {
    double d = (centroid - states[i].x.head<3>()).norm();
    if (d < min_d) { min_d = d; idx = i; }
  }
  return idx;
}

// --- polyfit replicating numpy.polyfit (decreasing-power coeffs, column-scaled Vandermonde + lstsq) ---
inline Eigen::VectorXd polyfit(const Eigen::VectorXd& x, const Eigen::VectorXd& y, int degree) {
  int n = static_cast<int>(x.size());
  int order = degree + 1;
  // Vandermonde with decreasing powers: column k = x^(degree-k)
  Eigen::MatrixXd V(n, order);
  for (int i = 0; i < n; ++i) {
    double xi = x[i], p = 1.0;
    for (int k = order - 1; k >= 0; --k) { V(i, k) = p; p *= xi; }
  }
  // column scaling for conditioning (numpy.polyfit does exactly this)
  Eigen::VectorXd scale(order);
  for (int k = 0; k < order; ++k) {
    double s = std::sqrt(V.col(k).squaredNorm());
    scale[k] = (s == 0.0) ? 1.0 : s;
    V.col(k) /= scale[k];
  }
  // lstsq via SVD with numpy's rcond = n * eps (relative to largest singular value)
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(V, Eigen::ComputeThinU | Eigen::ComputeThinV);
  svd.setThreshold(static_cast<double>(n) * std::numeric_limits<double>::epsilon());
  Eigen::VectorXd c = svd.solve(y);
  for (int k = 0; k < order; ++k) c[k] /= scale[k];
  return c;  // decreasing powers
}

// numpy.polyval (Horner) with decreasing-power coeffs
inline double polyval(const Eigen::VectorXd& beta, double t) {
  double v = 0.0;
  for (int i = 0; i < beta.size(); ++i) v = v * t + beta[i];
  return v;
}
inline double calculate_variance(const Eigen::VectorXd& t, const Eigen::VectorXd& y,
                                 const Eigen::VectorXd& beta, int degree) {
  int n = static_cast<int>(t.size());
  if (n - degree - 1 <= 0) return 0.0;
  double ss = 0.0;
  for (int i = 0; i < n; ++i) { double r = y[i] - polyval(beta, t[i]); ss += r * r; }
  return ss / static_cast<double>(n - degree - 1);
}

// --- clipped constant-accel extrapolation (the sample loop in _publish_predictions) ---
struct PredSamples { std::vector<double> t, x, y, z; };
inline double clip(double v, double lo, double hi) { return std::min(hi, std::max(lo, v)); }
inline PredSamples predict_samples(const Eigen::Matrix<double, 9, 1>& state, int num_steps, double dt) {
  Eigen::Vector3d pos = state.head<3>();
  Eigen::Vector3d vel = state.segment<3>(3);
  Eigen::Vector3d acc;
  for (int i = 0; i < 3; ++i) acc[i] = clip(state[6 + i], -2.0, 2.0);
  PredSamples s;
  s.t.push_back(0.0); s.x.push_back(pos[0]); s.y.push_back(pos[1]); s.z.push_back(pos[2]);
  for (int step = 0; step < num_steps; ++step) {
    for (int i = 0; i < 3; ++i) vel[i] = clip(vel[i], -0.5, 0.5);
    for (int i = 0; i < 3; ++i) acc[i] = clip(acc[i], -0.8, 0.8);
    double t = step * dt;
    Eigen::Vector3d future = pos + vel * t + 0.5 * acc * t * t;
    s.t.push_back(t); s.x.push_back(future[0]); s.y.push_back(future[1]); s.z.push_back(future[2]);
    pos = future;
    vel = vel + acc * dt;
  }
  return s;
}

}  // namespace obstacle_tracker
}  // namespace sando
