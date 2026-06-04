// utils_math — faithful C++ port of the NUMERIC (non-ROS) helpers in sando_py/utils.py.
// ROS message conversions / Timer / tf-stamped variants are intentionally omitted (glue).
#pragma once
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <limits>
#include <array>

namespace sando {
constexpr double PI_ = 3.141592653589793;  // M_PI not portable on MinGW

inline double angle_wrap(double a) {  // wrap to (-pi, pi]  (Python floor-mod)
  double x = a + PI_, m = 2.0 * PI_;
  double r = std::fmod(x, m);
  if (r < 0.0) r += m;
  return r - PI_;
}
inline double angle_diff(double target, double current) { return angle_wrap(target - current); }

inline double yaw_from_quat(double qx, double qy, double qz, double qw) {
  double siny = 2.0 * (qw * qz + qx * qy);
  double cosy = 1.0 - 2.0 * (qy * qy + qz * qz);
  return std::atan2(siny, cosy);
}
inline std::array<double, 4> quat_from_yaw(double yaw) {  // (x,y,z,w)
  double half = 0.5 * yaw;
  return {0.0, 0.0, std::sin(half), std::cos(half)};
}
inline double clampv(double x, double lo, double hi) { return x < lo ? lo : (x > hi ? hi : x); }
inline Eigen::Vector3d saturate(const Eigen::Vector3d& v, double vmax) {
  double n = v.norm();
  if (n > vmax && n > 1e-9) return v * (vmax / n);
  return v;
}

inline Eigen::Vector3d project_point_to_sphere(const Eigen::Vector3d& P1, const Eigen::Vector3d& P2, double radius) {
  Eigen::Vector3d diff = P2 - P1;
  double n = diff.norm();
  if (n <= radius) return P2;
  return P1 + diff * (radius / n);
}

inline Eigen::Vector3d project_point_to_box(const Eigen::Vector3d& P1, const Eigen::Vector3d& P2,
                                            double wdx, double wdy, double wdz) {
  double x_max = P1(0) + wdx / 2.0, x_min = P1(0) - wdx / 2.0;
  double y_max = P1(1) + wdy / 2.0, y_min = P1(1) - wdy / 2.0;
  double z_max = P1(2) + wdz / 2.0, z_min = P1(2) - wdz / 2.0;
  if (x_min < P2(0) && P2(0) < x_max && y_min < P2(1) && P2(1) < y_max && z_min < P2(2) && P2(2) < z_max)
    return P2;
  Eigen::Vector3d v = P2 - P1;
  if (v.norm() < 1e-12) return P1;
  double best_t = std::numeric_limits<double>::infinity();
  Eigen::Vector3d best_pt = P2;
  const Eigen::Vector3d nrm[6] = {{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
  const double dpl[6] = {-x_max, x_min, -y_max, y_min, -z_max, z_min};
  for (int i = 0; i < 6; ++i) {
    double denom = nrm[i].dot(v);
    if (std::fabs(denom) < 1e-12) continue;
    double t = -(nrm[i].dot(P1) + dpl[i]) / denom;
    if (0.0 < t && t < best_t) { best_t = t; best_pt = P1 + v * t; }
  }
  return best_pt;
}

inline double min_time_double_integrator_1d(double p0, double v0, double pf, double vf,
                                            double v_max, double a_max) {
  double x1 = v0, x2 = p0, x1r = vf, x2r = pf;
  double k1 = a_max, k2 = 1.0, x1_bar = v_max;
  double s = (-x1 + x1r) > 0 ? 1.0 : ((-x1 + x1r) < 0 ? -1.0 : 0.0);
  double B = (k2 / (2 * k1)) * s * (x1 * x1 - x1r * x1r) + x2r;
  double C = (k2 / (2 * k1)) * (x1 * x1 + x1r * x1r) - (k2 / k1) * x1_bar * x1_bar + x2r;
  double D = (-k2 / (2 * k1)) * (x1 * x1 + x1r * x1r) + (k2 / k1) * x1_bar * x1_bar + x2r;
  double t;
  if (x2 <= B && x2 >= C) {
    double inside = (k2 * k2) * (x1 * x1) - k1 * k2 * ((k2 / (2 * k1)) * (x1 * x1 - x1r * x1r) + x2 - x2r);
    inside = std::max(0.0, inside);
    t = (-k2 * (x1 + x1r) + 2 * std::sqrt(inside)) / (k1 * k2);
  } else if (x2 <= B && x2 < C) {
    t = (x1_bar - x1 - x1r) / k1 + (x1 * x1 + x1r * x1r) / (2 * k1 * x1_bar) + (x2r - x2) / (k2 * x1_bar);
  } else if (x2 > B && x2 <= D) {
    double inside = (k2 * k2) * (x1 * x1) + k1 * k2 * ((k2 / (2 * k1)) * (-x1 * x1 + x1r * x1r) + x2 - x2r);
    inside = std::max(0.0, inside);
    t = (k2 * (x1 + x1r) + 2 * std::sqrt(inside)) / (k1 * k2);
  } else {
    t = (x1_bar + x1 + x1r) / k1 + (x1 * x1 + x1r * x1r) / (2 * k1 * x1_bar) + (-x2r + x2) / (k2 * x1_bar);
  }
  return t;
}
inline double min_time_double_integrator_3d(const Eigen::Vector3d& p0, const Eigen::Vector3d& v0,
                                            const Eigen::Vector3d& pf, const Eigen::Vector3d& vf,
                                            const Eigen::Vector3d& v_max, const Eigen::Vector3d& a_max) {
  double tx = min_time_double_integrator_1d(p0(0), v0(0), pf(0), vf(0), v_max(0), a_max(0));
  double ty = min_time_double_integrator_1d(p0(1), v0(1), pf(1), vf(1), v_max(1), a_max(1));
  double tz = min_time_double_integrator_1d(p0(2), v0(2), pf(2), vf(2), v_max(2), a_max(2));
  return std::max({tx, ty, tz});
}

inline std::vector<Eigen::Vector3d> create_more_vertexes(const std::vector<Eigen::Vector3d>& path, double d) {
  if (path.size() < 2 || d <= 0.0) return path;
  std::vector<Eigen::Vector3d> out = {path[0]};
  for (size_t i = 0; i + 1 < path.size(); ++i) {
    Eigen::Vector3d a = path[i], b = path[i + 1], seg = b - a;
    double L = seg.norm();
    if (L <= 1e-9) continue;
    int n = (int)std::floor(L / d);
    Eigen::Vector3d u = seg / L;
    for (int k = 1; k <= n; ++k) out.push_back(a + u * (k * d));
    if ((out.back() - b).norm() > 1e-6) out.push_back(b);
  }
  return out;
}

inline Eigen::Matrix4d transform_inverse_se3(const Eigen::Matrix4d& M) {
  Eigen::Matrix3d R = M.block<3, 3>(0, 0);
  Eigen::Vector3d t = M.block<3, 1>(0, 3);
  Eigen::Matrix4d out = Eigen::Matrix4d::Identity();
  out.block<3, 3>(0, 0) = R.transpose();
  out.block<3, 1>(0, 3) = -R.transpose() * t;
  return out;
}

// translation + quaternion (x,y,z,w) -> 4x4 (non-ROS variant of transform_stamped_to_matrix)
inline Eigen::Matrix4d transform_from_tq(double tx, double ty, double tz,
                                         double qx, double qy, double qz, double qw) {
  Eigen::Matrix4d M = Eigen::Matrix4d::Identity();
  M(0,0)=1-2*(qy*qy+qz*qz); M(0,1)=2*(qx*qy-qz*qw); M(0,2)=2*(qx*qz+qy*qw);
  M(1,0)=2*(qx*qy+qz*qw); M(1,1)=1-2*(qx*qx+qz*qz); M(1,2)=2*(qy*qz-qx*qw);
  M(2,0)=2*(qx*qz-qy*qw); M(2,1)=2*(qy*qz+qx*qw); M(2,2)=1-2*(qx*qx+qy*qy);
  M(0,3)=tx; M(1,3)=ty; M(2,3)=tz;
  return M;
}

}  // namespace sando
