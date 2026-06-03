// Obstacles — faithful C++ port of sando_py/local/obstacles.py.
// SphereObstacle: CA motion model (centre0 + vel*t + 0.5*accel*t^2), signed_dist = ||p-c(t)||-r.
// AABBObstacle: static box; signed_dist = euclidean outside / L-inf penetration inside.
// 1:1 with Python; golden-verified.
#pragma once
#include <Eigen/Dense>
#include <string>
#include <cmath>
#include <algorithm>

namespace sando {

class Obstacle {
 public:
  std::string class_name;
  virtual ~Obstacle() = default;
  virtual double signed_dist(const Eigen::Vector3d& p, double t) const = 0;
};

class SphereObstacle : public Obstacle {
 public:
  Eigen::Vector3d centre0, vel, accel;
  double radius;

  SphereObstacle(const Eigen::Vector3d& c0, double r,
                 const Eigen::Vector3d& v = Eigen::Vector3d::Zero(),
                 const std::string& cls = "human",
                 const Eigen::Vector3d& a = Eigen::Vector3d::Zero())
      : centre0(c0), vel(v), accel(a), radius(r) { class_name = cls; }

  // centre(t) = centre0 + vel*t + 0.5*accel*t^2
  Eigen::Vector3d predict(double t) const {
    return centre0 + vel * t + 0.5 * accel * (t * t);
  }
  // velocity_at(t) = vel + accel*t
  Eigen::Vector3d velocity_at(double t) const { return vel + accel * t; }

  double signed_dist(const Eigen::Vector3d& p, double t) const override {
    return (p - predict(t)).norm() - radius;
  }
};

class AABBObstacle : public Obstacle {
 public:
  Eigen::Vector3d lo, hi;

  AABBObstacle(const Eigen::Vector3d& lo_, const Eigen::Vector3d& hi_,
               const std::string& cls = "wall")
      : lo(lo_), hi(hi_) {
    class_name = cls;
    if ((hi.array() < lo.array()).any())
      throw std::invalid_argument("AABB hi must be >= lo elementwise");
  }

  double signed_dist(const Eigen::Vector3d& p, double /*t*/ = 0.0) const override {
    // outside[axis] = how far p is beyond the box on each axis (0 if inside)
    Eigen::Vector3d outside =
        (lo - p).cwiseMax(0.0) + (p - hi).cwiseMax(0.0);
    if ((outside.array() > 0.0).any()) return outside.norm();
    // inside: L-inf penetration depth (<= 0)
    Eigen::Vector3d pen = (lo - p).cwiseMax(p - hi);
    return pen.maxCoeff();
  }
};

}  // namespace sando
