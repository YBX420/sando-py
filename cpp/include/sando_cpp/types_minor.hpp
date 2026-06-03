// types_minor — StateDeriv + Polytope (faithful structs from sando_py/types.py).
// (BasisConverter is only consumed by the Gurobi baseline path -> deferred with that decision.)
#pragma once
#include <Eigen/Dense>

namespace sando {

struct StateDeriv {
  Eigen::Vector3d pos = Eigen::Vector3d::Zero();
  Eigen::Vector3d vel = Eigen::Vector3d::Zero();
  Eigen::Vector3d accel = Eigen::Vector3d::Zero();
  Eigen::Vector3d jerk = Eigen::Vector3d::Zero();
};

// Convex polytope in half-space form: { x : A x <= b }.
struct Polytope {
  Eigen::MatrixXd A = Eigen::MatrixXd(0, 3);
  Eigen::VectorXd b = Eigen::VectorXd(0);
  bool contains(const Eigen::Vector3d& x) const {
    if (A.rows() == 0) return true;
    return ((A * x).array() <= b.array()).all();
  }
};

}  // namespace sando
