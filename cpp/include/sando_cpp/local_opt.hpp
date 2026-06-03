// local_opt — faithful C++ port of sando_py/local/local_opt.py (MINCO analytic-gradient core).
// Ported incrementally, bottom-up; each piece golden-verified. This file starts with the
// geometric primitive _signed_dist_and_grad and the quadrature power-basis helpers.
#pragma once
#include <Eigen/Dense>
#include <vector>
#include <map>
#include <cmath>
#include "sando_cpp/obstacles.hpp"

namespace sando {

struct DistGrad {
  double d;
  Eigen::Vector3d dd_dp;  // gradient of signed distance w.r.t. point
  double dd_dt;           // gradient w.r.t. time (moving obstacle)
};

// 1:1 with local_opt.py:_signed_dist_and_grad. Caller keeps samples off the kinks.
inline DistGrad signed_dist_and_grad(const Obstacle* obs, const Eigen::Vector3d& p, double t) {
  if (auto s = dynamic_cast<const SphereObstacle*>(obs)) {
    Eigen::Vector3d c = s->predict(t);
    Eigen::Vector3d diff = p - c;
    double nrm = diff.norm();
    Eigen::Vector3d n = diff / nrm;
    double d = nrm - s->radius;
    double dd_dt = -n.dot(s->velocity_at(t));
    return {d, n, dd_dt};
  }
  auto b = dynamic_cast<const AABBObstacle*>(obs);
  Eigen::Vector3d over = (b->lo - p).cwiseMax(0.0) + (p - b->hi).cwiseMax(0.0);
  if ((over.array() > 0.0).any()) {
    double nrm = over.norm();
    Eigen::Vector3d sign;
    for (int k = 0; k < 3; ++k)
      sign(k) = (p(k) > b->hi(k) ? 1.0 : 0.0) - (p(k) < b->lo(k) ? 1.0 : 0.0);
    Eigen::Vector3d dd_dp = over.cwiseProduct(sign) / nrm;
    return {nrm, dd_dp, 0.0};
  }
  // inside: L-inf penetration subgradient
  Eigen::Vector3d cand = (b->lo - p).cwiseMax(p - b->hi);
  int ax = 0;
  cand.maxCoeff(&ax);  // first index of max (matches np.argmax tie-break)
  Eigen::Vector3d grad = Eigen::Vector3d::Zero();
  grad(ax) = ((b->lo(ax) - p(ax)) >= (p(ax) - b->hi(ax))) ? -1.0 : 1.0;
  return {cand(ax), grad, 0.0};
}

// _seg_vander: power-basis rows for derivative orders 0..3 at local times taus (k,).
// Returns 4 matrices each (k,6): B_o[j,p] = (p!/(p-o)!) * tau_j^(p-o) for p>=o else 0.
inline std::array<Eigen::MatrixXd, 4> seg_vander(const Eigen::VectorXd& taus) {
  const int k = (int)taus.size();
  std::array<Eigen::MatrixXd, 4> out;
  for (int o = 0; o < 4; ++o) {
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(k, 6);
    for (int p = o; p < 6; ++p) {
      double coeff = 1.0;
      for (int m = 0; m < o; ++m) coeff *= double(p - m);
      for (int j = 0; j < k; ++j) B(j, p) = coeff * std::pow(taus(j), p - o);
    }
    out[o] = B;
  }
  return out;
}

}  // namespace sando
