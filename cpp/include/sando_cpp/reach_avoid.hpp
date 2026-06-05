// reach_avoid.hpp — the always-on SAFE-half MONITOR (ABS/gatekeeper reach-avoid value), done
// deterministically. It reuses the SAME geometry the per-class certificate uses
// (Obstacle::signed_dist over predicted time) — there is no second safety source.
//
//   reach_avoid_clearance(p, obs, t0, horizon)   : min signed-distance of point p to the
//       predicted obstacle set over [t0, t0+horizon]. >= 0 safe; this IS the monitor value.
//   reach_avoid_move_clearance(p, q, obs, t0, dt) : min clearance of a straight move p->q over
//       [t0, t0+dt] (drone interpolated, obstacles predicted) — used to score recovery candidates.
#pragma once
#include <vector>
#include <algorithm>
#include <Eigen/Dense>
#include "sando_cpp/obstacles.hpp"

namespace sando {

inline double reach_avoid_clearance(const Eigen::Vector3d& p,
                                    const std::vector<const Obstacle*>& obs,
                                    double t0, double horizon, int K = 8) {
  double mn = 1e18;
  for (int k = 0; k <= K; ++k) {
    double t = t0 + (K > 0 ? horizon * double(k) / K : 0.0);
    for (auto* o : obs) mn = std::min(mn, o->signed_dist(p, t));
  }
  return mn;
}

inline double reach_avoid_move_clearance(const Eigen::Vector3d& p, const Eigen::Vector3d& q,
                                         const std::vector<const Obstacle*>& obs,
                                         double t0, double dt, int K = 10) {
  double mn = 1e18;
  for (int k = 0; k <= K; ++k) {
    double f = (K > 0 ? double(k) / K : 0.0);
    Eigen::Vector3d x = p + f * (q - p);
    for (auto* o : obs) mn = std::min(mn, o->signed_dist(x, t0 + f * dt));
  }
  return mn;
}

}  // namespace sando
