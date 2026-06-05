// recovery.hpp — the SAFE-half RECOVERY maneuver. When no feasible forward plan exists, the
// drone must NOT freeze on the stale plan (an approaching human would walk into it). Instead it
// actively YIELDS: among a small candidate set it picks the move that MAXIMISES min-clearance to
// the predicted obstacle set over the next horizon (reusing reach_avoid). Validated in
// cpp/test/bench_recovery.cpp (freeze min_human -0.29 -> yield +2.97/+3.70).
#pragma once
#include <vector>
#include <Eigen/Dense>
#include "sando_cpp/obstacles.hpp"
#include "sando_cpp/reach_avoid.hpp"

namespace sando {

// Returns the chosen yield target point (== p if staying is genuinely the safest).
// path_dir: current heading toward the goal (used for the back-off / lateral candidates).
inline Eigen::Vector3d yield_target(const Eigen::Vector3d& p, const Eigen::Vector3d& path_dir,
                                    const std::vector<const Obstacle*>& obs,
                                    double t0, double horizon, double step) {
  // away = +gradient of the nearest obstacle's signed distance (the outward/escape direction)
  Eigen::Vector3d away(0, 1, 0); double dn = 1e18;
  for (auto* o : obs) {
    double d = o->signed_dist(p, t0);
    if (d < dn) {
      dn = d; const double h = 1e-3;
      Eigen::Vector3d g(o->signed_dist(p + Eigen::Vector3d(h, 0, 0), t0) - d,
                        o->signed_dist(p + Eigen::Vector3d(0, h, 0), t0) - d,
                        o->signed_dist(p + Eigen::Vector3d(0, 0, h), t0) - d);
      g(2) = 0.0;                                  // horizontal yield only
      if (g.norm() > 1e-9) away = g;
    }
  }
  if (away.norm() > 1e-6) away.normalize(); else away = Eigen::Vector3d(0, 1, 0);
  Eigen::Vector3d fwd = path_dir; fwd(2) = 0;
  if (fwd.norm() > 1e-6) fwd.normalize(); else fwd = Eigen::Vector3d(1, 0, 0);
  Eigen::Vector3d lat(-fwd(1), fwd(0), 0.0);

  std::vector<Eigen::Vector3d> cand = {
      p,                       // stay (brake in place) — only wins if genuinely safest
      p + step * away,         // away from the nearest obstacle
      p - step * fwd,          // back off along the path
      p + step * lat,          // sidestep
      p - step * lat};
  double best = -1e18; Eigen::Vector3d q = p;
  for (const auto& c : cand) {
    double clr = reach_avoid_move_clearance(p, c, obs, t0, horizon);
    if (clr > best) { best = clr; q = c; }
  }
  return q;
}

}  // namespace sando
