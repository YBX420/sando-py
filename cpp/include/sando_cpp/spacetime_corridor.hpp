// spacetime_corridor.hpp — Safe-Interval SPACE-TIME safe flight corridor for dynamic obstacles.
//
// Per interior MINCO waypoint we carve an axis-aligned cuboid free of STATIC obstacles, and compute a
// collision-free TIME WINDOW [t_l, t_u] from the ANALYTIC predicted moving-obstacle positions
// (SphereObstacle::predict(t)) — exact for oscillating motion, unlike constant-velocity Minkowski
// inflation. The local solve is then kept inside cuboid_i while its time window is active (a CONVEX
// "stay-in-box" conditioner that makes the non-convex hard-ALM converge instead of stalling -> removes
// the dense-crowd stop-and-go). All time is TRAJECTORY-LOCAL (t=0 == current replan time), matching the
// cost-grad convention. Default OFF (use_spacetime_corridor) -> golden byte-identical.
#pragma once
#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include <cmath>
#include "sando_cpp/obstacles.hpp"

namespace sando {

struct SpaceTimeCuboid {
  Eigen::Vector3d lo = Eigen::Vector3d::Zero();
  Eigen::Vector3d hi = Eigen::Vector3d::Zero();
  double t_l = 0.0;          // collision-free LOCAL-time window (t=0 == current replan time)
  double t_u = 1.0e9;
  int    seg = -1;           // interior-waypoint index it gates (== q0 row)
  bool   valid = true;       // false -> honest-degrade (skip; hard-ALM alone there)
};

struct StcParams {
  double r_init = 0.8;       // initial half-extent (= minco_sfc_radius)
  double r_cap  = 1.6;       // max half-extent (= 2*sfc_radius)
  double step   = 0.15;      // face grow step (m)
  double drone_radius = 0.0; // min slack required around the centre
  double d_safe_dyn   = 0.5; // mover keep-out used for the time window
  double time_dt = 0.1;      // window sampling cadence (s)
  double Th      = 1.5;      // window horizon (s, local time)
};

// distance from point p to an AABB surface (0 if inside).
inline double aabb_point_dist(const Eigen::Vector3d& lo, const Eigen::Vector3d& hi,
                              const Eigen::Vector3d& p) {
  Eigen::Vector3d c = p.cwiseMax(lo).cwiseMin(hi);
  return (p - c).norm();
}

// Grow a symmetric box from centre c outward (per face, fixed step) until a face would overlap a STATIC
// AABB OR a MOVER's keep-out at the arrival time tau (predict(tau) +- radius + d_safe), or reaches r_cap.
// Growing against movers makes the box a TIGHT mover-free GAP at arrival (a huge static-only box is useless
// in a mostly-dynamic scene: it always contains a mover so its window collapses, and never constrains q).
inline SpaceTimeCuboid carve_cuboid(const Eigen::Vector3d& c,
                                    const std::vector<const Obstacle*>& static_obs,
                                    const std::vector<const Obstacle*>& movers,
                                    double tau, const StcParams& sp) {
  SpaceTimeCuboid cb;
  cb.lo = c.array() - sp.r_init;
  cb.hi = c.array() + sp.r_init;
  // precompute mover keep-out AABBs at arrival time tau (cull to those near c)
  std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> mboxes;
  for (const Obstacle* o : movers) {
    const SphereObstacle* s = dynamic_cast<const SphereObstacle*>(o);
    if (!s) continue;
    Eigen::Vector3d p = s->predict(tau);
    if ((p - c).norm() > sp.r_cap + s->radius + sp.d_safe_dyn + 1.0) continue;  // cull far movers
    double rr = s->radius + sp.d_safe_dyn;
    mboxes.emplace_back(p.array() - rr, p.array() + rr);
  }
  auto blocked = [&](const Eigen::Vector3d& lo, const Eigen::Vector3d& hi) -> bool {
    for (const Obstacle* o : static_obs) {
      const AABBObstacle* a = dynamic_cast<const AABBObstacle*>(o);
      if (!a) continue;
      if ((lo.array() <= a->hi.array()).all() && (hi.array() >= a->lo.array()).all()) return true;
    }
    for (const auto& m : mboxes)
      if ((lo.array() <= m.second.array()).all() && (hi.array() >= m.first.array()).all()) return true;
    return false;
  };
  if (blocked(cb.lo, cb.hi)) { cb.valid = false; return cb; }   // seed itself inside a keep-out -> degrade
  const double room = sp.r_cap - sp.r_init;
  for (int ax = 0; ax < 3; ++ax) {
    for (double g = 0.0; g < room; g += sp.step) {              // grow +face
      Eigen::Vector3d h2 = cb.hi; h2(ax) += sp.step;
      if (blocked(cb.lo, h2)) break;
      cb.hi = h2;
    }
    for (double g = 0.0; g < room; g += sp.step) {              // grow -face
      Eigen::Vector3d l2 = cb.lo; l2(ax) -= sp.step;
      if (blocked(l2, cb.hi)) break;
      cb.lo = l2;
    }
  }
  if (((c - cb.lo).minCoeff() < sp.drone_radius) || ((cb.hi - c).minCoeff() < sp.drone_radius))
    cb.valid = false;        // no room for the body here -> skip (hard-ALM handles it)
  return cb;
}

// Maximal mover-free LOCAL-time interval [t_l, t_u] around the nominal arrival tau_star, using the
// ANALYTIC predicted mover positions (exact for oscillation). If occupied AT tau_star -> invalid.
inline void compute_window(SpaceTimeCuboid& cb, double tau_star,
                           const std::vector<const Obstacle*>& movers, const StcParams& sp) {
  if (!cb.valid) return;
  auto occupied = [&](double t) -> bool {
    for (const Obstacle* m : movers) {
      const SphereObstacle* s = dynamic_cast<const SphereObstacle*>(m);
      if (!s) continue;
      if (aabb_point_dist(cb.lo, cb.hi, s->predict(t)) < s->radius + sp.d_safe_dyn) return true;
    }
    return false;
  };
  double tau = std::min(std::max(tau_star, 0.0), sp.Th);
  if (occupied(tau)) { cb.valid = false; return; }             // the gap is shut at arrival -> degrade
  double tl = 0.0, tu = sp.Th;
  for (double t = tau; t >= 0.0;   t -= sp.time_dt) if (occupied(std::max(t, 0.0)))   { tl = t; break; }
  for (double t = tau; t <= sp.Th; t += sp.time_dt) if (occupied(std::min(t, sp.Th))) { tu = t; break; }
  cb.t_l = tl; cb.t_u = tu;
}

// smooth activeness gate: 1 inside [t_l,t_u], linear ramp over margin m, 0 outside. Stop-gradient on T.
inline double window_gate(double t, double t_l, double t_u, double m = 0.15) {
  if (t <= t_l - m || t >= t_u + m) return 0.0;
  if (t >= t_l && t <= t_u) return 1.0;
  return (t < t_l) ? (t - (t_l - m)) / m : ((t_u + m) - t) / m;
}

// Build one cuboid per interior waypoint q0 (rows 0..M-2). tau0[i] = seed cumulative time at waypoint i.
inline std::vector<SpaceTimeCuboid> build_corridor(
    const Eigen::MatrixXd& q0, const std::vector<double>& tau0,
    const std::vector<const Obstacle*>& static_obs,
    const std::vector<const Obstacle*>& movers, const StcParams& sp) {
  std::vector<SpaceTimeCuboid> out;
  out.reserve(static_cast<size_t>(q0.rows()));
  for (int i = 0; i < q0.rows(); ++i) {
    double tau = (i < static_cast<int>(tau0.size()) ? tau0[i] : 0.0);
    SpaceTimeCuboid cb = carve_cuboid(q0.row(i).transpose(), static_obs, movers, tau, sp);
    cb.seg = i;
    compute_window(cb, tau, movers, sp);
    out.push_back(cb);
  }
  return out;
}

}  // namespace sando
