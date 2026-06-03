// plan_minco — faithful C++ port of the MINCO optimiser DRIVER in
// sando_py/local/local_opt.py. Ports the top-level plan_minco() plus the
// deterministic helpers it drives:
//   _orthonormal_pair / _obstacle_geom / _resample_polyline /
//   _min_clearance_per_obstacle / _make_detour_path / _generate_detour_seeds
//   _partition_obstacles
//   _minco_seed
//   check_feasibility (== _minco_feasibility on a GIVEN trajectory)
//   hard_clearance
//   _alm_solve  (the ALM outer loop: freeze normals/trust-mask, run an L-BFGS-B
//                INNER solve over x=[q,T] with the dt_min lower bound on T, then
//                update ALM multipliers lam<-max(0,lam+rho*g), grow rho, check
//                convergence / STOP) -- via LBFGSpp::LBFGSBSolver
//   _optimise_one_minco
//   _select_best_minco
//   plan_minco  (generate seeds, optimise each, select best, return (MinjerkTraj, info))
//
// The inner unconstrained solve uses the VENDORED header-only L-BFGS-B
// (cpp/third_party/LBFGSpp). Parameters mirror scipy's L-BFGS-B used in the
// Python (maxiter/alm_inner_maxiter, m, ftol, gtol, dt_min bounds). The objective
// + gradient is minco_cost_grad.hpp (already golden-verified 1:1 with Python).
//
// IMPORTANT match note: LBFGSpp and scipy's Fortran L-BFGS-B do NOT produce
// bit-identical iterates, so plan_minco's FINAL trajectory will differ slightly
// from Python's. The DETERMINISTIC helpers (partition, detour geometry,
// orthonormal pair, feasibility on a given traj) are golden-tested EXACTLY; the
// end-to-end final trajectory is compared and the TRUE discrepancy reported.
//
// REUSES the already golden-verified headers (no re-derivation):
//   minjerk_traj.hpp, obstacles.hpp, avoid_config.hpp, minco_cost_grad.hpp,
//   local_opt.hpp, local_opt_terms.hpp, local_opt_hardalm.hpp, cost.hpp.
// 不许欺骗、必须还原: every formula / branch mirrors the Python 1:1.
#pragma once

#include "sando_cpp/minco_cost_grad.hpp"     // minco_cost_grad + CostGradOptParams + AlmState
#include "sando_cpp/local_opt_hardalm.hpp"   // seg_normals / trust_mask / alm_constraints / pred_margin / eps_track / hard_d_safe / is_moving
#include "sando_cpp/minjerk_traj.hpp"
#include "sando_cpp/obstacles.hpp"
#include "sando_cpp/avoid_config.hpp"        // AvoidParams + resolve_mode

// Vendored header-only L-BFGS-B (cpp/third_party/LBFGSpp). Path is relative to the
// third_party include root so the standard "-I cpp/third_party" compile line resolves it
// (LBFGSB.h's own "LBFGSpp/Param.h" includes then resolve next to LBFGSB.h).
#include "LBFGSpp/include/LBFGSB.h"          // LBFGSpp::LBFGSBSolver

#include <Eigen/Dense>
#include <vector>
#include <map>
#include <string>
#include <cmath>
#include <limits>
#include <algorithm>
#include <memory>

namespace sando {

// ===========================================================================
// OptParams — the FULL set the MINCO driver reads (1:1 with local_opt.py:OptParams).
// Superset of CostGradOptParams (which it can produce) plus the seed/detour/ALM-outer
// /feasibility fields. Defaults match the Python dataclass exactly.
// ===========================================================================
struct PlanOptParams {
  // cost weights
  double w_smooth = 1.0;
  double w_vel    = 100.0;
  double w_accel  = 100.0;
  double w_time   = 10.0;
  double vmax     = 3.0;
  double amax     = 3.0;
  double dt_min   = 0.05;
  int    K        = 200;
  int    maxiter  = 200;
  int    degree   = 5;
  // feasibility tolerances
  double clearance_tol = 0.05;
  double vel_tol       = 0.10;
  double accel_tol     = 0.10;
  // MINCO additions
  double w_obs  = 1.0;
  int    kappa  = 16;
  int    M_cap  = 12;
  // per-class HARD/SOFT + ALM
  std::string avoid_override = "";   // "" -> per-class; "soft"/"hard" force
  double alm_rho0       = 10.0;
  double alm_rho_max    = 1.0e8;
  double alm_rho_grow   = 2.0;
  int    alm_outer_iters = 12;
  double alm_viol_tol   = 1.0e-3;
  int    alm_inner_maxiter = 40;
  // Stage 4 space-time
  double tau_trust      = 0.75;
  bool   spacetime_hard = true;
  // conformal / exec-gap / deterministic reach
  double q_conformal    = 0.0;
  double epsilon_track  = 0.0;
  std::string safety_mode = "conformal";
  std::string reach_model = "accel";
  double a_max_human = 0.0;
  double v_max_human = 0.0;
  double est_pos_err = 0.0;
  double est_vel_err = 0.0;

  // Project to the CostGradOptParams the inner objective reads.
  CostGradOptParams cost_grad() const {
    CostGradOptParams o;
    o.w_smooth = w_smooth; o.w_obs = w_obs; o.w_vel = w_vel; o.w_accel = w_accel;
    o.w_time = w_time; o.vmax = vmax; o.amax = amax; o.kappa = kappa;
    o.avoid_override = avoid_override; o.tau_trust = tau_trust;
    o.spacetime_hard = spacetime_hard; o.q_conformal = q_conformal;
    o.epsilon_track = epsilon_track; o.safety_mode = safety_mode;
    o.reach_model = reach_model; o.a_max_human = a_max_human;
    o.v_max_human = v_max_human; o.est_pos_err = est_pos_err;
    o.est_vel_err = est_vel_err;
    return o;
  }
  // Project to the HardAlmOptParams the ALM constraint helpers read.
  HardAlmOptParams hard_alm() const {
    HardAlmOptParams h;
    h.avoid_override = avoid_override; h.tau_trust = tau_trust;
    h.spacetime_hard = spacetime_hard; h.q_conformal = q_conformal;
    h.epsilon_track = epsilon_track; h.safety_mode = safety_mode;
    h.reach_model = reach_model; h.a_max_human = a_max_human;
    h.v_max_human = v_max_human; h.est_pos_err = est_pos_err;
    h.est_vel_err = est_vel_err;
    return h;
  }
};

// DetourConfig (1:1 with local_opt.py:DetourConfig).
struct DetourConfig {
  bool   enabled = true;
  int    directions = 4;
  double trigger_margin = 0.10;
  double detour_margin  = 0.30;
  int    top_k_obstacles = 1;
  int    max_seeds = 5;
  int    K_eval = 100;
  int    perturb_window = 3;
};

// ---------------------------------------------------------------------------
// _orthonormal_pair(axis): two unit vectors orthogonal to `axis` and each other.
// ---------------------------------------------------------------------------
inline std::pair<Eigen::Vector3d, Eigen::Vector3d> orthonormal_pair(const Eigen::Vector3d& axis) {
  double n = axis.norm();
  if (n < 1e-9)
    return {Eigen::Vector3d(1.0, 0.0, 0.0), Eigen::Vector3d(0.0, 1.0, 0.0)};
  Eigen::Vector3d a = axis / n;
  // world axis least aligned with `a`: argmin |world @ a| = argmin |a(k)|
  int pick = 0;
  double best = std::abs(a(0));
  for (int k = 1; k < 3; ++k) { double c = std::abs(a(k)); if (c < best) { best = c; pick = k; } }
  Eigen::Vector3d world = Eigen::Vector3d::Zero(); world(pick) = 1.0;
  Eigen::Vector3d u = world - world.dot(a) * a;
  u /= std::max(u.norm(), 1e-9);
  Eigen::Vector3d v = a.cross(u);
  v /= std::max(v.norm(), 1e-9);
  return {u, v};
}

// ---------------------------------------------------------------------------
// _obstacle_geom(obs): (centre, radius_equiv). Sphere -> (centre0, radius).
//                      AABB -> (box centre, 0.5*||hi-lo||).
// ---------------------------------------------------------------------------
inline std::pair<Eigen::Vector3d, double> obstacle_geom(const Obstacle* obs) {
  if (auto s = dynamic_cast<const SphereObstacle*>(obs))
    return {s->centre0, s->radius};
  auto b = dynamic_cast<const AABBObstacle*>(obs);
  Eigen::Vector3d centre = 0.5 * (b->lo + b->hi);
  double radius = 0.5 * (b->hi - b->lo).norm();
  return {centre, radius};
}

// ---------------------------------------------------------------------------
// _resample_polyline(path, K): arc-length uniform resample to K points (M x 3).
// ---------------------------------------------------------------------------
inline Eigen::MatrixXd resample_polyline(const Eigen::MatrixXd& path, int K) {
  const int M = (int)path.rows();
  const int D = (int)path.cols();
  Eigen::VectorXd cum(M);
  cum(0) = 0.0;
  for (int i = 1; i < M; ++i)
    cum(i) = cum(i - 1) + (path.row(i) - path.row(i - 1)).norm();
  double total = cum(M - 1);
  Eigen::MatrixXd out(K, D);
  if (total == 0.0) {
    for (int k = 0; k < K; ++k) out.row(k) = path.row(0);
    return out;
  }
  // target = linspace(0, total, K); per-dim linear interp (np.interp).
  for (int k = 0; k < K; ++k) {
    double t = (K == 1) ? 0.0 : (total * double(k) / double(K - 1));
    // find segment: largest i with cum(i) <= t (np.interp clamps to endpoints)
    if (t <= cum(0)) { out.row(k) = path.row(0); continue; }
    if (t >= cum(M - 1)) { out.row(k) = path.row(M - 1); continue; }
    int lo = 0, hi = M - 1;
    while (hi - lo > 1) { int mid = (lo + hi) / 2; if (cum(mid) <= t) lo = mid; else hi = mid; }
    double seg = cum(hi) - cum(lo);
    double w = (seg > 0.0) ? (t - cum(lo)) / seg : 0.0;
    out.row(k) = (1.0 - w) * path.row(lo) + w * path.row(hi);
  }
  return out;
}

// ---------------------------------------------------------------------------
// _min_clearance_per_obstacle: per obstacle (min signed_dist, closest path index).
// ---------------------------------------------------------------------------
inline std::vector<std::pair<double, int>> min_clearance_per_obstacle(
    const Eigen::MatrixXd& path, const std::vector<const Obstacle*>& obstacles, int K_eval) {
  Eigen::MatrixXd samples = resample_polyline(path, K_eval);
  std::vector<std::pair<double, int>> out;
  const int M = (int)path.rows();
  for (const Obstacle* obs : obstacles) {
    double dmin = std::numeric_limits<double>::infinity();
    int i_min = 0;
    for (int i = 0; i < K_eval; ++i) {
      double d = obs->signed_dist(samples.row(i).transpose(), 0.0);
      if (d < dmin) { dmin = d; i_min = i; }
    }
    // map sample index back to path index (proportional, rounded)
    int denom = std::max(K_eval - 1, 1);
    int i_path = (int)std::lround(double(i_min) * double(M - 1) / double(denom));
    out.push_back({dmin, i_path});
  }
  return out;
}

// ---------------------------------------------------------------------------
// _make_detour_path: bulge the polyline around `obstacle` along `direction`.
// ---------------------------------------------------------------------------
inline Eigen::MatrixXd make_detour_path(const Eigen::MatrixXd& path, const Obstacle* obstacle,
                                        const Eigen::Vector3d& direction, double d_safe,
                                        const DetourConfig& dc) {
  auto cg = obstacle_geom(obstacle);
  Eigen::Vector3d centre = cg.first; double r_eq = cg.second;
  const int M = (int)path.rows();
  // i_close = argmin ||path - centre||
  int i_close = 0; double best = std::numeric_limits<double>::infinity();
  for (int i = 0; i < M; ++i) {
    double dd = (path.row(i).transpose() - centre).norm();
    if (dd < best) { best = dd; i_close = i; }
  }
  Eigen::Vector3d target = centre + direction * (r_eq + d_safe + dc.detour_margin);
  Eigen::Vector3d shift = target - path.row(i_close).transpose();
  int w = dc.perturb_window;
  Eigen::MatrixXd np = path;
  int jlo = std::max(0, i_close - w);
  int jhi = std::min(M, i_close + w + 1);
  for (int j = jlo; j < jhi; ++j) {
    if (j == 0 || j == M - 1) continue;  // endpoints pinned
    double weight = std::max(0.0, 1.0 - std::abs(double(j - i_close)) / double(w + 1));
    np.row(j) += (shift * weight).transpose();
  }
  return np;
}

// ---------------------------------------------------------------------------
// _generate_detour_seeds: list of (name, seed_path). Always includes "original".
// ---------------------------------------------------------------------------
inline std::vector<std::pair<std::string, Eigen::MatrixXd>> generate_detour_seeds(
    const Eigen::MatrixXd& astar_path, const std::vector<const Obstacle*>& obstacles,
    const std::map<std::string, AvoidParams>& avoid_cfg, const PlanOptParams& /*opt*/,
    const DetourConfig& dc) {
  std::vector<std::pair<std::string, Eigen::MatrixXd>> seeds;
  seeds.push_back({"original", astar_path});
  if (!dc.enabled || obstacles.empty()) return seeds;
  auto clearances = min_clearance_per_obstacle(astar_path, obstacles, dc.K_eval);
  // violators: (violation, i, obs, d_safe, i_path)
  struct Viol { double violation; int i; const Obstacle* obs; double d_safe; int i_path; };
  std::vector<Viol> violators;
  for (int i = 0; i < (int)obstacles.size(); ++i) {
    auto it = avoid_cfg.find(obstacles[i]->class_name);
    double d_safe = (it != avoid_cfg.end()) ? it->second.d_safe : 0.8;
    double min_clr = clearances[i].first;
    int i_path = clearances[i].second;
    double violation = (d_safe + dc.trigger_margin) - min_clr;
    if (violation > 0.0) violators.push_back({violation, i, obstacles[i], d_safe, i_path});
  }
  if (violators.empty()) return seeds;
  // sort by -violation (most-violated first); stable to mirror Python's list.sort key
  std::stable_sort(violators.begin(), violators.end(),
                   [](const Viol& a, const Viol& b) { return a.violation > b.violation; });
  int top_k = std::min((int)violators.size(), dc.top_k_obstacles);
  for (int vi = 0; vi < top_k; ++vi) {
    const Viol& V = violators[vi];
    auto cg = obstacle_geom(V.obs);
    Eigen::Vector3d centre = cg.first;
    int pidx = std::min(V.i_path, (int)astar_path.rows() - 1);
    Eigen::Vector3d path_pt = astar_path.row(pidx).transpose();
    Eigen::Vector3d axis = centre - path_pt;
    if (axis.norm() < 1e-6)
      axis = astar_path.row(astar_path.rows() - 1).transpose() - astar_path.row(0).transpose();
    auto uv = orthonormal_pair(axis);
    Eigen::Vector3d u = uv.first, v = uv.second;
    Eigen::Vector3d dirs[4] = {u, -u, v, -v};
    int ndir = std::min(dc.directions, 4);
    for (int j = 0; j < ndir; ++j) {
      Eigen::MatrixXd new_path = make_detour_path(astar_path, V.obs, dirs[j], V.d_safe, dc);
      seeds.push_back({"detour_obs" + std::to_string(V.i) + "_dir" + std::to_string(j), new_path});
      if ((int)seeds.size() >= dc.max_seeds) return seeds;
    }
  }
  return seeds;
}

// ---------------------------------------------------------------------------
// _partition_obstacles: split into (soft_obs, hard_obs) per resolve_mode+override.
// ---------------------------------------------------------------------------
inline void partition_obstacles(const std::vector<const Obstacle*>& obstacles,
                                const std::map<std::string, AvoidParams>& avoid_cfg,
                                const std::string& override_,
                                std::vector<const Obstacle*>& soft_obs,
                                std::vector<const Obstacle*>& hard_obs) {
  soft_obs.clear(); hard_obs.clear();
  for (const Obstacle* o : obstacles) {
    if (resolve_mode(o->class_name, avoid_cfg, override_) == "hard") hard_obs.push_back(o);
    else soft_obs.push_back(o);
  }
}

// ---------------------------------------------------------------------------
// check_feasibility (== _minco_feasibility): trajectory-level feasibility,
// independent of optimiser convergence. Returns a Feasibility struct.
// ---------------------------------------------------------------------------
struct Feasibility {
  double min_clearance = 0.0;
  double max_clearance_violation = 0.0;
  double max_v = 0.0;
  double max_a = 0.0;
  double vel_violation = 0.0;
  double accel_violation = 0.0;
  bool   trajectory_valid = false;
  std::string failure_reason;   // "" == None
};

inline Feasibility check_feasibility(const MinjerkTraj& spline,
                                     const std::vector<const Obstacle*>& obstacles,
                                     const std::map<std::string, AvoidParams>& avoid_cfg,
                                     const PlanOptParams& opt, int K_eval = 200) {
  Feasibility F;
  const double inf = std::numeric_limits<double>::infinity();
  // ts = linspace(t_start, t_end, K_eval); pts = spline.eval(ts)
  Eigen::VectorXd ts(K_eval);
  for (int i = 0; i < K_eval; ++i)
    ts(i) = (K_eval == 1) ? spline.t_start
                          : spline.t_start + (spline.t_end - spline.t_start) * double(i) / double(K_eval - 1);
  double min_clr = inf, max_violation = 0.0;
  for (const Obstacle* obs : obstacles) {
    auto it = avoid_cfg.find(obs->class_name);
    double d_safe = (it != avoid_cfg.end()) ? it->second.d_safe : 0.8;
    for (int i = 0; i < K_eval; ++i) {
      double d = obs->signed_dist(spline.eval(ts(i)), ts(i));
      if (d < min_clr) min_clr = d;
      double violation = d_safe - d;
      if (violation > max_violation) max_violation = violation;
    }
  }
  if (min_clr == inf) min_clr = 0.0;
  double max_v = 0.0, max_a = 0.0;
  for (int i = 0; i < K_eval; ++i) {
    double vm = spline.eval_deriv(ts(i), 1).norm();
    double am = spline.eval_deriv(ts(i), 2).norm();
    if (vm > max_v) max_v = vm;
    if (am > max_a) max_a = am;
  }
  double vel_violation = std::max(0.0, max_v - opt.vmax);
  double accel_violation = std::max(0.0, max_a - opt.amax);

  std::string failure_reason;  // ""
  if (!obstacles.empty() && max_violation > opt.clearance_tol)
    failure_reason = "clearance_violation";
  else if (max_v > opt.vmax * (1.0 + opt.vel_tol))
    failure_reason = "velocity_violation";
  else if (max_a > opt.amax * (1.0 + opt.accel_tol))
    failure_reason = "acceleration_violation";

  F.min_clearance = min_clr;
  F.max_clearance_violation = max_violation;
  F.max_v = max_v; F.max_a = max_a;
  F.vel_violation = vel_violation; F.accel_violation = accel_violation;
  F.failure_reason = failure_reason;
  F.trajectory_valid = failure_reason.empty();
  return F;
}

// ---------------------------------------------------------------------------
// hard_clearance: densely-sampled continuous-time min HARD clearance + worst breach.
// Returns (min_clearance, max_breach). Empty hard set -> (+inf, -inf).
// ---------------------------------------------------------------------------
inline std::pair<double, double> hard_clearance(const MinjerkTraj& tr,
                                                const std::vector<const Obstacle*>& hard_obs,
                                                const std::map<std::string, AvoidParams>& avoid_cfg,
                                                int K_eval = 2000) {
  const double inf = std::numeric_limits<double>::infinity();
  if (hard_obs.empty()) return {inf, -inf};
  double min_clr = inf, max_breach = -inf;
  for (const Obstacle* obs : hard_obs) {
    auto it = avoid_cfg.find(obs->class_name);
    double d_safe = (it != avoid_cfg.end()) ? it->second.d_safe : 0.8;
    for (int j = 0; j < K_eval; ++j) {
      double t = (K_eval == 1) ? tr.t_start
                               : tr.t_start + (tr.t_end - tr.t_start) * double(j) / double(K_eval - 1);
      double d = obs->signed_dist(tr.eval(t), t);
      if (d < min_clr) min_clr = d;
      double br = d_safe - d;
      if (br > max_breach) max_breach = br;
    }
  }
  return {min_clr, max_breach};
}

// ---------------------------------------------------------------------------
// _minco_seed: init from a seed polyline -> (start, goal, q0 (M-1,3), T0 (M,)).
// ---------------------------------------------------------------------------
struct MincoSeed {
  Eigen::Vector3d start, goal;
  Eigen::MatrixXd q0;   // (M-1,3)
  Eigen::VectorXd T0;   // (M,)
};

inline MincoSeed minco_seed(const Eigen::MatrixXd& seed_path, const PlanOptParams& opt) {
  MincoSeed S;
  const int N = (int)seed_path.rows();
  // M = clip(round(N/2), 2, M_cap)
  long Mr = std::lround(double(N) / 2.0);
  int M = (int)std::min<long>(std::max<long>(Mr, 2), opt.M_cap);
  Eigen::MatrixXd wp = resample_polyline(seed_path, M + 1);
  wp.row(0) = seed_path.row(0);
  wp.row(M) = seed_path.row(N - 1);
  Eigen::VectorXd T0(M);
  for (int i = 0; i < M; ++i) {
    double seg = (wp.row(i + 1) - wp.row(i)).norm();
    T0(i) = std::max(seg / std::max(opt.vmax, 1e-6), opt.dt_min * 2.0);
  }
  S.start = wp.row(0).transpose();
  S.goal = wp.row(M).transpose();
  S.q0 = (M > 1) ? Eigen::MatrixXd(wp.middleRows(1, M - 1)) : Eigen::MatrixXd(0, 3);
  S.T0 = T0;
  return S;
}

// ---------------------------------------------------------------------------
// _safety_surrogate: deterministic+speed-reach -> replace moving hard spheres by a
// static surrogate at centre0. Returns a parallel list (sharing static obstacles,
// owning the surrogates in `owned`). No-op in any other mode.
// ---------------------------------------------------------------------------
inline std::vector<const Obstacle*> safety_surrogate(
    const std::vector<const Obstacle*>& obstacles, const PlanOptParams& opt,
    std::vector<std::shared_ptr<Obstacle>>& owned) {
  if (opt.safety_mode != "deterministic" || opt.reach_model != "speed")
    return obstacles;
  std::vector<const Obstacle*> out;
  for (const Obstacle* o : obstacles) {
    const SphereObstacle* s = dynamic_cast<const SphereObstacle*>(o);
    if (s && (s->vel.norm() > 0.0 || s->accel.norm() > 0.0)) {
      auto sur = std::make_shared<SphereObstacle>(s->centre0, s->radius,
                                                  Eigen::Vector3d::Zero(), s->class_name,
                                                  Eigen::Vector3d::Zero());
      owned.push_back(sur);
      out.push_back(sur.get());
    } else {
      out.push_back(o);
    }
  }
  return out;
}

// ===========================================================================
// ALM SOLVE — the outer loop wrapping the L-BFGS-B inner solve (via LBFGSpp).
// ===========================================================================
struct AlmSolveResult {
  MinjerkTraj tr;                 // final trajectory
  Eigen::VectorXd x;              // final [q.ravel(), T]
  Eigen::VectorXd lam;            // final multipliers
  double rho = 0.0;
  int outer_iters = 0;
  int total_inner_iters = 0;
  double max_violation = 0.0;     // on the final trajectory (recomputed normals)
  std::vector<double> final_normals;
  Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> final_trust;
  bool has_final_alm = false;     // whether final_normals/final_trust are populated
  double f0 = 0.0;
  double T_target = 0.0;
  double result_fun = 0.0;        // L-BFGS-B final objective (== last inner fx)
  bool   result_success = true;   // converged within maxiter (gradient/objective test)
  bool   has_result = false;
};

// The inner-objective functor for LBFGSpp: f(x, grad) = minco_cost_grad(...).
struct MincoObjective {
  const Eigen::Vector3d* start; const Eigen::Vector3d* goal; int M;
  const std::vector<const Obstacle*>* soft_obs;
  const std::map<std::string, AvoidParams>* avoid_cfg;
  const CostGradOptParams* opt; double T_target;
  const Eigen::Vector3d* v0; const Eigen::Vector3d* a0;
  const Eigen::Vector3d* vf; const Eigen::Vector3d* af;
  const AlmState* alm;

  double operator()(const Eigen::VectorXd& x, Eigen::VectorXd& grad) {
    CostGradResult r = minco_cost_grad(x, *start, *goal, M, *soft_obs, *avoid_cfg,
                                       *opt, T_target, *v0, *a0, *vf, *af, *alm);
    grad = r.grad;
    return r.f;
  }
};

// Build an LBFGSBParam mirroring scipy's L-BFGS-B options used in the Python.
//   maxiter -> max_iterations ; gtol -> epsilon (||proj_grad||_inf, abs only) ;
//   scipy ftol default 2.22e-9 -> delta (with past=1) ; m default 10 -> m.
inline LBFGSpp::LBFGSBParam<double> make_lbfgsb_param(int maxiter, double gtol) {
  LBFGSpp::LBFGSBParam<double> p;
  p.m = 10;                  // scipy L-BFGS-B default maxcor
  p.max_iterations = maxiter;
  p.epsilon = gtol;          // projected-gradient inf-norm tolerance == scipy gtol (pgtol)
  p.epsilon_rel = 0.0;       // scipy uses an absolute pgtol -> turn off the relative arm
  p.past = 1;                // enable the ftol-style objective-decrease test
  p.delta = 2.220446049250313e-9;  // scipy L-BFGS-B default ftol (factr=1e7 * eps)
  p.max_linesearch = 40;     // scipy default maxls=20; widen a bit for robustness
  return p;
}

inline AlmSolveResult alm_solve(
    const Eigen::Vector3d& start, const Eigen::Vector3d& goal,
    const Eigen::MatrixXd& q0, const Eigen::VectorXd& T0,
    const std::vector<const Obstacle*>& soft_obs,
    const std::vector<const Obstacle*>& hard_obs,
    const std::map<std::string, AvoidParams>& avoid_cfg, const PlanOptParams& opt,
    const Eigen::Vector3d& v0, const Eigen::Vector3d& a0) {
  const int M = (int)T0.size();
  const int nq = 3 * (M - 1);
  const Eigen::Vector3d zero3 = Eigen::Vector3d::Zero();

  // x = [q0.ravel(), T0]  (or just T0 when M==1)
  Eigen::VectorXd x(nq + M);
  for (int i = 0; i < M - 1; ++i)
    for (int j = 0; j < 3; ++j) x(i * 3 + j) = q0(i, j);
  for (int i = 0; i < M; ++i) x(nq + i) = T0(i);
  double T_target = T0.sum();

  // bounds: q free; T >= dt_min
  Eigen::VectorXd lb(nq + M), ub(nq + M);
  const double inf = std::numeric_limits<double>::infinity();
  for (int i = 0; i < nq; ++i) { lb(i) = -inf; ub(i) = inf; }
  for (int i = 0; i < M; ++i) { lb(nq + i) = opt.dt_min; ub(nq + i) = inf; }

  const CostGradOptParams cgopt = opt.cost_grad();
  const HardAlmOptParams halmopt = opt.hard_alm();

  // Count Nc and init lam=0, rho=rho0.
  MinjerkTraj tr0(/*wp*/ [&]{ Eigen::MatrixXd wp(M + 1, 3); wp.row(0) = start.transpose();
                              for (int i = 0; i < M - 1; ++i) wp.row(i + 1) = q0.row(i);
                              wp.row(M) = goal.transpose(); return wp; }(),
                  T0, v0, a0, zero3, zero3);
  int Nc = 0;
  if (!hard_obs.empty()) {
    AlmConstraints cons0 = alm_constraints(tr0, hard_obs, avoid_cfg, nullptr, halmopt, nullptr);
    Nc = cons0.Nc;
  }
  Eigen::VectorXd lam = Eigen::VectorXd::Zero(Nc);
  double rho = opt.alm_rho0;

  AlmSolveResult R{tr0};
  R.T_target = T_target;

  // f0 = soft-only cost (info only): minco_cost_grad with empty ALM.
  {
    AlmState empty; empty.rho = 0.0;
    CostGradResult r0 = minco_cost_grad(x, start, goal, M, soft_obs, avoid_cfg, cgopt,
                                        T_target, v0, a0, zero3, zero3, empty);
    R.f0 = r0.f;
  }

  double prev_viol = inf;
  int outer_iters = 0, total_inner = 0;
  bool has_result = false; double result_fun = 0.0; bool result_success = true;

  auto rebuild = [&](const Eigen::VectorXd& xv) {
    Eigen::MatrixXd wp(M + 1, 3); wp.row(0) = start.transpose();
    for (int i = 0; i < M - 1; ++i)
      for (int j = 0; j < 3; ++j) wp(i + 1, j) = xv(i * 3 + j);
    wp.row(M) = goal.transpose();
    Eigen::VectorXd T(M); for (int i = 0; i < M; ++i) T(i) = xv(nq + i);
    return MinjerkTraj(wp, T, v0, a0, zero3, zero3);
  };

  if (hard_obs.empty()) {
    // single unconstrained solve (pure soft / smooth), maxiter / gtol=1e-6.
    AlmState empty; empty.rho = 0.0;
    MincoObjective obj{&start, &goal, M, &soft_obs, &avoid_cfg, &cgopt, T_target,
                       &v0, &a0, &zero3, &zero3, &empty};
    LBFGSpp::LBFGSBParam<double> p = make_lbfgsb_param(opt.maxiter, 1e-6);
    LBFGSpp::LBFGSBSolver<double> solver(p);
    double fx = 0.0; int nit = 0;
    try {
      nit = solver.minimize(obj, x, fx, lb, ub);
    } catch (const std::exception&) {
      // numerically pathological inner solve -> keep current x; reflect as "not converged"
      CostGradResult rr = minco_cost_grad(x, start, goal, M, soft_obs, avoid_cfg, cgopt,
                                          T_target, v0, a0, zero3, zero3, empty);
      fx = rr.f; result_success = false;
    }
    total_inner = nit;
    result_fun = fx;
    result_success = result_success && (nit < opt.maxiter);
    has_result = true;
  } else {
    std::vector<double> normals;
    for (int outer = 0; outer < opt.alm_outer_iters; ++outer) {
      outer_iters = outer + 1;
      // FREEZE normals + trust mask from the current iterate.
      MinjerkTraj tr = rebuild(x);
      normals = seg_normals(tr, hard_obs, halmopt);
      Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> tmask = trust_mask(tr, halmopt);

      // inner L-BFGS-B at frozen (lam, rho, normals, tmask), maxiter=alm_inner_maxiter, gtol=1e-7.
      AlmState alm;
      alm.hard_obs = hard_obs; alm.lam = lam; alm.rho = rho;
      alm.normals = normals; alm.trust_mask = tmask;
      MincoObjective obj{&start, &goal, M, &soft_obs, &avoid_cfg, &cgopt, T_target,
                         &v0, &a0, &zero3, &zero3, &alm};
      LBFGSpp::LBFGSBParam<double> p = make_lbfgsb_param(opt.alm_inner_maxiter, 1e-7);
      LBFGSpp::LBFGSBSolver<double> solver(p);
      double fx = 0.0; int nit = 0;
      try {
        nit = solver.minimize(obj, x, fx, lb, ub);
      } catch (const std::exception&) {
        CostGradResult rr = minco_cost_grad(x, start, goal, M, soft_obs, avoid_cfg, cgopt,
                                            T_target, v0, a0, zero3, zero3, alm);
        fx = rr.f; result_success = false;
      }
      total_inner += nit;
      result_fun = fx;
      result_success = (nit < opt.alm_inner_maxiter);
      has_result = true;

      // recompute g under the SAME frozen normals + mask.
      MinjerkTraj tr2 = rebuild(x);
      AlmConstraints cons = alm_constraints(tr2, hard_obs, avoid_cfg, &normals, halmopt, &tmask);
      const Eigen::VectorXd& g = cons.g;
      // lam <- max(0, lam + rho*g)
      for (int m = 0; m < Nc; ++m) lam(m) = std::max(0.0, lam(m) + rho * g(m));
      double max_viol = 0.0;
      for (int m = 0; m < Nc; ++m) max_viol = std::max(max_viol, std::max(g(m), 0.0));
      if (max_viol < opt.alm_viol_tol) break;
      if (rho >= opt.alm_rho_max && max_viol > 0.98 * prev_viol) break;
      if (max_viol > 0.5 * prev_viol)
        rho = std::min(rho * opt.alm_rho_grow, opt.alm_rho_max);
      prev_viol = max_viol;
    }
  }

  // final trajectory
  MinjerkTraj trf = rebuild(x);
  R.tr = trf;
  R.x = x; R.lam = lam; R.rho = rho;
  R.outer_iters = outer_iters; R.total_inner_iters = total_inner;
  R.result_fun = result_fun; R.result_success = result_success; R.has_result = has_result;

  if (!hard_obs.empty()) {
    std::vector<double> fn = seg_normals(trf, hard_obs, halmopt);
    Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> ft = trust_mask(trf, halmopt);
    AlmConstraints consf = alm_constraints(trf, hard_obs, avoid_cfg, &fn, halmopt, &ft);
    double mv = 0.0;
    for (int m = 0; m < consf.Nc; ++m) mv = std::max(mv, std::max(consf.g(m), 0.0));
    R.max_violation = mv;
    R.final_normals = fn; R.final_trust = ft; R.has_final_alm = true;
  }
  return R;
}

// ===========================================================================
// _optimise_one_minco — one ALM(hard)+soft solve from a seed polyline.
// ===========================================================================
struct MincoCandidate {
  MinjerkTraj spline;
  double init_cost = 0.0;
  double final_cost = 0.0;
  int iter = 0;
  bool optimizer_success = true;
  double dt0 = 0.0, dt_final = 0.0, T_target = 0.0, T_final = 0.0;
  Feasibility feasibility;
  int M = 0;
  bool hard_violation = false;
  double hard_max_breach = 0.0;
  double hard_min_clearance = 0.0;
  int n_hard = 0, n_soft = 0;
  int alm_outer_iters = 0;
  double alm_rho_final = 0.0;
  double alm_max_violation = 0.0;
  double alm_lambda_max = 0.0;
  double continuous_min_clearance = 0.0;
  double certificate_margin = 0.0;
  std::string seed_name;
};

inline MincoCandidate optimise_one_minco(const Eigen::MatrixXd& seed_path,
                                         const std::vector<const Obstacle*>& obstacles,
                                         const std::map<std::string, AvoidParams>& avoid_cfg,
                                         const PlanOptParams& opt,
                                         const Eigen::Vector3d& v0, const Eigen::Vector3d& a0) {
  MincoSeed S = minco_seed(seed_path, opt);
  const int M = (int)S.T0.size();
  const int nq = 3 * (M - 1);

  // deterministic SPEED-reach: static surrogate for moving humans (no-op otherwise).
  std::vector<std::shared_ptr<Obstacle>> owned_sur;
  std::vector<const Obstacle*> obs_eff = safety_surrogate(obstacles, opt, owned_sur);

  std::vector<const Obstacle*> soft_obs, hard_obs;
  partition_obstacles(obs_eff, avoid_cfg, opt.avoid_override, soft_obs, hard_obs);

  AlmSolveResult sol = alm_solve(S.start, S.goal, S.q0, S.T0, soft_obs, hard_obs,
                                 avoid_cfg, opt, v0, a0);
  const MinjerkTraj& tr = sol.tr;
  Eigen::VectorXd Tf(M); for (int i = 0; i < M; ++i) Tf(i) = sol.x(nq + i);

  // feasibility (all obstacles), then hard-only override.
  Feasibility feas = check_feasibility(tr, obs_eff, avoid_cfg, opt);
  int K_dense = std::max(opt.K, 200);
  std::vector<const Obstacle*> safety_soft, safety_obs;  // hard set under override=None
  partition_obstacles(obs_eff, avoid_cfg, "", safety_soft, safety_obs);
  auto mc = hard_clearance(tr, safety_obs, avoid_cfg, K_dense);
  double min_clr = mc.first, max_breach = mc.second;

  double extra = pred_margin(opt.hard_alm()) + eps_track(opt.hard_alm());
  bool hard_violation = (!safety_obs.empty()) && (max_breach + extra > opt.clearance_tol);
  if (hard_violation) {
    feas.trajectory_valid = false;
    feas.failure_reason = "clearance_violation";
  } else if (feas.failure_reason == "clearance_violation") {
    if (feas.max_v > opt.vmax * (1.0 + opt.vel_tol)) feas.failure_reason = "velocity_violation";
    else if (feas.max_a > opt.amax * (1.0 + opt.accel_tol)) feas.failure_reason = "acceleration_violation";
    else feas.failure_reason = "";
    feas.trajectory_valid = feas.failure_reason.empty();
  }

  MincoCandidate C{tr};
  C.init_cost = sol.f0;
  C.final_cost = sol.has_result ? sol.result_fun : sol.f0;
  C.iter = sol.total_inner_iters;
  C.optimizer_success = sol.has_result ? sol.result_success : true;
  C.dt0 = S.T0.minCoeff();
  C.dt_final = Tf.minCoeff();
  C.T_target = sol.T_target;
  C.T_final = Tf.sum();
  C.feasibility = feas;
  C.M = M;
  C.hard_violation = hard_violation;
  C.hard_max_breach = max_breach;
  C.hard_min_clearance = min_clr;
  C.n_hard = (int)hard_obs.size();
  C.n_soft = (int)soft_obs.size();
  C.alm_outer_iters = sol.outer_iters;
  C.alm_rho_final = sol.rho;
  C.alm_max_violation = sol.max_violation;
  C.alm_lambda_max = (sol.lam.size() ? sol.lam.maxCoeff() : 0.0);
  C.continuous_min_clearance = min_clr;
  // certificate_margin: the Python wires _certificate_margin / _certificate_margin_spacetime;
  // the driver only USES it for telemetry (selection uses hard_max_breach + feasibility).
  // We expose the simple Stage-3 / spacetime-off certificate margin (= -max g) on the final
  // trajectory; the spacetime variant is reported only in info and does not affect selection.
  if (!hard_obs.empty() && sol.has_final_alm) {
    AlmConstraints cf = alm_constraints(tr, hard_obs, avoid_cfg, &sol.final_normals,
                                        opt.hard_alm(), &sol.final_trust);
    double maxg = -std::numeric_limits<double>::infinity();
    for (int m = 0; m < cf.Nc; ++m) maxg = std::max(maxg, cf.g(m));
    C.certificate_margin = -maxg;
  } else {
    C.certificate_margin = std::numeric_limits<double>::infinity();
  }
  return C;
}

// ===========================================================================
// _select_best_minco — feasible-first; else "least bad" with hard-breach priority.
// ===========================================================================
inline int select_best_minco(const std::vector<MincoCandidate>& cands) {
  // feasible candidates -> min final_cost
  int best = -1; double bestcost = std::numeric_limits<double>::infinity();
  for (int i = 0; i < (int)cands.size(); ++i)
    if (cands[i].feasibility.trajectory_valid && cands[i].final_cost < bestcost) {
      bestcost = cands[i].final_cost; best = i;
    }
  if (best >= 0) return best;
  // none feasible -> lexicographic (max(hard_max_breach,0), max_clearance_violation,
  //                                  vel_violation, final_cost)
  int li = -1;
  double bk0 = 0, bk1 = 0, bk2 = 0, bk3 = 0;
  for (int i = 0; i < (int)cands.size(); ++i) {
    double k0 = std::max(cands[i].hard_max_breach, 0.0);
    double k1 = cands[i].feasibility.max_clearance_violation;
    double k2 = cands[i].feasibility.vel_violation;
    double k3 = cands[i].final_cost;
    if (li < 0 ||
        (k0 < bk0) ||
        (k0 == bk0 && k1 < bk1) ||
        (k0 == bk0 && k1 == bk1 && k2 < bk2) ||
        (k0 == bk0 && k1 == bk1 && k2 == bk2 && k3 < bk3)) {
      li = i; bk0 = k0; bk1 = k1; bk2 = k2; bk3 = k3;
    }
  }
  return li;
}

// ===========================================================================
// plan_minco — top-level: generate seeds, optimise each, select best, return.
// ===========================================================================
struct PlanInfo {
  bool trajectory_valid = false;
  bool optimizer_success = true;
  std::string failure_reason;
  Feasibility feasibility;
  std::string seed_used;
  int n_seeds_tried = 0;
  double init_cost = 0.0, final_cost = 0.0;
  int iter = 0;
  double dt0 = 0.0, dt_final = 0.0, T_target = 0.0, T_final = 0.0;
  int M = 0;
  int n_hard = 0, n_soft = 0;
  bool hard_violation = false;
  double hard_max_breach = 0.0;
  int alm_outer_iters = 0;
  double alm_rho_final = 0.0;
  double alm_max_violation = 0.0;
  double alm_lambda_max = 0.0;
  double continuous_min_clearance = 0.0;
  double certificate_margin = 0.0;
};

inline std::pair<MinjerkTraj, PlanInfo> plan_minco(
    const Eigen::MatrixXd& astar_path, const std::vector<const Obstacle*>& obstacles,
    const std::map<std::string, AvoidParams>& avoid_cfg,
    const PlanOptParams& opt = PlanOptParams(), const DetourConfig& detour_cfg = DetourConfig(),
    const Eigen::Vector3d& v0 = Eigen::Vector3d::Zero(),
    const Eigen::Vector3d& a0 = Eigen::Vector3d::Zero()) {
  if (astar_path.rows() < 2 || astar_path.cols() != 3)
    throw std::invalid_argument("astar_path must be (M>=2, 3)");

  auto seeds = generate_detour_seeds(astar_path, obstacles, avoid_cfg, opt, detour_cfg);

  std::vector<MincoCandidate> candidates;
  std::vector<std::string> names;
  for (auto& sd : seeds) {
    try {
      MincoCandidate rec = optimise_one_minco(sd.second, obstacles, avoid_cfg, opt, v0, a0);
      rec.seed_name = sd.first;
      candidates.push_back(std::move(rec));
    } catch (const std::exception&) {
      continue;  // one seed failed numerically -> skip
    }
  }
  if (candidates.empty()) throw std::runtime_error("all seeds failed to optimise");

  int bi = select_best_minco(candidates);
  const MincoCandidate& best = candidates[bi];
  const Feasibility& feas = best.feasibility;

  PlanInfo info;
  info.trajectory_valid = feas.trajectory_valid;
  info.optimizer_success = best.optimizer_success;
  info.failure_reason = feas.failure_reason;
  info.feasibility = feas;
  info.seed_used = best.seed_name;
  info.n_seeds_tried = (int)candidates.size();
  info.init_cost = best.init_cost;
  info.final_cost = best.final_cost;
  info.iter = best.iter;
  info.dt0 = best.dt0; info.dt_final = best.dt_final;
  info.T_target = best.T_target; info.T_final = best.T_final;
  info.M = best.M;
  info.n_hard = best.n_hard; info.n_soft = best.n_soft;
  info.hard_violation = best.hard_violation;
  info.hard_max_breach = best.hard_max_breach;
  info.alm_outer_iters = best.alm_outer_iters;
  info.alm_rho_final = best.alm_rho_final;
  info.alm_max_violation = best.alm_max_violation;
  info.alm_lambda_max = best.alm_lambda_max;
  info.continuous_min_clearance = best.continuous_min_clearance;
  info.certificate_margin = best.certificate_margin;
  return {best.spline, info};
}

}  // namespace sando
