// minco_cost_grad — faithful C++ port of sando_py/local/local_opt.py:_minco_cost_grad
// (the analytic objective+gradient L-BFGS-B calls inside the MINCO solve).
//
// 1:1 with the Python. Given the optimisation vector x = [q.ravel() ((M-1)*3), T (M)],
// it rebuilds a MinjerkTraj from (start, goal, q, T, v0/a0/vf/af), assembles the TOTAL
// cost and its gradient w.r.t. x, and returns (f, grad) with grad ordered identically
// to x: [dCost/dq.ravel(), dCost/dT].
//
//   f = w_obs   * obstacle_soft(EGO field)
//     + 1.0     * vel_hinge        (the dynamic term already carries w_vel internally)
//     + 1.0     * accel_hinge      (the dynamic term already carries w_accel internally)
//     + w_time  * ((Ttot - T_target)/max(T_target,1e-9))^2          (time anchor)
//     + (hard ALM PHR term, only if hard_obs nonempty AND alm_rho > 0)
//     + w_smooth* energy(min-jerk)
//
// The soft-obstacle and the two dynamic-hinge terms each return (cost, dcost_dc (6M,3),
// dcost_dT_explicit (M,)); these accumulate into ONE pair (dcost_dc, dT_exp). The
// time-anchor is a pure explicit-T term. The hard-ALM term (frozen multipliers / penalty
// / normals / trust mask) adds its own (cost, dcost_dc, dT_exp). The accumulated
// (dcost_dc, dT_exp) is back-propagated through the MINCO map ONCE via
// MinjerkTraj::grad_from_dcost_dc (the c = M(T)^{-1} b implicit dependence). Finally the
// energy term is added with its own MINCO-internal energy_grad (w_smooth weight).
//
// REUSES the already golden-verified headers (no re-derivation):
//   minjerk_traj.hpp     MinjerkTraj (from_endpoints == ctor on stacked waypoints,
//                        energy_grad, grad_from_dcost_dc, control_points, ...)
//   local_opt_terms.hpp  minco_obstacle_term (soft EGO) + minco_dynamic_term (vel/accel hinge)
//   local_opt_hardalm.hpp alm_term (PHR augmented-Lagrangian hard convex-hull constraint)
//   avoid_config.hpp     AvoidParams (mode/d_safe/weight) + resolve_mode + default_config
//   obstacles.hpp / local_opt.hpp  (pulled in transitively; geometry + signed_dist)
//
// 不许欺骗、必须还原: every weight / formula / soft-vs-hard split mirrors the Python
// 1:1; golden test (test_minco_cost_grad.cpp) verifies against real _minco_cost_grad.
#pragma once

// ---------------------------------------------------------------------------
// ODR fix (no shim): local_opt_terms.hpp now includes avoid_config.hpp and uses the
// CANONICAL `sando::AvoidParams {name; mode; d_safe; weight;}` (its private minimal
// struct was removed). Both terms- and hard-ALM headers therefore agree on the single
// canonical type, so they co-include cleanly with no #define rename.
// ---------------------------------------------------------------------------
#include "sando_cpp/local_opt_terms.hpp"
#include "sando_cpp/local_opt_hardalm.hpp"  // brings avoid_config.hpp's full AvoidParams
#include "sando_cpp/minjerk_traj.hpp"
#include "sando_cpp/obstacles.hpp"

#include <Eigen/Dense>
#include <vector>
#include <map>
#include <string>
#include <cmath>
#include <algorithm>

namespace sando {

// ---------------------------------------------------------------------------
// OptParams subset that _minco_cost_grad reads (1:1 names with local_opt.py:OptParams).
// Defaults match the Python dataclass exactly. The hard-ALM sub-fields live in a nested
// HardAlmOptParams (see local_opt_hardalm.hpp) so the inner ALM evaluation is reused
// verbatim. avoid_override is shared (used both for the soft/hard split AND inside ALM).
// ---------------------------------------------------------------------------
struct CostGradOptParams {
  // cost weights
  double w_smooth = 1.0;
  double w_obs    = 1.0;
  double w_vel    = 100.0;
  double w_accel  = 100.0;
  double w_time   = 10.0;
  // dynamic limits
  double vmax     = 3.0;
  double amax     = 3.0;
  // quadrature samples per segment
  int    kappa    = 16;
  // soft/hard partition + hard-ALM fields (mirrors OptParams; "" => per-class)
  std::string avoid_override = "";
  double tau_trust     = 0.75;
  bool   spacetime_hard = true;
  double q_conformal   = 0.0;
  double epsilon_track = 0.0;
  std::string safety_mode = "conformal";
  std::string reach_model = "accel";
  double a_max_human = 0.0;
  double v_max_human = 0.0;
  double est_pos_err = 0.0;
  double est_vel_err = 0.0;
  // Soft flight-corridor (SFC): keep interior waypoints inside a tube of radius `corridor_radius`
  // around `corridor_centers` (the guide/seed waypoints, (M-1,3)). OFF when centers null or
  // w_corridor<=0 or radius<=0 -> term skipped -> golden byte-identical. Counters soft-field drift.
  const Eigen::MatrixXd* corridor_centers = nullptr;
  double corridor_radius = 0.0;
  double w_corridor = 0.0;

  // Build the HardAlmOptParams used by alm_term (every field the inner ALM reads).
  HardAlmOptParams hard_alm() const {
    HardAlmOptParams h;
    h.avoid_override = avoid_override;
    h.tau_trust      = tau_trust;
    h.spacetime_hard = spacetime_hard;
    h.q_conformal    = q_conformal;
    h.epsilon_track  = epsilon_track;
    h.safety_mode    = safety_mode;
    h.reach_model    = reach_model;
    h.a_max_human    = a_max_human;
    h.v_max_human    = v_max_human;
    h.est_pos_err    = est_pos_err;
    h.est_vel_err    = est_vel_err;
    return h;
  }

  // Build the TermOptParams the two cost terms read (kappa + vel/accel limits+weights).
  // NOTE: w_obs is NOT applied inside the soft term — _minco_cost_grad multiplies it on
  // the OUTSIDE (parity with Python). w_vel / w_accel ARE applied inside the dynamic
  // term (passed as the `weight` argument), so they are NOT re-multiplied outside.
  TermOptParams term_opt() const {
    TermOptParams t;
    t.kappa   = kappa;
    t.w_obs   = w_obs;   // kept for parity; the obstacle term itself ignores it
    t.vmax    = vmax;
    t.amax    = amax;
    t.w_vel   = w_vel;
    t.w_accel = w_accel;
    return t;
  }
};

// ---------------------------------------------------------------------------
// ALM frozen state fed into _minco_cost_grad: the hard obstacle set + FIXED multipliers
// (lambda) / penalty (rho) / FROZEN supporting-halfspace normals (M,6,H,3 flat, or empty
// to recompute from tr) / FROZEN (M,6) S4b trust mask (or empty to recompute). Mirrors
// the (hard_obs, alm_lambda, alm_rho, alm_normals, alm_trust_mask) kwargs exactly.
// ---------------------------------------------------------------------------
struct AlmState {
  std::vector<const Obstacle*> hard_obs;          // HARD set (humans / forced-hard walls)
  Eigen::VectorXd lam;                            // (Nc) FIXED multipliers (>= 0)
  double rho = 0.0;                               // FIXED penalty
  std::vector<double> normals;                    // (M,6,H,3) flat, empty => recompute
  Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> trust_mask;  // (M,6), size 0 => recompute
};

struct CostGradResult {
  double f = 0.0;
  Eigen::VectorXd grad;   // [dCost/dq.ravel() ((M-1)*3), dCost/dT (M)] (M=1 => just dT)
};

// ---------------------------------------------------------------------------
// _minco_cost_grad — THE analytic value + gradient fed to scipy L-BFGS-B.
//
//   x            optimisation vector [q.ravel() ((M-1)*3), T (M)]
//   start, goal  fixed endpoints (3,)
//   M            number of segments
//   soft_obs     the SOFT obstacle set (EGO field) the caller already partitioned out
//                (== Python `obstacles`); avoid_cfg must contain each soft obstacle's
//                class_name (the soft term looks up d_safe/weight, no .get()/fallback).
//   avoid_cfg    full per-class AvoidParams config (avoid_config.hpp form).
//   opt          weights + limits + ALM/space-time fields.
//   T_target     time-anchor target (Ttot pulled toward this).
//   v0,a0,vf,af  start/goal boundary velocity & acceleration.
//   alm          frozen hard-ALM state (see AlmState). The PHR penalty is added iff
//                alm.hard_obs nonempty AND alm.rho > 0 (mirrors `if hard_obs and alm_rho>0`).
//
// IMPORTANT: matching Python, this function does NOT itself partition `obstacles` into
// soft/hard. The Python _minco_cost_grad receives the SOFT set as `obstacles` and the
// HARD set as `hard_obs` (the caller _alm_solve / _minco path partitions once via
// _partition_obstacles and freezes the ALM state). We mirror that: pass `soft_obs`
// (already soft-only) and the frozen `alm.hard_obs`. A convenience overload below does
// the partition for callers that have the raw obstacle list.
// ---------------------------------------------------------------------------
inline CostGradResult minco_cost_grad(
    const Eigen::VectorXd& x, const Eigen::Vector3d& start,
    const Eigen::Vector3d& goal, int M,
    const std::vector<const Obstacle*>& soft_obs,
    const std::map<std::string, AvoidParams>& avoid_cfg,
    const CostGradOptParams& opt, double T_target,
    const Eigen::Vector3d& v0, const Eigen::Vector3d& a0,
    const Eigen::Vector3d& vf, const Eigen::Vector3d& af,
    const AlmState& alm) {
  // --- split x into interior waypoints q and segment durations T, rebuild MINCO ------
  const int nq = 3 * (M - 1);
  Eigen::MatrixXd q(std::max(M - 1, 0), 3);
  for (int i = 0; i < M - 1; ++i)
    for (int j = 0; j < 3; ++j) q(i, j) = x(i * 3 + j);
  Eigen::VectorXd T(M);
  for (int i = 0; i < M; ++i) T(i) = x(nq + i);

  // from_endpoints == ctor on stacked [start; q; goal]
  Eigen::MatrixXd wp(M + 1, 3);
  wp.row(0) = start.transpose();
  for (int i = 0; i < M - 1; ++i) wp.row(i + 1) = q.row(i);
  wp.row(M) = goal.transpose();
  MinjerkTraj tr(wp, T, v0, a0, vf, af);

  // --- one accumulator pair: dCost/dc (6M,3) and explicit dCost/dT (M) ---------------
  const int n = tr.NC * M;
  Eigen::MatrixXd dcost_dc = Eigen::MatrixXd::Zero(n, 3);
  Eigen::VectorXd dT_exp   = Eigen::VectorXd::Zero(M);
  double f = 0.0;

  // ODR fix: the terms header now uses the CANONICAL sando::AvoidParams, so the full
  // avoid_cfg is passed straight through (no per-field {d_safe, weight} repack needed).
  const TermOptParams topt = opt.term_opt();

  // --- soft obstacle (EGO field): cost & grad scaled by w_obs on the OUTSIDE ---------
  {
    TermResult r = minco_obstacle_term(tr, soft_obs, avoid_cfg, topt);
    dcost_dc += opt.w_obs * r.dcost_dc;
    dT_exp   += opt.w_obs * r.dcost_dT_explicit;
    f         = opt.w_obs * r.cost;
  }

  // --- velocity hinge (order=1, limit=vmax, weight=w_vel applied inside) --------------
  {
    TermResult r = minco_dynamic_term(tr, 1, opt.vmax, opt.w_vel, topt);
    dcost_dc += r.dcost_dc;
    dT_exp   += r.dcost_dT_explicit;
    f        += r.cost;
  }

  // --- acceleration hinge (order=2, limit=amax, weight=w_accel applied inside) --------
  {
    TermResult r = minco_dynamic_term(tr, 2, opt.amax, opt.w_accel, topt);
    dcost_dc += r.dcost_dc;
    dT_exp   += r.dcost_dT_explicit;
    f        += r.cost;
  }

  // --- time anchor: pure explicit-T term, depends only on total duration -------------
  const double Ttot = T.sum();
  const double denom = std::max(T_target, 1e-9);
  const double rel = (Ttot - T_target) / denom;
  f += opt.w_time * rel * rel;
  const double dt_anchor = 2.0 * opt.w_time * (Ttot - T_target) / (denom * denom);
  for (int i = 0; i < M; ++i) dT_exp(i) += dt_anchor;

  // --- hard-ALM PHR term (only if hard_obs nonempty AND alm_rho > 0) ------------------
  if (!alm.hard_obs.empty() && alm.rho > 0.0) {
    const HardAlmOptParams hopt = opt.hard_alm();
    const std::vector<double>* normals_ptr =
        alm.normals.empty() ? nullptr : &alm.normals;
    const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>* tmask_ptr =
        (alm.trust_mask.size() == 0) ? nullptr : &alm.trust_mask;
    AlmTermResult ar = alm_term(tr, alm.hard_obs, avoid_cfg, alm.lam, alm.rho,
                                normals_ptr, hopt, tmask_ptr);
    dcost_dc += ar.dcost_dc;
    dT_exp   += ar.dT_exp;
    f        += ar.cost;
  }

  // --- single back-prop through the MINCO map (c = M(T)^{-1} b implicit dependence) ---
  auto g = tr.grad_from_dcost_dc(dcost_dc, &dT_exp);
  Eigen::MatrixXd gq      = g.first;    // (M-1,3)
  Eigen::VectorXd gT_total = g.second;  // (M,)

  // --- smooth (min-jerk energy) term via MINCO's own energy_grad (w_smooth weight) ----
  auto eg = tr.energy_grad();             // (J, gq_e (M-1,3), gT_e (M,))
  const double J = std::get<0>(eg);
  const Eigen::MatrixXd& gq_e = std::get<1>(eg);
  const Eigen::VectorXd& gT_e = std::get<2>(eg);
  f        += opt.w_smooth * J;
  gq        = gq + opt.w_smooth * gq_e;
  gT_total  = gT_total + opt.w_smooth * gT_e;

  // --- soft flight-corridor (SFC): penalise interior waypoints leaving the guide tube ----
  // cost = w · Σ_i max(0, ||q_i - centre_i|| - R)^2 ; acts directly on q (no MINCO back-prop).
  // OFF by default (centres null / w<=0 / R<=0) -> golden byte-identical.
  if (opt.w_corridor > 0.0 && opt.corridor_radius > 0.0 && opt.corridor_centers &&
      opt.corridor_centers->rows() == M - 1 && opt.corridor_centers->cols() == 3) {
    const Eigen::MatrixXd& Cc = *opt.corridor_centers;
    const double R = opt.corridor_radius;
    for (int i = 0; i < M - 1; ++i) {
      Eigen::Vector3d delta = q.row(i).transpose() - Cc.row(i).transpose();
      const double d = delta.norm();
      if (d > R) {
        const double over = d - R;
        f += opt.w_corridor * over * over;
        const Eigen::Vector3d gdir = (2.0 * opt.w_corridor * over / std::max(d, 1e-12)) * delta;
        gq(i, 0) += gdir(0); gq(i, 1) += gdir(1); gq(i, 2) += gdir(2);
      }
    }
  }

  // --- assemble grad in x-order: [gq.ravel(), gT_total] (M=1 => just gT_total) --------
  CostGradResult res;
  res.f = f;
  res.grad.resize(nq + M);
  for (int i = 0; i < M - 1; ++i)
    for (int j = 0; j < 3; ++j) res.grad(i * 3 + j) = gq(i, j);
  for (int i = 0; i < M; ++i) res.grad(nq + i) = gT_total(i);
  return res;
}

}  // namespace sando
