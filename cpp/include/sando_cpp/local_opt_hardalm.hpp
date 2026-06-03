// local_opt_hardalm — faithful C++ port of the HARD per-class convex-hull ALM
// constraint cost + analytic gradient from sando_py/local/local_opt.py.
//
// Ports ONE inner ALM evaluation at FIXED multipliers (lambda) and penalty (rho)
// with FROZEN supporting-halfspace normals + trust mask. NOT the outer
// multiplier-update loop (_alm_solve), NOT L-BFGS. Concretely it ports the
// chain that _minco_cost_grad invokes for the hard set:
//
//   _pred_margin / _eps_track / _hard_d_safe   (R = r + d_safe + pred + eps helpers)
//   _is_moving / _seg_rep_time / _trust_mask
//   _seg_normals      -> frozen supporting-halfspace normals (M,6,H,3)
//   _alm_constraints  -> per-(seg i, ctrl-pt k, hard-obs h) g / n / dist / R / dgdt / trust
//   _alm_term         -> (cost, dcost_dc (6M,3), dcost_dT_explicit (M,), cons)
//
// Reuses minjerk_traj.hpp (MinjerkTraj: control_points / control_point_times /
// grad_from_dcost_dc / control_points_dT_explicit / C2B_matrix), obstacles.hpp
// (SphereObstacle / AABBObstacle), avoid_config.hpp, and signed_dist_and_grad from
// local_opt.hpp.
//
// 不许欺骗、必须还原:每个公式逐项对照 Python,golden 测试验证。
#pragma once
#include <Eigen/Dense>
#include <vector>
#include <map>
#include <string>
#include <cmath>
#include "sando_cpp/minjerk_traj.hpp"
#include "sando_cpp/obstacles.hpp"
#include "sando_cpp/avoid_config.hpp"
#include "sando_cpp/local_opt.hpp"  // signed_dist_and_grad, DistGrad

namespace sando {

// ---------------------------------------------------------------------------
// OptParams — only the fields the HARD-ALM path actually reads (1:1 names with
// local_opt.py:OptParams). Defaults match the Python dataclass exactly.
// ---------------------------------------------------------------------------
struct HardAlmOptParams {
  // ALM master switch / override
  std::string avoid_override = "";   // "" -> per-class; "soft"/"hard" -> force-all
  // Stage-4 space-time + trust window
  double tau_trust = 0.75;
  bool spacetime_hard = true;
  // conformal continuous-time certificate margin (spheres only)
  double q_conformal = 0.0;
  // execution-gap tube radius (every hard obstacle)
  double epsilon_track = 0.0;
  // deterministic worst-case reachability
  std::string safety_mode = "conformal";   // "conformal" | "deterministic"
  std::string reach_model = "accel";        // "accel" | "speed"
  double a_max_human = 0.0;
  double v_max_human = 0.0;
  double est_pos_err = 0.0;
  double est_vel_err = 0.0;
  // (other OptParams fields — w_*, vmax, K, kappa, alm_rho0, ... — are not read
  //  by the inner HARD-ALM evaluation, so they are intentionally omitted.)
};

// ---------------------------------------------------------------------------
// margin helpers (local_opt.py:_pred_margin / _eps_track / _hard_d_safe)
// ---------------------------------------------------------------------------

// _pred_margin: human-prediction margin added to a hard SPHERE's R.
//   conformal     -> statistical Q_alpha (q_conformal)
//   deterministic -> physical worst-case reachable-set radius (constant at tau_trust)
//                    'speed' : est_pos_err + v_max_human * tau
//                    'accel' : est_pos_err + est_vel_err*tau + 0.5*a_max_human*tau^2
inline double pred_margin(const HardAlmOptParams& opt) {
  if (opt.safety_mode == "deterministic") {
    const double tau = opt.tau_trust;
    const double pos = opt.est_pos_err;
    if (opt.reach_model == "speed")
      return pos + opt.v_max_human * tau;
    return pos + opt.est_vel_err * tau + 0.5 * opt.a_max_human * tau * tau;
  }
  return opt.q_conformal;
}

// _eps_track: plan->flown tube radius, added to EVERY hard obstacle's R.
inline double eps_track(const HardAlmOptParams& opt) { return opt.epsilon_track; }

// _hard_d_safe: d_safe of a hard obstacle; fail-safe to the human default 0.8.
inline double hard_d_safe(const Obstacle* obs,
                          const std::map<std::string, AvoidParams>& cfg) {
  auto it = cfg.find(obs->class_name);
  if (it == cfg.end()) return 0.8;
  return it->second.d_safe;
}

// _is_moving: sphere-human with nonzero CV velocity OR nonzero accel.
inline bool is_moving(const Obstacle* obs) {
  if (auto s = dynamic_cast<const SphereObstacle*>(obs))
    return s->vel.norm() > 0.0 || s->accel.norm() > 0.0;
  return false;
}

// _seg_rep_time: Stage-3 representative wall-clock time for segment i = segment END.
inline double seg_rep_time(const MinjerkTraj& tr, int i) { return tr.cum(i + 1); }

// _trust_mask: (M,6) bool: control point (i,k) trusted-HARD iff t_{i,k} <= tau_trust.
inline Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> trust_mask(
    const MinjerkTraj& tr, const HardAlmOptParams& opt) {
  Eigen::MatrixXd t = tr.control_point_times();   // (M,6)
  return (t.array() <= opt.tau_trust);
}

// ---------------------------------------------------------------------------
// _seg_normals — FROZEN supporting-halfspace normals, shape (M,6,H,3).
// Stored row-major flat: A[((i*6 + k)*H + h)*3 + d].
// ---------------------------------------------------------------------------
inline std::vector<double> seg_normals(const MinjerkTraj& tr,
                                       const std::vector<const Obstacle*>& hard_obs,
                                       const HardAlmOptParams& opt) {
  const int M = tr.M;
  const int H = (int)hard_obs.size();
  auto P = tr.control_points();                       // M blocks of (6,3)
  const bool spacetime = opt.spacetime_hard;

  // per-segment centroid (M,3)
  std::vector<Eigen::Vector3d> centroid(M);
  for (int i = 0; i < M; ++i) centroid[i] = P[i].colwise().mean().transpose();

  // representative times (M) and control-point times (M,6)
  Eigen::VectorXd t_rep(M);
  for (int i = 0; i < M; ++i) t_rep(i) = seg_rep_time(tr, i);
  Eigen::MatrixXd t_pt = tr.control_point_times();    // (M,6)

  std::vector<double> A((size_t)M * 6 * H * 3, 0.0);
  const Eigen::Vector3d fallback(1.0, 0.0, 0.0);

  auto Aidx = [&](int i, int k, int h, int dd) -> double& {
    return A[(((size_t)i * 6 + k) * H + h) * 3 + dd];
  };

  for (int h = 0; h < H; ++h) {
    const Obstacle* obs = hard_obs[h];
    const SphereObstacle* sph = dynamic_cast<const SphereObstacle*>(obs);
    const bool moving_st = spacetime && is_moving(obs);
    for (int i = 0; i < M; ++i) {
      for (int k = 0; k < 6; ++k) {
        Eigen::Vector3d a;
        if (moving_st) {
          Eigen::Vector3d c = sph->predict(t_pt(i, k));    // per-control-point time
          a = P[i].row(k).transpose() - c;
        } else {
          Eigen::Vector3d c;
          if (sph) c = sph->predict(t_rep(i));             // segment-END single time
          else {
            auto b = dynamic_cast<const AABBObstacle*>(obs);
            c = 0.5 * (b->lo + b->hi);                      // wall centre
          }
          a = centroid[i] - c;                             // per-segment centroid normal
        }
        double nrm = a.norm();
        Eigen::Vector3d an = (nrm > 1e-12) ? (a / nrm) : fallback;
        for (int dd = 0; dd < 3; ++dd) Aidx(i, k, h, dd) = an(dd);
      }
    }
  }
  return A;
}

// ---------------------------------------------------------------------------
// _alm_constraints — every (seg i, ctrl-pt k, hard-obs h) TIGHT constraint.
// Flat index m = (i*6 + k)*H + h.
// ---------------------------------------------------------------------------
struct AlmConstraints {
  int M = 0, H = 0, Nc = 0;
  Eigen::VectorXd g;                 // (Nc)
  Eigen::MatrixXd n;                 // (Nc,3) outward unit normal (= -dg/dP)
  std::vector<int> seg, kpt, hidx;   // (Nc) reverse-index tables
  Eigen::VectorXd dist;              // (Nc) raw geometric ||p-c||-r (sphere) / signed_dist (AABB)
  Eigen::VectorXd R;                 // (Nc) enforced effective radius
  Eigen::VectorXd dgdt;              // (Nc) dg/dt_{i,k} (nonzero only for trusted moving spheres)
  std::vector<char> trust;           // (Nc) per-constraint S4b bool mask
  std::vector<Eigen::Matrix<double, 6, 3>> P;  // (M,6,3) control points
};

// trust_mask_in: optional FROZEN (M,6) mask. Pass nullptr to recompute from current T.
inline AlmConstraints alm_constraints(
    const MinjerkTraj& tr, const std::vector<const Obstacle*>& hard_obs,
    const std::map<std::string, AvoidParams>& avoid_cfg,
    const std::vector<double>* normals, const HardAlmOptParams& opt,
    const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>* trust_mask_in) {
  AlmConstraints out;
  const int M = tr.M;
  const int H = (int)hard_obs.size();
  out.M = M; out.H = H;
  out.P = tr.control_points();

  std::vector<double> A_local;
  const std::vector<double>* A = normals;
  if (A == nullptr) { A_local = seg_normals(tr, hard_obs, opt); A = &A_local; }
  auto Aidx = [&](int i, int k, int h, int dd) -> double {
    return (*A)[(((size_t)i * 6 + k) * H + h) * 3 + dd];
  };

  const bool spacetime = opt.spacetime_hard;
  const int Nc = M * 6 * H;
  out.Nc = Nc;
  out.g = Eigen::VectorXd::Zero(Nc);
  out.n = Eigen::MatrixXd::Zero(Nc, 3);
  out.dist = Eigen::VectorXd::Zero(Nc);
  out.R = Eigen::VectorXd::Zero(Nc);
  out.dgdt = Eigen::VectorXd::Zero(Nc);
  out.trust.assign(Nc, 1);
  out.seg.resize(Nc); out.kpt.resize(Nc); out.hidx.resize(Nc);
  for (int i = 0; i < M; ++i)
    for (int k = 0; k < 6; ++k)
      for (int h = 0; h < H; ++h) {
        int m = (i * 6 + k) * H + h;
        out.seg[m] = i; out.kpt[m] = k; out.hidx[m] = h;
      }

  const double q_alpha = pred_margin(opt);
  const double eps = eps_track(opt);

  Eigen::VectorXd t_rep(M);
  for (int i = 0; i < M; ++i) t_rep(i) = seg_rep_time(tr, i);
  Eigen::MatrixXd t_pt = tr.control_point_times();   // (M,6)
  const double G_NEG = -1.0e9;

  for (int h = 0; h < H; ++h) {
    const Obstacle* obs = hard_obs[h];
    const double d_safe = hard_d_safe(obs, avoid_cfg);
    const SphereObstacle* sph = dynamic_cast<const SphereObstacle*>(obs);
    if (sph) {                                         // sphere -> supporting halfspace
      const double R = sph->radius + d_safe + q_alpha + eps;
      const bool moving_st = spacetime && is_moving(obs);
      const bool has_accel = sph->accel.norm() > 0.0;  // np.any(accel) over float vec
      for (int i = 0; i < M; ++i) {
        for (int k = 0; k < 6; ++k) {
          int m = (i * 6 + k) * H + h;
          Eigen::Vector3d c =
              moving_st ? sph->predict(t_pt(i, k)) : sph->predict(t_rep(i));
          Eigen::Vector3d a(Aidx(i, k, h, 0), Aidx(i, k, h, 1), Aidx(i, k, h, 2));
          Eigen::Vector3d diff = out.P[i].row(k).transpose() - c;
          double proj = a.dot(diff);                   // a^T(P-c)
          double g_h = R - proj;
          double dist_h = diff.norm() - sph->radius;
          out.n.row(m) = a.transpose();
          out.dist(m) = dist_h;
          out.R(m) = R;
          if (moving_st) {
            // dg/dt_{i,k} = +a . velocity_at(t_{i,k}); CV fallback when accel==0
            double dgdt_h = has_accel ? a.dot(sph->velocity_at(t_pt(i, k)))
                                      : a.dot(sph->vel);
            // S4b trust mask: frozen if provided, else t_pt <= tau_trust
            bool tm = trust_mask_in ? (*trust_mask_in)(i, k)
                                    : (t_pt(i, k) <= opt.tau_trust);
            out.trust[m] = tm ? 1 : 0;
            out.g(m) = tm ? g_h : G_NEG;
            out.dgdt(m) = tm ? dgdt_h : 0.0;
          } else {
            out.g(m) = g_h;
            out.dgdt(m) = 0.0;
          }
        }
      }
    } else {                                           // AABB wall -> signed-dist surrogate
      for (int i = 0; i < M; ++i) {
        for (int k = 0; k < 6; ++k) {
          int m = (i * 6 + k) * H + h;
          DistGrad dg =
              signed_dist_and_grad(obs, out.P[i].row(k).transpose(), t_rep(i));
          out.g(m) = (d_safe + eps) - dg.d;
          out.n.row(m) = dg.dd_dp.transpose();
          out.dist(m) = dg.d;
          out.R(m) = d_safe + eps;
        }
      }
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// _alm_term — PHR augmented-Lagrangian penalty cost + analytic gradient at a
// FIXED (lambda, rho) with FROZEN normals + trust mask.
// Returns (cost, dcost_dc (6M,3), dcost_dT_explicit (M,)) and (optionally) cons.
// ---------------------------------------------------------------------------
struct AlmTermResult {
  double cost = 0.0;
  Eigen::MatrixXd dcost_dc;   // (6M,3)
  Eigen::VectorXd dT_exp;     // (M)
  AlmConstraints cons;        // valid only if has_cons
  bool has_cons = false;
};

inline AlmTermResult alm_term(
    const MinjerkTraj& tr, const std::vector<const Obstacle*>& hard_obs,
    const std::map<std::string, AvoidParams>& avoid_cfg,
    const Eigen::VectorXd& lam, double rho,
    const std::vector<double>* normals, const HardAlmOptParams& opt,
    const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>* trust_mask_in) {
  AlmTermResult res;
  const int M = tr.M;
  const int n6 = tr.NC * M;
  res.dcost_dc = Eigen::MatrixXd::Zero(n6, 3);
  res.dT_exp = Eigen::VectorXd::Zero(M);
  if (hard_obs.empty()) return res;

  AlmConstraints cons =
      alm_constraints(tr, hard_obs, avoid_cfg, normals, opt, trust_mask_in);
  const int Nc = cons.Nc;
  const Eigen::VectorXd& g = cons.g;

  // z = lam + rho*g ; active = z > 0
  Eigen::VectorXd z = lam + rho * g;

  // PHR cost: sum over active of (0.5/rho) z^2  -  sum over all of lam^2/(2 rho)
  double cost = 0.0;
  for (int m = 0; m < Nc; ++m)
    if (z(m) > 0.0) cost += (0.5 / rho) * z(m) * z(m);
  for (int m = 0; m < Nc; ++m) cost -= (lam(m) * lam(m)) / (2.0 * rho);
  res.cost = cost;

  // w = z where active else 0
  Eigen::VectorXd w = Eigen::VectorXd::Zero(Nc);
  for (int m = 0; m < Nc; ++m) if (z(m) > 0.0) w(m) = z(m);

  // dL/dP_{i,k} = sum_h z*(-n)  -> accumulate per (segment, ctrl-pt)
  std::vector<Eigen::Matrix<double, 6, 3>> dLdP_seg(M, Eigen::Matrix<double, 6, 3>::Zero());
  for (int m = 0; m < Nc; ++m) {
    int i = cons.seg[m], k = cons.kpt[m];
    Eigen::Vector3d contrib = (-w(m)) * cons.n.row(m).transpose();
    dLdP_seg[i].row(k) += contrib.transpose();
  }

  // (A) implicit-c chain: dL/dc_i = D(T_i) .* (C2B^T @ G_i), then reshape into dcost_dc
  const Eigen::Matrix<double, 6, 6> C2B = C2B_matrix();
  const Eigen::VectorXd& T = tr.T;
  for (int i = 0; i < M; ++i) {
    const double t1 = T(i), t2 = t1 * t1, t3 = t2 * t1, t4 = t2 * t2, t5 = t4 * t1;
    Eigen::Matrix<double, 6, 1> Dvec;
    Dvec << 1.0, t1, t2, t3, t4, t5;
    Eigen::Matrix<double, 6, 3> CtG = C2B.transpose() * dLdP_seg[i];   // (6,3)
    Eigen::Matrix<double, 6, 3> dc = Dvec.asDiagonal() * CtG;          // (6,3)
    res.dcost_dc.block(6 * i, 0, 6, 3) += dc;
  }

  // (B) explicit-T Source 1 (curve geometry): sum_k dLdP_seg . dP/dT
  auto dP_dT = tr.control_points_dT_explicit();   // M blocks of (6,3)
  for (int i = 0; i < M; ++i)
    res.dT_exp(i) += (dLdP_seg[i].array() * dP_dT[i].array()).sum();

  // (B) explicit-T Source 2 (moving-human absolute time): w*dgdt chained via dt_{i,k}/dT_j
  bool any_dgdt = false;
  for (int m = 0; m < Nc; ++m) if (cons.dgdt(m) != 0.0) { any_dgdt = true; break; }
  if (any_dgdt) {
    Eigen::MatrixXd S = Eigen::MatrixXd::Zero(M, 6);   // sum over obstacles h of w*dgdt
    for (int m = 0; m < Nc; ++m) {
      int i = cons.seg[m], k = cons.kpt[m];
      S(i, k) += w(m) * cons.dgdt(m);
    }
    Eigen::VectorXd dT_abs = Eigen::VectorXd::Zero(M);
    // same-segment fraction k/5
    for (int i = 0; i < M; ++i) {
      double acc = 0.0;
      for (int k = 0; k < 6; ++k) acc += (double(k) / MinjerkTraj::DEG) * S(i, k);
      dT_abs(i) = acc;
    }
    // cross-segment: dt_{i,k}/dT_j = 1 for all j<i -> reverse-cumsum of seg_sum into dT_abs[:-1]
    Eigen::VectorXd seg_sum(M);
    for (int i = 0; i < M; ++i) seg_sum(i) = S.row(i).sum();
    if (M > 1) {
      // np: dT_abs[:-1] += np.cumsum(seg_sum[::-1])[::-1][1:]
      // suffix[i] = sum_{j>i} seg_sum[j]  -> add to dT_abs[i] for i in 0..M-2
      double suffix = 0.0;
      for (int i = M - 1; i >= 1; --i) {
        suffix += seg_sum(i);
        dT_abs(i - 1) += suffix;
      }
    }
    res.dT_exp += dT_abs;
  }

  res.cons = std::move(cons);
  res.has_cons = true;
  return res;
}

}  // namespace sando
