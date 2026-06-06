// local_opt_terms — faithful C++ port of the two analytic cost-term functions in
// sando_py/local/local_opt.py:
//   _minco_obstacle_term(tr, obstacles, avoid_cfg, opt) -> (cost, dcost_dc (6M,3), dcost_dT_explicit (M,))
//   _minco_dynamic_term(tr, order, limit, weight, opt)  -> (cost, dcost_dc (6M,3), dcost_dT_explicit (M,))
//
// Reuses the already-ported helpers in local_opt.hpp (signed_dist_and_grad, seg_vander)
// and the MinjerkTraj backbone (minjerk_traj.hpp). 1:1 with the Python; golden-verified.
// 不许欺骗、必须还原:每个公式逐项对照 Python 移植,无近似无桩。
#pragma once
#include <Eigen/Dense>
#include <vector>
#include <map>
#include <string>
#include <cmath>
#include <algorithm>
#include <tuple>
#include "sando_cpp/obstacles.hpp"
#include "sando_cpp/minjerk_traj.hpp"
#include "sando_cpp/local_opt.hpp"  // signed_dist_and_grad, seg_vander
#include "sando_cpp/avoid_config.hpp"  // canonical sando::AvoidParams {name,mode,d_safe,weight}

namespace sando {

// ODR fix: use the CANONICAL avoid_config.hpp sando::AvoidParams (name/mode/d_safe/weight)
// instead of a private minimal struct that clashed when co-included with avoid_config.hpp.
// The obstacle term only reads .d_safe and .weight, both present on the canonical struct.
using AvoidCfg = std::map<std::string, AvoidParams>;

// OptParams fields read by these two terms (subset of local_opt.py:OptParams).
struct TermOptParams {
  int kappa = 16;
  double w_obs = 1.0;   // (kept for parity; the caller multiplies, not these terms)
  double vmax = 3.0;
  double amax = 3.0;
  double w_vel = 100.0;
  double w_accel = 100.0;
};

// Result triple shared by both terms.
struct TermResult {
  double cost = 0.0;
  Eigen::MatrixXd dcost_dc;          // (6M, 3)
  Eigen::VectorXd dcost_dT_explicit; // (M,)
};

// ---------------------------------------------------------------------------
// _fixed_basis(kappa): midpoint fractions s=(j+0.5)/kappa and fixed power basis
// Bs=seg_vander(s); idx[o][p]=max(p-o,0). (No cache needed in C++; recomputed.)
// ---------------------------------------------------------------------------
struct FixedBasis {
  Eigen::VectorXd s;                  // (kappa,)
  std::array<Eigen::MatrixXd, 4> Bs;  // each (kappa,6) at tau = s
  std::array<std::array<int, 6>, 4> idx;
};

inline FixedBasis fixed_basis(int kappa) {
  FixedBasis fb;
  fb.s.resize(kappa);
  for (int j = 0; j < kappa; ++j) fb.s(j) = (double(j) + 0.5) / double(kappa);
  fb.Bs = seg_vander(fb.s);
  for (int o = 0; o < 4; ++o)
    for (int p = 0; p < 6; ++p)
      fb.idx[o][p] = std::max(p - o, 0);
  return fb;
}

// _scale_basis(Bs, idx, Ti): B_o(s*Ti)[:,p] = Bs_o[:,p] * Ti^(p-o).
inline std::array<Eigen::MatrixXd, 4> scale_basis(const FixedBasis& fb, double Ti) {
  double tp[6] = {1.0, Ti, Ti * Ti, std::pow(Ti, 3), std::pow(Ti, 4), std::pow(Ti, 5)};
  std::array<Eigen::MatrixXd, 4> out;
  for (int o = 0; o < 4; ++o) {
    Eigen::MatrixXd B = fb.Bs[o];  // copy (kappa,6)
    for (int p = 0; p < 6; ++p) {
      double f = tp[fb.idx[o][p]];
      if (f != 1.0) B.col(p) *= f;
    }
    out[o] = B;
  }
  return out;
}

// ---------------------------------------------------------------------------
// _signed_dist_and_grad_batch: vectorised (d (k,), dd/dp (k,3), dd/dt (k,)) over
// k sample points P at absolute times t_abs. Same math as signed_dist_and_grad.
// For the sphere with accel==0 it uses the CV fallback dd_dt = -(n @ vel)
// (bit-identical to -n·velocity_at(t) when accel==0).
// ---------------------------------------------------------------------------
struct BatchDistGrad {
  Eigen::VectorXd d;     // (k,)
  Eigen::MatrixXd ndp;   // (k,3)
  Eigen::VectorXd dddt;  // (k,)
};

inline BatchDistGrad signed_dist_and_grad_batch(const Obstacle* obs,
                                                const Eigen::MatrixXd& P,
                                                const Eigen::VectorXd& t_abs) {
  const int k = (int)P.rows();
  BatchDistGrad r;
  r.d.resize(k);
  r.ndp = Eigen::MatrixXd::Zero(k, 3);
  r.dddt = Eigen::VectorXd::Zero(k);

  if (auto sph = dynamic_cast<const SphereObstacle*>(obs)) {
    const bool accel_nz = (sph->accel.array() != 0.0).any();
    for (int j = 0; j < k; ++j) {
      double t = t_abs(j);
      Eigen::Vector3d c = sph->predict(t);
      Eigen::Vector3d diff = P.row(j).transpose() - c;
      double nrm = std::max(diff.norm(), 1e-9);   // guard p≡c(t) centre singularity -> NaN; golden-untouched
      Eigen::Vector3d n = diff / nrm;
      r.d(j) = nrm - sph->radius;
      r.ndp.row(j) = n.transpose();
      if (!accel_nz)
        r.dddt(j) = -(n.dot(sph->vel));               // CV fallback
      else
        r.dddt(j) = -(n.dot(sph->velocity_at(t)));    // CA, per-sample
    }
    return r;
  }
  // AABB: vectorised over k samples, same math as the scalar path. #11: motion-aware (lo+vel*t /
  // hi+vel*t) + dddt = -dd_dp·vel so cost/gradient match obstacles.hpp::signed_dist for a moving box;
  // vel==0 -> static, dddt 0 -> byte-identical to before (golden-safe).
  auto b = dynamic_cast<const AABBObstacle*>(obs);
  for (int j = 0; j < k; ++j) {
    Eigen::Vector3d p = P.row(j).transpose();
    Eigen::Vector3d lo_t = b->lo + b->vel * t_abs(j);
    Eigen::Vector3d hi_t = b->hi + b->vel * t_abs(j);
    Eigen::Vector3d over = (lo_t - p).cwiseMax(0.0) + (p - hi_t).cwiseMax(0.0);
    if ((over.array() > 0.0).any()) {
      double nrm = over.norm();
      Eigen::Vector3d sign;
      for (int a = 0; a < 3; ++a)
        sign(a) = (p(a) > hi_t(a) ? 1.0 : 0.0) - (p(a) < lo_t(a) ? 1.0 : 0.0);
      r.d(j) = nrm;
      Eigen::Vector3d dd_dp = over.cwiseProduct(sign) / nrm;
      r.ndp.row(j) = dd_dp.transpose();
      r.dddt(j) = -dd_dp.dot(b->vel);
    } else {
      Eigen::Vector3d cand = (lo_t - p).cwiseMax(p - hi_t);
      int ax = 0;
      cand.maxCoeff(&ax);
      r.d(j) = cand(ax);
      double loP = lo_t(ax) - p(ax), Phi = p(ax) - hi_t(ax);
      Eigen::Vector3d g = Eigen::Vector3d::Zero();
      g(ax) = (loP >= Phi) ? -1.0 : 1.0;
      r.ndp.row(j) = g.transpose();
      r.dddt(j) = -g.dot(b->vel);
    }
  }
  return r;
}

// ---------------------------------------------------------------------------
// _minco_obstacle_term: soft EGO obstacle cost + analytic gradient.
//   cost = Σ_i (T_i/κ) Σ_j Σ_obs W·φ(d),  φ(d)=(d_safe-d)^3 for d<d_safe.
// `obstacles` is the SOFT obstacle set (walls). avoid_cfg must contain each
// obstacle's class_name (matches Python avoid_cfg[obs.class_name], no .get()).
// ---------------------------------------------------------------------------
inline TermResult minco_obstacle_term(const MinjerkTraj& tr,
                                      const std::vector<const Obstacle*>& obstacles,
                                      const AvoidCfg& avoid_cfg,
                                      const TermOptParams& opt) {
  const int M = tr.M;
  const int n = tr.NC * M;
  const int kap = std::max(opt.kappa, 1);
  TermResult res;
  res.dcost_dc = Eigen::MatrixXd::Zero(n, 3);
  res.dcost_dT_explicit = Eigen::VectorXd::Zero(M);
  if (obstacles.empty()) return res;

  FixedBasis fb = fixed_basis(kap);
  const Eigen::VectorXd& s = fb.s;
  double cost = 0.0;

  for (int i = 0; i < M; ++i) {
    const double Ti = tr.T(i);
    Eigen::VectorXd taus = s * Ti;                  // (κ,)
    auto B = scale_basis(fb, Ti);
    const Eigen::MatrixXd& B0 = B[0];
    const Eigen::MatrixXd& B1 = B[1];
    Eigen::Matrix<double, 6, 3> ci = tr.c.block(6 * i, 0, 6, 3);
    Eigen::MatrixXd Pm = B0 * ci;                   // (κ,3) positions
    Eigen::MatrixXd Vm = B1 * ci;                   // (κ,3) velocities
    Eigen::VectorXd t_abs = taus.array() + tr.cum(i);  // (κ,)
    const double wq = Ti / double(kap);

    for (const Obstacle* obs : obstacles) {
      const AvoidParams& params = avoid_cfg.at(obs->class_name);
      const double d_safe = params.d_safe;
      const double W = params.weight;

      BatchDistGrad bg = signed_dist_and_grad_batch(obs, Pm, t_abs);
      // diff = d_safe - d; active where diff > 0.
      double sum_phi = 0.0;                 // Σ φ over active
      double sum_Phi = 0.0;                 // Σ W·φ over active
      double sum_term_b = 0.0;              // Σ W·dphi·(n·V)·s over active
      double sum_A = 0.0;                   // Σ A over active   (A = wq·W·dphi·dddt)
      double sum_sA = 0.0;                  // Σ s·A over active
      bool any_active = false;
      bool any_dddt = false;
      Eigen::Matrix<double, 6, 3> dcc_seg = Eigen::Matrix<double, 6, 3>::Zero();

      for (int j = 0; j < kap; ++j) {
        double diff = d_safe - bg.d(j);
        if (diff <= 0.0) continue;
        any_active = true;
        double phi = diff * diff * diff;
        double dphi = -3.0 * diff * diff;        // φ'(d)
        sum_phi += phi;
        double Phi = W * phi;
        sum_Phi += Phi;
        // implicit-c: Σ_j (wq W dphi_j) outer(B0_j, ndp_j)
        double coef = (wq * W) * dphi;           // scalar
        Eigen::Vector3d ndpj = bg.ndp.row(j).transpose();
        Eigen::Matrix<double, 1, 6> B0j = B0.row(j);
        dcc_seg += B0j.transpose() * (coef * ndpj).transpose();
        // (b) moving-local-sample term
        double ndp_dot_V = ndpj.dot(Vm.row(j).transpose());
        sum_term_b += W * dphi * ndp_dot_V * s(j);
        // (c) abs-time term (moving obstacle): A = wq·W·dphi·dddt
        double dddt = bg.dddt(j);
        if (dddt != 0.0) {
          any_dddt = true;
          double A = (wq * W) * dphi * dddt;
          sum_A += A;
          sum_sA += s(j) * A;
        }
      }
      if (!any_active) continue;
      cost += wq * W * sum_phi;
      res.dcost_dc.block(6 * i, 0, 6, 3) += dcc_seg;
      // (a) quadrature-weight term
      res.dcost_dT_explicit(i) += (1.0 / double(kap)) * sum_Phi;
      // (b) moving-local-sample term
      res.dcost_dT_explicit(i) += wq * sum_term_b;
      // (c) abs-time term
      if (any_dddt) {
        res.dcost_dT_explicit(i) += sum_sA;                 // (c1) same-segment
        if (i > 0)
          for (int kpre = 0; kpre < i; ++kpre)
            res.dcost_dT_explicit(kpre) += sum_A;           // (c2) cross-segment
      }
    }
  }
  res.cost = cost;
  return res;
}

// ---------------------------------------------------------------------------
// _minco_dynamic_term: vel(order=1)/accel(order=2) hinge penalty + gradient.
//   integrand h = max(||deriv||-limit, 0)^2, cost = Σ_i (T_i/κ) Σ_j weight·h.
//   ∂h/∂deriv = 2·over·(deriv/||deriv||). Explicit-T = (a) qweight + (b) moving sample.
// ---------------------------------------------------------------------------
inline TermResult minco_dynamic_term(const MinjerkTraj& tr, int order, double limit,
                                     double weight, const TermOptParams& opt) {
  const int M = tr.M;
  const int n = tr.NC * M;
  const int kap = std::max(opt.kappa, 1);
  TermResult res;
  res.dcost_dc = Eigen::MatrixXd::Zero(n, 3);
  res.dcost_dT_explicit = Eigen::VectorXd::Zero(M);

  FixedBasis fb = fixed_basis(kap);
  const Eigen::VectorXd& s = fb.s;
  double cost = 0.0;

  for (int i = 0; i < M; ++i) {
    const double Ti = tr.T(i);
    auto B = scale_basis(fb, Ti);
    // Bo = order-th, Bnext = (order+1)-th
    const Eigen::MatrixXd& Bo = B[order];
    const Eigen::MatrixXd& Bnext = B[order + 1];
    Eigen::Matrix<double, 6, 3> ci = tr.c.block(6 * i, 0, 6, 3);
    Eigen::MatrixXd D = Bo * ci;       // (κ,3) limited derivative
    Eigen::MatrixXd Dn = Bnext * ci;   // (κ,3) its tau-derivative
    const double wq = Ti / double(kap);

    double sum_h = 0.0;
    double sum_dh_dtau_s = 0.0;
    bool any_active = false;
    Eigen::Matrix<double, 6, 3> dcc_seg = Eigen::Matrix<double, 6, 3>::Zero();

    for (int j = 0; j < kap; ++j) {
      Eigen::Vector3d Dj = D.row(j).transpose();
      double mag = Dj.norm();
      double over = mag - limit;
      if (!(over > 0.0 && mag > 0.0)) continue;
      any_active = true;
      Eigen::Vector3d u = Dj / mag;                 // unit direction
      double h = over * over;
      sum_h += h;
      Eigen::Vector3d dh_dderiv = (2.0 * over) * u; // ∂h/∂deriv
      // implicit-c
      Eigen::Matrix<double, 1, 6> Boj = Bo.row(j);
      dcc_seg += Boj.transpose() * dh_dderiv.transpose();
      // (b) dh/dtau = dh_dderiv · d(deriv)/dtau
      double dh_dtau = dh_dderiv.dot(Dn.row(j).transpose());
      sum_dh_dtau_s += dh_dtau * s(j);
    }
    if (!any_active) continue;
    cost += wq * weight * sum_h;
    res.dcost_dc.block(6 * i, 0, 6, 3) += (wq * weight) * dcc_seg;
    // (a) quadrature-weight term
    res.dcost_dT_explicit(i) += (1.0 / double(kap)) * weight * sum_h;
    // (b) moving-sample term
    res.dcost_dT_explicit(i) += wq * weight * sum_dh_dtau_s;
  }
  res.cost = cost;
  return res;
}

}  // namespace sando
