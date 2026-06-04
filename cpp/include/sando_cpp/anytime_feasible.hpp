#pragma once
// anytime_feasible.hpp — direction A in C++: feasible-direction anytime-feasible inner solve.
// 每个迭代保持人体硬约束 g<=0(用 alm_constraints 的解析 g + 外法向 n + 一次性 dP/dq);
// 方向子问题用 Gauss-Seidel 投影(替 scipy SLSQP)-> 快,目标 <50ms/replan。
// deadline 一到立即停,停在哪都可行(computation-invariant 行人安全)。
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <chrono>
#include <algorithm>
#include "sando_cpp/minjerk_traj.hpp"
#include "sando_cpp/obstacles.hpp"
#include "sando_cpp/avoid_config.hpp"
#include "sando_cpp/local_opt_hardalm.hpp"
#include "sando_cpp/minco_cost_grad.hpp"
#include "sando_cpp/plan_minco.hpp"

namespace sando {

inline MinjerkTraj af_make_tr(const Eigen::Vector3d& start, const Eigen::Vector3d& goal,
                              const Eigen::VectorXd& q, const Eigen::VectorXd& T) {
  const int M = (int)T.size();
  Eigen::MatrixXd wp(M + 1, 3);
  wp.row(0) = start.transpose();
  for (int i = 0; i < M - 1; ++i) wp.row(i + 1) = q.segment(3 * i, 3).transpose();
  wp.row(M) = goal.transpose();
  Eigen::Vector3d z = Eigen::Vector3d::Zero();
  return MinjerkTraj(wp, T, z, z, z, z);
}

// sampled continuous-time min clearance to all humans (sound for moving spheres)
inline double af_min_clr(const MinjerkTraj& tr, const std::vector<const Obstacle*>& hum, int K = 50) {
  double mn = 1e18;
  for (int j = 0; j < K; ++j) {
    double t = tr.t_start + (tr.t_end - tr.t_start) * (double)j / (K - 1);
    Eigen::Vector3d p = tr.eval(t);
    for (auto* h : hum) mn = std::min(mn, h->signed_dist(p, t));
  }
  return mn;
}

// dP/dq (M*6*3, nq) by one-time finite diff (fixed T -> linear; cheap, no clearance)
inline Eigen::MatrixXd af_dPdq(const Eigen::Vector3d& start, const Eigen::Vector3d& goal,
                               const Eigen::VectorXd& q, const Eigen::VectorXd& T) {
  const int M = (int)T.size(), nq = (int)q.size();
  auto flatP = [&](const Eigen::VectorXd& qq) {
    auto P = af_make_tr(start, goal, qq, T).control_points();
    Eigen::VectorXd v(M * 6 * 3);
    for (int i = 0; i < M; ++i)
      for (int k = 0; k < 6; ++k)
        for (int d = 0; d < 3; ++d) v[(i * 6 + k) * 3 + d] = P[i](k, d);
    return v;
  };
  Eigen::VectorXd base = flatP(q);
  Eigen::MatrixXd J(M * 6 * 3, nq);
  const double e = 1e-6;
  for (int j = 0; j < nq; ++j) {
    Eigen::VectorXd qp = q; qp[j] += e; J.col(j) = (flatP(qp) - base) / e;
  }
  return J;
}

// B: time-aware topology seed — per waypoint pick y nearest prev that clears all humans at pass-time.
inline Eigen::VectorXd af_time_aware_seed(const Eigen::Vector3d& start, const Eigen::Vector3d& goal,
                                          const std::vector<const Obstacle*>& hum, int M,
                                          const Eigen::VectorXd& T0, double d_safe, double ywin = 3.2) {
  Eigen::VectorXd cum(M + 1); cum(0) = 0; for (int i = 0; i < M; ++i) cum(i + 1) = cum(i) + T0(i);
  Eigen::VectorXd q(3 * (M - 1));
  double prev = 0.0;
  const int NY = 161;
  for (int i = 0; i < M - 1; ++i) {
    double frac = (double)(i + 1) / M;
    Eigen::Vector3d xb = start + frac * (goal - start);
    double ti = cum(i + 1);
    double best_y = 0, best_c = -1e18, ok_y = 0; bool have_ok = false; double ok_pen = 1e18;
    for (int n = 0; n < NY; ++n) {
      double y = -ywin + 2 * ywin * n / (NY - 1);
      Eigen::Vector3d p(xb[0], y, start[2]);
      double c = 1e18; for (auto* h : hum) c = std::min(c, h->signed_dist(p, ti));
      if (c > best_c) { best_c = c; best_y = y; }
      if (c >= d_safe + 0.15) { double pen = std::abs(y - prev); if (pen < ok_pen) { ok_pen = pen; ok_y = y; have_ok = true; } }
    }
    double yy = have_ok ? ok_y : best_y;
    q.segment(3 * i, 3) = Eigen::Vector3d(xb[0], yy, start[2]);
    prev = yy;
  }
  return q;
}

// Phase-1: feasibility restoration — gradient-descend total violation until min_clr >= d_safe.
inline Eigen::VectorXd af_restore(const Eigen::Vector3d& start, const Eigen::Vector3d& goal,
                                  const Eigen::VectorXd& q0, const Eigen::VectorXd& T,
                                  const std::vector<const Obstacle*>& hum, double d_safe, int iters = 200) {
  auto Vf = [&](const Eigen::VectorXd& q) {
    double mn = 1e18, sum = 0;
    MinjerkTraj tr = af_make_tr(start, goal, q, T);
    int K = 50;
    for (auto* h : hum) {
      double m = 1e18;
      for (int j = 0; j < K; ++j) { double t = tr.t_start + (tr.t_end - tr.t_start) * j / (K - 1); m = std::min(m, h->signed_dist(tr.eval(t), t)); }
      double v = std::max(d_safe - m, 0.0); sum += v * v; mn = std::min(mn, m);
    }
    return std::make_pair(sum, mn);
  };
  Eigen::VectorXd q = q0; const int nq = (int)q.size(); const double e = 1e-4;
  for (int it = 0; it < iters; ++it) {
    auto v0 = Vf(q); if (v0.second >= d_safe) return q;
    Eigen::VectorXd g(nq);
    for (int j = 0; j < nq; ++j) { Eigen::VectorXd qp = q; qp[j] += e; g[j] = (Vf(qp).first - v0.first) / e; }
    double ng = g.norm(); if (ng < 1e-9) break;
    Eigen::VectorXd d = -g / ng; double t = 0.6; bool stp = false;
    for (int bt = 0; bt < 34; ++bt) { Eigen::VectorXd qn = q + t * d; if (Vf(qn).first < v0.first - 1e-12) { q = qn; stp = true; break; } t *= 0.5; }
    if (!stp) break;
  }
  return q;
}

struct AFResult { Eigen::VectorXd q; std::vector<double> hist; double ms = 0; bool truncated = false; int iters = 0; };

// A: fast feasible-direction from feasible q0. deadline_ms<=0 => no deadline.
inline AFResult af_feasible_direction(const Eigen::Vector3d& start, const Eigen::Vector3d& goal,
                                      const Eigen::VectorXd& q0, const Eigen::VectorXd& T,
                                      const std::vector<const Obstacle*>& hum,
                                      const std::map<std::string, AvoidParams>& cfg,
                                      const CostGradOptParams& copt, const HardAlmOptParams& hopt,
                                      double d_safe, int iters = 25, double alpha = 0.5,
                                      double deadline_ms = -1, const Eigen::MatrixXd* corridor_ctr = nullptr) {
  using clk = std::chrono::high_resolution_clock;
  auto t0 = clk::now();
  const int M = (int)T.size(), nq = (int)q0.size(), H = (int)hum.size();
  Eigen::MatrixXd J = af_dPdq(start, goal, q0, T);   // one-time
  Eigen::VectorXd q = q0;
  Eigen::Vector3d z = Eigen::Vector3d::Zero();
  CostGradOptParams co = copt;
  if (corridor_ctr) { co.corridor_centers = corridor_ctr; co.corridor_radius = 0.5; co.w_corridor = 40.0; }
  AlmState alm;  // empty -> cost = smooth+vel+accel+time (+corridor); no hard penalty
  AFResult R; R.hist.push_back(af_min_clr(af_make_tr(start, goal, q, T), hum));
  for (int it = 0; it < iters; ++it) {
    if (deadline_ms > 0 && std::chrono::duration<double, std::milli>(clk::now() - t0).count() >= deadline_ms) { R.truncated = true; break; }
    Eigen::VectorXd x(nq + M); x.head(nq) = q; x.tail(M) = T;
    CostGradResult cg = minco_cost_grad(x, start, goal, M, {}, cfg, co, T.sum(), z, z, z, z, alm);
    Eigen::VectorXd gf = cg.grad.head(nq);
    MinjerkTraj tr = af_make_tr(start, goal, q, T);
    AlmConstraints C = alm_constraints(tr, hum, cfg, nullptr, hopt, nullptr);
    // direction u = -gf, Gauss-Seidel project onto active linearized constraints
    Eigen::VectorXd u = -gf / (gf.norm() + 1e-9);
    std::vector<int> act;
    for (int m = 0; m < C.Nc; ++m) if (C.g(m) > -0.2) act.push_back(m);
    for (int sweep = 0; sweep < 6; ++sweep) {
      for (int m : act) {
        int i = C.seg[m], k = C.kpt[m]; int b3 = (i * 6 + k) * 3;
        // grad g_m wrt q = -(n . dP/dq)  (n = outward normal = -dg/dP)
        Eigen::VectorXd Gm = -(C.n(m, 0) * J.row(b3) + C.n(m, 1) * J.row(b3 + 1) + C.n(m, 2) * J.row(b3 + 2)).transpose();
        double bm = -alpha * C.g(m);
        double viol = Gm.dot(u) - bm;
        if (viol > 0) u -= (viol / (Gm.squaredNorm() + 1e-12)) * Gm;
      }
    }
    if (u.norm() < 1e-7) break;
    // backtrack to REAL continuous-time feasibility + cost decrease
    double t = 1.0; bool moved = false; double f0 = cg.f;
    for (int bt = 0; bt < 12; ++bt) {
      Eigen::VectorXd qn = q + t * u;
      MinjerkTraj trn = af_make_tr(start, goal, qn, T);
      if (af_min_clr(trn, hum) >= d_safe - 1e-9) {
        Eigen::VectorXd xn(nq + M); xn.head(nq) = qn; xn.tail(M) = T;
        CostGradResult cn = minco_cost_grad(xn, start, goal, M, {}, cfg, co, T.sum(), z, z, z, z, alm);
        if (cn.f <= f0 + 1e-9) { q = qn; moved = true; break; }
      }
      t *= 0.5;
    }
    R.hist.push_back(af_min_clr(af_make_tr(start, goal, q, T), hum));
    R.iters = it + 1;
    if (!moved) break;
  }
  R.q = q;
  R.ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
  return R;
}

}  // namespace sando
