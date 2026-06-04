// MinjerkTraj — faithful C++ port of sando_py/local/minco.py:MinjerkTraj.
// MINCO s=3 quintic minimum-jerk trajectory: parameterized by waypoints + per-segment
// durations; coefficients recovered by solving the banded system M(T) c = b(q).
// 1:1 with the Python: same constraint-row layout (see minco.py docstring), same C2B.
// 必须还原:每个公式都对照 Python 原样移植;由 golden 测试逐项验证。
#pragma once
#include <Eigen/Dense>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <utility>
#include <tuple>

namespace sando {

// Power -> Bernstein constant matrix for a degree-5 polynomial (minco.py C2B).
inline Eigen::Matrix<double, 6, 6> C2B_matrix() {
  Eigen::Matrix<double, 6, 6> C;
  C << 60, 0, 0, 0, 0, 0,
       60, 12, 0, 0, 0, 0,
       60, 24, 6, 0, 0, 0,
       60, 36, 18, 6, 0, 0,
       60, 48, 36, 24, 12, 0,
       60, 60, 60, 60, 60, 60;
  return C / 60.0;
}

class MinjerkTraj {
 public:
  static constexpr int DEG = 5;
  static constexpr int NC = 6;

  int M = 0;
  int d = 3;
  Eigen::VectorXd T;          // (M)
  Eigen::MatrixXd waypoints;  // (M+1, 3)
  Eigen::Vector3d p_start, p_goal, v0, a0, vf, af;
  Eigen::MatrixXd q;          // (M-1, 3) interior
  double t_start = 0.0, t_end = 0.0;
  Eigen::VectorXd cum;        // (M+1) cumulative segment-start times
  Eigen::MatrixXd c;          // (6M, 3) coefficients ascending in power per segment
  Eigen::MatrixXd A_;         // dense M(T), kept for the M1 adjoint solve M^T x = rhs

  MinjerkTraj(const Eigen::MatrixXd& wp, const Eigen::VectorXd& durations,
              const Eigen::Vector3d& v0_ = Eigen::Vector3d::Zero(),
              const Eigen::Vector3d& a0_ = Eigen::Vector3d::Zero(),
              const Eigen::Vector3d& vf_ = Eigen::Vector3d::Zero(),
              const Eigen::Vector3d& af_ = Eigen::Vector3d::Zero()) {
    M = static_cast<int>(durations.size());
    if (wp.rows() != M + 1 || wp.cols() != 3)
      throw std::invalid_argument("waypoints_full must be (M+1, 3)");
    if (M < 1) throw std::invalid_argument("need at least one segment");
    for (int i = 0; i < M; ++i)
      if (durations(i) <= 0.0) throw std::invalid_argument("all durations T_i must be > 0");

    T = durations;
    waypoints = wp;
    p_start = wp.row(0).transpose();
    p_goal = wp.row(M).transpose();
    q = (M > 1) ? Eigen::MatrixXd(wp.middleRows(1, M - 1)) : Eigen::MatrixXd(0, 3);
    v0 = v0_; a0 = a0_; vf = vf_; af = af_;

    cum.resize(M + 1);
    cum(0) = 0.0;
    for (int i = 0; i < M; ++i) cum(i + 1) = cum(i) + T(i);
    t_start = 0.0;
    t_end = cum(M);

    solve();
  }

  // ---- banded M(T) c = b, solved densely (same linear system as scipy solve_banded) ----
  void solve() {
    const int n = NC * M;
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n, n);
    Eigen::MatrixXd b = Eigen::MatrixXd::Zero(n, 3);

    // head block: p(0)=p_start, v(0)=v0, a(0)=a0
    A(0, 0) = 1.0;
    A(1, 1) = 1.0;
    A(2, 2) = 2.0;
    b.row(0) = p_start.transpose();
    b.row(1) = v0.transpose();
    b.row(2) = a0.transpose();

    // interior junctions
    for (int i = 0; i < M - 1; ++i) {
      const double t1 = T(i), t2 = t1 * t1, t3 = t2 * t1, t4 = t2 * t2, t5 = t4 * t1;
      const int base = 6 * i;
      A(base + 3, base + 3) = 6.0;
      A(base + 3, base + 4) = 24.0 * t1;
      A(base + 3, base + 5) = 60.0 * t2;
      A(base + 3, base + 9) = -6.0;

      A(base + 4, base + 4) = 24.0;
      A(base + 4, base + 5) = 120.0 * t1;
      A(base + 4, base + 10) = -24.0;

      A(base + 5, base + 0) = 1.0;
      A(base + 5, base + 1) = t1;
      A(base + 5, base + 2) = t2;
      A(base + 5, base + 3) = t3;
      A(base + 5, base + 4) = t4;
      A(base + 5, base + 5) = t5;

      A(base + 6, base + 0) = 1.0;
      A(base + 6, base + 1) = t1;
      A(base + 6, base + 2) = t2;
      A(base + 6, base + 3) = t3;
      A(base + 6, base + 4) = t4;
      A(base + 6, base + 5) = t5;
      A(base + 6, base + 6) = -1.0;

      A(base + 7, base + 1) = 1.0;
      A(base + 7, base + 2) = 2.0 * t1;
      A(base + 7, base + 3) = 3.0 * t2;
      A(base + 7, base + 4) = 4.0 * t3;
      A(base + 7, base + 5) = 5.0 * t4;
      A(base + 7, base + 7) = -1.0;

      A(base + 8, base + 2) = 2.0;
      A(base + 8, base + 3) = 6.0 * t1;
      A(base + 8, base + 4) = 12.0 * t2;
      A(base + 8, base + 5) = 20.0 * t3;
      A(base + 8, base + 8) = -2.0;

      b.row(base + 5) = q.row(i);
    }

    // tail block: p(T_last)=p_goal, v=vf, a=af  (segment M-1)
    const int last = M - 1;
    const int bN = 6 * M;
    const double t1 = T(last), t2 = t1 * t1, t3 = t2 * t1, t4 = t2 * t2, t5 = t4 * t1;
    A(bN - 3, bN - 6) = 1.0;
    A(bN - 3, bN - 5) = t1;
    A(bN - 3, bN - 4) = t2;
    A(bN - 3, bN - 3) = t3;
    A(bN - 3, bN - 2) = t4;
    A(bN - 3, bN - 1) = t5;

    A(bN - 2, bN - 5) = 1.0;
    A(bN - 2, bN - 4) = 2.0 * t1;
    A(bN - 2, bN - 3) = 3.0 * t2;
    A(bN - 2, bN - 2) = 4.0 * t3;
    A(bN - 2, bN - 1) = 5.0 * t4;

    A(bN - 1, bN - 4) = 2.0;
    A(bN - 1, bN - 3) = 6.0 * t1;
    A(bN - 1, bN - 2) = 12.0 * t2;
    A(bN - 1, bN - 1) = 20.0 * t3;

    b.row(bN - 3) = p_goal.transpose();
    b.row(bN - 2) = vf.transpose();
    b.row(bN - 1) = af.transpose();

    A_ = A;                          // keep M(T) for the adjoint M^T x = rhs
    c = A.partialPivLu().solve(b);  // dense LU == scipy banded LU solution
  }

  // searchsorted(cum, tc, side='right') - 1, clipped to [0, M-1]  (matches minco.py)
  int locate_idx(double tc) const {
    const double* first = cum.data();
    const double* last = cum.data() + (M + 1);
    int j = static_cast<int>(std::upper_bound(first, last, tc) - first);  // count of cum<=tc
    int idx = j - 1;
    if (idx < 0) idx = 0;
    if (idx > M - 1) idx = M - 1;
    return idx;
  }

  // order-th derivative at scalar t. order 0 = position.
  Eigen::Vector3d eval_deriv(double t, int order = 0) const {
    if (order < 0) throw std::invalid_argument("order must be >= 0");
    if (order > DEG) return Eigen::Vector3d::Zero();
    // only float-noise overshoot (<=1e-6) is clamped; beyond that = segment misalignment, surface it
    if (t < t_start - 1e-6 || t > t_end + 1e-6)
      throw std::out_of_range("eval_deriv: t outside [t_start, t_end] beyond 1e-6 float noise");
    const double tc = std::min(std::max(t, t_start), t_end);
    const int idx = locate_idx(tc);
    const double tau = tc - cum(idx);
    Eigen::Matrix<double, 1, 6> B = Eigen::Matrix<double, 1, 6>::Zero();
    for (int p = order; p < 6; ++p) {
      double coeff = 1.0;
      for (int m = 0; m < order; ++m) coeff *= static_cast<double>(p - m);
      B(p) = coeff * std::pow(tau, p - order);
    }
    Eigen::Matrix<double, 6, 3> cseg = c.block(6 * idx, 0, 6, 3);
    return (B * cseg).transpose();
  }

  Eigen::Vector3d eval(double t) const { return eval_deriv(t, 0); }

  // Bernstein control points of every segment, (M,6,3) -> returned as M blocks of 6x3.
  std::vector<Eigen::Matrix<double, 6, 3>> control_points() const {
    const Eigen::Matrix<double, 6, 6> C2B = C2B_matrix();
    std::vector<Eigen::Matrix<double, 6, 3>> out(M);
    for (int i = 0; i < M; ++i) {
      const double t1 = T(i), t2 = t1 * t1, t3 = t2 * t1, t4 = t2 * t2, t5 = t4 * t1;
      Eigen::Matrix<double, 6, 1> D;
      D << 1.0, t1, t2, t3, t4, t5;
      Eigen::Matrix<double, 6, 3> cseg = c.block(6 * i, 0, 6, 3);
      Eigen::Matrix<double, 6, 3> a = D.asDiagonal() * cseg;  // a_{j} = c[6i+j] * T_i^j
      out[i] = C2B * a;
    }
    return out;
  }

  // Wall-clock time of every Bernstein control point, (M,6): cum[i] + (k/5)*T_i.
  Eigen::MatrixXd control_point_times() const {
    Eigen::MatrixXd out(M, 6);
    for (int i = 0; i < M; ++i)
      for (int k = 0; k < 6; ++k)
        out(i, k) = cum(i) + (static_cast<double>(k) / DEG) * T(i);
    return out;
  }

  // ===================================================================
  // M1 — analytic gradient through the MINCO map  M(T) c = b(q)
  // ===================================================================
  // (J, dJ/dc) closed form; dJ/dc shape (6M,3).
  std::pair<double, Eigen::MatrixXd> energy_and_gradc() const {
    Eigen::MatrixXd gdC = Eigen::MatrixXd::Zero(6 * M, 3);
    double J = 0.0;
    for (int i = 0; i < M; ++i) {
      const double t1 = T(i), t2 = t1 * t1, t3 = t2 * t1, t4 = t2 * t2, t5 = t4 * t1;
      const int b = 6 * i;
      Eigen::Vector3d c3 = c.row(b + 3).transpose();
      Eigen::Vector3d c4 = c.row(b + 4).transpose();
      Eigen::Vector3d c5 = c.row(b + 5).transpose();
      J += 36.0 * t1 * c3.dot(c3) + 144.0 * t2 * c3.dot(c4) + 192.0 * t3 * c4.dot(c4) +
           240.0 * t3 * c3.dot(c5) + 720.0 * t4 * c4.dot(c5) + 720.0 * t5 * c5.dot(c5);
      gdC.row(b + 3) = (72.0 * t1 * c3 + 144.0 * t2 * c4 + 240.0 * t3 * c5).transpose();
      gdC.row(b + 4) = (144.0 * t2 * c3 + 384.0 * t3 * c4 + 720.0 * t4 * c5).transpose();
      gdC.row(b + 5) = (240.0 * t3 * c3 + 720.0 * t4 * c4 + 1440.0 * t5 * c5).transpose();
    }
    return {J, gdC};
  }

  double energy() const { return energy_and_gradc().first; }

  Eigen::VectorXd energy_grad_time_explicit() const {
    Eigen::VectorXd gdT = Eigen::VectorXd::Zero(M);
    for (int i = 0; i < M; ++i) {
      const double t1 = T(i), t2 = t1 * t1, t3 = t2 * t1, t4 = t2 * t2;
      const int b = 6 * i;
      Eigen::Vector3d c3 = c.row(b + 3).transpose();
      Eigen::Vector3d c4 = c.row(b + 4).transpose();
      Eigen::Vector3d c5 = c.row(b + 5).transpose();
      gdT(i) = 36.0 * c3.dot(c3) + 288.0 * t1 * c3.dot(c4) + 576.0 * t2 * c4.dot(c4) +
               720.0 * t2 * c3.dot(c5) + 2880.0 * t3 * c4.dot(c5) + 3600.0 * t4 * c5.dot(c5);
    }
    return gdT;
  }

  // solve M(T)^T x = rhs  (rhs (6M,3) -> x (6M,3))
  Eigen::MatrixXd solve_adjoint(const Eigen::MatrixXd& rhs) const {
    return A_.transpose().partialPivLu().solve(rhs);
  }

  // (dM/dT_k) c, shape (6M,3) — only rows containing T_k are nonzero.
  Eigen::MatrixXd dM_dT_times_c(int k) const {
    const int n = 6 * M;
    Eigen::MatrixXd out = Eigen::MatrixXd::Zero(n, 3);
    const double Tk = T(k);
    auto ci = [&](int base, int j) { return Eigen::Vector3d(c.row(base + j).transpose()); };
    if (k < M - 1) {
      const int b = 6 * k;
      const double T2 = Tk * Tk, T3 = T2 * Tk, T4 = T2 * T2;
      out.row(b + 3) = (24.0 * ci(b, 4) + 120.0 * Tk * ci(b, 5)).transpose();
      out.row(b + 4) = (120.0 * ci(b, 5)).transpose();
      out.row(b + 5) = (ci(b, 1) + 2.0 * Tk * ci(b, 2) + 3.0 * T2 * ci(b, 3) +
                        4.0 * T3 * ci(b, 4) + 5.0 * T4 * ci(b, 5)).transpose();
      out.row(b + 6) = out.row(b + 5);
      out.row(b + 7) = (2.0 * ci(b, 2) + 6.0 * Tk * ci(b, 3) + 12.0 * T2 * ci(b, 4) +
                        20.0 * T3 * ci(b, 5)).transpose();
      out.row(b + 8) = (6.0 * ci(b, 3) + 24.0 * Tk * ci(b, 4) + 60.0 * T2 * ci(b, 5)).transpose();
    } else {
      const int b = 6 * (M - 1), bN = n;
      const double T2 = Tk * Tk, T3 = T2 * Tk, T4 = T2 * T2;
      out.row(bN - 3) = (ci(b, 1) + 2.0 * Tk * ci(b, 2) + 3.0 * T2 * ci(b, 3) +
                         4.0 * T3 * ci(b, 4) + 5.0 * T4 * ci(b, 5)).transpose();
      out.row(bN - 2) = (2.0 * ci(b, 2) + 6.0 * Tk * ci(b, 3) + 12.0 * T2 * ci(b, 4) +
                         20.0 * T3 * ci(b, 5)).transpose();
      out.row(bN - 1) = (6.0 * ci(b, 3) + 24.0 * Tk * ci(b, 4) + 60.0 * T2 * ci(b, 5)).transpose();
    }
    return out;
  }

  // backprop external dCost/dc (+ optional explicit dCost/dT) -> (gq (M-1,3), gT (M))
  std::pair<Eigen::MatrixXd, Eigen::VectorXd> grad_from_dcost_dc(
      const Eigen::MatrixXd& dcost_dc, const Eigen::VectorXd* dcost_dT_explicit = nullptr) const {
    Eigen::MatrixXd lam = solve_adjoint(dcost_dc);  // (6M,3)
    Eigen::MatrixXd gq = Eigen::MatrixXd::Zero(std::max(M - 1, 0), 3);
    for (int i = 0; i < M - 1; ++i) gq.row(i) = lam.row(6 * i + 5);
    Eigen::VectorXd gT = Eigen::VectorXd::Zero(M);
    if (dcost_dT_explicit) gT = *dcost_dT_explicit;
    for (int k = 0; k < M; ++k) {
      Eigen::MatrixXd dMc = dM_dT_times_c(k);
      gT(k) -= (lam.array() * dMc.array()).sum();
    }
    return {gq, gT};
  }

  std::tuple<double, Eigen::MatrixXd, Eigen::VectorXd> energy_grad() const {
    auto eg = energy_and_gradc();
    Eigen::VectorXd gdT = energy_grad_time_explicit();
    auto g = grad_from_dcost_dc(eg.second, &gdT);
    return {eg.first, g.first, g.second};
  }

  // explicit dP_i/dT_i (holding c fixed), per segment 6x3.
  std::vector<Eigen::Matrix<double, 6, 3>> control_points_dT_explicit() const {
    const Eigen::Matrix<double, 6, 6> C2B = C2B_matrix();
    std::vector<Eigen::Matrix<double, 6, 3>> out(M);
    for (int i = 0; i < M; ++i) {
      const double t1 = T(i);
      Eigen::Matrix<double, 6, 1> dD;
      dD << 0.0, 1.0, 2.0 * t1, 3.0 * t1 * t1, 4.0 * std::pow(t1, 3), 5.0 * std::pow(t1, 4);
      Eigen::Matrix<double, 6, 3> cseg = c.block(6 * i, 0, 6, 3);
      Eigen::Matrix<double, 6, 3> da = dD.asDiagonal() * cseg;
      out[i] = C2B * da;
    }
    return out;
  }
};

}  // namespace sando
