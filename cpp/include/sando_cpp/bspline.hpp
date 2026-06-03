// UniformBSpline — faithful C++ port of sando_py/local/bspline.py (legacy pre-MINCO backbone).
// Uniform-knot B-spline: De Boor eval, derivative-via-difference-control-points, Cox-de Boor
// nonzero basis, LSQ fit_path, endpoint clamp. 1:1 with Python; golden-verified.
#pragma once
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace sando {

class UniformBSpline {
 public:
  Eigen::MatrixXd ctrl;   // (N+1, d)
  int p;                  // degree
  double dt;
  int n;                  // last index of P (N)
  int d;
  int m;                  // last knot index
  Eigen::VectorXd knots;  // (m+1)
  double t_start, t_end, T;

  UniformBSpline(const Eigen::MatrixXd& ctrl_, int degree = 5, double dt_ = 1.0) {
    if (ctrl_.cols() < 1) throw std::invalid_argument("ctrl must be (N+1,d)");
    if (degree < 1) throw std::invalid_argument("degree must be >= 1");
    if (dt_ <= 0.0) throw std::invalid_argument("dt must be > 0");
    ctrl = ctrl_; p = degree; dt = dt_;
    n = (int)ctrl.rows() - 1; d = (int)ctrl.cols();
    if (n < p) throw std::invalid_argument("need >= p+1 control points");
    m = n + p + 1;
    knots.resize(m + 1);
    for (int i = 0; i <= m; ++i) knots(i) = (double(i) - p) * dt;
    t_start = knots(p); t_end = knots(n + 1); T = t_end - t_start;
  }

  int find_span(double t) const {
    t = std::min(std::max(t, t_start), t_end);
    if (t >= t_end - 1e-12) return n;
    int lo = p, hi = n;
    while (lo < hi) { int mid = (lo + hi + 1) / 2; if (knots(mid) <= t) lo = mid; else hi = mid - 1; }
    return lo;
  }

  // De Boor on arbitrary (ctrl, p, knots) with the spline's [t_start,t_end] clamp.
  static Eigen::VectorXd de_boor_with(double t, const Eigen::MatrixXd& C, int p,
                                      const Eigen::VectorXd& kn, double t_start, double t_end) {
    t = std::min(std::max(t, t_start), t_end);
    int n_local = (int)C.rows() - 1, k;
    if (t >= t_end - 1e-12) k = n_local;
    else { int lo = p, hi = n_local;
      while (lo < hi) { int mid = (lo + hi + 1) / 2; if (kn(mid) <= t) lo = mid; else hi = mid - 1; }
      k = lo; }
    std::vector<Eigen::VectorXd> dd(p + 1);
    for (int j = 0; j <= p; ++j) dd[j] = C.row(k - p + j).transpose();
    for (int r = 1; r <= p; ++r)
      for (int j = p; j >= r; --j) {
        int i_lo = k - p + j;
        double denom = kn(i_lo + p + 1 - r) - kn(i_lo);
        double alpha = (denom == 0.0) ? 0.0 : (t - kn(i_lo)) / denom;
        dd[j] = (1.0 - alpha) * dd[j - 1] + alpha * dd[j];
      }
    return dd[p];
  }

  Eigen::VectorXd eval(double t) const { return de_boor_with(t, ctrl, p, knots, t_start, t_end); }

  Eigen::VectorXd eval_deriv(double t, int order = 1) const {
    if (order < 0) throw std::invalid_argument("order must be >= 0");
    if (order == 0) return eval(t);
    if (order > p) return Eigen::VectorXd::Zero(d);
    Eigen::MatrixXd Q = ctrl;
    Eigen::VectorXd kn = knots;
    int pp = p;
    for (int o = 0; o < order; ++o) {
      Eigen::MatrixXd Qn(Q.rows() - 1, Q.cols());
      for (int i = 0; i < Qn.rows(); ++i) Qn.row(i) = (Q.row(i + 1) - Q.row(i)) / dt;
      Q = Qn;
      kn = kn.segment(1, kn.size() - 2).eval();  // knots[1:-1]
      pp -= 1;
    }
    return de_boor_with(t, Q, pp, kn, t_start, t_end);
  }

  // (span k, basis values [N_{k-p,p}..N_{k,p}]) — Cox-de Boor triangular recursion.
  std::pair<int, Eigen::VectorXd> nonzero_basis(double t) const {
    t = std::min(std::max(t, t_start), t_end);
    int k = find_span(t);
    Eigen::VectorXd N = Eigen::VectorXd::Zero(p + 1);
    N(0) = 1.0;
    Eigen::VectorXd left = Eigen::VectorXd::Zero(p + 1), right = Eigen::VectorXd::Zero(p + 1);
    for (int j = 1; j <= p; ++j) {
      left(j) = t - knots(k + 1 - j);
      right(j) = knots(k + j) - t;
      double saved = 0.0;
      for (int r = 0; r < j; ++r) {
        double denom = right(r + 1) + left(j - r);
        double temp = (denom != 0.0) ? N(r) / denom : 0.0;
        N(r) = saved + right(r + 1) * temp;
        saved = left(j - r) * temp;
      }
      N(j) = saved;
    }
    return {k, N};
  }

  void lock_endpoint(const std::string& side, const Eigen::VectorXd& position) {
    if (side == "start") for (int i = 0; i <= p; ++i) ctrl.row(i) = position.transpose();
    else if (side == "end") for (int i = 0; i <= p; ++i) ctrl.row(n - p + i) = position.transpose();
    else throw std::invalid_argument("side must be 'start' or 'end'");
  }

  static UniformBSpline fit_path(const Eigen::MatrixXd& path, int num_ctrl, int degree = 5,
                                 double dt = 1.0, bool clamp_endpoints = false) {
    int M = (int)path.rows(), dd = (int)path.cols(), N = num_ctrl - 1;
    if (N < degree) throw std::invalid_argument("num_ctrl >= degree+1");
    if (clamp_endpoints && num_ctrl < 2 * (degree + 1))
      throw std::invalid_argument("clamp needs num_ctrl >= 2*(degree+1)");
    UniformBSpline bs(Eigen::MatrixXd::Zero(num_ctrl, dd), degree, dt);
    Eigen::VectorXd ts(M);
    for (int i = 0; i < M; ++i)
      ts(i) = (M == 1) ? bs.t_start : (bs.t_start + (bs.t_end - bs.t_start) * (double(i) / (M - 1)));
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(M, N + 1);
    for (int i = 0; i < M; ++i) {
      auto kn = bs.nonzero_basis(ts(i));
      for (int c = 0; c <= degree; ++c) B(i, kn.first - degree + c) = kn.second(c);
    }
    // least squares B ctrl ~= path (SVD, matches numpy.linalg.lstsq rcond=None)
    Eigen::MatrixXd ctrl = B.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(path);
    bs.ctrl = ctrl;
    if (clamp_endpoints) {
      bs.lock_endpoint("start", path.row(0).transpose());
      bs.lock_endpoint("end", path.row(M - 1).transpose());
    }
    return bs;
  }
};

}  // namespace sando
