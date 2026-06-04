// cost — faithful C++ port of sando_py/local/cost.py.
// EGO cubic penalty phi(d) = (d_safe - d)^3 if d<d_safe else 0; obstacle_cost averages it
// over K uniform-time samples of the trajectory, per-class weighted, divided by K (K-invariant).
#pragma once
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <map>
#include "sando_cpp/obstacles.hpp"
#include "sando_cpp/avoid_config.hpp"

namespace sando {

inline double penalty(double d, double d_safe) {
  const double diff = d_safe - d;
  return diff > 0.0 ? diff * diff * diff : 0.0;
}

// `Spline` must provide: eval(double t) -> Vector3d, members t_start, t_end.
// K must be given explicitly here (Python's K=None default uses spline.n; the MINCO
// backbone has no .n, so callers pass K — golden uses an explicit K too).
template <class Spline>
double obstacle_cost(const Spline& spline, const std::vector<const Obstacle*>& obstacles,
                     const std::map<std::string, AvoidParams>& config, int K) {
  if (obstacles.empty()) return 0.0;
  // ts = linspace(t_start, t_end, K)
  std::vector<double> ts(K);
  const double a = spline.t_start, b = spline.t_end;
  for (int i = 0; i < K; ++i)
    ts[i] = (K == 1) ? a : (a + (b - a) * (double(i) / (K - 1)));
  // precompute positions (same as Python pts = spline.eval(ts))
  std::vector<Eigen::Vector3d> pts(K);
  for (int i = 0; i < K; ++i) pts[i] = spline.eval(ts[i]);

  double total = 0.0;
  for (const Obstacle* obs : obstacles) {       // obs outer, i inner (matches Python order)
    const AvoidParams& p = config.at(obs->class_name);
    const double d_safe = p.d_safe, w = p.weight;
    for (int i = 0; i < K; ++i) {
      const double d = obs->signed_dist(pts[i], ts[i]);
      const double diff = d_safe - d;
      if (diff > 0.0) total += w * diff * diff * diff;
    }
  }
  return total / K;
}

}  // namespace sando
