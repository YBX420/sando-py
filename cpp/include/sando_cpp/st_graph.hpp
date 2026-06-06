// st_graph.hpp — space-time FRONT-END: station-time (ST-graph) speed planner.
//
// The heat-A* guide owns WHERE / which-side (lateral); this owns WHEN (timing). Project the analytic
// predicted movers onto the (s = arc-length along the guide, t) plane -> a station is BLOCKED at time t
// if a mover is within d_safe of the guide point there. A 1-D DP (monotone in s = a DAG) finds the
// fastest velocity-bounded s(t) profile that avoids the blocked cells: it SLOWS where the gap is shut
// and GOES when it opens. The profile rewrites the MINCO seed segment TIMES T0 (and T_target=T0.sum(),
// the dominant anchor under w_time=2500) so the solve TIMES the gap instead of braking to 0 and waiting.
// Deterministic, O(NS*NT^2) ~ sub-ms, no MIQP/learning. q0 (lateral) is UNCHANGED. Default OFF.
#pragma once
#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>
#include "sando_cpp/obstacles.hpp"

namespace sando {

struct StGraphParams {
  double d_safe_dyn = 0.5;   // mover keep-out for the blocked test
  double Th = 1.5;           // time horizon (s)
  double vmax = 12.0;
  double dt_min = 0.05;      // min segment time
  int    NS = 12;            // stations along the guide (need NT >> NS so vmax is representable)
  int    NT = 48;            // time levels in [0, Th]
  double w_wait = 1.0;       // extra cost per second of slow/wait (tie-break toward progress)
};

struct StProfile {
  std::vector<double> s;     // station arc-lengths
  std::vector<double> tau;   // chosen arrival time at each station (monotone)
  std::vector<char>   waited;// 1 if this station's edge was a slow/wait
  bool   feasible = false;
  double t_total = 0.0;
};

// Resample a guide polyline (rows = points) into NS stations by arc-length; return points + cum-s.
inline std::vector<std::pair<Eigen::Vector3d, double>> sample_stations(
    const Eigen::MatrixXd& guide, int NS) {
  std::vector<std::pair<Eigen::Vector3d, double>> out;
  const int n = (int)guide.rows();
  if (n == 0) return out;
  std::vector<double> cum(n, 0.0);
  for (int i = 1; i < n; ++i)
    cum[i] = cum[i - 1] + (guide.row(i) - guide.row(i - 1)).norm();
  double total = cum[n - 1];
  if (total < 1e-9 || n == 1 || NS < 2) {              // degenerate -> single station
    out.emplace_back(guide.row(0).transpose(), 0.0);
    return out;
  }
  out.reserve(NS);
  int seg = 0;
  for (int k = 0; k < NS; ++k) {
    double sk = total * double(k) / double(NS - 1);
    while (seg < n - 2 && cum[seg + 1] < sk) ++seg;
    double seglen = std::max(cum[seg + 1] - cum[seg], 1e-9);
    double a = std::min(std::max((sk - cum[seg]) / seglen, 0.0), 1.0);
    Eigen::Vector3d p = (1.0 - a) * guide.row(seg).transpose() + a * guide.row(seg + 1).transpose();
    out.emplace_back(p, sk);
  }
  return out;
}

// blocked[k][j] = a mover is within radius+d_safe of station k at time t_j = Th*j/(NT-1).
inline std::vector<std::vector<char>> build_blocked(
    const std::vector<std::pair<Eigen::Vector3d, double>>& stations,
    const std::vector<const Obstacle*>& movers, const StGraphParams& gp) {
  const int NS = (int)stations.size();
  std::vector<std::vector<char>> blk(NS, std::vector<char>(gp.NT, 0));
  for (int k = 0; k < NS; ++k) {
    const Eigen::Vector3d& p = stations[k].first;
    for (int j = 0; j < gp.NT; ++j) {
      double t = (gp.NT > 1) ? gp.Th * double(j) / double(gp.NT - 1) : 0.0;
      for (const Obstacle* o : movers) {
        const SphereObstacle* s = dynamic_cast<const SphereObstacle*>(o);
        if (!s) continue;
        if ((p - s->predict(t)).norm() < s->radius + gp.d_safe_dyn) { blk[k][j] = 1; break; }
      }
    }
  }
  return blk;
}

// DP for the fastest velocity-bounded s(t) avoiding blocked cells. Start at (station 0, t=0).
inline StProfile solve_st_graph(
    const std::vector<std::pair<Eigen::Vector3d, double>>& stations,
    const std::vector<std::vector<char>>& blk, const StGraphParams& gp) {
  StProfile prof;
  const int NS = (int)stations.size();
  if (NS < 2) return prof;
  const int NT = gp.NT;
  auto tlev = [&](int j) { return (NT > 1) ? gp.Th * double(j) / double(NT - 1) : 0.0; };
  const double INF = std::numeric_limits<double>::infinity();
  std::vector<std::vector<double>> cost(NS, std::vector<double>(NT, INF));
  std::vector<std::vector<int>>    par(NS, std::vector<int>(NT, -1));
  if (blk[0][0]) return prof;                          // drone's own start blocked -> infeasible
  cost[0][0] = 0.0;
  for (int k = 0; k < NS - 1; ++k) {
    double ds = stations[k + 1].second - stations[k].second;
    for (int j = 0; j < NT; ++j) {
      if (!std::isfinite(cost[k][j])) continue;
      for (int jp = j + 1; jp < NT; ++jp) {            // time strictly advances
        if (blk[k + 1][jp]) continue;
        double dt = tlev(jp) - tlev(j);
        if (dt < gp.dt_min) continue;
        double v = ds / dt;
        if (v > gp.vmax) continue;                     // too fast for this gap -> skip (slower jp later)
        double slow = (v < 0.5 * gp.vmax) ? gp.w_wait * dt : 0.0;   // gentle bias toward progress
        double nc = cost[k][j] + dt + slow;
        if (nc < cost[k + 1][jp]) { cost[k + 1][jp] = nc; par[k + 1][jp] = j; }
      }
    }
  }
  int bestj = -1; double bestc = INF;
  for (int j = 0; j < NT; ++j) if (cost[NS - 1][j] < bestc) { bestc = cost[NS - 1][j]; bestj = j; }
  if (bestj < 0) return prof;                          // last station unreachable within Th -> honest hold
  // backtrack
  std::vector<int> jpath(NS, 0);
  int j = bestj;
  for (int k = NS - 1; k >= 0; --k) { jpath[k] = j; if (k > 0) j = par[k][j]; }
  prof.s.resize(NS); prof.tau.resize(NS); prof.waited.assign(NS, 0);
  for (int k = 0; k < NS; ++k) { prof.s[k] = stations[k].second; prof.tau[k] = tlev(jpath[k]); }
  for (int k = 0; k + 1 < NS; ++k) {
    double dt = prof.tau[k + 1] - prof.tau[k], ds = prof.s[k + 1] - prof.s[k];
    if (dt > 1e-9 && ds / dt < 0.5 * gp.vmax) prof.waited[k] = 1;
  }
  prof.feasible = true; prof.t_total = prof.tau[NS - 1];
  return prof;
}

// Map the s(t) profile onto per-segment seed times T0 for the M+1 MINCO waypoints `wp` (cum arc-length).
inline Eigen::VectorXd profile_to_T0(const StProfile& prof, const Eigen::MatrixXd& wp,
                                     int M, double dt_min) {
  Eigen::VectorXd T0(M);
  std::vector<double> sw(M + 1, 0.0);
  for (int i = 1; i <= M; ++i) sw[i] = sw[i - 1] + (wp.row(i) - wp.row(i - 1)).norm();
  auto interp_tau = [&](double s) -> double {          // monotone interp of tau(s)
    if (prof.s.empty()) return 0.0;
    if (s <= prof.s.front()) return prof.tau.front();
    if (s >= prof.s.back())  return prof.tau.back();
    int lo = 0, hi = (int)prof.s.size() - 1;
    while (hi - lo > 1) { int mid = (lo + hi) / 2; (prof.s[mid] <= s ? lo : hi) = mid; }
    double a = (s - prof.s[lo]) / std::max(prof.s[hi] - prof.s[lo], 1e-9);
    return (1.0 - a) * prof.tau[lo] + a * prof.tau[hi];
  };
  for (int i = 0; i < M; ++i) {
    double t0 = interp_tau(sw[i]), t1 = interp_tau(sw[i + 1]);
    T0(i) = std::max(t1 - t0, dt_min * 2.0);
  }
  return T0;
}

}  // namespace sando
