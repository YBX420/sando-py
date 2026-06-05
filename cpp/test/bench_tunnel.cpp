// bench_tunnel — A+B 闭环 + tunnel(y-band)约束:扫不同 lane 宽度,看能否在 tunnel 内穿过移动人群。
// 证明结构:安全(人体 + tunnel)对任意截断恒成立(构造性);能否到达 = lane 内是否存在时机缝(场景相关)。
#include <cstdio>
#include <vector>
#include <algorithm>
#include <memory>
#include "sando_cpp/anytime_feasible.hpp"
using namespace sando;

struct Spec { double x, y0, vy; };
static std::vector<Spec> SPEC = {{3,-1.5,0.55},{5,1.5,-0.55},{7,-1.5,0.55},{9,1.5,-0.55},{11,-1.5,0.55},{13,1.5,-0.55}};
static void humans_at(double t, std::vector<std::unique_ptr<SphereObstacle>>& own, std::vector<const Obstacle*>& hum) {
  own.clear(); hum.clear();
  for (auto& s : SPEC) { Eigen::Vector3d c(s.x, s.y0 + s.vy * t, 2.0);
    own.push_back(std::make_unique<SphereObstacle>(c, 0.3, Eigen::Vector3d(0, s.vy, 0), "human")); hum.push_back(own.back().get()); }
}
static double true_clr(const Eigen::Vector3d& p, double t) {
  double mn = 1e18; for (auto& s : SPEC) { Eigen::Vector3d c(s.x, s.y0 + s.vy * t, 2.0); mn = std::min(mn, (p - c).norm() - 0.3); } return mn;
}

int main() {
  auto cfg = default_config(); double d_safe = cfg["human"].d_safe;
  Eigen::Vector3d start(0, 0, 2), goal(14, 0, 2);
  CostGradOptParams copt; copt.vmax = 4; copt.amax = 8; copt.spacetime_hard = true; copt.tau_trust = 100;
  HardAlmOptParams hopt; hopt.spacetime_hard = true; hopt.tau_trust = 100;
  const double RD = 0.25, T_MAX = 18.0, GOAL_R = 0.7, BUDGET = 50.0, vmax = 4.0, d_plan = d_safe + 0.12;

  printf("d_safe=%.2f  thread-the-crowd WITHIN a tunnel (y-band)?  humans cross y in [-1.5,1.5]\n", d_safe);
  printf("%-8s | %-8s | %-9s | %-10s | %-9s | %s\n", "tunnel_W", "reached", "final_x", "min_clr", "max|y|", "in-tunnel & safe?");
  for (double W : {0.6, 0.9, 1.2, 1.6}) {
    Eigen::Vector3d p = start; double t = 0; bool reached = false, held = false;
    double fmin = true_clr(start, 0.0), ymax = 0, msmax = 0;
    while (t < T_MAX && !reached) {
      std::vector<std::unique_ptr<SphereObstacle>> own; std::vector<const Obstacle*> hum; humans_at(t, own, hum);
      double dist = (goal - p).norm(); int M = std::max(4, std::min(12, (int)std::lround(dist / 1.4)));
      Eigen::VectorXd T0 = Eigen::VectorXd::Constant(M, std::max(dist / M / vmax, 0.12));
      Eigen::VectorXd seed = af_time_aware_seed(p, goal, hum, M, T0, d_plan, W);   // 种子只在 lane 内搜
      Eigen::VectorXd qf = af_restore(p, goal, seed, T0, hum, d_plan, 200, W);
      AFResult R = af_feasible_direction(p, goal, qf, T0, hum, cfg, copt, hopt, d_plan, 60, 0.5, BUDGET, nullptr, W);
      msmax = std::max(msmax, R.ms);
      MinjerkTraj tr = af_make_tr(p, goal, R.q, T0);
      // gatekeeper:只看【将要提交的那一段 [0,RD]】是否人体安全 + 在 lane 内(远端会被下拍重规划)
      bool chunk_ok = true;
      for (double tau = 0; tau <= RD + 1e-9; tau += 0.02) {
        Eigen::Vector3d pp = tr.eval(std::min(tau, tr.t_end));
        if (std::abs(pp[1]) > W + 0.05) { chunk_ok = false; break; }
        for (auto* h : hum) if (h->signed_dist(pp, tau) < d_safe - 1e-3) { chunk_ok = false; break; }
        if (!chunk_ok) break;
      }
      if (!chunk_ok) { held = true; fmin = std::min(fmin, true_clr(p, t)); break; }
      int ns = std::max(1, (int)(RD / 0.02)); double dt = RD / ns;
      Eigen::Vector3d p_prev = p;
      for (int s = 0; s < ns; ++s) { double tau = std::min((s + 1) * dt, tr.t_end); p = tr.eval(tau);
        fmin = std::min(fmin, true_clr(p, t + tau)); ymax = std::max(ymax, std::abs(p[1])); }
      t += RD;
      if ((p - goal).norm() < GOAL_R) reached = true;
      if ((p - p_prev).norm() < 1e-3 && t > 1.0) break;   // 卡住(没法在 lane 内前进)-> 停
    }
    bool safe = fmin >= d_safe - 0.05; bool inlane = ymax <= W + 0.05;
    const char* status = reached ? "THREADED ✓" : (held ? "no gap in lane → held safe (gatekeeper)" : "timeout");
    printf("%-8.1f | %-8s | %-9.1f | %+9.3f | %-9.3f | %s/%s | %s  (max ms=%.1f)\n",
           W, reached ? "YES" : "no", p[0], fmin, ymax,
           inlane ? "in-lane" : "OUT-OF-LANE", safe ? "safe" : "UNSAFE", status, msmax);
  }
  printf("\n证明:安全(人体+tunnel)对任意截断构造性成立(每个迭代两类约束 g<=0);\n");
  printf("到达 = lane 内存在时机缝(场景相关):窄 tunnel 无缝 -> 卡住但仍安全(不撞、不出 lane)。\n");
  return 0;
}
