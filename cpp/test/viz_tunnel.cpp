// viz_tunnel — 跑 tunnel 闭环并把飞行轨迹 dump 成 CSV(给 python 画图)。
// 输出 media/_tunnel_W<xx>.csv: 每行 t,x,y,clr,W,status(reached/held)
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

  for (double W : {1.6, 0.9}) {
    char fn[128]; std::snprintf(fn, sizeof(fn), "../python/media/_tunnel_W%02d.csv", (int)std::lround(W * 10));
    FILE* f = std::fopen(fn, "w"); std::fprintf(f, "t,x,y,clr,W\n");
    Eigen::Vector3d p = start; double t = 0; bool reached = false, held = false;
    std::fprintf(f, "%.3f,%.4f,%.4f,%.4f,%.2f\n", 0.0, p[0], p[1], true_clr(p, 0), W);
    while (t < T_MAX && !reached) {
      std::vector<std::unique_ptr<SphereObstacle>> own; std::vector<const Obstacle*> hum; humans_at(t, own, hum);
      double dist = (goal - p).norm(); int M = std::max(4, std::min(12, (int)std::lround(dist / 1.4)));
      Eigen::VectorXd T0 = Eigen::VectorXd::Constant(M, std::max(dist / M / vmax, 0.12));
      Eigen::VectorXd seed = af_time_aware_seed(p, goal, hum, M, T0, d_plan, W);
      Eigen::VectorXd qf = af_restore(p, goal, seed, T0, hum, d_plan, 200, W);
      AFResult R = af_feasible_direction(p, goal, qf, T0, hum, cfg, copt, hopt, d_plan, 60, 0.5, BUDGET, nullptr, W);
      MinjerkTraj tr = af_make_tr(p, goal, R.q, T0);
      bool chunk_ok = true;
      for (double tau = 0; tau <= RD + 1e-9; tau += 0.02) {
        Eigen::Vector3d pp = tr.eval(std::min(tau, tr.t_end));
        if (std::abs(pp[1]) > W + 0.05) { chunk_ok = false; break; }
        for (auto* h : hum) if (h->signed_dist(pp, tau) < d_safe - 1e-3) { chunk_ok = false; break; }
        if (!chunk_ok) break;
      }
      if (!chunk_ok) { held = true; break; }
      int ns = std::max(1, (int)(RD / 0.02)); double dt = RD / ns;
      for (int s = 0; s < ns; ++s) { double tau = std::min((s + 1) * dt, tr.t_end); p = tr.eval(tau);
        std::fprintf(f, "%.3f,%.4f,%.4f,%.4f,%.2f\n", t + tau, p[0], p[1], true_clr(p, t + tau), W); }
      t += RD;
      if ((p - goal).norm() < GOAL_R) reached = true;
    }
    std::fclose(f);
    std::printf("W=%.1f -> %s  final_x=%.1f  (wrote %s)\n", W, reached ? "THREADED" : "held", p[0], fn);
  }
  return 0;
}
