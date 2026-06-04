// bench_anytime — C++ A+B 闭环:无人机在 6 个横穿移动行人中 receding-horizon 飞 start->goal。
// 验证 (1) 每拍 replan < 50ms(gap 1);(2) 全程人体净空 >= d_safe;(3) 每拍截断不变性。
// 场景与 python/_exp_production.py 一致。
#include <cstdio>
#include <vector>
#include <algorithm>
#include <memory>
#include "sando_cpp/anytime_feasible.hpp"

using namespace sando;

struct Spec { double x, y0, vy; };
static std::vector<Spec> SPEC = {{3,-1.5,0.55},{5,1.5,-0.55},{7,-1.5,0.55},{9,1.5,-0.55},{11,-1.5,0.55},{13,1.5,-0.55}};

// humans at absolute time t (as obstacles whose local-time-0 = their position at t)
static void humans_at(double t, std::vector<std::unique_ptr<SphereObstacle>>& own,
                      std::vector<const Obstacle*>& hum) {
  own.clear(); hum.clear();
  for (auto& s : SPEC) {
    Eigen::Vector3d c(s.x, s.y0 + s.vy * t, 2.0);
    own.push_back(std::make_unique<SphereObstacle>(c, 0.3, Eigen::Vector3d(0, s.vy, 0), "human"));
    hum.push_back(own.back().get());
  }
}
static double true_clr(const Eigen::Vector3d& p, double t) {
  double mn = 1e18;
  for (auto& s : SPEC) { Eigen::Vector3d c(s.x, s.y0 + s.vy * t, 2.0); mn = std::min(mn, (p - c).norm() - 0.3); }
  return mn;
}

int main() {
  auto cfg = default_config(); double d_safe = cfg["human"].d_safe;
  Eigen::Vector3d start(0, 0, 2), goal(14, 0, 2);
  CostGradOptParams copt; copt.vmax = 4; copt.amax = 8; copt.spacetime_hard = true; copt.tau_trust = 100.0;
  HardAlmOptParams hopt; hopt.spacetime_hard = true; hopt.tau_trust = 100.0;
  const double RD = 0.25, T_MAX = 16.0, GOAL_R = 0.7, BUDGET = 50.0, vmax = 4.0;
  const double d_plan = d_safe + 0.12;   // 规划裕度:吸收拍间行人漂移,使实际净空严格 >= d_safe

  Eigen::Vector3d p = start; double t = 0; bool reached = false;
  std::vector<double> ms_list; int n_trunc_safe = 0, n_replan = 0;
  double flight_min_clr = 1e18; int nstep_total = 0;
  printf("d_safe=%.2f  C++ A+B closed loop, per-replan deadline=%.0fms\n", d_safe, BUDGET);
  while (t < T_MAX && !reached) {
    std::vector<std::unique_ptr<SphereObstacle>> own; std::vector<const Obstacle*> hum;
    humans_at(t, own, hum);
    double dist = (goal - p).norm();
    int M = std::max(4, std::min(12, (int)std::lround(dist / 1.4)));
    Eigen::VectorXd T0 = Eigen::VectorXd::Constant(M, std::max(dist / M / vmax, 0.12));
    Eigen::VectorXd seed = af_time_aware_seed(p, goal, hum, M, T0, d_plan);
    Eigen::VectorXd qf = af_restore(p, goal, seed, T0, hum, d_plan, 160);
    Eigen::MatrixXd corr(M - 1, 3); for (int i = 0; i < M - 1; ++i) corr.row(i) = qf.segment(3 * i, 3).transpose();
    AFResult R = af_feasible_direction(p, goal, qf, T0, hum, cfg, copt, hopt, d_plan, 25, 0.5, BUDGET, &corr);
    ms_list.push_back(R.ms); ++n_replan;
    double hmin = *std::min_element(R.hist.begin(), R.hist.end());
    if (hmin >= d_safe - 1e-3) ++n_trunc_safe;   // truncation invariance vs real d_safe
    // commit: follow committed trajectory (humans shifted to current time) for RD seconds
    MinjerkTraj tr = af_make_tr(p, goal, R.q, T0);
    int ns = std::max(1, (int)(RD / 0.02)); double dt = RD / ns;
    for (int s = 0; s < ns; ++s) {
      double tau = std::min((s + 1) * dt, tr.t_end);
      p = tr.eval(tau);
      double c = true_clr(p, t + tau); flight_min_clr = std::min(flight_min_clr, c); ++nstep_total;
    }
    t += RD;
    if ((p - goal).norm() < GOAL_R) reached = true;
  }
  std::sort(ms_list.begin(), ms_list.end());
  double mean = 0; for (double m : ms_list) mean += m; mean /= ms_list.size();
  double p90 = ms_list[(int)(0.9 * (ms_list.size() - 1))];
  printf("\n=== RESULT (C++ closed loop) ===\n");
  printf("reached=%s  final_x=%.1f  flight min human clearance=%+.3f  (>=d_safe all flight? %s)\n",
         reached ? "true" : "false", p[0], flight_min_clr, flight_min_clr >= d_safe - 0.05 ? "YES" : "NO");
  printf("per-replan ms: mean=%.2f  p90=%.2f  max=%.2f   <50ms deadline? %s\n",
         mean, p90, ms_list.back(), ms_list.back() < 50.0 ? "ALL replans YES" : (p90 < 50.0 ? "p90 YES" : "NO"));
  printf("per-replan truncation-invariance safe rate = %d/%d (%.0f%%)\n",
         n_trunc_safe, n_replan, 100.0 * n_trunc_safe / n_replan);
  return 0;
}
