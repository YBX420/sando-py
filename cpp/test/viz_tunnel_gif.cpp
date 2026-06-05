// viz_tunnel_gif — 高复杂度(密集动态硬行人,SANDO dynamic-hard 强度)+ tunnel,
// 闭环跑并 dump 每个时间步的 (无人机 + 所有行人位置) 给 python 做 gif。
#include <cstdio>
#include <vector>
#include <algorithm>
#include <memory>
#include <cmath>
#include "sando_cpp/anytime_feasible.hpp"
using namespace sando;

struct Spec { double x, y0, vy, ay, r; };
static std::vector<Spec> SPEC;
static void build_spec() {
  // 16 个密集行人(SANDO dynamic-hard 强度):随机相位 + 抛物运动(vy 横穿 + ay 折返 -> 持续在 lane 里来回挤)
  unsigned s = 777;
  auto rnd = [&]() { s = s * 1103515245 + 12345; return ((s >> 16) & 0x7fff) / 32768.0; };
  for (int i = 0; i < 16; ++i) {
    double x = 1.8 + i * 0.82;
    double y0 = (rnd() * 2 - 1) * 1.7;                 // lane 内任意起点
    double vy = (rnd() * 2 - 1) * 0.7;                  // 任意方向横穿
    double ay = -(vy >= 0 ? 1.0 : -1.0) * (0.10 + 0.12 * rnd());  // 反向加速 -> 折返,持续占据 lane
    SPEC.push_back({x, y0, vy, ay, 0.26 + 0.07 * rnd()});
  }
}
static double hy_at(const Spec& s, double t) { return s.y0 + s.vy * t + 0.5 * s.ay * t * t; }
static void humans_at(double t, std::vector<std::unique_ptr<SphereObstacle>>& own, std::vector<const Obstacle*>& hum) {
  own.clear(); hum.clear();
  for (auto& s : SPEC) { Eigen::Vector3d c(s.x, hy_at(s, t), 2.0);
    own.push_back(std::make_unique<SphereObstacle>(c, s.r, Eigen::Vector3d(0, s.vy + s.ay * t, 0), "human", Eigen::Vector3d(0, s.ay, 0))); hum.push_back(own.back().get()); }
}

int main() {
  build_spec();
  auto cfg = default_config(); double d_safe = cfg["human"].d_safe;
  Eigen::Vector3d start(0, 0, 2), goal(15, 0, 2);
  CostGradOptParams copt; copt.vmax = 4; copt.amax = 8; copt.spacetime_hard = true; copt.tau_trust = 100;
  HardAlmOptParams hopt; hopt.spacetime_hard = true; hopt.tau_trust = 100;
  const double RD = 0.2, T_MAX = 22.0, GOAL_R = 0.7, BUDGET = 50.0, vmax = 4.0, d_plan = d_safe + 0.12, W = 2.2;

  FILE* fp = std::fopen("../python/media/_gif_path.csv", "w"); std::fprintf(fp, "t,x,y,z,clr,ms\n");
  FILE* fh = std::fopen("../python/media/_gif_hum.csv", "w"); std::fprintf(fh, "t,hid,x,y,r\n");
  FILE* fm = std::fopen("../python/media/_gif_meta.csv", "w"); std::fprintf(fm, "W,d_safe,goalx\n%.2f,%.2f,%.1f\n", W, d_safe, goal[0]);
  std::fclose(fm);

  Eigen::Vector3d p = start; double t = 0; bool reached = false, held = false; double msmax = 0;
  auto dump = [&](double tt) {
    double mn = 1e18; for (auto& s : SPEC) { Eigen::Vector3d c(s.x, hy_at(s, tt), 2.0); mn = std::min(mn, (p - c).norm() - s.r); }
    std::fprintf(fp, "%.3f,%.4f,%.4f,%.4f,%.4f,%.2f\n", tt, p[0], p[1], p[2], mn, msmax);
    int hid = 0; for (auto& s : SPEC) { std::fprintf(fh, "%.3f,%d,%.4f,%.4f,%.3f\n", tt, hid++, s.x, hy_at(s, tt), s.r); }
  };
  dump(0);
  while (t < T_MAX && !reached) {
    std::vector<std::unique_ptr<SphereObstacle>> own; std::vector<const Obstacle*> hum; humans_at(t, own, hum);
    double dist = (goal - p).norm(); int M = std::max(4, std::min(12, (int)std::lround(dist / 1.4)));
    Eigen::VectorXd T0 = Eigen::VectorXd::Constant(M, std::max(dist / M / vmax, 0.12));
    Eigen::VectorXd seed = af_time_aware_seed(p, goal, hum, M, T0, d_plan, W);
    Eigen::VectorXd qf = af_restore(p, goal, seed, T0, hum, d_plan, 200, W);
    AFResult R = af_feasible_direction(p, goal, qf, T0, hum, cfg, copt, hopt, d_plan, 60, 0.5, BUDGET, nullptr, W);
    msmax = R.ms;
    MinjerkTraj tr = af_make_tr(p, goal, R.q, T0);
    bool chunk_ok = true;
    for (double tau = 0; tau <= RD + 1e-9; tau += 0.02) {
      Eigen::Vector3d pp = tr.eval(std::min(tau, tr.t_end));
      if (std::abs(pp[1]) > W + 0.05) { chunk_ok = false; break; }
      for (auto* h : hum) if (h->signed_dist(pp, tau) < d_safe - 1e-3) { chunk_ok = false; break; }
      if (!chunk_ok) break;
    }
    if (!chunk_ok) {  // 无缝:held（原地等下一拍，行人移动可能开缝）—— dump 当前(停住)帧
      held = true; dump(t + RD); t += RD; continue;
    }
    held = false;
    int ns = std::max(1, (int)(RD / 0.04)); double dt = RD / ns;
    for (int s = 0; s < ns; ++s) { double tau = std::min((s + 1) * dt, tr.t_end); p = tr.eval(tau); dump(t + tau); }
    t += RD;
    if ((p - goal).norm() < GOAL_R) reached = true;
  }
  std::fclose(fp); std::fclose(fh);
  std::printf("complexity=%d humans, tunnel W=%.1f -> %s  final_x=%.1f  (dumped gif CSVs)\n",
              (int)SPEC.size(), W, reached ? "THREADED" : "held/timeout", p[0]);
  return 0;
}
