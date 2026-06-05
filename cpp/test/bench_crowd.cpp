// bench_crowd — CLOSED-LOOP receding-horizon rollout through a MOVING crowd, driving the
// PRODUCTION plan_minco (warm-started). C++ port of python/test/stage_agile_dynamic_crowd.py.
// Compares corridor OFF vs ON (the new default 50/0.6) on real closed-loop clearance + reach.
//
// Build:  g++ -std=c++17 -O2 -I cpp/include -I cpp/third_party/eigen -I cpp/third_party \
//             cpp/test/bench_crowd.cpp -o cpp/build/bench_crowd.exe
#include <cstdio>
#include <vector>
#include <memory>
#include <cmath>
#include "sando_cpp/plan_minco.hpp"
#include "sando_cpp/types.hpp"
using namespace sando;

static const double D_HUMAN = 0.8, HR = 0.3, TOL = 0.05;

static std::map<std::string, AvoidParams> make_cfg() {
  auto c = default_config();
  c["human"] = AvoidParams{"human", "hard", D_HUMAN, 1.0e4};
  c["wall"]  = AvoidParams{"wall",  "soft", 0.4, 1.0e1};
  return c;
}
struct Mov { Eigen::Vector3d p0, v; };
struct Roll { double min_h, min_w; bool reached; int ticks, fails; };

static Roll rollout(const Eigen::Vector3d& start, const Eigen::Vector3d& goal,
                    std::vector<Mov> crowd,
                    const std::vector<std::shared_ptr<AABBObstacle>>& walls,
                    double w_corr, double sfc_r,
                    double dt = 0.30, int max_ticks = 90, double vmax = 3.0) {
  auto cfg = make_cfg();
  Eigen::Vector3d p = start, v = Eigen::Vector3d::Zero(), a = Eigen::Vector3d::Zero();
  std::vector<Eigen::Vector3d> hp; for (auto& m : crowd) hp.push_back(m.p0);
  Roll R{1e18, 1e18, false, 0, 0};
  for (int tick = 0; tick < max_ticks; ++tick) {
    R.ticks = tick + 1;
    std::vector<std::shared_ptr<SphereObstacle>> own;
    std::vector<const Obstacle*> obs;
    for (auto& w : walls) obs.push_back(w.get());
    for (size_t i = 0; i < crowd.size(); ++i) {
      own.push_back(std::make_shared<SphereObstacle>(hp[i], HR, crowd[i].v, "human"));
      obs.push_back(own.back().get());
    }
    int NP = 9; Eigen::MatrixXd ap(NP, 3);
    for (int i = 0; i < NP; ++i) { double f = double(i) / (NP - 1); ap.row(i) = (p + f * (goal - p)).transpose(); }
    PlanOptParams opt; opt.vmax = vmax; opt.amax = 3.0; opt.w_corridor = w_corr; opt.sfc_radius = sfc_r;
    DetourConfig dc; dc.enabled = true;
    bool valid = false; double a1 = dt;
    Eigen::Vector3d np = p, nv = Eigen::Vector3d::Zero(), na = Eigen::Vector3d::Zero();
    std::vector<Eigen::Vector3d> samp;          // drone positions over the committed slice
    try {
      auto pr = plan_minco(ap, obs, cfg, opt, dc, v, a);
      if (pr.second.trajectory_valid) {
        valid = true; const MinjerkTraj& tr = pr.first;
        a1 = std::min(dt, tr.t_end);
        for (int s = 0; s <= 12; ++s) samp.push_back(tr.eval(std::min(a1 * s / 12.0, tr.t_end)));
        np = tr.eval(a1); nv = tr.eval_deriv(a1, 1); na = tr.eval_deriv(a1, 2);
      }
    } catch (...) { valid = false; }
    if (!valid) { R.fails++; a1 = dt; for (int s = 0; s <= 12; ++s) samp.push_back(p); }  // HOLD (freeze)
    // closed-loop clearance over the slice (humans move through it)
    for (int s = 0; s <= 12; ++s) {
      double tau = a1 * s / 12.0;
      for (size_t i = 0; i < crowd.size(); ++i) {
        Eigen::Vector3d hc = hp[i] + crowd[i].v * tau;
        R.min_h = std::min(R.min_h, (samp[s] - hc).norm() - HR);
      }
      for (auto& w : walls) R.min_w = std::min(R.min_w, w->signed_dist(samp[s], tau));
    }
    p = np; v = nv; a = na;
    for (size_t i = 0; i < crowd.size(); ++i) hp[i] += crowd[i].v * dt;
    if ((p.head<2>() - goal.head<2>()).norm() < 0.4) { R.reached = true; break; }
  }
  return R;
}

static void report(const char* scen, const Roll& off, const Roll& on, bool walls) {
  auto line = [&](const char* tag, const Roll& r) {
    bool breach = r.min_h < D_HUMAN - TOL;
    if (walls)
      printf("  %-12s | reached=%-3s | min_human=%+.3f %-8s | min_wall=%+.3f | ticks=%d fails=%d\n",
             tag, r.reached ? "YES" : "no", r.min_h, breach ? "(BREACH)" : "(safe)", r.min_w, r.ticks, r.fails);
    else
      printf("  %-12s | reached=%-3s | min_human=%+.3f %-8s | ticks=%d fails=%d\n",
             tag, r.reached ? "YES" : "no", r.min_h, breach ? "(BREACH)" : "(safe)", r.ticks, r.fails);
  };
  printf("--- %s ---\n", scen);
  line("corridor OFF", off);
  line("corridor ON ", on);
}

int main() {
  { Parameters _p; printf("Parameters default: minco_w_corridor=%.1f sfc_radius=%.2f  (d_safe_human=%.2f)\n\n",
                          _p.minco_w_corridor, _p.minco_sfc_radius, D_HUMAN); }
  Eigen::Vector3d start(0, 0, 1.5), goal(10, 0, 1.5);
  std::vector<std::shared_ptr<AABBObstacle>> nowall;

  // Scenario sweep: 4-human wall drifting +y (the y=0 slot closes)
  std::vector<Mov> sweep;
  for (double y : {-3.6, -1.2, 1.2, 3.6}) sweep.push_back({Eigen::Vector3d(5, y, 1.5), Eigen::Vector3d(0, 1.2, 0)});
  report("sweep crowd (4 moving humans, no walls)",
         rollout(start, goal, sweep, nowall, 0.0, 0.0),
         rollout(start, goal, sweep, nowall, 50.0, 0.6), false);

  // Scenario tunnel: soft-wall lane |y|<=1.3 + 3 humans crossing inside
  std::vector<std::shared_ptr<AABBObstacle>> walls = {
    std::make_shared<AABBObstacle>(Eigen::Vector3d(2, -3.0, 0), Eigen::Vector3d(8, -1.3, 3), "wall"),
    std::make_shared<AABBObstacle>(Eigen::Vector3d(2,  1.3, 0), Eigen::Vector3d(8,  3.0, 3), "wall")};
  std::vector<Mov> tun = {
    {Eigen::Vector3d(4, -1.0, 1.5), Eigen::Vector3d(0,  0.6, 0)},
    {Eigen::Vector3d(6,  1.0, 1.5), Eigen::Vector3d(0, -0.6, 0)},
    {Eigen::Vector3d(5, -1.2, 1.5), Eigen::Vector3d(0,  0.5, 0)}};
  report("tunnel (soft-wall lane |y|<=1.3) + 3 crossing humans",
         rollout(start, goal, tun, walls, 0.0, 0.0),
         rollout(start, goal, tun, walls, 50.0, 0.6), true);

  printf("\n(closed-loop: each tick warm-started replan, commit dt=0.3, humans walk, dense sub-step clearance)\n");
  return 0;
}
