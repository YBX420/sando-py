// bench_maps — SPARSE vs DENSE map validation sweep, closed-loop, driving PRODUCTION plan_minco.
// C++ companion to bench_crowd.cpp: same receding-horizon rollout (warm-started replan, commit dt,
// obstacles move, dense sub-step clearance), but a MATRIX of map families x {sparse, dense} so we can
// see at a glance whether (a) sparse stays live & not over-conservative, (b) dense stays safe (zero
// human penetration) and still reaches.
//
// Pure experiment: production code & golden untouched. Production config = SFC corridor 50/0.6
// (the shipping default). On the DENSE maps we also A/B the just-landed Safe-Interval space-time
// corridor + ST-graph timing front-end (default OFF) for their first map validation.
//
// Build (Strawberry g++):
//   g++ -std=c++17 -O2 -I cpp/include -I cpp/third_party/eigen -I cpp/third_party \
//       cpp/test/bench_maps.cpp -o cpp/build/bench_maps.exe
//   cpp/build/bench_maps.exe
#include <cstdio>
#include <vector>
#include <memory>
#include <cmath>
#include <algorithm>
#include <chrono>
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

struct Mov { Eigen::Vector3d p0, v; };               // a (possibly moving) human
struct Mode { const char* tag; double w_corr, sfc_r; bool st_corr, st_graph; };
struct Roll {
  double min_h = 1e18, min_w = 1e18;
  bool   reached = false;
  int    ticks = 0, fails = 0;
  std::vector<double> ms;                            // per-replan wall time
};

static double pct(std::vector<double> v, double q) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  size_t i = std::min(v.size() - 1, (size_t)std::llround(q * (v.size() - 1)));
  return v[i];
}

static Roll rollout(const Eigen::Vector3d& start, const Eigen::Vector3d& goal,
                    std::vector<Mov> crowd,
                    const std::vector<std::shared_ptr<AABBObstacle>>& walls,
                    const Mode& mode,
                    double dt = 0.30, int max_ticks = 80, double vmax = 3.0) {
  auto cfg = make_cfg();
  Eigen::Vector3d p = start, v = Eigen::Vector3d::Zero(), a = Eigen::Vector3d::Zero();
  std::vector<Eigen::Vector3d> hp; for (auto& m : crowd) hp.push_back(m.p0);
  Roll R;
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
    PlanOptParams opt; opt.vmax = vmax; opt.amax = 3.0;
    opt.w_corridor = mode.w_corr; opt.sfc_radius = mode.sfc_r;
    opt.use_spacetime_corridor = mode.st_corr; opt.use_st_graph = mode.st_graph;
    DetourConfig dc; dc.enabled = true;
    bool valid = false; double a1 = dt;
    Eigen::Vector3d np = p, nv = Eigen::Vector3d::Zero(), na = Eigen::Vector3d::Zero();
    std::vector<Eigen::Vector3d> samp;
    auto t0 = std::chrono::steady_clock::now();
    try {
      auto pr = plan_minco(ap, obs, cfg, opt, dc, v, a);
      if (pr.second.trajectory_valid) {
        valid = true; const MinjerkTraj& tr = pr.first;
        a1 = std::min(dt, tr.t_end);
        for (int s = 0; s <= 12; ++s) samp.push_back(tr.eval(std::min(a1 * s / 12.0, tr.t_end)));
        np = tr.eval(a1); nv = tr.eval_deriv(a1, 1); na = tr.eval_deriv(a1, 2);
      }
    } catch (...) { valid = false; }
    auto t1 = std::chrono::steady_clock::now();
    R.ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    if (!valid) { R.fails++; a1 = dt; for (int s = 0; s <= 12; ++s) samp.push_back(p); }  // HOLD (freeze)
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

static void row(const char* density, const Roll& r, bool has_walls) {
  bool has_humans = r.min_h < 1e17;                 // sentinel => no humans on this map
  bool breach = has_humans && r.min_h < D_HUMAN - TOL;
  const char* verdict = (!r.reached) ? "NO-REACH" : (breach ? "BREACH" : "PASS");
  char hcol[32];
  if (has_humans) snprintf(hcol, sizeof hcol, "%+.3f %-8s", r.min_h, breach ? "(BREACH)" : "(safe)");
  else            snprintf(hcol, sizeof hcol, "  n/a  %-8s", "");
  if (has_walls)
    printf("  %-8s | %-8s | min_human=%-16s | min_wall=%+.3f | reach=%-3s | ticks=%2d fails=%2d | "
           "replan med=%5.1f p95=%5.1f ms\n",
           density, verdict, hcol, r.min_w,
           r.reached ? "YES" : "no", r.ticks, r.fails, pct(r.ms, 0.5), pct(r.ms, 0.95));
  else
    printf("  %-8s | %-8s | min_human=%-16s | reach=%-3s | ticks=%2d fails=%2d | "
           "replan med=%5.1f p95=%5.1f ms\n",
           density, verdict, hcol,
           r.reached ? "YES" : "no", r.ticks, r.fails, pct(r.ms, 0.5), pct(r.ms, 0.95));
}

// ----- map families ---------------------------------------------------------
static std::vector<std::shared_ptr<AABBObstacle>> pillar(double cx, double cy, double half = 0.4) {
  return {std::make_shared<AABBObstacle>(Eigen::Vector3d(cx - half, cy - half, 0),
                                         Eigen::Vector3d(cx + half, cy + half, 3), "wall")};
}

int main() {
  Parameters pp;
  printf("Production defaults: minco_w_corridor=%.1f minco_sfc_radius=%.2f  (d_safe_human=%.2f, vmax=3)\n",
         pp.minco_w_corridor, pp.minco_sfc_radius, D_HUMAN);
  printf("Verdict: PASS = reached & min_human>=d_safe-tol (zero penetration). Soft-wall grazing is allowed by design.\n\n");

  Eigen::Vector3d start(0, 0, 1.5), goal(12, 0, 1.5);
  std::vector<std::shared_ptr<AABBObstacle>> nowall;
  Mode PROD{"prod", pp.minco_w_corridor, pp.minco_sfc_radius, false, false};
  Mode STON{"+ST",  pp.minco_w_corridor, pp.minco_sfc_radius, true,  true};

  // ---- Map 1: dynamic crossing crowd (moving hard humans, open) ----
  printf("=== Map 1: dynamic crossing crowd (moving hard humans, no walls) ===\n");
  {
    std::vector<Mov> sparse = {{Eigen::Vector3d(6, -2.5, 1.5), {0, 1.0, 0}},
                               {Eigen::Vector3d(6,  2.5, 1.5), {0, -1.0, 0}}};
    std::vector<Mov> dense;
    for (double y : {-4.0, -2.4, -0.8, 0.8, 2.4, 4.0})
      dense.push_back({Eigen::Vector3d(5.0 + 0.4 * (int)(y), y, 1.5), {0, (y < 0 ? 1.1 : -1.1), 0}});
    row("sparse", rollout(start, goal, sparse, nowall, PROD), false);
    row("dense",  rollout(start, goal, dense,  nowall, PROD), false);
  }

  // ---- Map 2: static human wall with a center gap (thread the slot) ----
  printf("\n=== Map 2: standing-human wall @ x=6 with a center gap (thread it) ===\n");
  {
    std::vector<Mov> sparse = {{Eigen::Vector3d(6, -2.2, 1.5), {0, 0, 0}},
                               {Eigen::Vector3d(6,  2.2, 1.5), {0, 0, 0}}};
    std::vector<Mov> dense;
    for (double y : {-4.0, -2.6, -1.2, 1.2, 2.6, 4.0})
      dense.push_back({Eigen::Vector3d(6, y, 1.5), {0, 0, 0}});   // gap |y|<1.2 -> center min_h=0.9
    row("sparse", rollout(start, goal, sparse, nowall, PROD), false);
    row("dense",  rollout(start, goal, dense,  nowall, PROD), false);
  }

  // ---- Map 3: pillar forest (static soft walls) ----
  printf("\n=== Map 3: pillar forest (static soft-wall columns) ===\n");
  {
    std::vector<std::shared_ptr<AABBObstacle>> sparse;
    for (auto pr : {std::pair<double,double>{4, 1.0}, {6, -1.2}, {8, 0.8}})
      sparse.push_back(pillar(pr.first, pr.second)[0]);
    std::vector<std::shared_ptr<AABBObstacle>> dense;
    for (double cx : {3.5, 5.5, 7.5, 9.5})
      for (double cy : {-1.6, 0.0, 1.6})
        if (!(std::abs(cx - 5.5) < 0.1 && std::abs(cy) < 0.1))   // leave a lane near center
          dense.push_back(pillar(cx, cy)[0]);
    row("sparse", rollout(start, goal, {}, sparse, PROD), true);
    row("dense",  rollout(start, goal, {}, dense,  PROD), true);
  }

  // ---- Map 4: soft-wall corridor (lane |y|<=1.3) + crossing humans ----
  printf("\n=== Map 4: soft-wall corridor (lane |y|<=1.3) + crossing humans ===\n");
  std::vector<std::shared_ptr<AABBObstacle>> lane = {
    std::make_shared<AABBObstacle>(Eigen::Vector3d(2, -3.0, 0), Eigen::Vector3d(10, -1.3, 3), "wall"),
    std::make_shared<AABBObstacle>(Eigen::Vector3d(2,  1.3, 0), Eigen::Vector3d(10,  3.0, 3), "wall")};
  std::vector<Mov> c4_sparse = {{Eigen::Vector3d(5, -1.0, 1.5), {0, 0.6, 0}},
                                {Eigen::Vector3d(7,  1.0, 1.5), {0, -0.6, 0}}};
  std::vector<Mov> c4_dense = {{Eigen::Vector3d(4, -1.0, 1.5), {0,  0.6, 0}},
                               {Eigen::Vector3d(5,  1.1, 1.5), {0, -0.5, 0}},
                               {Eigen::Vector3d(6, -1.2, 1.5), {0,  0.5, 0}},
                               {Eigen::Vector3d(7,  1.0, 1.5), {0, -0.6, 0}},
                               {Eigen::Vector3d(8, -0.9, 1.5), {0,  0.5, 0}}};
  row("sparse", rollout(start, goal, c4_sparse, lane, PROD), true);
  row("dense",  rollout(start, goal, c4_dense,  lane, PROD), true);

  // ---- A/B the just-landed Safe-Interval ST-corridor + ST-graph on the DENSE maps ----
  printf("\n=== A/B: Safe-Interval space-time corridor + ST-graph (default OFF) on DENSE maps ===\n");
  {
    std::vector<Mov> d1;
    for (double y : {-4.0, -2.4, -0.8, 0.8, 2.4, 4.0})
      d1.push_back({Eigen::Vector3d(5.0 + 0.4 * (int)(y), y, 1.5), {0, (y < 0 ? 1.1 : -1.1), 0}});
    printf("Map1 dense (moving crowd):\n");
    row("prod",  rollout(start, goal, d1, nowall, PROD), false);
    row("+ST",   rollout(start, goal, d1, nowall, STON), false);
    printf("Map4 dense (corridor+crowd):\n");
    row("prod",  rollout(start, goal, c4_dense, lane, PROD), true);
    row("+ST",   rollout(start, goal, c4_dense, lane, STON), true);
  }

  printf("\n(closed-loop: each tick warm-started replan, commit dt=0.3, obstacles walk, dense sub-step clearance)\n");
  return 0;
}
