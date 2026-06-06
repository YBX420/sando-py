// bench_recovery — SAFE half: a truly-blocking corridor where plan_minco FAILS, so the drone
// must wait. Three modes:
//   baseline  : on failure FREEZE on the stale plan (today) -> oncoming human walks in (HIT).
//   recovery  : on failure ACTIVELY YIELD (max-clearance move) instead of freezing.
//   recov+mon : ALSO a proactive monitor -> yield EARLY when clearance drops (don't wait for
//               the plan to fail), so there is room to keep d_safe.
// C++ port of python/test/stage_agile_corridor_trap.py baseline + the ABS/gatekeeper fix.
//
// Build:  g++ -std=c++17 -O2 -I cpp/include -I cpp/third_party/eigen -I cpp/third_party \
//             cpp/test/bench_recovery.cpp -o cpp/build/bench_recovery.exe
#include <cstdio>
#include <vector>
#include <memory>
#include <cmath>
#include "sando_cpp/plan_minco.hpp"
#include "sando_cpp/types.hpp"
using namespace sando;

static const double D_HUMAN = 0.8, HR = 0.3, TOL = 0.05, LANE = 1.3, MON_BUF = 0.5;

static std::map<std::string, AvoidParams> make_cfg() {
  auto c = default_config();
  c["human"] = AvoidParams{"human", "hard", D_HUMAN, 1.0e4};
  c["wall"]  = AvoidParams{"wall",  "hard", 0.15, 1.0e4};   // HARD: confine the lane, force a real block
  return c;
}
struct Mov { Eigen::Vector3d p0, v; };

static double slice_clr(const Eigen::Vector3d& p, const Eigen::Vector3d& q,
                        const std::vector<Mov>& crowd, const std::vector<Eigen::Vector3d>& hp, double dt) {
  double mn = 1e18;
  for (int s = 0; s <= 10; ++s) {
    double f = s / 10.0; Eigen::Vector3d dp = p + f * (q - p);
    for (size_t i = 0; i < crowd.size(); ++i)
      mn = std::min(mn, (dp - (hp[i] + crowd[i].v * (f * dt))).norm() - HR);
  }
  return mn;
}
static double cur_clr(const Eigen::Vector3d& p, const std::vector<Mov>& crowd, const std::vector<Eigen::Vector3d>& hp) {
  double mn = 1e18; for (size_t i = 0; i < crowd.size(); ++i) mn = std::min(mn, (p - hp[i]).norm() - HR); return mn;
}
// pick the in-lane move that maximises min-clearance over the next dt (never freeze)
static Eigen::Vector3d yield_move(const Eigen::Vector3d& p, const std::vector<Mov>& crowd,
                                  const std::vector<Eigen::Vector3d>& hp, double dt, double vmax) {
  size_t jn = 0; double dn = 1e18;
  for (size_t i = 0; i < crowd.size(); ++i) { double d = (hp[i] - p).norm(); if (d < dn) { dn = d; jn = i; } }
  Eigen::Vector3d away = p - hp[jn]; away[2] = 0; if (away.norm() > 1e-6) away.normalize();
  double step = 0.7 * vmax * dt;
  std::vector<Eigen::Vector3d> cand = { p, p + step * away, p + step * Eigen::Vector3d(-1,0,0),
                                        p + step * Eigen::Vector3d(0,1,0), p + step * Eigen::Vector3d(0,-1,0) };
  double best = -1e18; Eigen::Vector3d q = p;
  for (auto c : cand) {
    c[0] = std::max(-2.0, c[0]); c[2] = p[2];          // may retreat into the open area before the lane
    if (c[0] >= 1.9 && c[0] <= 9.1)                    // only stay in-lane while INSIDE the corridor
      c[1] = std::max(-LANE + 0.05, std::min(LANE - 0.05, c[1]));
    double clr = slice_clr(p, c, crowd, hp, dt);
    if (clr > best) { best = clr; q = c; }
  }
  return q;
}

struct Roll { double min_h; bool reached; int ticks, waits; bool hit; };

static Roll rollout(const Eigen::Vector3d& start, const Eigen::Vector3d& goal, std::vector<Mov> crowd,
                    bool recovery, bool monitor, double dt = 0.30, int max_ticks = 140, double vmax = 3.0) {
  auto cfg = make_cfg();
  std::vector<std::shared_ptr<AABBObstacle>> walls = {
    std::make_shared<AABBObstacle>(Eigen::Vector3d(2,-4.0,0), Eigen::Vector3d(9,-LANE,3), "wall"),
    std::make_shared<AABBObstacle>(Eigen::Vector3d(2, LANE,0), Eigen::Vector3d(9, 4.0,3), "wall")};
  Eigen::Vector3d p = start, v = Eigen::Vector3d::Zero(), a = Eigen::Vector3d::Zero();
  std::vector<Eigen::Vector3d> hp; for (auto& m : crowd) hp.push_back(m.p0);
  Roll R{1e18, false, 0, 0, false};
  for (int tick = 0; tick < max_ticks; ++tick) {
    R.ticks = tick + 1;
    std::vector<std::shared_ptr<SphereObstacle>> own; std::vector<const Obstacle*> obs;
    for (auto& w : walls) obs.push_back(w.get());
    for (size_t i = 0; i < crowd.size(); ++i) { own.push_back(std::make_shared<SphereObstacle>(hp[i], HR, crowd[i].v, "human")); obs.push_back(own.back().get()); }
    int NP = 9; Eigen::MatrixXd ap(NP, 3);
    for (int i = 0; i < NP; ++i) { double f = double(i) / (NP - 1); ap.row(i) = (p + f * (goal - p)).transpose(); }
    PlanOptParams opt; opt.vmax = vmax; opt.amax = 3.0; opt.w_corridor = 50.0; opt.sfc_radius = 0.6;
    DetourConfig dc; dc.enabled = true;

    // forward (agile) candidate
    bool valid = false; Eigen::Vector3d fp = p, fv = Eigen::Vector3d::Zero(), fa = Eigen::Vector3d::Zero();
    double fclr = -1e18, a1 = dt;
    try {
      auto pr = plan_minco(ap, obs, cfg, opt, dc, v, a);
      if (pr.second.trajectory_valid) {
        valid = true; const MinjerkTraj& tr = pr.first; a1 = std::min(dt, tr.t_end);
        fclr = 1e18;
        for (int s = 0; s <= 10; ++s) { double tau = a1 * s / 10.0; Eigen::Vector3d dp = tr.eval(std::min(tau, tr.t_end));
          for (size_t i = 0; i < crowd.size(); ++i) fclr = std::min(fclr, (dp - (hp[i] + crowd[i].v * tau)).norm() - HR); }
        fp = tr.eval(a1); fv = tr.eval_deriv(a1, 1); fa = tr.eval_deriv(a1, 2);
      }
    } catch (...) { valid = false; }

    // monitor: yield EARLY when current clearance is low, even if a forward plan exists
    bool yield_now = monitor && (cur_clr(p, crowd, hp) < D_HUMAN + MON_BUF);
    bool do_forward = valid && !yield_now;

    Eigen::Vector3d np; double mclr;
    if (do_forward) { np = fp; v = fv; a = fa; mclr = fclr; }
    else {                                            // WAIT (blocked or monitor-triggered)
      R.waits++;
      Eigen::Vector3d q = recovery ? yield_move(p, crowd, hp, dt, vmax) : p;   // yield vs freeze
      mclr = slice_clr(p, q, crowd, hp, dt);
      np = q; v = (q - p) / dt; a = Eigen::Vector3d::Zero();
    }
    R.min_h = std::min(R.min_h, mclr);
    if (mclr < D_HUMAN - TOL) R.hit = true;
    p = np;
    for (size_t i = 0; i < crowd.size(); ++i) hp[i] += crowd[i].v * dt;
    if ((p.head<2>() - goal.head<2>()).norm() < 0.4) { R.reached = true; break; }
  }
  return R;
}

int main() {
  Eigen::Vector3d start(0, 0, 1.5), goal(11, 0, 1.5);
  std::vector<Mov> crowd = {
    {Eigen::Vector3d(4.0, 0.0,1.5), Eigen::Vector3d(-1,0,0)}, {Eigen::Vector3d(5.0, 0.7,1.5), Eigen::Vector3d(-1,0,0)},
    {Eigen::Vector3d(6.0,-0.7,1.5), Eigen::Vector3d(-1,0,0)}, {Eigen::Vector3d(7.0, 0.4,1.5), Eigen::Vector3d(-1,0,0)},
    {Eigen::Vector3d(8.0,-0.4,1.5), Eigen::Vector3d(-1,0,0)}};
  auto pr = [&](const char* tag, const Roll& r) {
    printf("  %-22s | min_human=%+.3f %-9s | reached=%-3s | ticks=%d waits=%d %s\n",
           tag, r.min_h, r.min_h < D_HUMAN - TOL ? "(BREACH!)" : "(>=d_safe)", r.reached ? "YES" : "no",
           r.ticks, r.waits, r.hit ? "HIT" : "");
  };
  std::vector<Mov> crowd3 = {
    {Eigen::Vector3d(4.0, 0.0,1.5), Eigen::Vector3d(-1,0,0)},
    {Eigen::Vector3d(6.0, 0.5,1.5), Eigen::Vector3d(-1,0,0)},
    {Eigen::Vector3d(8.0,-0.5,1.5), Eigen::Vector3d(-1,0,0)}};

  printf("=== BRUTAL: hard lane |y|<=1.3, 5 oncoming humans (near-unavoidable), d_safe=0.8 ===\n");
  pr("baseline (freeze)",   rollout(start, goal, crowd, false, false));
  pr("recovery (yield)",    rollout(start, goal, crowd, true,  false));
  pr("recovery + monitor",  rollout(start, goal, crowd, true,  true));
  printf("\n=== MODERATE: same lane, 3 oncoming humans (escapable) ===\n");
  pr("baseline (freeze)",   rollout(start, goal, crowd3, false, false));
  pr("recovery (yield)",    rollout(start, goal, crowd3, true,  false));
  pr("recovery + monitor",  rollout(start, goal, crowd3, true,  true));
  printf("\nmonitor yields EARLY (clearance < d_safe+%.1f); yield may retreat into the open area.\n", MON_BUF);
  return 0;
}
