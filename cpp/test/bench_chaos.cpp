// bench_chaos — THE HEADLINE acceptance test for the "hold-or-thread in chaos" milestone.
// Closed-loop, driving PRODUCTION plan_minco (agile half) + the PRODUCTION safe-half functions
// reach_avoid_clearance (monitor) and yield_target (recovery) — the SAME calls recovery_yield()
// makes in planner.hpp, with the SAME gate constants. This is NOT a harness re-implementation
// (bench_recovery re-implemented the yield); here the safe-half is the real shipping code.
//
// Two milestone clauses, scored jointly per scenario:
//   SAFE : min HARD-human clearance >= d_safe - tol over the WHOLE rollout (incl. the wait/yield
//          phase) — zero penetration, even while waiting.
//   LIVE : the drone THREADS once a gap opens and REACHES the goal (not frozen forever).
// Baseline (freeze-on-block) is shown per scenario so a BREACH proves the test is non-vacuous.
//
// Build (Strawberry g++):
//   g++ -std=c++17 -O2 -I cpp/include -I cpp/third_party/eigen -I cpp/third_party \
//       cpp/test/bench_chaos.cpp -o cpp/build/bench_chaos.exe
//   cpp/build/bench_chaos.exe
#include <cstdio>
#include <vector>
#include <memory>
#include <cmath>
#include "sando_cpp/plan_minco.hpp"
#include "sando_cpp/reach_avoid.hpp"   // PRODUCTION monitor (reach_avoid_clearance)
#include "sando_cpp/recovery.hpp"      // PRODUCTION recovery (yield_target)
#include "sando_cpp/types.hpp"
using namespace sando;

// gate constants == recovery_yield() in planner.hpp (horizon/step/mon_buffer)
static const double D_HUMAN = 0.8, HR = 0.3, TOL = 0.05, LANE = 1.3;
static const double HORIZON = 0.6, STEP = 0.6, MON_BUF = 0.5;

static std::map<std::string, AvoidParams> make_cfg() {
  auto c = default_config();
  c["human"] = AvoidParams{"human", "hard", D_HUMAN, 1.0e4};
  c["wall"]  = AvoidParams{"wall",  "hard", 0.15, 1.0e4};   // HARD walls: confine the lane (real block)
  return c;
}
struct Mov { Eigen::Vector3d p0, v; };
struct Roll { double min_h, min_w; bool reached, hit; int ticks, waits, threads; Eigen::Vector3d pf; };
// FREEZE   = baseline: forward if a valid plan exists, else hold on the stale plan (today's freeze).
// PROD     = the ACTUAL production wiring (planner.hpp): agile-FIRST (commit any certified plan),
//            recovery_yield() ONLY when the agile plan fails. No monitor veto of a valid plan.
// MONITOR  = aspirational ABS always-on monitor: veto a valid forward plan when the monitor flags
//            danger (clearance < d_safe+buffer) -> yield early. (Shows the hysteresis gap.)
enum class Mode { FREEZE, PROD, MONITOR };

static Roll rollout(const Eigen::Vector3d& start, const Eigen::Vector3d& goal,
                    std::vector<Mov> crowd,
                    const std::vector<std::shared_ptr<AABBObstacle>>& walls,
                    Mode mode, double dt = 0.30, int max_ticks = 160, double vmax = 3.0) {
  auto cfg = make_cfg();
  Eigen::Vector3d p = start, v = Eigen::Vector3d::Zero(), a = Eigen::Vector3d::Zero();
  std::vector<Eigen::Vector3d> hp; for (auto& m : crowd) hp.push_back(m.p0);
  Roll R{1e18, 1e18, false, false, 0, 0, 0, start};
  for (int tick = 0; tick < max_ticks; ++tick) {
    R.ticks = tick + 1;
    std::vector<std::shared_ptr<SphereObstacle>> own;
    std::vector<const Obstacle*> obs, humans;
    for (auto& w : walls) obs.push_back(w.get());
    for (size_t i = 0; i < crowd.size(); ++i) {
      own.push_back(std::make_shared<SphereObstacle>(hp[i], HR, crowd[i].v, "human"));
      obs.push_back(own.back().get()); humans.push_back(own.back().get());
    }
    int NP = 9; Eigen::MatrixXd ap(NP, 3);
    for (int i = 0; i < NP; ++i) { double f = double(i) / (NP - 1); ap.row(i) = (p + f * (goal - p)).transpose(); }
    PlanOptParams opt; opt.vmax = vmax; opt.amax = 3.0; opt.w_corridor = 50.0; opt.sfc_radius = 0.6;
    DetourConfig dc; dc.enabled = true;

    // ---- AGILE half (production plan_minco) ----
    bool valid = false; double a1 = dt;
    Eigen::Vector3d fp = p, fv = Eigen::Vector3d::Zero(), fa = Eigen::Vector3d::Zero();
    std::shared_ptr<MinjerkTraj> ftr;
    try {
      auto pr = plan_minco(ap, obs, cfg, opt, dc, v, a);
      if (pr.second.trajectory_valid) {
        valid = true; ftr = std::make_shared<MinjerkTraj>(pr.first);
        a1 = std::min(dt, ftr->t_end);
        fp = ftr->eval(a1); fv = ftr->eval_deriv(a1, 1); fa = ftr->eval_deriv(a1, 2);
      }
    } catch (...) { valid = false; }

    // ---- MONITOR (production reach_avoid_clearance, humans only; == recovery_yield gate) ----
    double mon = reach_avoid_clearance(p, humans, 0.0, HORIZON);
    bool danger = mon < D_HUMAN + MON_BUF;
    // agile-first for FREEZE/PROD (commit any certified plan); MONITOR vetoes a valid plan on danger.
    bool do_forward = valid && !(mode == Mode::MONITOR && danger);

    // ---- commit: forward (thread) OR recovery (yield/hold) ----
    std::vector<Eigen::Vector3d> samp; double span;
    Eigen::Vector3d np;
    if (do_forward) {
      R.threads++; np = fp; v = fv; a = fa; span = a1;
      for (int s = 0; s <= 12; ++s) samp.push_back(ftr->eval(std::min(span * s / 12.0, ftr->t_end)));
    } else {
      R.waits++; span = dt;
      Eigen::Vector3d q = p;
      if (mode != Mode::FREEZE && danger) {            // PRODUCTION recovery_yield(): gate on human
        Eigen::Vector3d path_dir = goal - p;           // danger, yield vs ALL obs (won't retreat into wall)
        q = yield_target(p, path_dir, obs, 0.0, HORIZON, STEP);
      }                                                // FREEZE, or no danger -> hold at p
      np = q; v = (q - p) / dt; a = Eigen::Vector3d::Zero();
      for (int s = 0; s <= 12; ++s) samp.push_back(p + (double(s) / 12.0) * (q - p));
    }

    // ---- closed-loop clearance over the committed slice (obstacles move) ----
    for (int s = 0; s <= 12; ++s) {
      double tau = span * s / 12.0;
      for (size_t i = 0; i < crowd.size(); ++i)
        R.min_h = std::min(R.min_h, (samp[s] - (hp[i] + crowd[i].v * tau)).norm() - HR);
      for (auto& w : walls) R.min_w = std::min(R.min_w, w->signed_dist(samp[s], tau));
    }
    if (R.min_h < D_HUMAN - TOL) R.hit = true;
    p = np; R.pf = p;
    for (size_t i = 0; i < crowd.size(); ++i) hp[i] += crowd[i].v * dt;
    if ((p.head<2>() - goal.head<2>()).norm() < 0.4) { R.reached = true; break; }
  }
  return R;
}

static void row(const char* tag, const Roll& r) {
  bool safe = (r.min_h >= D_HUMAN - TOL);
  bool live = r.reached && r.threads > 0;
  const char* verdict = !safe ? "BREACH" : (live ? "PASS" : (r.reached ? "PASS*" : "STUCK"));
  printf("  %-20s | %-6s | min_human=%+.3f %-9s | reach=%-3s | "
         "wait=%3d thread=%3d | final=(%+.1f,%+.1f) %s\n",
         tag, verdict, r.min_h, safe ? "(>=d_safe)" : "(BREACH!)",
         r.reached ? "YES" : "no", r.waits, r.threads, r.pf.x(), r.pf.y(), r.hit ? "<HIT>" : "");
}

int main() {
  printf("Headline acceptance: HOLD-OR-THREAD in chaos. d_safe=%.2f, hard lane |y|<=%.1f, vmax=3.\n", D_HUMAN, LANE);
  printf("PASS = SAFE (zero penetration whole rollout) AND LIVE (threaded + reached).  "
         "PASS* = safe+reached, no thread needed.\n");
  printf("Safe-half = PRODUCTION reach_avoid_clearance + yield_target (the recovery_yield() calls).\n\n");

  std::vector<std::shared_ptr<AABBObstacle>> lane = {
    std::make_shared<AABBObstacle>(Eigen::Vector3d(2, -4.0, 0), Eigen::Vector3d(9, -LANE, 3), "wall"),
    std::make_shared<AABBObstacle>(Eigen::Vector3d(2,  LANE, 0), Eigen::Vector3d(9,  4.0, 3), "wall")};

  // S1: drone starts OUTSIDE the lane (x=0, open area behind it -> can retreat). 5 oncoming humans
  //     fill the lane then walk past and exit -> lane clears -> must WAIT then THREAD.
  printf("=== S1: gate-that-opens (start OUTSIDE lane @ x=0, can retreat) ===\n");
  {
    Eigen::Vector3d start(0, 0, 1.5), goal(11, 0, 1.5);
    std::vector<Mov> crowd = {
      {Eigen::Vector3d(4.0, 0.0, 1.5), {-1, 0, 0}}, {Eigen::Vector3d(5.0, 0.7, 1.5), {-1, 0, 0}},
      {Eigen::Vector3d(6.0,-0.7, 1.5), {-1, 0, 0}}, {Eigen::Vector3d(7.0, 0.4, 1.5), {-1, 0, 0}},
      {Eigen::Vector3d(8.0,-0.4, 1.5), {-1, 0, 0}}};
    row("baseline (freeze)",   rollout(start, goal, crowd, lane, Mode::FREEZE));
    row("production wiring",    rollout(start, goal, crowd, lane, Mode::PROD));
    row("+ always-on monitor",  rollout(start, goal, crowd, lane, Mode::MONITOR));
  }

  // S2: drone starts INSIDE the hard lane (x=3.5, walls on both sides -> NO retreat). 5 oncoming
  //     humans. The honest no-escape clause: yield is confined to |y|<1.3; may be insufficient.
  printf("\n=== S2: no-escape (start INSIDE hard lane @ x=3.5, confined) ===\n");
  {
    Eigen::Vector3d start(3.5, 0, 1.5), goal(11, 0, 1.5);
    std::vector<Mov> crowd = {
      {Eigen::Vector3d(5.0, 0.0, 1.5), {-1, 0, 0}}, {Eigen::Vector3d(6.0, 0.6, 1.5), {-1, 0, 0}},
      {Eigen::Vector3d(6.5,-0.6, 1.5), {-1, 0, 0}}, {Eigen::Vector3d(7.5, 0.3, 1.5), {-1, 0, 0}},
      {Eigen::Vector3d(8.0,-0.3, 1.5), {-1, 0, 0}}};
    row("baseline (freeze)",   rollout(start, goal, crowd, lane, Mode::FREEZE));
    row("production wiring",    rollout(start, goal, crowd, lane, Mode::PROD));
    row("+ always-on monitor",  rollout(start, goal, crowd, lane, Mode::MONITOR));
  }

  printf("\n(closed-loop: each tick warm-started plan_minco; on block/danger the PRODUCTION yield_target moves;\n");
  printf(" humans walk -x at 1.0 m/s; clearance densely sampled over each committed slice.)\n");
  return 0;
}
