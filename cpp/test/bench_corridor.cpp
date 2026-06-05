// bench_corridor — does turning the SFC corridor ON stop the detour from relaxing back to
// the straight "pocket"? (Audit root cause #1 soft-field zero-gradient + #2 corridor OFF.)
//
// This drives the PRODUCTION plan_minco (NOT the A+B af_* solver that bench_tunnel uses), so
// it actually exercises the corridor knob. Pure experiment: production DEFAULTS are untouched
// (golden-safe); we set w_corridor/sfc_radius locally per run.
//
// Build (Strawberry g++):
//   g++ -std=c++17 -O2 -I cpp/include -I cpp/third_party/eigen -I cpp/third_party \
//       cpp/test/bench_corridor.cpp -o cpp/build/bench_corridor.exe
//   cpp/build/bench_corridor.exe
#include <cstdio>
#include <vector>
#include <memory>
#include <cmath>
#include "sando_cpp/plan_minco.hpp"
using namespace sando;

static std::map<std::string, AvoidParams> make_cfg() {
  std::map<std::string, AvoidParams> c = default_config();
  c["human"] = AvoidParams{"human", "hard", 0.8, 1.0e4};
  c["wall"]  = AvoidParams{"wall",  "soft", 0.4, 1.0e1};
  return c;
}

static Eigen::MatrixXd straight_path(const Eigen::Vector3d& a, const Eigen::Vector3d& b, int np) {
  Eigen::MatrixXd p(np, 3);
  for (int i = 0; i < np; ++i) { double f = double(i) / (np - 1); p.row(i) = (a + f * (b - a)).transpose(); }
  return p;
}

struct Metrics { double min_h, min_w, ymax, tend; bool valid; std::string seed; };

static Metrics run(const Eigen::MatrixXd& ap, const std::vector<const Obstacle*>& obs,
                   const Obstacle* wall, const Obstacle* human,
                   double w_corridor, double sfc_radius) {
  auto cfg = make_cfg();
  PlanOptParams opt;              // production defaults
  opt.vmax = 3.0; opt.amax = 3.0;
  opt.w_corridor = w_corridor; opt.sfc_radius = sfc_radius;   // the experiment knobs
  DetourConfig dc; dc.enabled = true;                          // +-u/+-v multistart (C++ default)
  Eigen::Vector3d z = Eigen::Vector3d::Zero();
  auto pr = plan_minco(ap, obs, cfg, opt, dc, z, z);
  const MinjerkTraj& tr = pr.first; const PlanInfo& info = pr.second;
  Metrics m{1e18, 1e18, 0.0, tr.t_end, info.trajectory_valid, info.seed_used};
  const int K = 300;
  for (int s = 0; s < K; ++s) {
    double t = tr.t_start + (tr.t_end - tr.t_start) * double(s) / (K - 1);
    Eigen::Vector3d p = tr.eval(t);
    if (human) m.min_h = std::min(m.min_h, human->signed_dist(p, t));
    if (wall)  m.min_w = std::min(m.min_w, wall->signed_dist(p, t));
    m.ymax = std::max(m.ymax, std::fabs(p[1]));
  }
  return m;
}

int main() {
  Eigen::Vector3d start(0, 0, 1.5), goal(8, 0, 1.5);
  Eigen::MatrixXd ap = straight_path(start, goal, 9);

  printf("=== Scenario A: a SOFT WALL straddling the straight path (d_safe_wall=0.4) ===\n");
  printf("    (#1 soft-field has zero gradient past d_safe -> detour should relax to graze w/o corridor)\n");
  auto wallA = std::make_shared<AABBObstacle>(Eigen::Vector3d(3.5, -0.5, 0.0),
                                              Eigen::Vector3d(4.5,  0.5, 3.0), "wall");
  std::vector<const Obstacle*> obsA{ wallA.get() };
  printf("%-22s | %-10s | %-8s | %-6s | %s\n", "config", "min_wall_clr", "max|y|", "valid", "seed");
  for (auto cw : std::vector<std::pair<double,double>>{{0,0},{50,0.6},{200,0.8}}) {
    Metrics m = run(ap, obsA, wallA.get(), nullptr, cw.first, cw.second);
    printf("w_corr=%-5.0f r=%-4.2f      | %+10.3f | %-8.3f | %-6d | %s\n",
           cw.first, cw.second, m.min_w, m.ymax, m.valid ? 1 : 0, m.seed.c_str());
  }

  printf("\n=== Scenario B: a tunnel (two soft walls, lane |y|<=1.3) + a HARD human off-centre ===\n");
  printf("    (does corridor keep it committed to the open -y side instead of relaxing centred?)\n");
  auto lo_wall = std::make_shared<AABBObstacle>(Eigen::Vector3d(2.0, -3.0, 0.0),
                                                Eigen::Vector3d(6.0, -1.3, 3.0), "wall");
  auto hi_wall = std::make_shared<AABBObstacle>(Eigen::Vector3d(2.0,  1.3, 0.0),
                                                Eigen::Vector3d(6.0,  3.0, 3.0), "wall");
  auto humanB = std::make_shared<SphereObstacle>(Eigen::Vector3d(4.0, 0.4, 1.5), 0.3,
                                                 Eigen::Vector3d(0, 0, 0), "human");
  std::vector<const Obstacle*> obsB{ lo_wall.get(), hi_wall.get(), humanB.get() };
  printf("%-22s | %-9s | %-9s | %-8s | %-6s | %s\n",
         "config", "min_human", "min_wall", "max|y|", "valid", "seed");
  for (auto cw : std::vector<std::pair<double,double>>{{0,0},{50,0.6},{200,0.8}}) {
    Metrics m = run(ap, obsB, lo_wall.get(), humanB.get(), cw.first, cw.second);
    // min_wall over BOTH walls:
    double mw = m.min_w;
    {
      auto cfg = make_cfg(); PlanOptParams opt; opt.vmax=3; opt.amax=3;
      opt.w_corridor=cw.first; opt.sfc_radius=cw.second; DetourConfig dc; dc.enabled=true;
      Eigen::Vector3d z=Eigen::Vector3d::Zero();
      auto pr = plan_minco(ap, obsB, cfg, opt, dc, z, z); const MinjerkTraj& tr=pr.first;
      for (int s=0;s<300;++s){ double t=tr.t_start+(tr.t_end-tr.t_start)*s/299.0; Eigen::Vector3d p=tr.eval(t);
        mw=std::min(mw, hi_wall->signed_dist(p,t)); }
    }
    printf("w_corr=%-5.0f r=%-4.2f      | %+9.3f | %+9.3f | %-8.3f | %-6d | %s\n",
           cw.first, cw.second, m.min_h, mw, m.ymax, m.valid ? 1 : 0, m.seed.c_str());
  }
  printf("\nReading: if corridor ON raises min clearance / commits the detour (vs w_corr=0),\n");
  printf("the #1+#2 'detour relaxes to straight' root cause is confirmed + mitigated.\n");
  return 0;
}
