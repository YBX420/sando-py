// Golden verification: C++ plan_minco.hpp must reproduce the Python
// sando_py.local.local_opt MINCO driver.
//
// Reads cpp/golden/plan_minco_cases.txt (gen_plan_minco_golden.py). Five case kinds:
//   ORTHO : _orthonormal_pair                 -> EXACT (abs tol)
//   PART  : _partition_obstacles              -> EXACT (soft/hard tags, counts)
//   SEEDS : _generate_detour_seeds            -> EXACT (names + path geometry)
//   FEAS  : check_feasibility(given traj)     -> EXACT (all fields)
//   PLAN  : plan_minco end-to-end             -> NOT bit-exact (LBFGSpp vs scipy
//           L-BFGS-B differ on iterates). We REPORT the true sampled-position max
//           error vs Python + whether trajectory_valid matches; this is informational
//           (not a hard pass/fail gate) and printed honestly.
//
// 不许欺骗、必须还原: the deterministic helpers must match exactly or FAIL.
#include "sando_cpp/plan_minco.hpp"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <cmath>
#include <limits>
#include <Eigen/Dense>

using std::string;
using std::vector;

static vector<double> nums(const string& rest) {
  vector<double> v; std::istringstream is(rest); double x;
  while (is >> x) v.push_back(x);
  return v;
}

int main(int argc, char** argv) {
  const char* path = (argc > 1) ? argv[1] : "golden/plan_minco_cases.txt";
  std::ifstream f(path);
  if (!f) { std::printf("CANNOT OPEN %s\n", path); return 2; }

  string line;
  std::map<string, vector<double>> num;
  std::map<string, string> str;
  // obstacle buckets keyed by prefix index
  std::map<int, std::map<string, string>> obs_str;
  std::map<int, std::map<string, vector<double>>> obs_num;
  std::map<int, string> seed_name; std::map<int, vector<double>> seed_path; std::map<int, int> seed_np;
  vector<std::tuple<string, double, double, string>> classes;

  int ncase = 0, nfail = 0;
  double global_max_err = 0.0;          // over EXACT helper cases
  const double TOL = 1e-7;

  // PLAN aggregate stats
  int plan_n = 0, plan_valid_match = 0;
  double plan_pos_err_max = 0.0, plan_pos_err_worst_case = 0.0;
  string plan_worst_name;
  double plan_cost_rel_max = 0.0;

  auto build_cfg = [&]() {
    std::map<string, sando::AvoidParams> cfg = sando::default_config();
    for (auto& cd : classes) {
      const string& cn = std::get<0>(cd);
      cfg[cn] = sando::AvoidParams{cn, std::get<3>(cd), std::get<1>(cd), std::get<2>(cd)};
    }
    return cfg;
  };

  // owns obstacles built per case
  vector<std::shared_ptr<sando::Obstacle>> owned;
  auto build_obs = [&](std::map<string, string>& sm, std::map<string, vector<double>>& nm)
      -> std::shared_ptr<sando::Obstacle> {
    if (sm["KIND"] == "sphere") {
      Eigen::Vector3d c0(nm["C0"][0], nm["C0"][1], nm["C0"][2]);
      double r = nm["R"][0];
      Eigen::Vector3d vel(nm["VEL"][0], nm["VEL"][1], nm["VEL"][2]);
      Eigen::Vector3d acc(nm["ACC"][0], nm["ACC"][1], nm["ACC"][2]);
      return std::make_shared<sando::SphereObstacle>(c0, r, vel, sm["CLASS"], acc);
    }
    Eigen::Vector3d lo(nm["LO"][0], nm["LO"][1], nm["LO"][2]);
    Eigen::Vector3d hi(nm["HI"][0], nm["HI"][1], nm["HI"][2]);
    return std::make_shared<sando::AABBObstacle>(lo, hi, sm["CLASS"]);
  };

  auto read_opt = [&]() {
    sando::PlanOptParams o;
    o.w_smooth = num["W_SMOOTH"][0]; o.w_vel = num["W_VEL"][0]; o.w_accel = num["W_ACCEL"][0];
    o.w_time = num["W_TIME"][0]; o.w_obs = num["W_OBS"][0];
    o.vmax = num["VMAX"][0]; o.amax = num["AMAX"][0]; o.dt_min = num["DT_MIN"][0];
    o.K = (int)num["K"][0]; o.maxiter = (int)num["MAXITER"][0]; o.kappa = (int)num["KAPPA"][0];
    o.M_cap = (int)num["M_CAP"][0];
    o.clearance_tol = num["CLEAR_TOL"][0]; o.vel_tol = num["VEL_TOL"][0]; o.accel_tol = num["ACCEL_TOL"][0];
    { string ov = str.count("AVOID_OVERRIDE") ? str["AVOID_OVERRIDE"] : "_";
      o.avoid_override = (ov == "_") ? "" : ov; }
    o.alm_rho0 = num["ALM_RHO0"][0]; o.alm_rho_max = num["ALM_RHO_MAX"][0];
    o.alm_rho_grow = num["ALM_RHO_GROW"][0]; o.alm_outer_iters = (int)num["ALM_OUTER"][0];
    o.alm_viol_tol = num["ALM_VIOL_TOL"][0]; o.alm_inner_maxiter = (int)num["ALM_INNER_MAXITER"][0];
    o.tau_trust = num["TAU_TRUST"][0]; o.spacetime_hard = (num["SPACETIME_HARD"][0] != 0.0);
    o.q_conformal = num["Q_CONFORMAL"][0]; o.epsilon_track = num["EPSILON_TRACK"][0];
    o.safety_mode = str["SAFETY_MODE"]; o.reach_model = str["REACH_MODEL"];
    o.a_max_human = num["A_MAX_HUMAN"][0]; o.v_max_human = num["V_MAX_HUMAN"][0];
    o.est_pos_err = num["EST_POS_ERR"][0]; o.est_vel_err = num["EST_VEL_ERR"][0];
    return o;
  };
  auto read_detour = [&]() {
    sando::DetourConfig d;
    d.enabled = (num["D_ENABLED"][0] != 0.0); d.directions = (int)num["D_DIRECTIONS"][0];
    d.trigger_margin = num["D_TRIGGER_MARGIN"][0]; d.detour_margin = num["D_DETOUR_MARGIN"][0];
    d.top_k_obstacles = (int)num["D_TOPK"][0]; d.max_seeds = (int)num["D_MAX_SEEDS"][0];
    d.K_eval = (int)num["D_KEVAL"][0]; d.perturb_window = (int)num["D_PWINDOW"][0];
    return d;
  };

  auto process = [&]() {
    const string kind = str["KIND"];
    const string name = str.count("NAME") ? str["NAME"] : "?";
    double e = 0.0;
    bool exact = true;   // ORTHO/PART/SEEDS/FEAS are exact; PLAN is informational

    if (kind == "ORTHO") {
      Eigen::Vector3d ax(num["AXIS"][0], num["AXIS"][1], num["AXIS"][2]);
      auto uv = sando::orthonormal_pair(ax);
      for (int j = 0; j < 3; ++j) {
        e = std::max(e, std::fabs(uv.first(j) - num["U"][j]));
        e = std::max(e, std::fabs(uv.second(j) - num["V"][j]));
      }
    } else if (kind == "PART") {
      auto cfg = build_cfg();
      string ov = str.count("OVERRIDE") ? str["OVERRIDE"] : "_";
      string override_ = (ov == "_") ? "" : ov;
      int N = (int)num["N"][0];
      owned.clear();
      vector<const sando::Obstacle*> obstacles;
      for (int i = 0; i < N; ++i) { auto o = build_obs(obs_str[i], obs_num[i]); owned.push_back(o); obstacles.push_back(o.get()); }
      vector<const sando::Obstacle*> soft, hard;
      sando::partition_obstacles(obstacles, cfg, override_, soft, hard);
      e = std::max(e, std::fabs((double)soft.size() - num["NSOFT"][0]));
      e = std::max(e, std::fabs((double)hard.size() - num["NHARD"][0]));
      for (int i = 0; i < N; ++i) {
        double tag = (sando::resolve_mode(obstacles[i]->class_name, cfg, override_) == "hard") ? 1.0 : 0.0;
        e = std::max(e, std::fabs(tag - num["TAGS"][i]));
      }
    } else if (kind == "SEEDS") {
      auto cfg = build_cfg();
      sando::PlanOptParams opt = read_opt();
      sando::DetourConfig dc = read_detour();
      int NP = (int)num["NP"][0];
      Eigen::MatrixXd ap(NP, 3);
      for (int i = 0; i < NP; ++i) for (int j = 0; j < 3; ++j) ap(i, j) = num["PATH"][i * 3 + j];
      int NOBS = (int)num["NOBS"][0];
      owned.clear();
      vector<const sando::Obstacle*> obstacles;
      for (int i = 0; i < NOBS; ++i) { auto o = build_obs(obs_str[i], obs_num[i]); owned.push_back(o); obstacles.push_back(o.get()); }
      auto seeds = sando::generate_detour_seeds(ap, obstacles, cfg, opt, dc);
      int NSEEDS = (int)num["NSEEDS"][0];
      e = std::max(e, std::fabs((double)seeds.size() - NSEEDS));
      int nchk = std::min((int)seeds.size(), NSEEDS);
      for (int si = 0; si < nchk; ++si) {
        if (seeds[si].first != seed_name[si]) e = std::max(e, 1.0);  // name mismatch
        int gnp = seed_np[si];
        if ((int)seeds[si].second.rows() != gnp) { e = std::max(e, 1.0); continue; }
        for (int i = 0; i < gnp; ++i)
          for (int j = 0; j < 3; ++j)
            e = std::max(e, std::fabs(seeds[si].second(i, j) - seed_path[si][i * 3 + j]));
      }
    } else if (kind == "FEAS") {
      auto cfg = build_cfg();
      sando::PlanOptParams opt = read_opt();
      int M = (int)num["M"][0];
      Eigen::MatrixXd wp(M + 1, 3);
      for (int i = 0; i < M + 1; ++i) for (int j = 0; j < 3; ++j) wp(i, j) = num["WP"][i * 3 + j];
      Eigen::VectorXd T = Eigen::Map<Eigen::VectorXd>(num["T"].data(), M);
      auto v3 = [&](const char* k) { return Eigen::Vector3d(num[k][0], num[k][1], num[k][2]); };
      sando::MinjerkTraj tr(wp, T, v3("V0"), v3("A0"), v3("VF"), v3("AF"));
      int NOBS = (int)num["NOBS"][0];
      owned.clear();
      vector<const sando::Obstacle*> obstacles;
      for (int i = 0; i < NOBS; ++i) { auto o = build_obs(obs_str[i], obs_num[i]); owned.push_back(o); obstacles.push_back(o.get()); }
      sando::Feasibility F = sando::check_feasibility(tr, obstacles, cfg, opt, opt.K);
      e = std::max(e, std::fabs(F.min_clearance - num["F_MIN_CLR"][0]));
      e = std::max(e, std::fabs(F.max_clearance_violation - num["F_MAX_CLR_VIOL"][0]));
      e = std::max(e, std::fabs(F.max_v - num["F_MAX_V"][0]));
      e = std::max(e, std::fabs(F.max_a - num["F_MAX_A"][0]));
      e = std::max(e, std::fabs(F.vel_violation - num["F_VEL_VIOL"][0]));
      e = std::max(e, std::fabs(F.accel_violation - num["F_ACCEL_VIOL"][0]));
      e = std::max(e, std::fabs((F.trajectory_valid ? 1.0 : 0.0) - num["F_VALID"][0]));
      string gr = str.count("F_REASON") ? str["F_REASON"] : "_";
      string cr = F.failure_reason.empty() ? "_" : F.failure_reason;
      if (gr != cr) e = std::max(e, 1.0);
    } else if (kind == "PLAN") {
      exact = false;  // informational
      auto cfg = build_cfg();
      sando::PlanOptParams opt = read_opt();
      sando::DetourConfig dc = read_detour();
      int NP = (int)num["NP"][0];
      Eigen::MatrixXd ap(NP, 3);
      for (int i = 0; i < NP; ++i) for (int j = 0; j < 3; ++j) ap(i, j) = num["PATH"][i * 3 + j];
      int NOBS = (int)num["NOBS"][0];
      owned.clear();
      vector<const sando::Obstacle*> obstacles;
      for (int i = 0; i < NOBS; ++i) { auto o = build_obs(obs_str[i], obs_num[i]); owned.push_back(o); obstacles.push_back(o.get()); }
      Eigen::Vector3d v0(num["V0"][0], num["V0"][1], num["V0"][2]);
      Eigen::Vector3d a0(num["A0"][0], num["A0"][1], num["A0"][2]);
      auto pr = sando::plan_minco(ap, obstacles, cfg, opt, dc, v0, a0);
      const sando::MinjerkTraj& tr = pr.first;
      const sando::PlanInfo& info = pr.second;
      int NSAMP = (int)num["NSAMP"][0];
      // Sample over the C++ trajectory's OWN [t_start, t_end] and compare positions to
      // Python's sampled positions (Python sampled over its own [t_start, t_end]). Because
      // both parameterise s in [0,1] over the same waypoint sequence, like-fraction samples
      // are the right comparison.
      double pos_err = 0.0;
      for (int s = 0; s < NSAMP; ++s) {
        double frac = (NSAMP == 1) ? 0.0 : double(s) / double(NSAMP - 1);
        double t = tr.t_start + (tr.t_end - tr.t_start) * frac;
        Eigen::Vector3d p = tr.eval(t);
        for (int j = 0; j < 3; ++j)
          pos_err = std::max(pos_err, std::fabs(p(j) - num["POS"][s * 3 + j]));
      }
      bool valid_match = ((info.trajectory_valid ? 1.0 : 0.0) == num["VALID"][0]);
      double pyc = num["FINAL_COST"][0];
      double cost_rel = std::fabs(info.final_cost - pyc) / (std::fabs(pyc) + 1.0);
      plan_n++;
      if (valid_match) plan_valid_match++;
      if (pos_err > plan_pos_err_worst_case) { plan_pos_err_worst_case = pos_err; plan_worst_name = name; }
      plan_pos_err_max = std::max(plan_pos_err_max, pos_err);
      plan_cost_rel_max = std::max(plan_cost_rel_max, cost_rel);
      std::printf("  PLAN %-20s : pos_err=%.3e valid_match=%d (cpp_valid=%d py_valid=%d) "
                  "cost_rel=%.3e seed[cpp=%s py=%s]\n",
                  name.c_str(), pos_err, valid_match ? 1 : 0,
                  info.trajectory_valid ? 1 : 0, (int)num["VALID"][0],
                  cost_rel, info.seed_used.c_str(),
                  str.count("SEED_USED") ? str["SEED_USED"].c_str() : "?");
      ncase++;
      return;  // do not gate on PLAN
    }

    global_max_err = std::max(global_max_err, e);
    bool ok = e < TOL;
    if (!ok) nfail++;
    std::printf("  %-6s %-20s : max_err=%.3e  %s\n", kind.c_str(), name.c_str(), e,
                ok ? "PASS" : "FAIL");
    ncase++;
    (void)exact;
  };

  while (std::getline(f, line)) {
    if (line == "CASE") {
      num.clear(); str.clear(); obs_str.clear(); obs_num.clear();
      seed_name.clear(); seed_path.clear(); seed_np.clear(); classes.clear();
      continue;
    }
    if (line == "END") { process(); continue; }
    std::istringstream is(line);
    string key; is >> key;
    string rest; std::getline(is, rest);
    if (!rest.empty() && rest[0] == ' ') rest.erase(0, 1);

    if (key == "KIND" || key == "NAME" || key == "OVERRIDE" || key == "AVOID_OVERRIDE" ||
        key == "SAFETY_MODE" || key == "REACH_MODEL" || key == "F_REASON" || key == "SEED_USED") {
      str[key] = rest; continue;
    }
    // class metadata: "CLASS<i> name d_safe weight mode"
    if (key.rfind("CLASS", 0) == 0 && key != "CLASS") {
      std::istringstream cs(rest); string cname, mode; double ds, w;
      cs >> cname >> ds >> w >> mode;
      classes.push_back({cname, ds, w, mode}); continue;
    }
    // obstacle keys: "OBS<i>_FIELD ..."
    if (key.rfind("OBS", 0) == 0 && key != "OBS" && key.find('_') != string::npos) {
      size_t us = key.find('_');
      int i = std::stoi(key.substr(3, us - 3));
      string field = key.substr(us + 1);
      if (field == "KIND" || field == "CLASS") obs_str[i][field] = rest;
      else obs_num[i][field] = nums(rest);
      continue;
    }
    // seed keys: "SEED<i>_NAME / _NP / _PATH"
    if (key.rfind("SEED", 0) == 0 && key.find('_') != string::npos) {
      size_t us = key.find('_');
      int i = std::stoi(key.substr(4, us - 4));
      string field = key.substr(us + 1);
      if (field == "NAME") seed_name[i] = rest;
      else if (field == "NP") seed_np[i] = (int)nums(rest)[0];
      else if (field == "PATH") seed_path[i] = nums(rest);
      continue;
    }
    num[key] = nums(rest);
  }

  std::printf("\nplan_minco golden (EXACT helpers): %d helper-fail, global_max_err=%.3e (tol=%.0e)\n",
              nfail, global_max_err, TOL);
  std::printf("plan_minco END-TO-END (informational, NOT bit-exact): %d cases, "
              "valid_match=%d/%d, pos_err_max=%.3e (worst=%s), cost_rel_max=%.3e\n",
              plan_n, plan_valid_match, plan_n, plan_pos_err_max, plan_worst_name.c_str(),
              plan_cost_rel_max);
  std::printf("%s\n", nfail == 0
              ? "ALL EXACT HELPERS PASS (C++ reproduces Python deterministic driver pieces)"
              : "FAILED (an EXACT helper case did not match Python)");
  return nfail == 0 ? 0 : 1;
}
