// Golden verification: C++ minco_cost_grad.hpp must reproduce Python
// local_opt._minco_cost_grad (the analytic objective+gradient L-BFGS-B calls).
// Reads cpp/golden/minco_cost_grad_cases.txt (gen_minco_cost_grad_golden.py),
// rebuilds the scenario (start/goal/M/x/T_target, soft+hard obstacles, OptParams,
// fixed lambda/rho, frozen-or-recomputed normals+trust mask), runs minco_cost_grad,
// and asserts cost (absolute) + grad (relative, through the MINCO adjoint) match.
// 不许欺骗、必须还原: 对不上就 FAIL。
#include "sando_cpp/minco_cost_grad.hpp"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <cmath>
#include <Eigen/Dense>

using std::string;
using std::vector;

static vector<double> nums(const string& rest) {
  vector<double> v;
  std::istringstream is(rest);
  double x;
  while (is >> x) v.push_back(x);
  return v;
}

int main(int argc, char** argv) {
  const char* path = (argc > 1) ? argv[1] : "golden/minco_cost_grad_cases.txt";
  std::ifstream f(path);
  if (!f) { std::printf("CANNOT OPEN %s\n", path); return 2; }

  string line;
  std::map<string, vector<double>> num;     // numeric keys
  std::map<string, string> str;             // string keys
  // per-case obstacle buckets, keyed by prefix tag (SOFT / HARD), index, field
  vector<std::map<string, string>> soft_str, hard_str;
  vector<std::map<string, vector<double>>> soft_num, hard_num;
  // per-class cfg: name -> (d_safe, weight, mode)
  vector<std::tuple<string, double, double, string>> classes;

  int ncase = 0, nfail = 0;
  double global_max_err = 0.0;
  // cost is an absolute-magnitude scalar -> absolute tol. grad passes through the LU
  // adjoint (grad_from_dcost_dc) where magnitudes can legitimately blow up, so grad
  // uses a relative band |a-b|/(|b|+1). (Same discipline as test_minjerk for the adjoint.)
  const double TOL_COST = 1e-6;
  const double TOL_GRAD = 1e-7;

  auto process = [&]() {
    int M = (int)num["M"][0];
    int nq = 3 * (M - 1);
    Eigen::VectorXd x = Eigen::Map<Eigen::VectorXd>(num["X"].data(), nq + M);
    // split x -> q, T (mirror the header)
    Eigen::VectorXd T(M);
    for (int i = 0; i < M; ++i) T(i) = x(nq + i);

    Eigen::Vector3d start(num["START"][0], num["START"][1], num["START"][2]);
    Eigen::Vector3d goal(num["GOAL"][0], num["GOAL"][1], num["GOAL"][2]);
    double T_target = num["TTARGET"][0];
    auto vec3 = [&](const char* k) {
      return Eigen::Vector3d(num[k][0], num[k][1], num[k][2]);
    };
    Eigen::Vector3d v0 = vec3("V0"), a0 = vec3("A0"), vf = vec3("VF"), af = vec3("AF");

    // OptParams
    sando::CostGradOptParams opt;
    opt.w_smooth = num["W_SMOOTH"][0];
    opt.w_obs    = num["W_OBS"][0];
    opt.w_vel    = num["W_VEL"][0];
    opt.w_accel  = num["W_ACCEL"][0];
    opt.w_time   = num["W_TIME"][0];
    opt.vmax     = num["VMAX"][0];
    opt.amax     = num["AMAX"][0];
    opt.kappa    = (int)num["KAPPA"][0];
    {
      string ov = str.count("AVOID_OVERRIDE") ? str["AVOID_OVERRIDE"] : "_";
      opt.avoid_override = (ov == "_") ? "" : ov;
    }
    opt.tau_trust     = num["TAU_TRUST"][0];
    opt.spacetime_hard = (num["SPACETIME_HARD"][0] != 0.0);
    opt.q_conformal   = num["Q_CONFORMAL"][0];
    opt.epsilon_track = num["EPSILON_TRACK"][0];
    opt.safety_mode   = str["SAFETY_MODE"];
    opt.reach_model   = str["REACH_MODEL"];
    opt.a_max_human   = num["A_MAX_HUMAN"][0];
    opt.v_max_human   = num["V_MAX_HUMAN"][0];
    opt.est_pos_err   = num["EST_POS_ERR"][0];
    opt.est_vel_err   = num["EST_VEL_ERR"][0];

    // optional SFC corridor (only present in cases that exercise it; absent -> term OFF -> parity).
    // corridor_centers must outlive the minco_cost_grad call below (declared in this case scope).
    Eigen::MatrixXd corridor_centers;
    if (num.count("W_CORRIDOR")) {
      opt.w_corridor      = num["W_CORRIDOR"][0];
      opt.corridor_radius = num["CORRIDOR_RADIUS"][0];
      const auto& cc = num["CORRIDOR_CENTERS"];   // (M-1)*3 flat
      corridor_centers.resize(M - 1, 3);
      for (int i = 0; i < M - 1; ++i)
        for (int j = 0; j < 3; ++j) corridor_centers(i, j) = cc[i * 3 + j];
      opt.corridor_centers = &corridor_centers;
    }

    // avoid config: start from default_config, override per dumped class so the lookups
    // (mode for the soft/hard split, d_safe/weight for the soft term) all match Python.
    std::map<string, sando::AvoidParams> cfg = sando::default_config();
    for (auto& cd : classes) {
      const string& cn = std::get<0>(cd);
      cfg[cn] = sando::AvoidParams{cn, std::get<3>(cd), std::get<1>(cd), std::get<2>(cd)};
    }

    // build obstacle objects
    auto build = [&](std::map<string, string>& sm, std::map<string, vector<double>>& nm)
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

    int S = (int)num["S"][0];
    int H = (int)num["H"][0];
    vector<std::shared_ptr<sando::Obstacle>> owned;
    vector<const sando::Obstacle*> soft_obs, hard_obs;
    for (int s = 0; s < S; ++s) {
      auto o = build(soft_str[s], soft_num[s]);
      owned.push_back(o); soft_obs.push_back(o.get());
    }
    for (int h = 0; h < H; ++h) {
      auto o = build(hard_str[h], hard_num[h]);
      owned.push_back(o); hard_obs.push_back(o.get());
    }

    // ALM frozen state
    sando::AlmState alm;
    alm.hard_obs = hard_obs;
    alm.rho = num["RHO"][0];
    int Nc = (int)num["NC"][0];
    bool freeze = (num.count("FREEZE") && num["FREEZE"][0] != 0.0);
    if (Nc > 0) {
      alm.lam = Eigen::Map<Eigen::VectorXd>(num["LAM"].data(), Nc);
      // normals + trust mask are always dumped for hard cases (frozen OR recomputed-from-tr0,
      // both passed explicitly to mirror the solver). Feed them in directly.
      alm.normals = num["NORMALS"];                 // (M,6,H,3) flat
      alm.trust_mask.resize(M, 6);
      for (int i = 0; i < M; ++i)
        for (int k = 0; k < 6; ++k)
          alm.trust_mask(i, k) = (num["TMASK"][i * 6 + k] != 0.0);
    }

    auto res = sando::minco_cost_grad(x, start, goal, M, soft_obs, cfg, opt, T_target,
                                      v0, a0, vf, af, alm);

    double e = 0.0;
    // cost (absolute)
    e = std::max(e, std::fabs(res.f - num["COST"][0]));
    double e_cost = e;
    // grad (relative, through the adjoint LU)
    double e_grad = 0.0;
    for (int i = 0; i < nq + M; ++i) {
      double a = res.grad(i), b = num["GRAD"][i];
      e_grad = std::max(e_grad, std::fabs(a - b) / (std::fabs(b) + 1.0));
    }

    global_max_err = std::max(global_max_err, std::max(e_cost, e_grad));
    bool ok = (e_cost < TOL_COST) && (e_grad < TOL_GRAD);
    std::printf("  case %-20s M=%d S=%d H=%d Nc=%d frz=%d : cost_err=%.3e grad_err=%.3e  %s\n",
                str.count("NAME") ? str["NAME"].c_str() : "?", M, S, H, Nc,
                freeze ? 1 : 0, e_cost, e_grad, ok ? "PASS" : "FAIL");
    if (!ok) nfail++;
    ncase++;
  };

  while (std::getline(f, line)) {
    if (line == "CASE") {
      num.clear(); str.clear();
      soft_str.clear(); soft_num.clear(); hard_str.clear(); hard_num.clear();
      classes.clear();
      continue;
    }
    if (line == "END") { process(); continue; }
    std::istringstream is(line);
    string key; is >> key;
    string rest; std::getline(is, rest);
    if (!rest.empty() && rest[0] == ' ') rest.erase(0, 1);

    if (key == "NAME" || key == "SAFETY_MODE" || key == "REACH_MODEL" ||
        key == "AVOID_OVERRIDE") {
      str[key] = rest;
      continue;
    }
    // class metadata: "CLASS<i> name d_safe weight mode"
    if (key.rfind("CLASS", 0) == 0 && key != "CLASS") {
      std::istringstream cs(rest);
      string cname, mode; double ds, w;
      cs >> cname >> ds >> w >> mode;
      classes.push_back({cname, ds, w, mode});
      continue;
    }
    // soft obstacle keys: "SOFT<i>_FIELD ..."
    if (key.rfind("SOFT", 0) == 0) {
      size_t us = key.find('_');
      int i = std::stoi(key.substr(4, us - 4));
      string field = key.substr(us + 1);
      if ((int)soft_str.size() <= i) { soft_str.resize(i + 1); soft_num.resize(i + 1); }
      if (field == "KIND" || field == "CLASS") soft_str[i][field] = rest;
      else soft_num[i][field] = nums(rest);
      continue;
    }
    // hard obstacle keys: "HARD<i>_FIELD ..."
    if (key.rfind("HARD", 0) == 0) {
      size_t us = key.find('_');
      int i = std::stoi(key.substr(4, us - 4));
      string field = key.substr(us + 1);
      if ((int)hard_str.size() <= i) { hard_str.resize(i + 1); hard_num.resize(i + 1); }
      if (field == "KIND" || field == "CLASS") hard_str[i][field] = rest;
      else hard_num[i][field] = nums(rest);
      continue;
    }
    num[key] = nums(rest);
  }

  std::printf("\nMINCO cost+grad golden: %d cases, %d fail, global_max_err=%.3e "
              "(tol_cost=%.0e tol_grad=%.0e)\n",
              ncase, nfail, global_max_err, TOL_COST, TOL_GRAD);
  std::printf("%s\n", nfail == 0 ? "ALL PASS (C++ reproduces Python _minco_cost_grad)"
                                 : "FAILED");
  return nfail == 0 ? 0 : 1;
}
