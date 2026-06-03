// Golden verification: C++ HARD per-class convex-hull ALM (local_opt_hardalm.hpp)
// must reproduce Python local_opt._alm_term / _alm_constraints / _seg_normals.
// Reads cpp/golden/local_opt_hardalm_cases.txt (gen_local_opt_hardalm_golden.py),
// rebuilds the trajectory + obstacles + OptParams + fixed (lambda, rho) + freeze,
// runs ONE inner ALM evaluation, and asserts cost / dcost_dc / dT_exp / per-point
// g/dist/R/dgdt/trust match within tolerance. 不许欺骗、必须还原:对不上就 FAIL.
#include "sando_cpp/local_opt_hardalm.hpp"
#include "sando_cpp/minjerk_traj.hpp"
#include "sando_cpp/obstacles.hpp"
#include "sando_cpp/avoid_config.hpp"
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
  const char* path = (argc > 1) ? argv[1] : "golden/local_opt_hardalm_cases.txt";
  std::ifstream f(path);
  if (!f) { std::printf("CANNOT OPEN %s\n", path); return 2; }

  string line;
  std::map<string, vector<double>> num;     // numeric keys
  std::map<string, string> str;             // string keys (NAME / SAFETY_MODE / ...)
  // per-case obstacle / class metadata captured during parse
  vector<std::map<string, string>> obs_str;
  vector<std::map<string, vector<double>>> obs_num;
  vector<std::pair<string, double>> classes;  // (class_name, d_safe)

  int ncase = 0, nfail = 0;
  double global_max_err = 0.0;
  // absolute tol for direct quantities; relative tol for gradients that pass through
  // the adjoint LU solve inside grad_from_dcost_dc — those are tested in test_minjerk;
  // here dcost_dc / dT_exp are the PRE-adjoint accumulations (no LU), so absolute works
  // and we keep one relative band only as a safety net where magnitudes are large.
  const double TOL = 1e-7;

  auto process = [&]() {
    int M = (int)num["M"][0];
    Eigen::VectorXd T = Eigen::Map<Eigen::VectorXd>(num["T"].data(), M);
    Eigen::Vector3d start(num["START"][0], num["START"][1], num["START"][2]);
    Eigen::Vector3d goal(num["GOAL"][0], num["GOAL"][1], num["GOAL"][2]);
    Eigen::MatrixXd q(std::max(M - 1, 0), 3);
    if (M > 1)
      for (int i = 0; i < M - 1; ++i)
        for (int j = 0; j < 3; ++j) q(i, j) = num["Q"][i * 3 + j];
    auto vec3 = [&](const char* k) {
      return Eigen::Vector3d(num[k][0], num[k][1], num[k][2]);
    };
    // build trajectory from endpoints == MinjerkTraj.from_endpoints
    Eigen::MatrixXd wp(M + 1, 3);
    wp.row(0) = start.transpose();
    for (int i = 0; i < M - 1; ++i) wp.row(i + 1) = q.row(i);
    wp.row(M) = goal.transpose();
    sando::MinjerkTraj tr(wp, T, vec3("V0"), vec3("A0"), vec3("VF"), vec3("AF"));

    // OptParams ALM fields
    sando::HardAlmOptParams opt;
    {
      string ov = str.count("AVOID_OVERRIDE") ? str["AVOID_OVERRIDE"] : "_";
      opt.avoid_override = (ov == "_") ? "" : ov;
    }
    opt.tau_trust = num["TAU_TRUST"][0];
    opt.spacetime_hard = (num["SPACETIME_HARD"][0] != 0.0);
    opt.q_conformal = num["Q_CONFORMAL"][0];
    opt.epsilon_track = num["EPSILON_TRACK"][0];
    opt.safety_mode = str["SAFETY_MODE"];
    opt.reach_model = str["REACH_MODEL"];
    opt.a_max_human = num["A_MAX_HUMAN"][0];
    opt.v_max_human = num["V_MAX_HUMAN"][0];
    opt.est_pos_err = num["EST_POS_ERR"][0];
    opt.est_vel_err = num["EST_VEL_ERR"][0];

    // avoid config (use the dumped per-class d_safe so the lookup matches Python)
    std::map<string, sando::AvoidParams> cfg = sando::default_config();
    for (auto& cd : classes)
      cfg[cd.first] = sando::AvoidParams{cd.first, "hard", cd.second, 1.0e4};

    // obstacles
    int H = (int)num["H"][0];
    vector<std::shared_ptr<sando::Obstacle>> owned;
    vector<const sando::Obstacle*> hard_obs;
    for (int h = 0; h < H; ++h) {
      auto& sm = obs_str[h];
      auto& nm = obs_num[h];
      if (sm["KIND"] == "sphere") {
        Eigen::Vector3d c0(nm["C0"][0], nm["C0"][1], nm["C0"][2]);
        double r = nm["R"][0];
        Eigen::Vector3d vel(nm["VEL"][0], nm["VEL"][1], nm["VEL"][2]);
        Eigen::Vector3d acc(nm["ACC"][0], nm["ACC"][1], nm["ACC"][2]);
        auto s = std::make_shared<sando::SphereObstacle>(c0, r, vel, sm["CLASS"], acc);
        owned.push_back(s);
        hard_obs.push_back(s.get());
      } else {
        Eigen::Vector3d lo(nm["LO"][0], nm["LO"][1], nm["LO"][2]);
        Eigen::Vector3d hi(nm["HI"][0], nm["HI"][1], nm["HI"][2]);
        auto a = std::make_shared<sando::AABBObstacle>(lo, hi, sm["CLASS"]);
        owned.push_back(a);
        hard_obs.push_back(a.get());
      }
    }

    int Nc = (int)num["NC"][0];
    Eigen::VectorXd lam = Eigen::Map<Eigen::VectorXd>(num["LAM"].data(), Nc);
    double rho = num["RHO"][0];
    bool freeze = (num["FREEZE"][0] != 0.0);

    // frozen normals + trust mask (or recompute)
    vector<double> normals;
    const vector<double>* normals_ptr = nullptr;
    Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> tmask;
    const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>* tmask_ptr = nullptr;
    if (freeze) {
      normals = num["NORMALS"];   // (M,6,H,3) flat row-major, matches our layout
      normals_ptr = &normals;
      tmask.resize(M, 6);
      for (int i = 0; i < M; ++i)
        for (int k = 0; k < 6; ++k)
          tmask(i, k) = (num["TMASK"][i * 6 + k] != 0.0);
      tmask_ptr = &tmask;
    }

    auto res = sando::alm_term(tr, hard_obs, cfg, lam, rho, normals_ptr, opt, tmask_ptr);

    double e = 0.0;
    auto absd = [&](double a, double b) { e = std::max(e, std::fabs(a - b)); };
    auto reld = [&](double a, double b) {
      e = std::max(e, std::fabs(a - b) / (std::fabs(b) + 1.0));
    };

    // cost
    reld(res.cost, num["COST"][0]);
    // dcost_dc (6M,3)
    for (int i = 0; i < 6 * M; ++i)
      for (int j = 0; j < 3; ++j) reld(res.dcost_dc(i, j), num["DCC"][i * 3 + j]);
    // dT_exp (M)
    for (int k = 0; k < M; ++k) reld(res.dT_exp(k), num["DTEXP"][k]);
    // per-point constraint records
    const auto& cons = res.cons;
    for (int m = 0; m < Nc; ++m) {
      absd(cons.g(m), num["G"][m]);
      for (int j = 0; j < 3; ++j) absd(cons.n(m, j), num["NRM"][m * 3 + j]);
      absd(cons.dist(m), num["DIST"][m]);
      absd(cons.R(m), num["RARR"][m]);
      absd(cons.dgdt(m), num["DGDT"][m]);
      absd(double(cons.trust[m]), num["TRUST"][m]);
    }

    global_max_err = std::max(global_max_err, e);
    bool ok = e < TOL;
    std::printf("  case %-22s M=%d H=%d Nc=%d freeze=%d : max_err=%.3e  %s\n",
                str.count("NAME") ? str["NAME"].c_str() : "?", M, H, Nc,
                freeze ? 1 : 0, e, ok ? "PASS" : "FAIL");
    if (!ok) nfail++;
    ncase++;
  };

  // parse: lines are "KEY rest". Some keys are strings, most numeric. Obstacle/class
  // keys are bucketed per-case.
  while (std::getline(f, line)) {
    if (line == "CASE") {
      num.clear(); str.clear(); obs_str.clear(); obs_num.clear(); classes.clear();
      continue;
    }
    if (line == "END") { process(); continue; }
    std::istringstream is(line);
    string key; is >> key;
    string rest; std::getline(is, rest);
    // trim leading space
    if (!rest.empty() && rest[0] == ' ') rest.erase(0, 1);

    // string-valued keys
    if (key == "NAME" || key == "SAFETY_MODE" || key == "REACH_MODEL" ||
        key == "AVOID_OVERRIDE") {
      str[key] = rest;
      continue;
    }
    // class metadata: "CLASS<i> name d_safe"
    if (key.rfind("CLASS", 0) == 0 && key != "CLASS") {
      std::istringstream cs(rest);
      string cname; double ds;
      cs >> cname >> ds;
      classes.push_back({cname, ds});
      continue;
    }
    // obstacle keys: "OBS<h>_FIELD ..."
    if (key.rfind("OBS", 0) == 0) {
      // parse h and field
      size_t us = key.find('_');
      int h = std::stoi(key.substr(3, us - 3));
      string field = key.substr(us + 1);
      if ((int)obs_str.size() <= h) { obs_str.resize(h + 1); obs_num.resize(h + 1); }
      if (field == "KIND" || field == "CLASS") {
        obs_str[h][field] = rest;
      } else {
        obs_num[h][field] = nums(rest);
      }
      continue;
    }
    // everything else numeric (may be empty, e.g. Q for M=1)
    num[key] = nums(rest);
  }

  std::printf("\nHARD-ALM golden: %d cases, %d fail, global_max_err=%.3e (tol=%.0e)\n",
              ncase, nfail, global_max_err, TOL);
  std::printf("%s\n", nfail == 0 ? "ALL PASS (C++ reproduces Python)" : "FAILED");
  return nfail == 0 ? 0 : 1;
}
