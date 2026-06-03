// Golden verification: C++ minco_obstacle_term / minco_dynamic_term must reproduce
// the Python local_opt.py terms. Reads cpp/golden/local_opt_terms_cases.txt
// (produced by gen_local_opt_terms_golden.py). 不许欺骗、必须还原:对不上就 FAIL.
#include "sando_cpp/local_opt_terms.hpp"
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
  const char* path = (argc > 1) ? argv[1] : "golden/local_opt_terms_cases.txt";
  std::ifstream f(path);
  if (!f) { std::printf("CANNOT OPEN %s\n", path); return 2; }

  string line;
  // raw text lines of the current case (we re-parse obstacle lines positionally)
  std::map<string, vector<double>> cur;     // numeric keys
  std::map<string, string> curs;            // string keys (KIND, NAME)
  vector<string> obs_lines;                 // OBS0.. raw text

  int ncase = 0, nfail = 0;
  double global_max_err = 0.0;
  // forward (cost) is absolute; gradients pass through a heavier chain -> relative.
  const double TOL_ABS = 1e-7;
  const double TOL_REL = 1e-7;

  // owns obstacles created per case
  vector<std::unique_ptr<sando::Obstacle>> owned;

  auto vec3 = [&](const char* k) {
    return Eigen::Vector3d(cur[k][0], cur[k][1], cur[k][2]);
  };

  auto build_traj = [&](int M) {
    Eigen::VectorXd T = Eigen::Map<Eigen::VectorXd>(cur["T"].data(), M);
    Eigen::MatrixXd wp(M + 1, 3);
    for (int i = 0; i < M + 1; ++i)
      for (int j = 0; j < 3; ++j) wp(i, j) = cur["WP"][i * 3 + j];
    return sando::MinjerkTraj(wp, T, vec3("V0"), vec3("A0"), vec3("VF"), vec3("AF"));
  };

  auto process = [&]() {
    const string kind = curs["KIND"];
    const string name = curs["NAME"];
    int M = (int)cur["M"][0];
    int kappa = (int)cur["KAPPA"][0];
    sando::TermOptParams opt;
    opt.kappa = kappa;

    sando::MinjerkTraj tr = build_traj(M);
    double e = 0.0;
    auto rel = [&](double a, double b) {
      e = std::max(e, std::fabs(a - b) / (std::fabs(b) + 1.0));
    };
    auto absd = [&](double a, double b) { e = std::max(e, std::fabs(a - b)); };

    sando::TermResult res;
    if (kind == "OBS") {
      owned.clear();
      vector<const sando::Obstacle*> obstacles;
      sando::AvoidCfg cfg;
      for (const string& ol : obs_lines) {
        std::istringstream is(ol);
        string tag, type;
        is >> tag >> type;  // OBSk  TYPE
        std::unique_ptr<sando::Obstacle> obs;
        string cls;
        double dsafe = 0, w = 0;
        if (type == "SPHERE") {
          double c0[3], radius, vel[3], acc[3];
          for (int j = 0; j < 3; ++j) is >> c0[j];
          is >> radius;
          for (int j = 0; j < 3; ++j) is >> vel[j];
          for (int j = 0; j < 3; ++j) is >> acc[j];
          string kw; is >> kw >> cls;       // CLASS <name>
          string kd; is >> kd >> dsafe;     // DSAFE <d>
          string kw2; is >> kw2 >> w;       // W <w>
          obs = std::make_unique<sando::SphereObstacle>(
              Eigen::Vector3d(c0[0], c0[1], c0[2]), radius,
              Eigen::Vector3d(vel[0], vel[1], vel[2]), cls,
              Eigen::Vector3d(acc[0], acc[1], acc[2]));
        } else {  // AABB
          double lo[3], hi[3];
          for (int j = 0; j < 3; ++j) is >> lo[j];
          for (int j = 0; j < 3; ++j) is >> hi[j];
          string kw; is >> kw >> cls;
          string kd; is >> kd >> dsafe;
          string kw2; is >> kw2 >> w;
          obs = std::make_unique<sando::AABBObstacle>(
              Eigen::Vector3d(lo[0], lo[1], lo[2]),
              Eigen::Vector3d(hi[0], hi[1], hi[2]), cls);
        }
        cfg[cls] = sando::AvoidParams{cls, "soft", dsafe, w};  // canonical 4-field struct
        obstacles.push_back(obs.get());
        owned.push_back(std::move(obs));
      }
      res = sando::minco_obstacle_term(tr, obstacles, cfg, opt);
    } else {  // DYN
      int order = (int)cur["ORDER"][0];
      double limit = cur["LIMIT"][0];
      double weight = cur["WEIGHT"][0];
      res = sando::minco_dynamic_term(tr, order, limit, weight, opt);
    }

    // cost: absolute
    absd(res.cost, cur["COST"][0]);
    // dcost_dc (6M,3): relative (passes through quadrature + cubic/φ' chain)
    for (int i = 0; i < 6 * M; ++i)
      for (int j = 0; j < 3; ++j)
        rel(res.dcost_dc(i, j), cur["DCC"][i * 3 + j]);
    // dcost_dT_explicit (M,): relative
    for (int k = 0; k < M; ++k)
      rel(res.dcost_dT_explicit(k), cur["DT"][k]);

    global_max_err = std::max(global_max_err, e);
    bool ok = e < (TOL_ABS > TOL_REL ? TOL_ABS : TOL_REL);
    std::printf("  case %-16s (%s M=%d kap=%d): max_err=%.3e  %s\n",
                name.c_str(), kind.c_str(), M, kappa, e, ok ? "PASS" : "FAIL");
    if (!ok) nfail++;
    ncase++;
  };

  while (std::getline(f, line)) {
    if (line == "CASE") { cur.clear(); curs.clear(); obs_lines.clear(); continue; }
    if (line == "END") { process(); continue; }
    std::istringstream is(line);
    string key; is >> key;
    string rest; std::getline(is, rest);
    if (key == "KIND" || key == "NAME") {
      // strip a single leading space
      if (!rest.empty() && rest[0] == ' ') rest.erase(0, 1);
      curs[key] = rest;
    } else if (key.rfind("OBS", 0) == 0 && key != "OBS") {
      // an obstacle data line "OBSk ..." (NOBS uses key "NOBS", excluded)
      obs_lines.push_back(line);
    } else {
      cur[key] = nums(rest);
    }
  }

  std::printf("\nlocal_opt_terms golden: %d cases, %d fail, global_max_err=%.3e (tol=%.0e)\n",
              ncase, nfail, global_max_err, TOL_ABS);
  std::printf("%s\n", nfail == 0 ? "ALL PASS (C++ reproduces Python)" : "FAILED");
  return nfail == 0 ? 0 : 1;
}
