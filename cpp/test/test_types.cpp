// Golden verification: C++ types.hpp must reproduce Python types.py outputs.
// Reads cpp/golden/types_cases.txt (produced by gen_types_golden.py), rebuilds each
// PieceWisePol / _AnalyticExpr / DynTraj in C++, and asserts eval/velocity/accel match
// within tolerance. 不许欺骗、必须还原: 对不上就 FAIL.
//
// Standalone build (no cmake):
//   g++ -std=c++17 -O2 -I .../cpp/third_party/eigen -I .../cpp/include \
//       test_types.cpp -o /tmp/test_types && /tmp/test_types .../golden/types_cases.txt
#include "sando_cpp/types.hpp"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
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

// Parse the full line after the first token into a raw string (keeps spaces; may be empty).
// We store these in a separate map because expression sources contain spaces.

int main(int argc, char** argv) {
  const char* path = (argc > 1) ? argv[1] : "golden/types_cases.txt";
  std::ifstream f(path);
  if (!f) { std::printf("CANNOT OPEN %s\n", path); return 2; }

  string line;
  std::map<string, vector<double>> num;   // numeric keys
  std::map<string, string> str;           // raw-string keys
  int ncase = 0, nfail = 0;
  double global_max_err = 0.0;
  const double TOL = 1e-7;

  auto build_pwp = [&](int nseg) {
    sando::PieceWisePol p;
    for (double v : num["TIMES"]) p.times.push_back(v);
    auto load = [&](const char* key, vector<Eigen::Vector4d>& dst) {
      const vector<double>& c = num[key];
      for (int i = 0; i < nseg; ++i)
        dst.push_back(Eigen::Vector4d(c[i * 4 + 0], c[i * 4 + 1], c[i * 4 + 2], c[i * 4 + 3]));
    };
    load("CX", p.coeff_x);
    load("CY", p.coeff_y);
    load("CZ", p.coeff_z);
    return p;
  };

  auto process = [&]() {
    const string kind = str.count("KIND") ? str["KIND"] : "";
    double e = 0.0;
    int M_id = ncase;

    if (kind == "PWP") {
      int nseg = (int)num["NSEG"][0];
      sando::PieceWisePol p = build_pwp(nseg);
      e = std::max(e, std::fabs(p.get_end_time() - num["GET_END_TIME"][0]));
      e = std::max(e, std::fabs(p.get_duration() - num["GET_DURATION"][0]));
      int nt = (int)num["NT"][0];
      const vector<double>& ts = num["TS"];
      for (int k = 0; k < nt; ++k) {
        Eigen::Vector3d v0 = p.eval(ts[k]);
        Eigen::Vector3d v1 = p.velocity(ts[k]);
        Eigen::Vector3d v2 = p.acceleration(ts[k]);
        for (int j = 0; j < 3; ++j) {
          e = std::max(e, std::fabs(v0(j) - num["E0"][k * 3 + j]));
          e = std::max(e, std::fabs(v1(j) - num["E1"][k * 3 + j]));
          e = std::max(e, std::fabs(v2(j) - num["E2"][k * 3 + j]));
        }
      }
    } else if (kind == "EXPR") {
      string src = str.count("SRC") ? str["SRC"] : "";
      sando::AnalyticExpr ex(src);
      int nt = (int)num["NT"][0];
      const vector<double>& ts = num["TS"];
      for (int k = 0; k < nt; ++k) {
        double v = ex(ts[k]);
        double ref = num["VALS"][k];
        // compare with relative tolerance: large-magnitude values (e.g. 2**9.87, exp)
        // pass through pow/exp -> use |a-b|/(|b|+1) like the minco gradient block.
        e = std::max(e, std::fabs(v - ref) / (std::fabs(ref) + 1.0));
      }
    } else if (kind == "DYNA") {
      sando::DynTraj d;
      d.mode = "Analytic";
      d.traj_x = str.count("SX") ? str["SX"] : "";
      d.traj_y = str.count("SY") ? str["SY"] : "";
      d.traj_z = str.count("SZ") ? str["SZ"] : "";
      d.traj_vx = str.count("SVX") ? str["SVX"] : "";
      d.traj_vy = str.count("SVY") ? str["SVY"] : "";
      d.traj_vz = str.count("SVZ") ? str["SVZ"] : "";
      d.compile_analytic();
      int nt = (int)num["NT"][0];
      const vector<double>& ts = num["TS"];
      for (int k = 0; k < nt; ++k) {
        Eigen::Vector3d ev = d.eval(ts[k]);
        Eigen::Vector3d vv = d.velocity(ts[k]);
        Eigen::Vector3d av = d.accel(ts[k]);
        for (int j = 0; j < 3; ++j) {
          auto rel = [&](double a, double b) {
            e = std::max(e, std::fabs(a - b) / (std::fabs(b) + 1.0));
          };
          rel(ev(j), num["EV"][k * 3 + j]);
          rel(vv(j), num["VV"][k * 3 + j]);
          rel(av(j), num["AV"][k * 3 + j]);
        }
      }
    } else if (kind == "DYNP") {
      int nseg = (int)num["NSEG"][0];
      sando::PieceWisePol p = build_pwp(nseg);
      sando::DynTraj d;
      d.set_piecewise(p);
      int nt = (int)num["NT"][0];
      const vector<double>& ts = num["TS"];
      for (int k = 0; k < nt; ++k) {
        Eigen::Vector3d ev = d.eval(ts[k]);
        Eigen::Vector3d vv = d.velocity(ts[k]);
        Eigen::Vector3d av = d.accel(ts[k]);
        for (int j = 0; j < 3; ++j) {
          e = std::max(e, std::fabs(ev(j) - num["EV"][k * 3 + j]));
          e = std::max(e, std::fabs(vv(j) - num["VV"][k * 3 + j]));
          e = std::max(e, std::fabs(av(j) - num["AV"][k * 3 + j]));
        }
      }
    } else {
      return;  // unknown kind, skip
    }

    global_max_err = std::max(global_max_err, e);
    bool ok = e < TOL;
    std::printf("  case #%d kind=%-4s : max_err=%.3e  %s\n", M_id, kind.c_str(), e,
                ok ? "PASS" : "FAIL");
    if (!ok) nfail++;
    ncase++;
  };

  while (std::getline(f, line)) {
    // strip trailing CR (Windows golden files)
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line == "CASE") { num.clear(); str.clear(); continue; }
    if (line == "END") { process(); continue; }
    std::istringstream is(line);
    string key; is >> key;
    string rest;
    std::getline(is, rest);
    if (!rest.empty() && rest[0] == ' ') rest.erase(0, 1);  // drop single leading space
    // Keys that hold raw expression / metadata strings:
    if (key == "KIND" || key == "SRC" || key == "SX" || key == "SY" || key == "SZ" ||
        key == "SVX" || key == "SVY" || key == "SVZ") {
      str[key] = rest;
    } else {
      num[key] = nums(rest);
    }
  }

  std::printf("\ntypes golden: %d cases, %d fail, global_max_err=%.3e (tol=%.0e)\n",
              ncase, nfail, global_max_err, TOL);
  std::printf("%s\n", nfail == 0 ? "PASS — ALL (C++ reproduces Python)" : "FAILED");
  return nfail == 0 ? 0 : 1;
}
