// Golden verification: C++ MinjerkTraj must reproduce Python minco.py outputs.
// Reads cpp/golden/minco_cases.txt (produced by gen_minco_golden.py), rebuilds each
// trajectory in C++, and asserts coefficients / derivatives / control points match
// within tolerance. 不许欺骗、必须还原:对不上就 FAIL。
#include "sando_cpp/minjerk_traj.hpp"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <tuple>
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
  const char* path = (argc > 1) ? argv[1] : "golden/minco_cases.txt";
  std::ifstream f(path);
  if (!f) { std::printf("CANNOT OPEN %s\n", path); return 2; }

  string line;
  std::map<string, vector<double>> cur;
  int ncase = 0, nfail = 0;
  double global_max_err = 0.0;
  const double TOL = 1e-7;

  auto process = [&]() {
    int M = (int)cur["M"][0];
    Eigen::VectorXd T = Eigen::Map<Eigen::VectorXd>(cur["T"].data(), M);
    Eigen::MatrixXd wp(M + 1, 3);
    for (int i = 0; i < M + 1; ++i)
      for (int j = 0; j < 3; ++j) wp(i, j) = cur["WP"][i * 3 + j];
    auto vec3 = [&](const char* k) {
      return Eigen::Vector3d(cur[k][0], cur[k][1], cur[k][2]);
    };
    sando::MinjerkTraj mj(wp, T, vec3("V0"), vec3("A0"), vec3("VF"), vec3("AF"));

    double e = 0.0;
    // t_end
    e = std::max(e, std::fabs(mj.t_end - cur["TEND"][0]));
    // coefficients c (6M,3)
    for (int i = 0; i < 6 * M; ++i)
      for (int j = 0; j < 3; ++j)
        e = std::max(e, std::fabs(mj.c(i, j) - cur["C"][i * 3 + j]));
    // derivatives at sampled times
    int K = (int)cur["K"][0];
    const char* ekeys[4] = {"E0", "E1", "E2", "E3"};
    for (int o = 0; o < 4; ++o)
      for (int k = 0; k < K; ++k) {
        Eigen::Vector3d v = mj.eval_deriv(cur["TS"][k], o);
        for (int j = 0; j < 3; ++j)
          e = std::max(e, std::fabs(v(j) - cur[ekeys[o]][k * 3 + j]));
      }
    // control points (M,6,3)
    auto cp = mj.control_points();
    for (int i = 0; i < M; ++i)
      for (int kk = 0; kk < 6; ++kk)
        for (int j = 0; j < 3; ++j)
          e = std::max(e, std::fabs(cp[i](kk, j) - cur["CP"][(i * 6 + kk) * 3 + j]));
    // control point times (M,6)
    Eigen::MatrixXd cpt = mj.control_point_times();
    for (int i = 0; i < M; ++i)
      for (int kk = 0; kk < 6; ++kk)
        e = std::max(e, std::fabs(cpt(i, kk) - cur["CPT"][i * 6 + kk]));

    // ---- M1 analytic gradient (energy_grad / grad_from_dcost_dc / dT-explicit) ----
    // gradient quantities go through a second (adjoint) LU solve + T^5 terms -> larger
    // magnitude, so compare by RELATIVE error |a-b|/(|b|+1) (forward stays absolute above).
    {
      auto rel = [&](double a, double b) { e = std::max(e, std::fabs(a - b) / (std::fabs(b) + 1.0)); };
      auto eg = mj.energy_grad();
      rel(std::get<0>(eg), cur["ENERGY"][0]);
      const Eigen::MatrixXd& gq = std::get<1>(eg);
      const Eigen::VectorXd& gT = std::get<2>(eg);
      for (int i = 0; i < M - 1; ++i)
        for (int j = 0; j < 3; ++j) rel(gq(i, j), cur["EGQ"][i * 3 + j]);
      for (int k = 0; k < M; ++k) rel(gT(k), cur["EGT"][k]);
      Eigen::MatrixXd dcc(6 * M, 3);
      for (int i = 0; i < 6 * M; ++i)
        for (int j = 0; j < 3; ++j) dcc(i, j) = cur["DCC"][i * 3 + j];
      Eigen::VectorXd dct = Eigen::Map<Eigen::VectorXd>(cur["DCT"].data(), M);
      auto g2 = mj.grad_from_dcost_dc(dcc, &dct);
      for (int i = 0; i < M - 1; ++i)
        for (int j = 0; j < 3; ++j) rel(g2.first(i, j), cur["GQ2"][i * 3 + j]);
      for (int k = 0; k < M; ++k) rel(g2.second(k), cur["GT2"][k]);
      auto cpdt = mj.control_points_dT_explicit();
      for (int i = 0; i < M; ++i)
        for (int kk = 0; kk < 6; ++kk)
          for (int j = 0; j < 3; ++j) rel(cpdt[i](kk, j), cur["CPDT"][(i * 6 + kk) * 3 + j]);
    }

    global_max_err = std::max(global_max_err, e);
    bool ok = e < TOL;
    std::printf("  case M=%d : max_err=%.3e  %s\n", M, e, ok ? "PASS" : "FAIL");
    if (!ok) nfail++;
    ncase++;
  };

  while (std::getline(f, line)) {
    if (line == "CASE") { cur.clear(); continue; }
    if (line == "END") { process(); continue; }
    std::istringstream is(line);
    string key; is >> key;
    string rest; std::getline(is, rest);
    cur[key] = nums(rest);
  }

  std::printf("\nMinjerkTraj golden: %d cases, %d fail, global_max_err=%.3e (tol=%.0e)\n",
              ncase, nfail, global_max_err, TOL);
  std::printf("%s\n", nfail == 0 ? "ALL PASS (C++ reproduces Python)" : "FAILED");
  return nfail == 0 ? 0 : 1;
}
