// Golden verification for avoid_config.hpp + cost.hpp vs Python.
#include "sando_cpp/minjerk_traj.hpp"
#include "sando_cpp/obstacles.hpp"
#include "sando_cpp/avoid_config.hpp"
#include "sando_cpp/cost.hpp"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <cmath>

using std::string;
using std::vector;

static vector<double> nums(const string& rest) {
  vector<double> v; std::istringstream is(rest); double x;
  while (is >> x) v.push_back(x);
  return v;
}

int main(int argc, char** argv) {
  const char* path = (argc > 1) ? argv[1] : "golden/cost_cases.txt";
  std::ifstream f(path);
  if (!f) { std::printf("CANNOT OPEN %s\n", path); return 2; }

  std::map<string, vector<double>> cur;
  string penalty_line, resolve_line, line;
  while (std::getline(f, line)) {
    if (line == "CASE" || line == "END") continue;
    std::istringstream is(line); string key; is >> key; string rest; std::getline(is, rest);
    if (key == "PENALTY") { penalty_line = rest; continue; }
    if (key == "RESOLVE") { resolve_line = rest; continue; }
    cur[key] = nums(rest);
  }

  int nfail = 0; double gmax = 0.0;

  // --- obstacle_cost ---
  int M = (int)cur["M"][0];
  Eigen::VectorXd T = Eigen::Map<Eigen::VectorXd>(cur["T"].data(), M);
  Eigen::MatrixXd wp(M + 1, 3);
  for (int i = 0; i < M + 1; ++i) for (int j = 0; j < 3; ++j) wp(i, j) = cur["WP"][i * 3 + j];
  auto v3 = [&](const char* k) { return Eigen::Vector3d(cur[k][0], cur[k][1], cur[k][2]); };
  sando::MinjerkTraj mj(wp, T, v3("V0"), v3("A0"), v3("VF"), v3("AF"));

  auto& S = cur["SPH"];  // c0(3) r v(3) a(3)
  sando::SphereObstacle sph(Eigen::Vector3d(S[0], S[1], S[2]), S[3],
                            Eigen::Vector3d(S[4], S[5], S[6]), "human",
                            Eigen::Vector3d(S[7], S[8], S[9]));
  auto& B = cur["AABB"];  // lo(3) hi(3)
  sando::AABBObstacle box(Eigen::Vector3d(B[0], B[1], B[2]), Eigen::Vector3d(B[3], B[4], B[5]), "wall");
  vector<const sando::Obstacle*> obs = {&sph, &box};
  int K = (int)cur["K"][0];
  double cost = sando::obstacle_cost(mj, obs, sando::default_config(), K);
  double cost_gold = cur["COST"][0];
  double rel = std::fabs(cost - cost_gold) / (std::fabs(cost_gold) + 1.0);
  gmax = std::max(gmax, rel);
  bool ok = rel < 1e-6;
  std::printf("  obstacle_cost: cpp=%.8g py=%.8g rel_err=%.3e  %s\n", cost, cost_gold, rel, ok ? "PASS" : "FAIL");
  if (!ok) nfail++;

  // --- penalty ---
  {
    std::istringstream is(penalty_line); double d, ds, ex; int bad = 0; double me = 0;
    while (is >> d >> ds >> ex) { me = std::max(me, std::fabs(sando::penalty(d, ds) - ex)); }
    bool pok = me < 1e-12;
    std::printf("  penalty: max_err=%.3e  %s\n", me, pok ? "PASS" : "FAIL");
    if (!pok) nfail++;
  }

  // --- resolve_mode ---
  {
    auto cfg = sando::default_config();
    std::istringstream is(resolve_line); string tok; int bad = 0, n = 0;
    while (is >> tok) {
      // cls|ov|mode
      size_t p1 = tok.find('|'), p2 = tok.rfind('|');
      string cls = tok.substr(0, p1);
      string ov = tok.substr(p1 + 1, p2 - p1 - 1);
      string mode = tok.substr(p2 + 1);
      string got = sando::resolve_mode(cls, cfg, ov == "NONE" ? "" : ov);
      if (got != mode) { bad++; std::printf("    resolve mismatch %s|%s : got %s want %s\n", cls.c_str(), ov.c_str(), got.c_str(), mode.c_str()); }
      n++;
    }
    bool rok = bad == 0;
    std::printf("  resolve_mode: %d/%d ok  %s\n", n - bad, n, rok ? "PASS" : "FAIL");
    if (!rok) nfail++;
  }

  std::printf("\navoid_config+cost golden: %d fail\n", nfail);
  std::printf("%s\n", nfail == 0 ? "ALL PASS (C++ reproduces Python)" : "FAILED");
  return nfail == 0 ? 0 : 1;
}
