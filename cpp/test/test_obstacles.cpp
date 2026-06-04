// Golden verification for obstacles.hpp vs obstacles.py.
#include "sando_cpp/obstacles.hpp"
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
  const char* path = (argc > 1) ? argv[1] : "golden/obstacles_cases.txt";
  std::ifstream f(path);
  if (!f) { std::printf("CANNOT OPEN %s\n", path); return 2; }

  string line, kind;
  std::map<string, vector<double>> cur;
  int ncase = 0, nfail = 0; double gmax = 0.0; const double TOL = 1e-10;
  auto v3 = [&](const char* k, int off = 0) {
    return Eigen::Vector3d(cur[k][off], cur[k][off + 1], cur[k][off + 2]);
  };

  auto process = [&]() {
    double e = 0.0;
    int N = (int)cur["N"][0];
    if (kind == "SPHERE") {
      sando::SphereObstacle s(v3("C0"), cur["R"][0], v3("V"), "human", v3("A"));
      for (int i = 0; i < N; ++i) {
        double t = cur["TS"][i];
        Eigen::Vector3d pr = s.predict(t), va = s.velocity_at(t);
        for (int j = 0; j < 3; ++j) {
          e = std::max(e, std::fabs(pr(j) - cur["PRED"][i * 3 + j]));
          e = std::max(e, std::fabs(va(j) - cur["VELAT"][i * 3 + j]));
        }
        Eigen::Vector3d p(cur["PTS"][i * 3], cur["PTS"][i * 3 + 1], cur["PTS"][i * 3 + 2]);
        e = std::max(e, std::fabs(s.signed_dist(p, t) - cur["SD"][i]));
      }
    } else {  // AABB
      sando::AABBObstacle b(v3("LO"), v3("HI"), "wall");
      for (int i = 0; i < N; ++i) {
        Eigen::Vector3d p(cur["PTS"][i * 3], cur["PTS"][i * 3 + 1], cur["PTS"][i * 3 + 2]);
        e = std::max(e, std::fabs(b.signed_dist(p, 0.0) - cur["SD"][i]));
      }
    }
    gmax = std::max(gmax, e);
    bool ok = e < TOL;
    std::printf("  %-6s : max_err=%.3e  %s\n", kind.c_str(), e, ok ? "PASS" : "FAIL");
    if (!ok) nfail++;
    ncase++;
  };

  while (std::getline(f, line)) {
    if (line == "SPHERE" || line == "AABB") { cur.clear(); kind = line; continue; }
    if (line == "END") { process(); continue; }
    std::istringstream is(line); string key; is >> key; string rest; std::getline(is, rest);
    cur[key] = nums(rest);
  }
  std::printf("\nObstacles golden: %d cases, %d fail, global_max_err=%.3e (tol=%.0e)\n",
              ncase, nfail, gmax, TOL);
  std::printf("%s\n", nfail == 0 ? "ALL PASS (C++ reproduces Python)" : "FAILED");
  return nfail == 0 ? 0 : 1;
}
