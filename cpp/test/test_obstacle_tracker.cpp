// Golden verification for obstacle_tracker_kernels.hpp vs Python obstacle_tracker.py.
#include "sando_cpp/obstacle_tracker_kernels.hpp"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>

using std::string;
using std::vector;
using namespace sando::obstacle_tracker;

static vector<double> nums(const string& rest) {
  vector<double> v; std::istringstream is(rest); double x;
  while (is >> x) v.push_back(x);
  return v;
}
static vector<Eigen::Vector3d> pts_of(const vector<double>& a) {
  vector<Eigen::Vector3d> p;
  for (size_t i = 0; i + 3 <= a.size(); i += 3) p.emplace_back(a[i], a[i + 1], a[i + 2]);
  return p;
}

int main(int argc, char** argv) {
  const char* path = (argc > 1) ? argv[1] : "golden/obstacle_tracker_cases.txt";
  std::ifstream f(path);
  if (!f) { std::printf("CANNOT OPEN %s\n", path); return 2; }
  std::map<string, vector<double>> cur;
  string line;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    std::istringstream is(line); string key; is >> key; string rest; std::getline(is, rest);
    cur[key] = nums(rest);
  }
  int nfail = 0;
  auto report = [&](const char* name, double err, double tol) {
    bool ok = err < tol; if (!ok) nfail++;
    std::printf("  %-18s max_err=%.3e  %s\n", name, err, ok ? "PASS" : "FAIL");
  };

  // --- voxel downsample ---
  {
    auto in = pts_of(cur["VOXEL_IN"]);
    double leaf = cur["VOXEL_LEAF"][0];
    auto out = voxel_downsample(in, leaf);
    auto gold = pts_of(cur["VOXEL_OUT"]);
    double err = 0;
    if (out.size() != gold.size()) { err = 1e9; }
    else for (size_t i = 0; i < out.size(); ++i) err = std::max(err, (out[i] - gold[i]).cwiseAbs().maxCoeff());
    std::printf("  voxel: cpp_n=%zu py_n=%zu\n", out.size(), gold.size());
    report("voxel_downsample", err, 1e-12);
  }

  // --- clustering ---
  {
    auto in = pts_of(cur["CLUST_IN"]);
    double eps = cur["CLUST_PARAMS"][0];
    int mn = (int)cur["CLUST_PARAMS"][1], mx = (int)cur["CLUST_PARAMS"][2];
    auto clusters = connected_components(in, eps, mn, mx);
    // build (centroid,bbox,size) and sort by centroid like the generator
    struct D { Eigen::Vector3d cen, box; int sz; };
    vector<D> desc;
    for (auto& c : clusters) {
      Eigen::Vector3d pmin = in[c[0]], pmax = in[c[0]];
      for (int idx : c) { pmin = pmin.cwiseMin(in[idx]); pmax = pmax.cwiseMax(in[idx]); }
      desc.push_back({(pmin + pmax) * 0.5, pmax - pmin, (int)c.size()});
    }
    std::sort(desc.begin(), desc.end(), [](const D& a, const D& b) {
      if (std::abs(a.cen[0] - b.cen[0]) > 1e-6) return a.cen[0] < b.cen[0];
      if (std::abs(a.cen[1] - b.cen[1]) > 1e-6) return a.cen[1] < b.cen[1];
      return a.cen[2] < b.cen[2];
    });
    int ncg = (int)cur["CLUST_NCLUST"][0];
    double err = 0;
    if ((int)desc.size() != ncg) err = 1e9;
    else for (int i = 0; i < ncg; ++i) {
      auto& g = cur["CLUST_" + std::to_string(i)];  // cen(3) box(3) size
      err = std::max(err, std::abs(desc[i].cen[0] - g[0]));
      err = std::max(err, std::abs(desc[i].cen[1] - g[1]));
      err = std::max(err, std::abs(desc[i].cen[2] - g[2]));
      err = std::max(err, std::abs(desc[i].box[0] - g[3]));
      err = std::max(err, std::abs(desc[i].box[1] - g[4]));
      err = std::max(err, std::abs(desc[i].box[2] - g[5]));
      if (desc[i].sz != (int)g[6]) err = std::max(err, 1.0);
    }
    std::printf("  cluster: cpp_n=%zu py_n=%d\n", desc.size(), ncg);
    report("connected_comp", err, 1e-12);
  }

  // --- EKF predict + adaptive update ---
  auto load9 = [&](const char* k) {
    Eigen::Matrix<double, 9, 9> M; auto& a = cur[k];
    for (int r = 0; r < 9; ++r) for (int c = 0; c < 9; ++c) M(r, c) = a[r * 9 + c];
    return M;
  };
  auto load3 = [&](const char* k) {
    Eigen::Matrix3d M; auto& a = cur[k];
    for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) M(r, c) = a[r * 3 + c];
    return M;
  };
  auto vec9 = [&](const char* k) {
    Eigen::Matrix<double, 9, 1> v; for (int i = 0; i < 9; ++i) v[i] = cur[k][i]; return v;
  };
  {
    EKFState s;
    s.x = vec9("EKF_IN_X"); s.P = load9("EKF_IN_P"); s.Q = load9("EKF_IN_Q"); s.R = load3("EKF_IN_R");
    ekf_predict(s, cur["EKF_DT"][0]);
    double ex = (s.x - vec9("EKF_PRED_X")).cwiseAbs().maxCoeff();
    double eP = (s.P - load9("EKF_PRED_P")).cwiseAbs().maxCoeff();
    report("ekf_predict_x", ex, 1e-12);
    report("ekf_predict_P", eP, 1e-12);

    Eigen::Vector3d z(cur["AEKF_Z"][0], cur["AEKF_Z"][1], cur["AEKF_Z"][2]);
    Eigen::Vector3d bbox(cur["AEKF_BBOX"][0], cur["AEKF_BBOX"][1], cur["AEKF_BBOX"][2]);
    aekf_update(s, z, cur["AEKF_ALPHA"][0], 12.3, bbox, true);
    report("aekf_x", (s.x - vec9("AEKF_X")).cwiseAbs().maxCoeff(), 1e-11);
    report("aekf_P", (s.P - load9("AEKF_P")).cwiseAbs().maxCoeff(), 1e-11);
    report("aekf_R", (s.R - load3("AEKF_R")).cwiseAbs().maxCoeff(), 1e-11);
    report("aekf_Q", (s.Q - load9("AEKF_Q")).cwiseAbs().maxCoeff(), 1e-11);
    Eigen::Vector3d bo(cur["AEKF_BBOX_OUT"][0], cur["AEKF_BBOX_OUT"][1], cur["AEKF_BBOX_OUT"][2]);
    report("aekf_bbox", (s.bbox - bo).cwiseAbs().maxCoeff(), 1e-12);
  }

  // --- polyfit + variance ---
  {
    auto& tv = cur["POLY_T"]; auto& yv = cur["POLY_Y"];
    Eigen::VectorXd t = Eigen::Map<Eigen::VectorXd>(tv.data(), tv.size());
    Eigen::VectorXd y = Eigen::Map<Eigen::VectorXd>(yv.data(), yv.size());
    auto relmax = [](const Eigen::VectorXd& a, const vector<double>& g) {
      double e = 0; for (int i = 0; i < a.size(); ++i) e = std::max(e, std::abs(a[i] - g[i]) / (std::abs(g[i]) + 1.0)); return e;
    };
    Eigen::VectorXd b3 = polyfit(t, y, 3);
    Eigen::VectorXd b5 = polyfit(t, y, 5);
    report("polyfit_deg3", relmax(b3, cur["POLY_B3"]), 1e-7);
    report("polyfit_deg5", relmax(b5, cur["POLY_B5"]), 1e-6);
    double v3 = calculate_variance(t, y, b3, 3), v5 = calculate_variance(t, y, b5, 5);
    report("variance_deg3", std::abs(v3 - cur["POLY_V3"][0]) / (cur["POLY_V3"][0] + 1.0), 1e-9);
    report("variance_deg5", std::abs(v5 - cur["POLY_V5"][0]) / (cur["POLY_V5"][0] + 1.0), 1e-9);
  }

  // --- predict_samples ---
  {
    Eigen::Matrix<double, 9, 1> st = vec9("PRED_STATE");
    int steps = (int)cur["PRED_STEPS"][0]; double dt = cur["PRED_DT"][0];
    auto s = predict_samples(st, steps, dt);
    auto cmp = [&](const vector<double>& v, const char* k) {
      auto& g = cur[k]; double e = 0;
      if (v.size() != g.size()) return 1e9;
      for (size_t i = 0; i < v.size(); ++i) e = std::max(e, std::abs(v[i] - g[i]));
      return e;
    };
    double e = std::max(std::max(cmp(s.t, "PRED_T"), cmp(s.x, "PRED_X")),
                        std::max(cmp(s.y, "PRED_Y"), cmp(s.z, "PRED_Z")));
    report("predict_samples", e, 1e-12);
  }

  std::printf("\nobstacle_tracker_kernels golden: %d fail\n", nfail);
  std::printf("%s\n", nfail == 0 ? "ALL PASS (C++ reproduces Python)" : "FAILED");
  return nfail == 0 ? 0 : 1;
}
