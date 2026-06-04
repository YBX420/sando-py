// Golden verification: C++ GraphSearch (heat-A*) must reproduce Python
// graph_search.py outputs. Reads cpp/golden/graph_search_cases.txt (produced by
// gen_graph_search_golden.py), rebuilds the SAME VoxelMapUtil via read_map(...)
// with identical knobs, runs the SAME GraphSearch.plan(...), and asserts:
//   - success flag matches,
//   - the raw integer path (cell node sequence) matches EXACTLY (node equality),
//   - the world path (cell centers) matches within float tol.
// 不许欺骗、必须还原:对不上就 FAIL。
#include "sando_cpp/graph_search.hpp"
#include "sando_cpp/voxel_map.hpp"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
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
  const char* path = (argc > 1) ? argv[1] : "golden/graph_search_cases.txt";
  std::ifstream f(path);
  if (!f) { std::printf("CANNOT OPEN %s\n", path); return 2; }

  string line;
  // store both numeric and raw-string forms (GSPLANNER is a string token)
  std::map<string, vector<double>> cur;
  std::map<string, string> curstr;
  int ncase = 0, nfail = 0;
  long node_mismatches = 0;
  long success_mismatches = 0;
  double world_max_err = 0.0;
  const double TOL = 2e-4;  // world path = cell centers via int_to_float (float-ish)

  auto process = [&]() {
    sando::VoxelMapUtil vm(cur["RES"][0]);

    // ---- VoxelMapUtil knobs ----
    const vector<double>& kb = cur["KBOOL"];
    vm.use_heat_map = kb[0] != 0;
    vm.dynamic_heat_enabled = kb[1] != 0;
    vm.dynamic_as_occupied_current = kb[2] != 0;
    vm.dynamic_as_occupied_future = kb[3] != 0;
    vm.static_heat_enabled = kb[4] != 0;
    vm.static_heat_boundary_only = kb[5] != 0;
    vm.static_heat_apply_on_unknown = kb[6] != 0;
    vm.static_heat_exclude_dynamic = kb[7] != 0;
    vm.use_soft_cost_obstacles = kb[8] != 0;

    const vector<double>& ki = cur["KINT"];
    vm.heat_p = (int)ki[0];
    vm.heat_q = (int)ki[1];
    vm.heat_num_samples = (int)ki[2];
    vm.static_heat_p = (int)ki[3];

    const vector<double>& kf = cur["KFLOAT"];
    vm.heat_alpha0 = kf[0];
    vm.heat_alpha1 = kf[1];
    vm.heat_tau_ratio = kf[2];
    vm.heat_gamma = kf[3];
    vm.heat_Hmax = kf[4];
    vm.dyn_base_inflation_m = kf[5];
    vm.dyn_heat_tube_radius_m = kf[6];
    vm.obst_max_vel = kf[7];
    vm.static_heat_alpha = kf[8];
    vm.static_heat_Hmax = kf[9];
    vm.static_heat_rmax_m = kf[10];
    vm.static_heat_default_radius_m = kf[11];
    vm.obstacle_soft_cost = kf[12];

    // ---- predicted times / samples (empty in these cases, but parse schema) ----
    const vector<double>& pt = cur["PREDTIMES"];
    int npt = (int)pt[0];
    if (npt >= 1) {
      vm.has_dyn_pred_times = true;
      vm.dyn_pred_times.assign(pt.begin() + 1, pt.begin() + 1 + npt);
    }
    const vector<double>& ps = cur["PREDSAMP"];
    if (!ps.empty() && ps[0] >= 0) {
      vm.has_dyn_pred_samples = true;
      int nob = (int)ps[0];
      vector<int> counts(nob);
      for (int i = 0; i < nob; ++i) counts[i] = (int)ps[1 + i];
      size_t off = 1 + nob;
      vm.dyn_pred_samples.resize(nob);
      for (int i = 0; i < nob; ++i) {
        vm.dyn_pred_samples[i].resize(counts[i]);
        for (int j = 0; j < counts[i]; ++j) {
          vm.dyn_pred_samples[i][j] =
              Eigen::Vector3d(ps[off], ps[off + 1], ps[off + 2]);
          off += 3;
        }
      }
    }

    // ---- read_map inputs ----
    const vector<double>& pr = cur["PARAMS"];
    long cells_x = (long)pr[0], cells_y = (long)pr[1], cells_z = (long)pr[2];
    Eigen::Vector3d center_map(pr[3], pr[4], pr[5]);
    double z_ground = pr[6], z_max = pr[7], inflation = pr[8], traj_max_time = pr[9];

    int nobst = (int)cur["NOBST"][0];
    vector<Eigen::Vector3d> obst_pos, obst_bbox;
    if (nobst > 0) {
      const vector<double>& op = cur["OBSTPOS"];
      const vector<double>& ob = cur["OBSTBBOX"];
      for (int i = 0; i < nobst; ++i) {
        obst_pos.emplace_back(op[i * 3], op[i * 3 + 1], op[i * 3 + 2]);
        obst_bbox.emplace_back(ob[i * 3], ob[i * 3 + 1], ob[i * 3 + 2]);
      }
    }
    int ncloud = (int)cur["NCLOUD"][0];
    vector<Eigen::Vector3d> cloud;
    if (ncloud > 0) {
      const vector<double>& cl = cur["CLOUD"];
      for (int i = 0; i < ncloud; ++i)
        cloud.emplace_back(cl[i * 3], cl[i * 3 + 1], cl[i * 3 + 2]);
    }

    vm.read_map(cells_x, cells_y, cells_z, center_map, cloud, z_ground, z_max,
                inflation, obst_pos, obst_bbox, traj_max_time);

    // ---- DIM cross-check (exact) ----
    long dimX = (long)cur["DIM"][0], dimY = (long)cur["DIM"][1], dimZ = (long)cur["DIM"][2];
    bool case_ok = true;
    if (vm.dimX != dimX || vm.dimY != dimY || vm.dimZ != dimZ) {
      std::printf("  DIM mismatch: got %ld %ld %ld expected %ld %ld %ld\n",
                  vm.dimX, vm.dimY, vm.dimZ, dimX, dimY, dimZ);
      case_ok = false;
    }

    // ---- GraphSearch config ----
    string planner = curstr.count("GSPLANNER") ? curstr["GSPLANNER"] : "astar_heat";
    const vector<double>& gsf = cur["GSFLOAT"];
    // GS_FLOAT order: eps, w_unknown, w_align, decay_len_cells, w_side,
    //                 heat_weight, obstacle_soft_cost
    double eps = gsf[0];
    sando::GraphSearch gs(vm, eps, planner, gsf[1], gsf[2], gsf[3], gsf[4],
                          false, gsf[5], gsf[6]);

    long start_i[3] = {(long)cur["START"][0], (long)cur["START"][1], (long)cur["START"][2]};
    long goal_i[3] = {(long)cur["GOAL"][0], (long)cur["GOAL"][1], (long)cur["GOAL"][2]};
    double initial_g = cur["INITG"][0];
    Eigen::Vector3d start_vel(cur["STARTVEL"][0], cur["STARTVEL"][1], cur["STARTVEL"][2]);
    long max_expand = (long)cur["MAXEXPAND"][0];

    bool success = gs.plan(start_i, goal_i, initial_g, start_vel, max_expand);

    // ---- compare success flag ----
    bool exp_success = cur["SUCCESS"][0] != 0;
    if (success != exp_success) {
      success_mismatches++;
      case_ok = false;
      std::printf("  SUCCESS mismatch: got %d expected %d\n", (int)success, (int)exp_success);
    }

    // ---- compare raw integer path (node sequence) EXACTLY ----
    int npath = (int)cur["NPATH"][0];
    const vector<double>& pth = cur.count("PATH") ? cur["PATH"] : vector<double>();
    if ((int)gs.path_.size() != npath) {
      node_mismatches++;
      case_ok = false;
      std::printf("  PATH length mismatch: got %d expected %d\n",
                  (int)gs.path_.size(), npath);
    }
    int ncmp = std::min((int)gs.path_.size(), npath);
    for (int i = 0; i < ncmp; ++i) {
      long gx = gs.path_[i]->x, gy = gs.path_[i]->y, gz = gs.path_[i]->z;
      long ex = (long)pth[i * 3], ey = (long)pth[i * 3 + 1], ez = (long)pth[i * 3 + 2];
      if (gx != ex || gy != ey || gz != ez) {
        node_mismatches++;
        case_ok = false;
        if (node_mismatches <= 8)
          std::printf("  NODE mismatch at %d: got (%ld,%ld,%ld) expected (%ld,%ld,%ld)\n",
                      i, gx, gy, gz, ex, ey, ez);
      }
    }

    // ---- compare world path (cell centers) within tol ----
    auto pw = gs.get_path_world();
    const vector<double>& pwexp = cur.count("PATHWORLD") ? cur["PATHWORLD"] : vector<double>();
    int nw = std::min((int)pw.size(), (int)(pwexp.size() / 3));
    double e = 0.0;
    for (int i = 0; i < nw; ++i)
      for (int j = 0; j < 3; ++j)
        e = std::max(e, std::fabs(pw[i](j) - pwexp[i * 3 + j]));
    world_max_err = std::max(world_max_err, e);
    if (e >= TOL) case_ok = false;

    bool ok = case_ok;
    std::printf("  case %d (%s, dim %ldx%ldx%ld): npath=%d success=%d world_err=%.3e  %s\n",
                ncase, planner.c_str(), vm.dimX, vm.dimY, vm.dimZ,
                (int)gs.path_.size(), (int)success, e, ok ? "PASS" : "FAIL");
    if (!ok) nfail++;
    ncase++;
  };

  while (std::getline(f, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line == "CASE") { cur.clear(); curstr.clear(); continue; }
    if (line == "END") { process(); continue; }
    std::istringstream is(line);
    string key; is >> key;
    string rest; std::getline(is, rest);
    // trim leading space
    size_t p0 = rest.find_first_not_of(' ');
    string trimmed = (p0 == string::npos) ? string() : rest.substr(p0);
    curstr[key] = trimmed;
    cur[key] = nums(rest);
  }

  std::printf("\nGraphSearch golden: %d cases, %d fail\n", ncase, nfail);
  std::printf("node_mismatches=%ld, success_mismatches=%ld, world_max_err=%.3e (tol=%.0e)\n",
              node_mismatches, success_mismatches, world_max_err, TOL);
  std::printf("%s\n", (nfail == 0) ? "ALL PASS (C++ reproduces Python path exactly)" : "FAILED");
  return nfail == 0 ? 0 : 1;
}
