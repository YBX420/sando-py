// Golden verification: C++ HGPPlanner must reproduce Python hgp_planner.py outputs.
// Reads cpp/golden/hgp_planner_cases.txt (produced by gen_hgp_planner_golden.py),
// rebuilds the SAME VoxelMapUtil via read_map(...) with identical knobs, builds the
// SAME HGPPlanner, runs the SAME plan(start, start_vel, goal, t, eps), and asserts:
//   - success flag matches,
//   - raw waypoint count + cell indices match EXACTLY; raw world coords abs 2e-4,
//   - processed waypoint count + cell indices match EXACTLY; world coords abs 2e-4,
//   - final_g matches (abs 2e-4).
// 不许欺骗、必须还原:对不上就 FAIL。
#include "sando_cpp/hgp_planner.hpp"
#include "sando_cpp/voxel_map.hpp"
#include "sando_cpp/voxel_map_helpers.hpp"
#include "sando_cpp/graph_search.hpp"
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
  const char* path = (argc > 1) ? argv[1] : "golden/hgp_planner_cases.txt";
  std::ifstream f(path);
  if (!f) { std::printf("CANNOT OPEN %s\n", path); return 2; }

  string line;
  std::map<string, vector<double>> cur;
  std::map<string, string> curstr;
  int ncase = 0, nfail = 0;
  long raw_node_mismatches = 0, proc_node_mismatches = 0, success_mismatches = 0;
  double world_max_err = 0.0, finalg_max_err = 0.0;
  const double TOL = 2e-4;

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

    // ---- predicted times / samples (empty here, parse schema) ----
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

    // ---- HGPPlanner config ----
    string planner = curstr.count("HGPPLANNER") ? curstr["HGPPLANNER"] : "astar_heat";
    const vector<double>& hf = cur["HGPFLOAT"];
    // HGP_FLOAT order: w_unknown, w_align, decay_len_cells, w_side,
    //                  min_len, min_turn, heat_weight, obstacle_soft_cost
    double w_unknown = hf[0], w_align = hf[1], decay_len_cells = hf[2], w_side = hf[3];
    double min_len = hf[4], min_turn = hf[5], heat_weight = hf[6], obstacle_soft_cost = hf[7];
    long los_cells = (long)cur["LOSCELLS"][0];

    sando::HGPPlanner hgp(planner, false, 3.0, 5.0, 20.0, 1000, w_unknown, w_align,
                          decay_len_cells, w_side, los_cells, min_len, min_turn,
                          heat_weight, obstacle_soft_cost);
    hgp.set_map_util(vm);
    long max_expand = (long)cur["MAXEXPAND"][0];
    hgp.set_max_expand(max_expand);

    double eps = cur["EPS"][0];
    Eigen::Vector3d start(cur["START"][0], cur["START"][1], cur["START"][2]);
    Eigen::Vector3d start_vel(cur["STARTVEL"][0], cur["STARTVEL"][1], cur["STARTVEL"][2]);
    Eigen::Vector3d goal(cur["GOAL"][0], cur["GOAL"][1], cur["GOAL"][2]);

    auto res = hgp.plan(start, start_vel, goal, 0.0, eps);
    bool success = res.first;
    double final_g = res.second;

    // ---- success flag ----
    bool exp_success = cur["SUCCESS"][0] != 0;
    if (success != exp_success) {
      success_mismatches++;
      case_ok = false;
      std::printf("  SUCCESS mismatch: got %d expected %d\n", (int)success, (int)exp_success);
    }

    // ---- final_g ----
    double eg = std::fabs(final_g - cur["FINALG"][0]);
    finalg_max_err = std::max(finalg_max_err, eg);
    if (eg >= TOL) case_ok = false;

    // ---- helper: compare a waypoint sequence (world abs tol + cell exact) ----
    auto cmp_seq = [&](const char* nkey, const char* wkey, const char* ckey,
                       const vector<Eigen::Vector3d>& got, long& node_mismatch,
                       const char* tag) {
      int nexp = (int)cur[nkey][0];
      if ((int)got.size() != nexp) {
        node_mismatch++;
        case_ok = false;
        std::printf("  %s length mismatch: got %d expected %d\n", tag,
                    (int)got.size(), nexp);
      }
      const vector<double>& w = cur.count(wkey) ? cur[wkey] : vector<double>();
      const vector<double>& c = cur.count(ckey) ? cur[ckey] : vector<double>();
      int ncmp = std::min((int)got.size(), nexp);
      for (int i = 0; i < ncmp; ++i) {
        // world coords abs tol
        for (int j = 0; j < 3; ++j) {
          double e = std::fabs(got[i](j) - w[i * 3 + j]);
          world_max_err = std::max(world_max_err, e);
          if (e >= TOL) {
            case_ok = false;
            if (node_mismatch <= 8)
              std::printf("  %s WORLD mismatch at %d.%d: got %.9g expected %.9g\n",
                          tag, i, j, got[i](j), w[i * 3 + j]);
          }
        }
        // cell index exact (via float_to_int)
        Eigen::Array<long, 3, 1> ci = vm.float_to_int(got[i]);
        long ex = (long)c[i * 3], ey = (long)c[i * 3 + 1], ez = (long)c[i * 3 + 2];
        if (ci(0) != ex || ci(1) != ey || ci(2) != ez) {
          node_mismatch++;
          case_ok = false;
          if (node_mismatch <= 8)
            std::printf("  %s CELL mismatch at %d: got (%ld,%ld,%ld) expected (%ld,%ld,%ld)\n",
                        tag, i, ci(0), ci(1), ci(2), ex, ey, ez);
        }
      }
    };

    cmp_seq("NRAW", "RAW", "RAWCELL", hgp.get_raw_path(), raw_node_mismatches, "RAW");
    cmp_seq("NPROC", "PROC", "PROCCELL", hgp.get_path(), proc_node_mismatches, "PROC");

    bool ok = case_ok;
    std::printf("  case %d (%s, dim %ldx%ldx%ld): nraw=%d nproc=%d success=%d final_g_err=%.3e  %s\n",
                ncase, planner.c_str(), vm.dimX, vm.dimY, vm.dimZ,
                (int)hgp.get_raw_path().size(), (int)hgp.get_path().size(),
                (int)success, eg, ok ? "PASS" : "FAIL");
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
    size_t p0 = rest.find_first_not_of(' ');
    string trimmed = (p0 == string::npos) ? string() : rest.substr(p0);
    curstr[key] = trimmed;
    cur[key] = nums(rest);
  }

  std::printf("\nHGPPlanner golden: %d cases, %d fail\n", ncase, nfail);
  std::printf("raw_node_mismatches=%ld, proc_node_mismatches=%ld, success_mismatches=%ld\n",
              raw_node_mismatches, proc_node_mismatches, success_mismatches);
  std::printf("world_max_err=%.3e, final_g_max_err=%.3e (tol=%.0e)\n",
              world_max_err, finalg_max_err, TOL);
  std::printf("%s\n", (nfail == 0) ? "ALL PASS (C++ reproduces Python raw+processed paths)" : "FAILED");
  return nfail == 0 ? 0 : 1;
}
