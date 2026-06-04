// Golden verification: C++ VoxelMapUtil geometric query helpers (voxel_map_helpers.hpp)
// must reproduce Python voxel_map.py outputs.
// Reads cpp/golden/voxel_map_helpers_cases.txt (gen_voxel_map_helpers_golden.py),
// rebuilds each map via read_map(...) with the SAME inputs+knobs, then asserts:
//   find_closest_free_point (float, exact world point)
//   has_non_free_neighbor   (exact bool)
//   is_blocked              (exact bool, val=100 and val=0)
//   line_of_sight_capsule   (exact bool, radii 0/1/2)
//   cells_world             (exact counts for free/occ/unk)
//   set_free_voxel_and_surroundings (exact n + exact OCC after carve)
// 不许欺骗、必须还原:对不上就 FAIL.
#include "sando_cpp/voxel_map.hpp"
#include "sando_cpp/voxel_map_helpers.hpp"
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

// Build a VoxelMapUtil from the per-case golden record and run read_map.
static void build_vm(sando::VoxelMapUtil& vm,
                     std::map<string, vector<double>>& cur) {
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
        vm.dyn_pred_samples[i][j] = Eigen::Vector3d(ps[off], ps[off + 1], ps[off + 2]);
        off += 3;
      }
    }
  }

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
}

int main(int argc, char** argv) {
  const char* path = (argc > 1) ? argv[1] : "golden/voxel_map_helpers_cases.txt";
  std::ifstream f(path);
  if (!f) { std::printf("CANNOT OPEN %s\n", path); return 2; }

  string line;
  std::map<string, vector<double>> cur;
  int ncase = 0, nfail = 0;
  double global_max_err = 0.0;     // float (find_closest_free_point world points)
  long bool_mismatches = 0;        // is_blocked / los / has_non_free_neighbor
  long count_mismatches = 0;       // cells_world counts
  long carve_mismatches = 0;       // set_free_voxel_and_surroundings OCC / n
  const double TOL = 1e-9;         // doubles all the way; cell-centers are exact

  auto process = [&]() {
    sando::VoxelMapUtil vm(cur["RES"][0]);
    build_vm(vm, cur);

    // ---- precarve: free regions applied BEFORE read-only queries (mirrors Python) ----
    int nprecarve = (int)cur["NPRECARVE"][0];
    for (int c = 0; c < nprecarve; ++c) {
      const vector<double>& pc = cur["PRECARVE_" + std::to_string(c)];
      Eigen::Vector3d center(pc[0], pc[1], pc[2]);
      sando::helpers::set_free_voxel_and_surroundings(vm, center, pc[3]);
    }

    double e = 0.0;
    bool case_ok = true;

    // ---- DIM sanity (exact) ----
    long dimX = (long)cur["DIM"][0], dimY = (long)cur["DIM"][1], dimZ = (long)cur["DIM"][2];
    if (vm.dimX != dimX || vm.dimY != dimY || vm.dimZ != dimZ) {
      std::printf("  DIM mismatch: got %ld %ld %ld expected %ld %ld %ld\n",
                  vm.dimX, vm.dimY, vm.dimZ, dimX, dimY, dimZ);
      case_ok = false;
    }

    // ---- sample points: find_closest_free_point + has_non_free_neighbor ----
    int npts = (int)cur["NPTS"][0];
    const vector<double>& pts = cur["PTS"];
    const vector<double>& fcfp = cur["FCFP"];
    const vector<double>& hnfn = cur["HNFN"];
    for (int i = 0; i < npts; ++i) {
      Eigen::Vector3d p(pts[i * 3], pts[i * 3 + 1], pts[i * 3 + 2]);
      Eigen::Vector3d w = sando::helpers::find_closest_free_point(vm, p);
      for (int j = 0; j < 3; ++j)
        e = std::max(e, std::fabs(w(j) - fcfp[i * 3 + j]));
      int got = sando::helpers::has_non_free_neighbor(vm, p) ? 1 : 0;
      if (got != (int)hnfn[i]) {
        bool_mismatches++;
        case_ok = false;
        if (bool_mismatches <= 8)
          std::printf("  HNFN mismatch pt %d (%.3f %.3f %.3f): got %d expected %d\n",
                      i, p(0), p(1), p(2), got, (int)hnfn[i]);
      }
    }

    // ---- segments: is_blocked (val=100, val=0) + line_of_sight_capsule ----
    int nseg = (int)cur["NSEG"][0];
    const vector<double>& segs = cur["SEGS"];
    const vector<double>& blk100 = cur["BLK100"];
    const vector<double>& blk0 = cur["BLK0"];
    const vector<double>& radii = cur["RADII"];
    const vector<double>& los = cur["LOS"];
    int nrad = (int)radii.size();
    for (int i = 0; i < nseg; ++i) {
      Eigen::Vector3d a(segs[i * 6 + 0], segs[i * 6 + 1], segs[i * 6 + 2]);
      Eigen::Vector3d b(segs[i * 6 + 3], segs[i * 6 + 4], segs[i * 6 + 5]);
      int g100 = sando::helpers::is_blocked(vm, a, b, 100) ? 1 : 0;
      int g0 = sando::helpers::is_blocked(vm, a, b, 0) ? 1 : 0;
      if (g100 != (int)blk100[i]) {
        bool_mismatches++;
        case_ok = false;
        if (bool_mismatches <= 8)
          std::printf("  BLK100 mismatch seg %d: got %d expected %d\n",
                      i, g100, (int)blk100[i]);
      }
      if (g0 != (int)blk0[i]) {
        bool_mismatches++;
        case_ok = false;
        if (bool_mismatches <= 8)
          std::printf("  BLK0 mismatch seg %d: got %d expected %d\n",
                      i, g0, (int)blk0[i]);
      }
      for (int r = 0; r < nrad; ++r) {
        int gl = sando::helpers::line_of_sight_capsule(vm, a, b, (int)radii[r]) ? 1 : 0;
        int expected = (int)los[i * nrad + r];
        if (gl != expected) {
          bool_mismatches++;
          case_ok = false;
          if (bool_mismatches <= 8)
            std::printf("  LOS mismatch seg %d radius %d: got %d expected %d\n",
                        i, (int)radii[r], gl, expected);
        }
      }
    }

    // ---- cells_world counts (exact int) ----
    const vector<double>& cw = cur["CWCOUNTS"];
    long g_free = (long)sando::helpers::cells_world(vm, 0).size();
    long g_occ = (long)sando::helpers::cells_world(vm, 100).size();
    long g_unk = (long)sando::helpers::cells_world(vm, -1).size();
    if (g_free != (long)cw[0] || g_occ != (long)cw[1] || g_unk != (long)cw[2]) {
      count_mismatches++;
      case_ok = false;
      std::printf("  CWCOUNTS mismatch: got free=%ld occ=%ld unk=%ld expected %ld %ld %ld\n",
                  g_free, g_occ, g_unk, (long)cw[0], (long)cw[1], (long)cw[2]);
    }

    // ---- set_free_voxel_and_surroundings: rebuild fresh, carve, compare OCC ----
    int ncarve = (int)cur["NCARVE"][0];
    // Re-read the carve blocks. They were stored as repeated keys; std::map keeps
    // only the LAST occurrence, so we instead stashed them flattened: re-parse from
    // the in-order vectors below. To handle repeated keys we collected them with
    // numeric suffix during parsing (see loop). Here keys are CARVE_*_<idx>.
    for (int c = 0; c < ncarve; ++c) {
      string sc = std::to_string(c);
      const vector<double>& cc = cur["CARVE_CENTER_" + sc];
      const vector<double>& cd = cur["CARVE_D_" + sc];
      const vector<double>& cn = cur["CARVE_N_" + sc];
      int ncell = (int)cur["CARVE_NCELL_" + sc][0];
      const vector<double>& cells = cur["CARVE_CELLS_" + sc];
      const vector<double>& occ = cur["CARVE_OCC_" + sc];

      sando::VoxelMapUtil vm2(cur["RES"][0]);
      build_vm(vm2, cur);
      // n check
      long n_got = sando::helpers::py_round_to_long(cd[0] / vm2.res + 0.5);
      if (n_got != (long)cn[0]) {
        carve_mismatches++;
        case_ok = false;
        if (carve_mismatches <= 8)
          std::printf("  CARVE_N mismatch carve %d: got %ld expected %ld (d=%.4f res=%.4f)\n",
                      c, n_got, (long)cn[0], cd[0], vm2.res);
      }
      Eigen::Vector3d center(cc[0], cc[1], cc[2]);
      sando::helpers::set_free_voxel_and_surroundings(vm2, center, cd[0]);
      for (int i = 0; i < ncell; ++i) {
        long x = (long)cells[i * 3], y = (long)cells[i * 3 + 1], z = (long)cells[i * 3 + 2];
        int got = (int)vm2.cmap[vm2.lin_index(x, y, z)];
        if (got != (int)occ[i]) {
          carve_mismatches++;
          case_ok = false;
          if (carve_mismatches <= 8)
            std::printf("  CARVE_OCC mismatch carve %d cell (%ld,%ld,%ld): got %d expected %d\n",
                        c, x, y, z, got, (int)occ[i]);
        }
      }
    }

    global_max_err = std::max(global_max_err, e);
    bool ok = case_ok && (e < TOL);
    std::printf("  case %d (dim %ldx%ldx%ld, npts=%d nseg=%d): float_max_err=%.3e  %s\n",
                ncase, vm.dimX, vm.dimY, vm.dimZ, npts, nseg, e, ok ? "PASS" : "FAIL");
    if (!ok) nfail++;
    ncase++;
  };

  // Parse: handle repeated CARVE_* / PRECARVE keys by appending a running index suffix.
  int carve_seen = 0;
  int precarve_seen = 0;
  while (std::getline(f, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line == "CASE") { cur.clear(); carve_seen = 0; precarve_seen = 0; continue; }
    if (line == "END") { process(); continue; }
    std::istringstream is(line);
    string key; is >> key;
    string rest; std::getline(is, rest);
    if (key == "PRECARVE") {
      cur["PRECARVE_" + std::to_string(precarve_seen)] = nums(rest);
      precarve_seen++;
    } else if (key == "CARVE_CENTER") {
      // start of a new carve block; assign suffix carve_seen
      string sc = std::to_string(carve_seen);
      cur["CARVE_CENTER_" + sc] = nums(rest);
    } else if (key == "CARVE_D") {
      cur["CARVE_D_" + std::to_string(carve_seen)] = nums(rest);
    } else if (key == "CARVE_N") {
      cur["CARVE_N_" + std::to_string(carve_seen)] = nums(rest);
    } else if (key == "CARVE_NCELL") {
      cur["CARVE_NCELL_" + std::to_string(carve_seen)] = nums(rest);
    } else if (key == "CARVE_CELLS") {
      cur["CARVE_CELLS_" + std::to_string(carve_seen)] = nums(rest);
    } else if (key == "CARVE_OCC") {
      cur["CARVE_OCC_" + std::to_string(carve_seen)] = nums(rest);
      carve_seen++;  // OCC is the last line of a carve block
    } else {
      cur[key] = nums(rest);
    }
  }

  std::printf("\nVoxelMapUtil helpers golden: %d cases, %d fail, float_max_err=%.3e (tol=%.0e)\n",
              ncase, nfail, global_max_err, TOL);
  std::printf("bool_mismatches=%ld, count_mismatches=%ld, carve_mismatches=%ld\n",
              bool_mismatches, count_mismatches, carve_mismatches);
  std::printf("%s\n", (nfail == 0) ? "ALL PASS (C++ reproduces Python)" : "FAILED");
  return nfail == 0 ? 0 : 1;
}
