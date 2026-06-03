// Golden verification: C++ VoxelMapUtil must reproduce Python voxel_map.py outputs.
// Reads cpp/golden/voxel_map_cases.txt (produced by gen_voxel_map_golden.py),
// rebuilds each map in C++ via read_map(...) with the SAME inputs+knobs, and asserts
// dims/origin, occupancy at sample cells, heat at sample cells, and float_to_int /
// int_to_float round-trips match. 不许欺骗、必须还原:对不上就 FAIL。
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
  const char* path = (argc > 1) ? argv[1] : "golden/voxel_map_cases.txt";
  std::ifstream f(path);
  if (!f) { std::printf("CANNOT OPEN %s\n", path); return 2; }

  string line;
  std::map<string, vector<double>> cur;
  int ncase = 0, nfail = 0;
  double global_max_err = 0.0;       // float (heat / origin)
  long occ_mismatches = 0;           // exact int comparison
  long f2i_mismatches = 0;           // exact int comparison
  const double TOL = 2e-4;           // float32 heat round-trips through float storage

  auto process = [&]() {
    sando::VoxelMapUtil vm(cur["RES"][0]);

    // ---- knobs ----
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

    // ---- predicted times / samples ----
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

    double e = 0.0;  // float error accumulator
    bool case_ok = true;

    // ---- DIM (exact) ----
    long dimX = (long)cur["DIM"][0], dimY = (long)cur["DIM"][1], dimZ = (long)cur["DIM"][2];
    if (vm.dimX != dimX || vm.dimY != dimY || vm.dimZ != dimZ) {
      std::printf("  DIM mismatch: got %ld %ld %ld expected %ld %ld %ld\n",
                  vm.dimX, vm.dimY, vm.dimZ, dimX, dimY, dimZ);
      case_ok = false;
    }
    // ---- ORIGIN (float) ----
    for (int i = 0; i < 3; ++i)
      e = std::max(e, std::fabs(vm.origin(i) - cur["ORIGIN"][i]));

    // ---- sample cells: occupancy (exact int) + heat (float) ----
    int ncell = (int)cur["NCELL"][0];
    const vector<double>& cells = cur["CELLS"];
    const vector<double>& occ = cur["OCC"];
    const vector<double>& heatv = cur["HEAT"];
    for (int i = 0; i < ncell; ++i) {
      long x = (long)cells[i * 3], y = (long)cells[i * 3 + 1], z = (long)cells[i * 3 + 2];
      int cval = (int)vm.cmap[vm.lin_index(x, y, z)];
      if (cval != (int)occ[i]) {
        occ_mismatches++;
        case_ok = false;
        if (occ_mismatches <= 6)
          std::printf("  OCC mismatch cell (%ld,%ld,%ld): got %d expected %d\n",
                      x, y, z, cval, (int)occ[i]);
      }
      double hval = vm.get_heat(x, y, z);
      e = std::max(e, std::fabs(hval - heatv[i]));
    }

    // ---- int_to_float round-trips (float) ----
    int nrt = (int)cur["NRT"][0];
    const vector<double>& rtcells = cur["RTCELLS"];
    const vector<double>& i2f = cur["I2F"];
    for (int i = 0; i < nrt; ++i) {
      Eigen::Array<long, 3, 1> p;
      p << (long)rtcells[i * 3], (long)rtcells[i * 3 + 1], (long)rtcells[i * 3 + 2];
      Eigen::Vector3d w = vm.int_to_float(p);
      for (int j = 0; j < 3; ++j)
        e = std::max(e, std::fabs(w(j) - i2f[i * 3 + j]));
    }

    // ---- float_to_int round-trips (exact int) ----
    int nf2i = (int)cur["NF2I"][0];
    const vector<double>& rtpoints = cur["RTPOINTS"];
    const vector<double>& f2i = cur["F2I"];
    for (int i = 0; i < nf2i; ++i) {
      Eigen::Vector3d pw(rtpoints[i * 3], rtpoints[i * 3 + 1], rtpoints[i * 3 + 2]);
      Eigen::Array<long, 3, 1> q = vm.float_to_int(pw);
      for (int j = 0; j < 3; ++j) {
        if (q(j) != (long)f2i[i * 3 + j]) {
          f2i_mismatches++;
          case_ok = false;
          if (f2i_mismatches <= 6)
            std::printf("  F2I mismatch pt %d axis %d: got %ld expected %ld\n",
                        i, j, (long)q(j), (long)f2i[i * 3 + j]);
        }
      }
    }

    global_max_err = std::max(global_max_err, e);
    bool ok = case_ok && (e < TOL);
    std::printf("  case %d (dim %ldx%ldx%ld, ncell=%d): float_max_err=%.3e  %s\n",
                ncase, vm.dimX, vm.dimY, vm.dimZ, ncell, e, ok ? "PASS" : "FAIL");
    if (!ok) nfail++;
    ncase++;
  };

  while (std::getline(f, line)) {
    // strip trailing CR (Windows line endings)
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line == "CASE") { cur.clear(); continue; }
    if (line == "END") { process(); continue; }
    std::istringstream is(line);
    string key; is >> key;
    string rest; std::getline(is, rest);
    cur[key] = nums(rest);
  }

  std::printf("\nVoxelMapUtil golden: %d cases, %d fail, float_max_err=%.3e (tol=%.0e)\n",
              ncase, nfail, global_max_err, TOL);
  std::printf("occ_mismatches=%ld, f2i_mismatches=%ld\n", occ_mismatches, f2i_mismatches);
  std::printf("%s\n", (nfail == 0) ? "ALL PASS (C++ reproduces Python)" : "FAILED");
  return nfail == 0 ? 0 : 1;
}
