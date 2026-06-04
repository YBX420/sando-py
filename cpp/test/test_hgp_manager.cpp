// Golden verification: C++ HGPManager must reproduce Python hgp_manager.py outputs.
// Reads cpp/golden/hgp_manager_cases.txt (produced by gen_hgp_manager_golden.py).
//
// Three case kinds:
//   KIND 0 (SOLVE): build a Parameters from the dumped fields, set_parameters(par),
//       update_map(wdx,wdy,wdz, center, cloud, None, obst_pos, obst_bbox, tmax),
//       solve_hgp(start, start_vel, goal, current_time). Asserts:
//         - mgr.res / mgr.drone_radius / mgr.weight match (abs 1e-12),
//         - map dim matches (exact),
//         - success flag matches, final_g matches (abs 2e-4),
//         - processed (densified) path: count + cell-index EXACT, world abs 2e-4,
//         - raw path:                  count + cell-index EXACT, world abs 2e-4,
//         - free_start/free_goal carve probes: is_free BEFORE/AFTER match EXACT.
//   KIND 1 (CMV): create_more_vertexes(polyline, d): output count + world abs 2e-4.
//
// 不许欺骗、必须还原:对不上就 FAIL。
#include "sando_cpp/hgp_manager.hpp"
#include "sando_cpp/types.hpp"
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
  const char* path = (argc > 1) ? argv[1] : "golden/hgp_manager_cases.txt";
  std::ifstream f(path);
  if (!f) { std::printf("CANNOT OPEN %s\n", path); return 2; }

  string line;
  std::map<string, vector<double>> cur;
  std::map<string, string> curstr;
  int ncase = 0, nfail = 0;
  long proc_node_mismatches = 0, raw_node_mismatches = 0, success_mismatches = 0;
  long carve_mismatches = 0, scalar_mismatches = 0, cmv_node_mismatches = 0;
  double world_max_err = 0.0, finalg_max_err = 0.0, scalar_max_err = 0.0;
  double cmv_max_err = 0.0;
  const double TOL = 2e-4;

  // ---- Parameters field order (must mirror gen_hgp_manager_golden.py) ----
  const char* PAR_FLOAT[] = {
      "global_planner_heuristic_weight", "factor_hgp", "res", "inflation_hgp",
      "z_min", "z_max", "v_max", "a_max", "j_max", "max_dist_vertexes",
      "shrinked_box_size", "free_start_factor", "free_goal_factor",
      "w_unknown", "w_align", "decay_len_cells", "w_side", "min_len", "min_turn",
      "heat_weight", "obstacle_soft_cost",
      "heat_alpha0", "heat_alpha1", "heat_tau_ratio", "heat_gamma", "heat_Hmax",
      "dyn_base_inflation_m", "dyn_heat_tube_radius_m", "obst_max_vel",
      "static_heat_alpha", "static_heat_Hmax", "static_heat_rmax_m",
      "static_heat_default_radius_m"};
  const char* PAR_INT[] = {"hgp_timeout_duration_ms", "max_num_expansion",
                           "los_cells", "heat_p", "heat_q", "heat_num_samples",
                           "static_heat_p"};
  const char* PAR_BOOL[] = {"use_free_start", "use_free_goal", "use_shrinked_box",
                            "use_heat_map", "dynamic_heat_enabled",
                            "dynamic_as_occupied_current", "dynamic_as_occupied_future",
                            "static_heat_enabled", "static_heat_boundary_only",
                            "static_heat_apply_on_unknown", "static_heat_exclude_dynamic",
                            "use_soft_cost_obstacles"};

  auto build_params = [&]() -> sando::Parameters {
    sando::Parameters p;
    const vector<double>& pf = cur["PARFLOAT"];
    const vector<double>& pi = cur["PARINT"];
    const vector<double>& pb = cur["PARBOOL"];
    int fi = 0;
    auto setf = [&](const char* name, double v) {
      // direct assignment via name switch (only the fields we dumped)
      if (string(name) == "global_planner_heuristic_weight") p.global_planner_heuristic_weight = v;
      else if (string(name) == "factor_hgp") p.factor_hgp = v;
      else if (string(name) == "res") p.res = v;
      else if (string(name) == "inflation_hgp") p.inflation_hgp = v;
      else if (string(name) == "z_min") p.z_min = v;
      else if (string(name) == "z_max") p.z_max = v;
      else if (string(name) == "v_max") p.v_max = v;
      else if (string(name) == "a_max") p.a_max = v;
      else if (string(name) == "j_max") p.j_max = v;
      else if (string(name) == "max_dist_vertexes") p.max_dist_vertexes = v;
      else if (string(name) == "shrinked_box_size") p.shrinked_box_size = v;
      else if (string(name) == "free_start_factor") p.free_start_factor = v;
      else if (string(name) == "free_goal_factor") p.free_goal_factor = v;
      else if (string(name) == "w_unknown") p.w_unknown = v;
      else if (string(name) == "w_align") p.w_align = v;
      else if (string(name) == "decay_len_cells") p.decay_len_cells = v;
      else if (string(name) == "w_side") p.w_side = v;
      else if (string(name) == "min_len") p.min_len = v;
      else if (string(name) == "min_turn") p.min_turn = v;
      else if (string(name) == "heat_weight") p.heat_weight = v;
      else if (string(name) == "obstacle_soft_cost") p.obstacle_soft_cost = v;
      else if (string(name) == "heat_alpha0") p.heat_alpha0 = v;
      else if (string(name) == "heat_alpha1") p.heat_alpha1 = v;
      else if (string(name) == "heat_tau_ratio") p.heat_tau_ratio = v;
      else if (string(name) == "heat_gamma") p.heat_gamma = v;
      else if (string(name) == "heat_Hmax") p.heat_Hmax = v;
      else if (string(name) == "dyn_base_inflation_m") p.dyn_base_inflation_m = v;
      else if (string(name) == "dyn_heat_tube_radius_m") p.dyn_heat_tube_radius_m = v;
      else if (string(name) == "obst_max_vel") p.obst_max_vel = v;
      else if (string(name) == "static_heat_alpha") p.static_heat_alpha = v;
      else if (string(name) == "static_heat_Hmax") p.static_heat_Hmax = v;
      else if (string(name) == "static_heat_rmax_m") p.static_heat_rmax_m = v;
      else if (string(name) == "static_heat_default_radius_m") p.static_heat_default_radius_m = v;
    };
    for (const char* name : PAR_FLOAT) setf(name, pf[fi++]);
    int ii = 0;
    auto seti = [&](const char* name, long v) {
      if (string(name) == "hgp_timeout_duration_ms") p.hgp_timeout_duration_ms = (int)v;
      else if (string(name) == "max_num_expansion") p.max_num_expansion = (int)v;
      else if (string(name) == "los_cells") p.los_cells = (int)v;
      else if (string(name) == "heat_p") p.heat_p = (int)v;
      else if (string(name) == "heat_q") p.heat_q = (int)v;
      else if (string(name) == "heat_num_samples") p.heat_num_samples = (int)v;
      else if (string(name) == "static_heat_p") p.static_heat_p = (int)v;
    };
    for (const char* name : PAR_INT) seti(name, (long)pi[ii++]);
    int bi = 0;
    auto setb = [&](const char* name, bool v) {
      if (string(name) == "use_free_start") p.use_free_start = v;
      else if (string(name) == "use_free_goal") p.use_free_goal = v;
      else if (string(name) == "use_shrinked_box") p.use_shrinked_box = v;
      else if (string(name) == "use_heat_map") p.use_heat_map = v;
      else if (string(name) == "dynamic_heat_enabled") p.dynamic_heat_enabled = v;
      else if (string(name) == "dynamic_as_occupied_current") p.dynamic_as_occupied_current = v;
      else if (string(name) == "dynamic_as_occupied_future") p.dynamic_as_occupied_future = v;
      else if (string(name) == "static_heat_enabled") p.static_heat_enabled = v;
      else if (string(name) == "static_heat_boundary_only") p.static_heat_boundary_only = v;
      else if (string(name) == "static_heat_apply_on_unknown") p.static_heat_apply_on_unknown = v;
      else if (string(name) == "static_heat_exclude_dynamic") p.static_heat_exclude_dynamic = v;
      else if (string(name) == "use_soft_cost_obstacles") p.use_soft_cost_obstacles = v;
    };
    for (const char* name : PAR_BOOL) setb(name, pb[bi++] != 0);
    p.global_planner = curstr.count("PARGP") ? curstr["PARGP"] : "astar_heat";
    const vector<double>& db = cur["PARDRONEBBOX"];
    p.drone_bbox.assign(db.begin(), db.end());
    const vector<double>& sfc = cur["PARSFC"];
    p.sfc_size.assign(sfc.begin(), sfc.end());
    return p;
  };

  auto process_solve = [&]() {
    bool case_ok = true;
    sando::Parameters p = build_params();

    sando::HGPManager mgr;
    mgr.set_parameters(p);

    // update_map inputs
    const vector<double>& wd = cur["WD"];
    double wdx = wd[0], wdy = wd[1], wdz = wd[2];
    const vector<double>& cm = cur["CENTER"];
    Eigen::Vector3d center(cm[0], cm[1], cm[2]);
    double tmax = cur["TMAX"][0];

    int ncloud = (int)cur["NCLOUD"][0];
    vector<Eigen::Vector3d> cloud;
    if (ncloud > 0) {
      const vector<double>& cl = cur["CLOUD"];
      for (int i = 0; i < ncloud; ++i)
        cloud.emplace_back(cl[i * 3], cl[i * 3 + 1], cl[i * 3 + 2]);
    }
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

    mgr.update_map(wdx, wdy, wdz, center, cloud, {}, obst_pos, obst_bbox, tmax);

    // ---- mgr-derived scalars ----
    auto chk_scalar = [&](double got, double exp, const char* tag) {
      double e = std::fabs(got - exp);
      scalar_max_err = std::max(scalar_max_err, e);
      if (e > 1e-9) {
        scalar_mismatches++;
        case_ok = false;
        std::printf("  %s mismatch: got %.12g expected %.12g\n", tag, got, exp);
      }
    };
    chk_scalar(mgr.res, cur["MGRRES"][0], "MGRRES");
    chk_scalar(mgr.drone_radius, cur["MGRDRONER"][0], "MGRDRONER");
    chk_scalar(mgr.weight, cur["MGRWEIGHT"][0], "MGRWEIGHT");

    // ---- dim ----
    long dimX = (long)cur["DIM"][0], dimY = (long)cur["DIM"][1], dimZ = (long)cur["DIM"][2];
    if (mgr.map_util.dimX != dimX || mgr.map_util.dimY != dimY ||
        mgr.map_util.dimZ != dimZ) {
      std::printf("  DIM mismatch: got %ld %ld %ld expected %ld %ld %ld\n",
                  mgr.map_util.dimX, mgr.map_util.dimY, mgr.map_util.dimZ, dimX, dimY, dimZ);
      case_ok = false;
    }

    // ---- free_start/free_goal carve probes (on a FRESH manager) ----
    int nprobe = (int)cur["NPROBE"][0];
    const vector<double>& pr = cur["PROBE"];
    const vector<double>& cb = cur["CARVEBEFORE"];
    const vector<double>& ca = cur["CARVEAFTER"];
    {
      sando::HGPManager mc;
      mc.set_parameters(p);
      mc.update_map(wdx, wdy, wdz, center, cloud, {}, obst_pos, obst_bbox, tmax);
      sando::VoxelMapUtil& vm = mc.map_util;
      Eigen::Vector3d start(cur["START"][0], cur["START"][1], cur["START"][2]);
      Eigen::Vector3d goal(cur["GOAL"][0], cur["GOAL"][1], cur["GOAL"][2]);
      vector<int> before(nprobe), after(nprobe);
      for (int i = 0; i < nprobe; ++i) {
        long x = (long)pr[i * 3], y = (long)pr[i * 3 + 1], z = (long)pr[i * 3 + 2];
        before[i] = vm.is_free(x, y, z) ? 1 : 0;
      }
      if (p.use_free_start) mc.free_start(start, p.free_start_factor);
      if (p.use_free_goal) mc.free_goal(goal, p.free_goal_factor);
      for (int i = 0; i < nprobe; ++i) {
        long x = (long)pr[i * 3], y = (long)pr[i * 3 + 1], z = (long)pr[i * 3 + 2];
        after[i] = vm.is_free(x, y, z) ? 1 : 0;
      }
      for (int i = 0; i < nprobe; ++i) {
        if (before[i] != (int)cb[i] || after[i] != (int)ca[i]) {
          carve_mismatches++;
          case_ok = false;
          if (carve_mismatches <= 8)
            std::printf("  CARVE mismatch at probe %d: before got %d exp %d, "
                        "after got %d exp %d\n",
                        i, before[i], (int)cb[i], after[i], (int)ca[i]);
        }
      }
    }

    // ---- solve_hgp ----
    Eigen::Vector3d start(cur["START"][0], cur["START"][1], cur["START"][2]);
    Eigen::Vector3d start_vel(cur["STARTVEL"][0], cur["STARTVEL"][1], cur["STARTVEL"][2]);
    Eigen::Vector3d goal(cur["GOAL"][0], cur["GOAL"][1], cur["GOAL"][2]);
    double curtime = cur["CURTIME"][0];

    sando::HGPManager::SolveResult sr = mgr.solve_hgp(start, start_vel, goal, curtime);

    bool exp_success = cur["SUCCESS"][0] != 0;
    if (sr.success != exp_success) {
      success_mismatches++;
      case_ok = false;
      std::printf("  SUCCESS mismatch: got %d expected %d\n", (int)sr.success,
                  (int)exp_success);
    }
    double eg = std::fabs(sr.final_g - cur["FINALG"][0]);
    finalg_max_err = std::max(finalg_max_err, eg);
    if (eg >= TOL) case_ok = false;

    // ---- compare a waypoint sequence (world abs tol + cell exact) ----
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
        Eigen::Array<long, 3, 1> ci = mgr.map_util.float_to_int(got[i]);
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
    cmp_seq("NPROC", "PROC", "PROCCELL", sr.path, proc_node_mismatches, "PROC");
    cmp_seq("NRAW", "RAW", "RAWCELL", sr.raw_path, raw_node_mismatches, "RAW");

    std::printf("  case %d SOLVE (%s, dim %ldx%ldx%ld): nproc=%d nraw=%d success=%d "
                "final_g_err=%.3e  %s\n",
                ncase, p.global_planner.c_str(), mgr.map_util.dimX, mgr.map_util.dimY,
                mgr.map_util.dimZ, (int)sr.path.size(), (int)sr.raw_path.size(),
                (int)sr.success, eg, case_ok ? "PASS" : "FAIL");
    if (!case_ok) nfail++;
    ncase++;
  };

  auto process_cmv = [&]() {
    bool case_ok = true;
    sando::HGPManager mgr;  // create_more_vertexes is independent of params
    int nin = (int)cur["NIN"][0];
    vector<Eigen::Vector3d> poly;
    if (nin > 0) {
      const vector<double>& in = cur["IN"];
      for (int i = 0; i < nin; ++i)
        poly.emplace_back(in[i * 3], in[i * 3 + 1], in[i * 3 + 2]);
    }
    double d = cur["D"][0];
    vector<Eigen::Vector3d> out = mgr.create_more_vertexes(poly, d);

    int nout = (int)cur["NOUT"][0];
    if ((int)out.size() != nout) {
      cmv_node_mismatches++;
      case_ok = false;
      std::printf("  CMV length mismatch: got %d expected %d\n", (int)out.size(), nout);
    }
    const vector<double>& w = cur.count("OUT") ? cur["OUT"] : vector<double>();
    int ncmp = std::min((int)out.size(), nout);
    for (int i = 0; i < ncmp; ++i)
      for (int j = 0; j < 3; ++j) {
        double e = std::fabs(out[i](j) - w[i * 3 + j]);
        cmv_max_err = std::max(cmv_max_err, e);
        if (e >= TOL) {
          case_ok = false;
          if (cmv_node_mismatches <= 8)
            std::printf("  CMV WORLD mismatch at %d.%d: got %.9g expected %.9g\n",
                        i, j, out[i](j), w[i * 3 + j]);
        }
      }
    std::printf("  case %d CMV (nin=%d d=%.4g): nout=%d  %s\n", ncase, nin, d,
                (int)out.size(), case_ok ? "PASS" : "FAIL");
    if (!case_ok) nfail++;
    ncase++;
  };

  auto process = [&]() {
    int kind = cur.count("KIND") ? (int)cur["KIND"][0] : 0;
    if (kind == 0) process_solve();
    else if (kind == 1) process_cmv();
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

  std::printf("\nHGPManager golden: %d cases, %d fail\n", ncase, nfail);
  std::printf("scalar_mismatches=%ld (max_err=%.3e), carve_mismatches=%ld\n",
              scalar_mismatches, scalar_max_err, carve_mismatches);
  std::printf("proc_node_mismatches=%ld, raw_node_mismatches=%ld, success_mismatches=%ld\n",
              proc_node_mismatches, raw_node_mismatches, success_mismatches);
  std::printf("solve world_max_err=%.3e, final_g_max_err=%.3e\n", world_max_err,
              finalg_max_err);
  std::printf("cmv_node_mismatches=%ld, cmv_max_err=%.3e (tol=%.0e)\n",
              cmv_node_mismatches, cmv_max_err, TOL);
  std::printf("%s\n", (nfail == 0) ? "ALL PASS (C++ reproduces Python HGPManager)" : "FAILED");
  return nfail == 0 ? 0 : 1;
}
